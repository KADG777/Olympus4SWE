"""Authentication guard contract for BlackNote.

Invariant implemented here: **INV-auth-session**.

* Every write operation (create/update/delete note, like, favorite, comment,
  moderation action, profile update) must be executed by an authenticated
  principal; anonymous access yields ``AUTH_REQUIRED`` (401).
* A session token is an opaque **UUIDv4** string (see :mod:`blacknote.contracts.ids`);
  the token is *not* a JWT and carries no data by itself — it must be resolved
  server side through a :class:`SessionResolver`.
* A resolved session may carry an expiry; expired sessions yield
  ``SESSION_EXPIRED`` (401) and the caller must re-authenticate.
* The guard never reveals whether a token exists: unknown and malformed tokens
  both produce ``AUTH_REQUIRED``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Final, Protocol, runtime_checkable

from .errors import ContractError, ErrorCode
from .ids import is_valid_id, new_id
from .roles import Role, parse_role
from .timeutil import is_expired

__all__ = [
    "ANONYMOUS",
    "BEARER",
    "Principal",
    "ResolverLike",
    "SessionResolver",
    "require_auth",
    "require_principal",
    "is_authenticated",
    "resolve_session",
    "anonymous_token",
]

#: A missing actor is always represented by ``None`` (never a fake user id).
ANONYMOUS: Final[None] = None
BEARER: Final[str] = "Bearer"


@dataclass(frozen=True)
class Principal:
    """The authenticated identity produced by :func:`require_auth`."""

    user_id: str
    token: str = ""
    role: Role = Role.USER
    issued_at: str | None = None
    expires_at: str | None = None
    email: str | None = None
    nickname: str | None = None

    def __post_init__(self) -> None:
        if not is_valid_id(self.user_id):
            # A principal without a valid id is by definition not authenticated.
            raise ContractError(
                ErrorCode.AUTH_REQUIRED, "principal requires a valid user id"
            )
        object.__setattr__(self, "role", parse_role(self.role))
        if self.token and not is_valid_id(self.token):
            raise ContractError(ErrorCode.INVALID_TOKEN, "session token must be a UUIDv4")

    # -- convenience ----------------------------------------------------------
    @property
    def is_anonymous(self) -> bool:
        return False

    @property
    def is_admin(self) -> bool:
        return self.role is Role.ADMIN

    def is_expired(self, *, now: Any = None) -> bool:
        return is_expired(self.expires_at, now=now)

    def to_dict(self) -> dict[str, Any]:
        """Public shape — deliberately omits the bearer token."""
        return {
            "user_id": self.user_id,
            "role": str(self.role),
            "issued_at": self.issued_at,
            "expires_at": self.expires_at,
        }

    def __str__(self) -> str:  # pragma: no cover - never leak the token
        return f"Principal(user_id={self.user_id!r}, role={str(self.role)!r})"


@runtime_checkable
class SessionResolver(Protocol):
    """Ports implemented by the account layer to map token → principal."""

    def resolve(self, token: str) -> "Principal | None":  # pragma: no cover - protocol
        ...


ResolverLike = "SessionResolver | Callable[[str], Principal | None] | None"


def _resolve(token: str, resolver: Any) -> Principal | None:
    if resolver is None:
        raise ContractError(
            ErrorCode.INTERNAL_ERROR,
            "require_auth() needs a session resolver to verify credentials",
        )
    principal = resolver.resolve(token) if hasattr(resolver, "resolve") else resolver(token)
    if principal is None:
        return None
    if isinstance(principal, Principal):
        return principal
    if isinstance(principal, dict):  # tolerate repository rows
        payload = dict(principal)
        payload.setdefault("user_id", payload.pop("id", None))
        return Principal(**payload)
    user_id = getattr(principal, "user_id", None)
    if user_id is None:
        raise ContractError(ErrorCode.INTERNAL_ERROR, "resolver returned an unusable principal")
    return Principal(
        user_id=user_id,
        token=token,
        role=getattr(principal, "role", Role.USER),
        issued_at=getattr(principal, "issued_at", None),
        expires_at=getattr(principal, "expires_at", None),
    )


def resolve_session(
    token: str | None,
    resolver: "SessionResolver | Callable[[str], Principal | None] | None" = None,
    *,
    now: Any = None,
) -> Principal | None:
    """Non-raising variant: ``None`` for anonymous / unknown / expired tokens."""
    try:
        return require_auth(token, resolver, now=now)
    except ContractError:
        return None


def require_auth(
    token: str | None,
    resolver: "SessionResolver | Callable[[str], Principal | None] | None" = None,
    *,
    now: Any = None,
) -> Principal:
    """Guard used by **every** write endpoint (INV-auth-session).

    Raises :class:`ContractError` with ``AUTH_REQUIRED`` when the token is
    missing / malformed / unknown, and ``SESSION_EXPIRED`` when a known session
    has passed its expiry.  ``resolver`` is mandatory for a real check (the
    contract has no session store of its own): passing ``None`` raises
    ``INTERNAL_ERROR`` so a mis-wired route can never silently authorise.
    """
    if token is None or (isinstance(token, str) and not token.strip()):
        raise ContractError(ErrorCode.AUTH_REQUIRED, "a session token is required")
    if not isinstance(token, str):
        raise ContractError(ErrorCode.AUTH_REQUIRED, "session token must be a string")
    if not is_valid_id(token.strip()):
        # Unknown-format tokens are indistinguishable from missing ones.
        raise ContractError(ErrorCode.AUTH_REQUIRED, "the session token is not recognised")
    principal = _resolve(token.strip(), resolver)
    if principal is None:
        raise ContractError(ErrorCode.AUTH_REQUIRED, "the session is no longer valid")
    if principal.token != token.strip():
        principal = Principal(
            user_id=principal.user_id,
            token=token.strip(),
            role=principal.role,
            issued_at=principal.issued_at,
            expires_at=principal.expires_at,
        )
    if principal.is_expired(now=now):
        raise ContractError(ErrorCode.SESSION_EXPIRED, "the session has expired")
    return principal


def require_principal(principal: Principal | None, *, reason: str | None = None) -> Principal:
    """Guard for layers that already resolved the session (middleware style)."""
    if principal is None:
        raise ContractError(ErrorCode.AUTH_REQUIRED, reason)
    if not isinstance(principal, Principal):
        raise ContractError(ErrorCode.INTERNAL_ERROR, "expected a Principal instance")
    if principal.is_expired():
        raise ContractError(ErrorCode.SESSION_EXPIRED, "the session has expired")
    return principal


def is_authenticated(
    token: str | None,
    resolver: "SessionResolver | Callable[[str], Principal | None] | None" = None,
    *,
    now: Any = None,
) -> bool:
    """Boolean helper used by read paths (``get_note(token_or_null, ...)``)."""
    return resolve_session(token, resolver, now=now) is not None


def anonymous_token() -> str:
    """Helper for tests/fixtures: a syntactically valid, unbound token."""
    return new_id()


#: Re-exported so fixtures do not have to import two modules.
NEW_TOKEN: Final[Callable[[], str]] = new_id
