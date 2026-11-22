"""Ports the note layer needs from the account layer (dependency injection).

``mod-note`` must not import the account implementation (Wave 1 runs in
parallel, and the dependency direction is enforced by the frozen contracts), so
everything it needs from an identity is described by two tiny ports that the web
layer wires at start-up:

:class:`AccountDirectory`
    nickname lookup (``NoteView.author_nickname``) and the follow graph needed
    by the ``following_only`` feed filter.
Session resolver adaptation
    anything able to turn a bearer token into a
    :class:`~blacknote.contracts.auth_guard.Principal`.  ``require_auth`` from
    the contract does the actual gating (INV-auth-session); this module only
    adapts the shape of the injected dependency so the guard always receives a
    usable resolver.

Tests (and dev wiring) may use :class:`InMemoryAccountDirectory`; production
wiring can wrap the account service with :func:`from_account_service`, which
duck-types ``get_profile`` / ``current_user`` and therefore never imports
``blacknote.account``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any, Protocol, runtime_checkable

from blacknote.contracts.auth_guard import Principal
from blacknote.contracts.errors import ContractError, ErrorCode
from blacknote.contracts.ids import is_valid_id
from blacknote.contracts.roles import Role, parse_role

__all__ = [
    "AccountDirectory",
    "InMemoryAccountDirectory",
    "NullAccountDirectory",
    "AccountServiceDirectory",
    "from_account_service",
    "adapt_session_resolver",
    "ANONYMOUS_NICKNAME",
]

ANONYMOUS_NICKNAME = "Anonymous"


@runtime_checkable
class AccountDirectory(Protocol):
    """Read-only identity port consumed by ``mod-note``."""

    def nickname_of(self, user_id: str) -> str:  # pragma: no cover - protocol
        ...

    def following_ids(self, user_id: str) -> Iterable[str]:  # pragma: no cover - protocol
        ...


class NullAccountDirectory:
    """Directory without data: nicknames fall back, nobody is followed."""

    def nickname_of(self, user_id: str) -> str:  # noqa: ARG002 - interface parity
        return ANONYMOUS_NICKNAME

    def following_ids(self, user_id: str) -> tuple[str, ...]:  # noqa: ARG002
        return ()


class InMemoryAccountDirectory(NullAccountDirectory):
    """Dictionary backed directory used by tests and the dev wiring."""

    def __init__(
        self,
        nicknames: Mapping[str, str] | None = None,
        following: Mapping[str, Iterable[str]] | None = None,
    ) -> None:
        self._nicknames: dict[str, str] = {str(k): str(v) for k, v in (nicknames or {}).items()}
        self._following: dict[str, tuple[str, ...]] = {
            str(key): tuple(str(item) for item in value)
            for key, value in (following or {}).items()
        }

    # -- writes (test / wiring helpers) ---------------------------------------
    def set_nickname(self, user_id: str, nickname: str) -> None:
        self._nicknames[str(user_id)] = str(nickname)

    def follow(self, user_id: str, target_user_id: str) -> None:
        current = list(self._following.get(str(user_id), ()))
        if str(target_user_id) not in current:
            current.append(str(target_user_id))
        self._following[str(user_id)] = tuple(current)

    # -- read port ------------------------------------------------------------
    def nickname_of(self, user_id: str) -> str:
        return self._nicknames.get(str(user_id), ANONYMOUS_NICKNAME)

    def following_ids(self, user_id: str) -> tuple[str, ...]:
        return self._following.get(str(user_id), ())


class AccountServiceDirectory(NullAccountDirectory):
    """Adapter over an object exposing ``get_profile`` / ``following_ids``.

    Duck typing keeps ``mod-note`` free of any sibling import: the account
    service may be anything with those methods.
    """

    def __init__(self, account: object) -> None:
        self._account = account

    def nickname_of(self, user_id: str) -> str:
        getter = getattr(self._account, "get_profile", None)
        if getter is None or not is_valid_id(user_id):
            return ANONYMOUS_NICKNAME
        try:
            profile = getter(user_id)
        except ContractError:
            return ANONYMOUS_NICKNAME
        if profile is None or isinstance(profile, ContractError):
            return ANONYMOUS_NICKNAME
        if isinstance(profile, Mapping):
            nickname = profile.get("nickname")
        else:
            nickname = getattr(profile, "nickname", None)
        return str(nickname) if nickname else ANONYMOUS_NICKNAME

    def following_ids(self, user_id: str) -> tuple[str, ...]:
        for name in ("following_ids", "list_following", "get_following"):
            getter = getattr(self._account, name, None)
            if getter is None:
                continue
            try:
                return tuple(str(item) for item in (getter(user_id) or ()))
            except (ContractError, TypeError):  # pragma: no cover - defensive
                return ()
        return ()


def from_account_service(account: object) -> AccountDirectory:
    """Build an :class:`AccountDirectory` out of any account facade."""
    if isinstance(account, NullAccountDirectory):
        return account  # type: ignore[return-value]
    return AccountServiceDirectory(account)


def _to_principal(candidate: object, token: str) -> Principal | None:
    """Normalise whatever the injected dependency returns into a ``Principal``."""
    if candidate is None or isinstance(candidate, ContractError):
        return None
    if isinstance(candidate, Principal):
        return candidate
    if isinstance(candidate, Mapping):
        payload = dict(candidate)
        user_id = payload.pop("user_id", None) or payload.pop("id", None)
        payload.pop("token", None)
        if not is_valid_id(user_id):
            return None
        try:
            role = parse_role(payload.pop("role", Role.USER))
        except ContractError:
            role = Role.USER
        return Principal(
            user_id=str(user_id),
            token=token,
            role=role,
            issued_at=payload.get("issued_at"),
            expires_at=payload.get("expires_at"),
            nickname=payload.get("nickname"),
        )
    user_id = getattr(candidate, "user_id", None) or getattr(candidate, "id", None)
    if not is_valid_id(user_id):
        return None
    return Principal(
        user_id=str(user_id),
        token=token,
        role=getattr(candidate, "role", Role.USER),
        issued_at=getattr(candidate, "issued_at", None),
        expires_at=getattr(candidate, "expires_at", None),
        nickname=getattr(candidate, "nickname", None),
    )


def adapt_session_resolver(resolver: object) -> Callable[[str], Principal | None] | None:
    """Turn any plausible session port into a ``contracts.require_auth`` resolver.

    Accepts ``None`` (no auth wiring — the contract guard then fails closed with
    ``INTERNAL_ERROR``), an object with ``resolve(token)``, an account facade
    with ``current_user(token)``, a ``token → session`` mapping (test fixtures
    and the account layer's session table), a fixed :class:`Principal`, or a
    plain callable.  The returned callable always produces a
    :class:`Principal` or ``None``, so the shared guard keeps ownership of the
    ``AUTH_REQUIRED`` / ``SESSION_EXPIRED`` decision.
    """
    if resolver is None:
        return None
    if isinstance(resolver, Principal):  # a fixed identity (tests / admin shell)
        fixed = resolver
        return lambda _token: fixed

    if hasattr(resolver, "resolve"):
        raw_resolve = resolver.resolve  # type: ignore[attr-defined]
    elif hasattr(resolver, "current_user"):
        raw_resolve = resolver.current_user  # type: ignore[attr-defined]
    elif isinstance(resolver, Mapping):
        table = dict(resolver)

        def raw_resolve(token: str, _table: dict = table) -> Any:  # type: ignore[misc]
            return _table.get(token)

    elif callable(resolver):
        raw_resolve = resolver
    else:
        raise ContractError(
            ErrorCode.CONFIG_ERROR,
            "session resolver must expose resolve()/current_user() or be callable",
        )

    def _resolve(token: str) -> Principal | None:
        return _to_principal(raw_resolve(token), token)

    return _resolve
