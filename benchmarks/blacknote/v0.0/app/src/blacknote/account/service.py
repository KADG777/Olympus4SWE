"""``mod-account`` — the account & identity service (``IF-account-*``).

Frozen signatures (identical parameter names to
:meth:`blacknote.contracts.interfaces.AccountAuthApi` /
:class:`~blacknote.contracts.interfaces.AccountProfileApi`):

``register(email, password)``      VALIDATION_ERROR → EMAIL_TAKEN → session
``login(email, password)``         INVALID_CREDENTIALS (no user enumeration)
``logout(token)``                  AUTH_REQUIRED once the session is revoked
``current_user(token)``            the session behind "sign in → home page"
``get_profile(user_id)``           public profile (USER_NOT_FOUND on miss)
``update_profile(token, patch)``   self-only PATCH, immediately readable

Behavioural guarantees pinned here:

* **contract errors, never exceptions** — the six public operations convert the
  internal :class:`~blacknote.contracts.errors.ContractError` into an
  :class:`~blacknote.contracts.errors.ErrorEnvelope` return value, so callers
  discriminate with :func:`~blacknote.contracts.errors.is_error`;
* **INV-auth-session** — passwords are salted hashes
  (:mod:`blacknote.account.security`), session tokens are opaque UUIDv4s, and a
  successful register/login opens a session immediately;
* **INV-role-check** — the resolver hands out a :class:`Principal` carrying the
  role, ``role`` is restricted to ``{user, admin}`` by schema and by
  :func:`~blacknote.contracts.roles.parse_role`, and admin-only routes go
  through :func:`blacknote.account.guards.require_admin_session`;
* **INV-config-db** — persistence is the SQLite file named by
  ``Settings.database_url``; a restart re-reads the same rows.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable, Mapping
from datetime import timedelta
from typing import Any, Final, TypeVar

from blacknote.contracts.auth_guard import Principal
from blacknote.contracts.config import Settings, load_settings
from blacknote.contracts.errors import ContractError, ErrorEnvelope, ErrorCode
from blacknote.contracts.ids import is_valid_id, new_id, normalize_id
from blacknote.contracts.interfaces import (
    ProfilePatch,
    ProfileView,
    SessionToken,
    UserSummary,
)
from blacknote.contracts.roles import Role
from blacknote.contracts.timeutil import now_iso, now_utc, to_iso
from blacknote.contracts.validation import DEFAULT_PAGE_SIZE, check_fields, normalize_email

from .db import Database
from .guards import require_admin_session, require_session
from .models import ProfileDetails, SessionGrant, UserRecord, profile_stats
from .patch import ProfilePatchPayload, normalise_profile_patch
from .repository import AccountRepository, session_is_live
from .security import (
    derive_username,
    generate_session_token,
    hash_password,
    password_needs_rehash,
    verify_password,
)
from .seed import seed_admin_account

__all__ = [
    "AccountService",
    "MAX_SESSION_RETRY",
    "ProfileStatsProvider",
    "create_account_service",
    "envelope_boundary",
]

logger = logging.getLogger("blacknote.account")

F = TypeVar("F", bound=Callable[..., Any])
MAX_SESSION_RETRY: Final[int] = 3

#: ``user_id -> {"notes_count": int, "likes_received": int, "favorites_received": int}``;
#: implemented by mod-web from mod-note / mod-interaction, never imported here.
ProfileStatsProvider = Callable[[str], Mapping[str, Any]]


def envelope_boundary(func: F) -> F:
    """Turn a raising service method into a ``payload | ErrorEnvelope`` method."""

    @functools.wraps(func)
    def inner(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except ContractError as exc:
            return exc.envelope
        except (TypeError, ValueError) as exc:  # defensive: never leak a traceback
            logger.exception("unexpected failure in %s", func.__name__)
            return ErrorEnvelope(ErrorCode.INTERNAL_ERROR, str(exc))

    return inner  # type: ignore[return-value]


class AccountService:
    """Email + password accounts, server side sessions and profiles."""

    def __init__(
        self,
        repository: AccountRepository,
        settings: Settings | None = None,
        *,
        stats: ProfileStatsProvider | None = None,
    ) -> None:
        """``stats`` lets the wiring layer inject the ``ProfileView`` counters.

        ``notes_count`` / ``likes_received`` / ``favorites_received`` belong to
        ``mod-note`` / ``mod-interaction``; :mod:`mod-account` must not read
        their tables, so it only forwards what the composition root provides.
        Without a provider the counters stay ``0`` (the contract default).
        """
        if not isinstance(repository, AccountRepository):
            raise ContractError(
                ErrorCode.CONFIG_ERROR, "AccountService requires an AccountRepository"
            )
        self.repository = repository
        self.settings = settings if settings is not None else repository.db.settings
        self.db = repository.db
        self._stats = stats

    # ------------------------------------------------------------------ #
    # SessionResolver port (INV-auth-session) — used by every other layer
    # ------------------------------------------------------------------ #
    def resolve(self, token: str) -> Principal | None:
        """Map a bearer token onto a live :class:`Principal` (``None`` if not).

        Unknown, malformed and expired tokens are indistinguishable from the
        outside, so the guard cannot be used to probe for session existence.
        """
        if not isinstance(token, str) or not is_valid_id(token.strip()):
            return None
        grant = self.repository.get_session(token.strip())
        if grant is None:
            return None
        if not session_is_live(grant):
            # Lazy GC: an expired credential is dropped as soon as it is used.
            self.repository.delete_session(grant.token)
            return None
        record = self.repository.get_user_by_id(grant.user_id)
        if record is None:  # orphaned session → drop it
            self.repository.delete_session(grant.token)
            return None
        return Principal(
            user_id=record.id,
            token=grant.token,
            role=record.role,
            issued_at=grant.created_at,
            expires_at=grant.expires_at,
            email=None,  # the principal travels to templates/logs: no e-mail
            nickname=record.username,
        )

    def authenticate(self, token: str | None) -> Principal:
        """Raising variant of :meth:`resolve` (``AUTH_REQUIRED`` / ``SESSION_EXPIRED``)."""
        return require_session(token, self)

    def is_authenticated(self, token: str | None) -> bool:
        return self.resolve(token if isinstance(token, str) else "") is not None

    def record_audit(
        self,
        event: str,
        *,
        actor_id: str | None = None,
        user_id: str | None = None,
        reason: str | None = None,
    ) -> None:
        """Security audit sink used by :mod:`blacknote.account.guards`."""
        self.repository.record_audit(
            event, actor_id=actor_id, user_id=user_id, reason=reason
        )

    # ------------------------------------------------------------------ #
    # IF-account-auth
    # ------------------------------------------------------------------ #
    @envelope_boundary
    def register(self, email: str, password: str) -> SessionToken | ErrorEnvelope:
        """Create an account and open a session for it immediately.

        ``VALIDATION_ERROR`` (bad/short input) → ``EMAIL_TAKEN`` (uniqueness) →
        ``SessionToken``.  Nothing is written when validation fails.
        """
        clean = self._validate_credentials(email, password, schema="register")
        email_value, password_value = clean
        if self.repository.email_exists(email_value):
            raise ContractError(
                ErrorCode.EMAIL_TAKEN,
                None,
                {"field": "email", "reason": "email must be unique"},
            )
        record = self._new_user(email_value, password_value)
        self.repository.insert_user(record)
        return self._issue_session(record)

    @envelope_boundary
    def login(self, email: str, password: str) -> SessionToken | ErrorEnvelope:
        """Verify credentials and open a fresh session.

        A missing account and a wrong password return the very same
        ``INVALID_CREDENTIALS`` envelope: existence is never disclosed.
        """
        clean = self._validate_credentials(email, password, schema="login")
        email_value, password_value = clean
        record = self.repository.get_user_by_email(email_value)
        if record is None or not verify_password(password_value, record.password_hash):
            raise ContractError(ErrorCode.INVALID_CREDENTIALS, None, {"field": "password"})
        if password_needs_rehash(record.password_hash, iterations=self.iterations):
            record = self.repository.update_user(
                record, changes={"password_hash": self._digest(password_value)}
            )
        return self._issue_session(record)

    @envelope_boundary
    def logout(self, token: str) -> None | ErrorEnvelope:
        """Revoke the presented session; a second use yields ``AUTH_REQUIRED``."""
        principal = require_session(token, self)
        self.repository.delete_session(principal.token)
        return None

    @envelope_boundary
    def current_user(self, token: str) -> UserSummary | ErrorEnvelope:
        """Resolve the session into the signed-in identity (home page header)."""
        principal = require_session(token, self)
        record = self.repository.get_user_by_id(principal.user_id)
        if record is None:  # pragma: no cover - the resolver just loaded it
            raise ContractError(ErrorCode.AUTH_REQUIRED)
        return record.summary()

    # ------------------------------------------------------------------ #
    # IF-account-profile
    # ------------------------------------------------------------------ #
    @envelope_boundary
    def get_profile(self, user_id: str) -> ProfileView | ErrorEnvelope:
        """Public profile of *user_id* (``VALIDATION_ERROR`` / ``USER_NOT_FOUND``)."""
        record = self._require_user(user_id)
        return record.profile_view(stats=self.profile_stats(record.id))

    @envelope_boundary
    def update_profile(
        self, token: str, patch: ProfilePatch | Mapping[str, Any] | Any
    ) -> ProfileView | ErrorEnvelope:
        """Self-service profile PATCH, readable immediately afterwards."""
        record = self._apply_profile_patch(token, patch)
        return record.profile_view(stats=self.profile_stats(record.id))

    # ------------------------------------------------------------------ #
    # Account-owned extensions (never redefine a contract shape)
    # ------------------------------------------------------------------ #
    @envelope_boundary
    def get_profile_details(self, user_id: str) -> ProfileDetails | ErrorEnvelope:
        """Profile including ``specialty`` / ``affiliation`` / ``expertise_tags``."""
        return self._require_user(user_id).details()

    @envelope_boundary
    def update_profile_details(
        self, token: str, patch: Any
    ) -> ProfileDetails | ErrorEnvelope:
        """Same as :meth:`update_profile` but returning the widened payload."""
        return self._apply_profile_patch(token, patch).details()

    @envelope_boundary
    def list_users(
        self, token: str, limit: int = DEFAULT_PAGE_SIZE
    ) -> list[UserSummary] | ErrorEnvelope:
        """Admin-only directory listing (``INV-role-check`` guard demonstration)."""
        require_admin_session(token, self)
        check_fields({"limit": limit}, "page_query")
        return [record.summary() for record in self.repository.list_users(limit=limit)]

    @envelope_boundary
    def current_principal(self, token: str) -> Principal | ErrorEnvelope:
        """The resolved session (role included) for guard-aware callers."""
        return require_session(token, self)

    # ------------------------------------------------------------------ #
    # seeding / maintenance helpers
    # ------------------------------------------------------------------ #
    def seed_admin(
        self,
        email: str | None = None,
        *,
        password: str | None = None,
        username: str | None = None,
    ) -> UserRecord:
        """Idempotently provision the pre-set ``admin`` account."""
        return seed_admin_account(
            self, email=email, password=password, username=username
        )

    def purge_expired_sessions(self) -> int:
        return self.repository.purge_expired_sessions()

    def close(self) -> None:
        self.repository.close()

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #
    @property
    def iterations(self) -> int:
        return int(self.settings.password_hash_iterations)

    def profile_stats(self, user_id: str) -> dict[str, int]:
        """Counters supplied by the injected port, sanitised via :func:`profile_stats`."""
        if self._stats is None:
            return profile_stats()
        try:
            return profile_stats(self._stats(user_id))
        except Exception:  # pragma: no cover - a profile page must still render
            logger.exception("profile stats provider failed for a user")
            return profile_stats()

    def _digest(self, password: str) -> str:
        return hash_password(password, iterations=self.iterations)

    def _validate_credentials(
        self, email: object, password: object, *, schema: str
    ) -> tuple[str, str]:
        check_fields({"email": email, "password": password}, schema)
        return normalize_email(email), password  # type: ignore[return-value]

    def _new_user(self, email: str, password: str) -> UserRecord:
        stamp = now_iso()
        return UserRecord(
            id=self._new_user_id(),
            email=email,
            password_hash=self._digest(password),
            role=Role.USER,
            username=derive_username(email),
            created_at=stamp,
            updated_at=stamp,
        )

    def _new_user_id(self) -> str:
        for _ in range(16):
            candidate = new_id()
            if self.repository.get_user_by_id(candidate) is None:
                return candidate
        raise ContractError(ErrorCode.DATABASE_ERROR, "cannot allocate a user id")

    def _issue_session(self, record: UserRecord) -> SessionToken:
        created = now_utc()
        expires = created + timedelta(seconds=int(self.settings.session_ttl_seconds))
        grant = self._create_session(record.id, to_iso(created), to_iso(expires))
        return SessionToken(
            token=grant.token,
            user=record.summary(),
            expires_at=grant.expires_at,
        )

    def _create_session(self, user_id: str, created_at: str, expires_at: str) -> SessionGrant:
        last_error: ContractError | None = None
        for _ in range(MAX_SESSION_RETRY):
            token = generate_session_token()
            try:
                return self.repository.create_session(
                    token=token, user_id=user_id, created_at=created_at, expires_at=expires_at
                )
            except ContractError as exc:
                last_error = exc
        raise last_error or ContractError(ErrorCode.DATABASE_ERROR, "cannot open a session")

    def _require_user(self, user_id: str) -> UserRecord:
        record = self.repository.get_user_by_id(normalize_id(user_id, "user_id"))
        if record is None:
            raise ContractError(
                ErrorCode.USER_NOT_FOUND, None, {"field": "user_id", "reason": "no such account"}
            )
        return record

    def _apply_profile_patch(self, token: str, patch: Any) -> UserRecord:
        principal = require_session(token, self)
        record = self._require_user(principal.user_id)
        payload: ProfilePatchPayload = normalise_profile_patch(patch)
        if payload.is_empty():  # pragma: no cover - normalise raises first
            raise ContractError(ErrorCode.VALIDATION_ERROR)
        return self.repository.update_user(record, changes=payload.to_dict())

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"AccountService(database={self.db.target!r})"


# --------------------------------------------------------------------------- #
# Wiring helper (what mod-web calls in its application factory)
# --------------------------------------------------------------------------- #
def create_account_service(
    settings: Settings | Mapping[str, Any] | None = None,
    *,
    database: Database | None = None,
    seed: bool = True,
    stats: ProfileStatsProvider | None = None,
) -> AccountService:
    """Build a ready service: schema created, admin seeded, connection open.

    ``settings`` may be a :class:`~blacknote.contracts.config.Settings` or a
    plain mapping of overrides; either way the connection string is read from
    the config layer only (``INV-config-db``).
    """
    resolved = _resolve_settings(settings)
    db = database if database is not None else Database(resolved)
    db.initialize()
    service = AccountService(AccountRepository(db), resolved, stats=stats)
    if seed:
        try:
            service.seed_admin()
        except ContractError as exc:  # pragma: no cover - never break start-up
            logger.warning("admin seed skipped: %s", exc.envelope)
    return service


def _resolve_settings(settings: Settings | Mapping[str, Any] | None) -> Settings:
    if isinstance(settings, Settings):
        return settings
    base = load_settings()
    if settings is None:
        return base
    if isinstance(settings, Mapping):
        return base.replace(**dict(settings))
    raise ContractError(ErrorCode.CONFIG_ERROR, "settings must be Settings or a mapping")
