"""AuthN / AuthZ guards built on the frozen contract helpers.

``INV-auth-session`` says the *account* layer implements the guard and every
other layer calls it.  Concretely that means:

* :func:`require_session` — thin, raising wrapper around
  :func:`blacknote.contracts.auth_guard.require_auth` bound to the account
  service's session resolver.  Missing / malformed / revoked / expired tokens
  all yield ``AUTH_REQUIRED`` (401) and no write happens afterwards, because
  the guard raises *before* any repository call.
* :func:`require_admin_session` — ordered authorisation: unauthenticated →
  ``AUTH_REQUIRED``, authenticated non-admin → ``PERMISSION_DENIED`` (**INV-role-check**),
  and every refusal logs exactly one audit record.  The token itself is never
  logged or stored.
* :func:`is_admin_session` — non-raising boolean helper for read paths
  (e.g. showing an admin-only nav entry on the feed page).

The contract layer owns the decision logic; this module only wires it to a
session resolver and records the attempt.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

from blacknote.contracts.auth_guard import (
    Principal,
    SessionResolver,
    require_auth,
    resolve_session,
)
from blacknote.contracts.errors import ContractError, ErrorCode
from blacknote.contracts.roles import Role, is_admin, require_admin, require_owner_or_admin

__all__ = [
    "AUDIT_UNAUTHORIZED_ADMIN_ATTEMPT",
    "GuardSessionProvider",
    "ROLES",
    "SECURITY_LOGGER",
    "is_admin_session",
    "require_admin_session",
    "require_owner_or_admin_session",
    "require_session",
    "resolve_optional",
    "session_resolver_of",
]

#: Security logger: one record per refused admin attempt (INV-role-check).
SECURITY_LOGGER = logging.getLogger("blacknote.security")

AUDIT_UNAUTHORIZED_ADMIN_ATTEMPT: str = "UNAUTHORIZED_ADMIN_ATTEMPT"


class GuardSessionProvider(SessionResolver, Protocol):
    """What a guard needs: something able to turn a token into a principal."""

    def resolve(self, token: str) -> "Principal | None": ...  # pragma: no cover


def session_resolver_of(candidate: Any) -> SessionResolver:
    """Normalise resolver-ish inputs (service, callable, guard bundle)."""
    if candidate is None:
        raise ContractError(
            ErrorCode.INTERNAL_ERROR,
            "a session resolver is required to guard this operation",
        )
    if hasattr(candidate, "resolve"):
        return candidate  # type: ignore[return-value]
    if hasattr(candidate, "session_resolver"):
        return candidate.session_resolver
    if callable(candidate):
        return _CallableResolver(candidate)
    raise ContractError(ErrorCode.INTERNAL_ERROR, "unsupported session resolver")


class _CallableResolver:
    """Adapter so a plain ``callable(token) -> Principal | None`` works."""

    def __init__(self, func: Any) -> None:
        self._func = func

    def resolve(self, token: str) -> "Principal | None":
        return self._func(token)


def _audit(resolver: Any, event: str, *, actor_id: str | None, reason: str) -> None:
    """Log + persist one attempt.  Never receives the bearer token."""
    SECURITY_LOGGER.warning(
        "event=%s actor_id=%s reason=%s", event, actor_id, reason
    )
    record = getattr(resolver, "record_audit", None)
    if callable(record):
        try:
            record(event, actor_id=actor_id, reason=reason)
        except Exception:  # pragma: no cover - audit must not break the request
            SECURITY_LOGGER.exception("failed to persist security audit record")


def require_session(
    token: str | None,
    resolver: Any,
    *,
    now: Any = None,
    reason: str | None = None,
) -> Principal:
    """``INV-auth-session`` entry point for every write operation.

    Raises ``AUTH_REQUIRED`` when nobody is authenticated.  The optional
    ``reason`` only rewrites the human readable message; the code stays
    ``AUTH_REQUIRED`` so clients can react uniformly.
    """
    target = session_resolver_of(resolver)
    try:
        return require_auth(token, target, now=now)
    except ContractError as exc:
        if exc.code == ErrorCode.AUTH_REQUIRED and reason:
            raise ContractError(exc.code, reason, exc.envelope.details) from exc
        raise


def resolve_optional(token: str | None, resolver: Any, *, now: Any = None) -> Principal | None:
    """Non-raising variant used by read paths that adapt to a logged-in viewer."""
    try:
        target = session_resolver_of(resolver)
    except ContractError:
        return None
    return resolve_session(token, target, now=now)


def is_admin_session(token: str | None, resolver: Any, *, now: Any = None) -> bool:
    """True only for a live session whose role is :attr:`Role.ADMIN`."""
    principal = resolve_optional(token, resolver, now=now)
    return bool(principal is not None and is_admin(principal))


def require_admin_session(
    token: str | None,
    resolver: Any,
    *,
    now: Any = None,
    reason: str | None = None,
) -> Principal:
    """Admin-only guard: ``AUTH_REQUIRED`` → ``PERMISSION_DENIED`` (ordered).

    A non-admin refusal records exactly one ``UNAUTHORIZED_ADMIN_ATTEMPT``
    audit entry before the error is returned to the caller.
    """
    principal = require_session(token, resolver, now=now)
    target = session_resolver_of(resolver)
    try:
        require_admin(principal, reason=reason)
    except ContractError as exc:
        if exc.code == "PERMISSION_DENIED":
            _audit(
                target,
                AUDIT_UNAUTHORIZED_ADMIN_ATTEMPT,
                actor_id=principal.user_id,
                reason=reason or "role is not admin",
            )
        raise
    return principal


def require_owner_or_admin_session(
    token: str | None,
    resolver: Any,
    owner_id: str | None,
    *,
    now: Any = None,
    reason: str | None = None,
) -> Principal:
    """Owner-or-admin guard (used by account writes scoped to a single user)."""
    principal = require_session(token, resolver, now=now)
    require_owner_or_admin(principal, owner_id, reason=reason)
    return principal


#: Re-exported for callers that want the enum without importing contracts.
ROLES: tuple[Role, Role] = (Role.USER, Role.ADMIN)
