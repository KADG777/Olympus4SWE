"""Role contract for BlackNote.

Invariant implemented here: **INV-role-check**.

* Two roles exist in the MVP: :attr:`Role.USER` (every registered account) and
  :attr:`Role.ADMIN` (the pre-provisioned moderator account).
* Authorisation checks are *ordered*: an unauthenticated caller always gets
  ``AUTH_REQUIRED`` (401) and only an authenticated non-admin gets
  ``PERMISSION_DENIED`` (403) — the difference leaks nothing but keeps the
  client UX correct.
* Unknown role strings fail closed: :func:`parse_role` raises,
  :func:`is_admin` returns ``False``.
* All privileged moderation paths MUST go through :func:`require_admin`; no
  module may compare ``role == "admin"`` inline.
"""

from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum
from types import MappingProxyType
from typing import Final, Mapping, Protocol, runtime_checkable

from .errors import ContractError, FieldError, ErrorCode

__all__ = [
    "Role",
    "ROLE_HIERARCHY",
    "DEFAULT_ROLE",
    "MODERATOR_ROLES",
    "HasRole",
    "HasActorId",
    "parse_role",
    "is_admin",
    "role_rank",
    "require_role",
    "require_admin",
    "require_owner_or_admin",
    "can_moderate",
]


class Role(StrEnum):
    """Account roles (lowest → highest privilege)."""

    USER = "user"
    ADMIN = "admin"

    @property
    def rank(self) -> int:
        return ROLE_HIERARCHY[self]

    @property
    def is_admin(self) -> bool:
        return self is Role.ADMIN


#: Explicit privilege ordering, kept as data so future roles slot in cleanly.
ROLE_HIERARCHY: Final[Mapping[Role, int]] = MappingProxyType(
    {Role.USER: 1, Role.ADMIN: 2}
)

DEFAULT_ROLE: Final[Role] = Role.USER
#: Roles that may moderate content (INV-role-check single source of truth).
MODERATOR_ROLES: Final[frozenset[Role]] = frozenset({Role.ADMIN})


@runtime_checkable
class HasRole(Protocol):
    """Structural view of an actor: something with ``user_id`` and ``role``."""

    @property
    def user_id(self) -> str:  # pragma: no cover - protocol declaration
        ...

    @property
    def role(self) -> str:  # pragma: no cover - protocol declaration
        ...


@runtime_checkable
class HasActorId(Protocol):
    """Structural view of an actor id only (anonymous actors are ``None``)."""

    @property
    def user_id(self) -> str:  # pragma: no cover - protocol declaration
        ...


def parse_role(value: object, *, field_name: str = "role") -> Role:
    """Coerce *value* into a :class:`Role` or raise ``VALIDATION_ERROR``."""
    if isinstance(value, Role):
        return value
    if not isinstance(value, str):
        raise ContractError.validation(
            [FieldError(field_name, "invalid_role", "role must be a string")]
        )
    try:
        return Role(value.strip().lower())
    except ValueError:
        raise ContractError.validation(
            [FieldError(field_name, "invalid_role",
                        f"role must be one of: {', '.join(r.value for r in Role)}")]
        ) from None


def role_rank(actor: HasRole | str | Role | None) -> int:
    """Rank of an actor / role value; ``-1`` for anonymous / unknown."""
    if actor is None:
        return -1
    raw = actor.role if isinstance(actor, HasRole) else actor
    try:
        return ROLE_HIERARCHY[parse_role(raw)]
    except ContractError:
        return -1


def is_admin(actor: HasRole | str | Role | None) -> bool:
    """Fail-closed admin check (anonymous, malformed or unknown → False)."""
    if actor is None:
        return False
    raw = actor.role if isinstance(actor, HasRole) else actor
    if isinstance(raw, Role):
        return raw is Role.ADMIN
    if not isinstance(raw, str):
        return False
    try:
        return parse_role(raw) is Role.ADMIN
    except ContractError:
        return False


def can_moderate(actor: HasRole | None) -> bool:
    """Alias used by the moderation layer: admins may moderate."""
    return is_admin(actor)


def require_role(
    actor: HasRole | None,
    allowed: Role | Iterable[Role] | None = None,
    *,
    reason: str | None = None,
) -> HasRole:
    """Guard: actor must exist and hold one of *allowed* roles.

    Raises ``AUTH_REQUIRED`` for anonymous callers and ``PERMISSION_DENIED``
    for authenticated but insufficiently privileged callers.
    """
    if actor is None:
        raise ContractError.auth_required(reason)
    if not isinstance(actor, HasRole):
        raise ContractError(
            ErrorCode.INTERNAL_ERROR, "actor does not expose a 'role' attribute"
        )
    permitted: tuple[Role, ...]
    if allowed is None:
        permitted = tuple(Role)
    elif isinstance(allowed, Role):
        permitted = (allowed,)
    else:
        permitted = tuple(allowed)
    try:
        held = parse_role(actor.role)
    except ContractError:
        raise ContractError(ErrorCode.PERMISSION_DENIED, "unknown role") from None
    if held not in permitted:
        raise ContractError(
            ErrorCode.PERMISSION_DENIED,
            reason,
            {"required_roles": [str(r) for r in permitted], "actual_role": str(held)},
        )
    return actor


def require_admin(actor: HasRole | None, *, reason: str | None = None) -> HasRole:
    """Guard for every admin-only operation (``IF-moderation-*``)."""
    return require_role(actor, Role.ADMIN, reason=reason)


def require_owner_or_admin(
    actor: HasRole | None,
    owner_id: object,
    *,
    reason: str | None = None,
    error_code: ErrorCode = ErrorCode.PERMISSION_DENIED,
) -> HasRole:
    """Guard for owner-only writes (edit/delete own note or comment)."""
    if actor is None:
        raise ContractError.auth_required(reason)
    require_role(actor, (Role.USER, Role.ADMIN), reason=reason)
    if is_admin(actor):
        return actor
    actor_id = getattr(actor, "user_id", None)
    if not isinstance(actor_id, str) or not isinstance(owner_id, str) or actor_id != owner_id:
        raise ContractError(error_code, reason, {"resource_owner_required": True})
    return actor
