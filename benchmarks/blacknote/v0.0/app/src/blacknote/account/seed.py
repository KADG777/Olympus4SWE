"""Pre-set administrator provisioning (**INV-role-check**).

The MVP has no UI for granting privileges: exactly one ``admin`` account
exists because this seed path creates it.  Rules enforced here:

* the e-mail comes from ``Settings.admin_email`` (``BLACKNOTE_ADMIN_EMAIL``)
  and the password from ``Settings.admin_password`` (``BLACKNOTE_ADMIN_PASSWORD``);
* outside ``dev``/``test`` a configured password is mandatory — and
  :class:`~blacknote.contracts.config.Settings` already refuses to start
  without one, so the dev-only bootstrap credential cannot reach production;
* seeding is **idempotent**: a restart reuses the existing row, only repairs
  the role if it drifted, and never opens a session;
* the stored credential is the same salted PBKDF2 digest used for normal
  accounts — seeding never writes plaintext.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

from blacknote.contracts.config import Environment
from blacknote.contracts.errors import ContractError, ErrorCode
from blacknote.contracts.ids import new_id
from blacknote.contracts.roles import Role
from blacknote.contracts.timeutil import now_iso
from blacknote.contracts.validation import check_fields, clean_text, normalize_email

from .models import UserRecord
from .security import hash_password

__all__ = [
    "ADMIN_DISPLAY_NAME",
    "ADMIN_EXPERTISE_TAGS",
    "DEV_ADMIN_PASSWORD",
    "SeedOutcome",
    "resolve_admin_password",
    "seed_admin_account",
]

#: Display name shown on the pre-set account's profile page.
ADMIN_DISPLAY_NAME: Final[str] = "BlackNote Admin"
ADMIN_EXPERTISE_TAGS: Final[tuple[str, ...]] = ("AI新闻", "AI工具", "AI产品")
#: Bootstrap credential, used **only** when ``environment`` is dev/test.
DEV_ADMIN_PASSWORD: Final[str] = "blacknote-admin"


@dataclass(frozen=True)
class SeedOutcome:
    """Summary of what the seed path did (never carries the credential)."""

    user_id: str
    created: bool
    role: str = str(Role.ADMIN)

    def to_dict(self) -> dict[str, Any]:
        return {"user_id": self.user_id, "created": self.created, "role": self.role}


def resolve_admin_password(settings: Any, password: str | None = None) -> str:
    """Explicit argument → ``BLACKNOTE_ADMIN_PASSWORD`` → dev bootstrap value."""
    candidate = password or getattr(settings, "admin_password", None)
    if isinstance(candidate, str) and candidate.strip():
        return candidate
    environment = str(getattr(settings, "environment", Environment.DEV))
    if environment in (Environment.DEV, Environment.TEST):
        return DEV_ADMIN_PASSWORD
    raise ContractError(
        ErrorCode.CONFIG_ERROR,
        "admin_password is required to seed the administrator account",
        {"field": "admin_password", "hint": "set BLACKNOTE_ADMIN_PASSWORD"},
    )


def seed_admin_account(
    service: Any,
    email: str | None = None,
    *,
    password: str | None = None,
    username: str | None = None,
) -> UserRecord:
    """Ensure the pre-set admin row exists and return it.

    ``service`` is anything exposing ``settings`` and ``repository`` (in
    practice :class:`blacknote.account.service.AccountService`); it is typed
    loosely on purpose so this module stays free of an import cycle.
    """
    settings = service.settings
    repository = service.repository

    admin_email = normalize_email(email or getattr(settings, "admin_email", None))
    admin_password = resolve_admin_password(settings, password)
    check_fields({"email": admin_email, "password": admin_password}, "register")

    display = clean_text(username) or ADMIN_DISPLAY_NAME
    existing = repository.get_user_by_email(admin_email)
    if existing is not None:
        if existing.role is Role.ADMIN:
            return existing
        # The address already belongs to a normal account: promote it and keep
        # the stored digest untouched (its owner may have set a real password).
        return repository.set_role(existing.id, Role.ADMIN)

    digest = hash_password(
        admin_password, iterations=int(settings.password_hash_iterations)
    )
    stamp = now_iso()
    record = UserRecord(
        id=new_id(),
        email=admin_email,
        password_hash=digest,
        role=Role.ADMIN,
        username=display,
        bio="Platform administrator.",
        specialty="平台治理",
        affiliation=clean_text(getattr(settings, "app_name", "BlackNote")) or "BlackNote",
        expertise_tags=ADMIN_EXPERTISE_TAGS,
        created_at=stamp,
        updated_at=stamp,
    )
    repository.insert_user(record)
    seeded = repository.get_user_by_email(admin_email)
    if seeded is None:  # pragma: no cover - defensive
        raise ContractError(ErrorCode.DATABASE_ERROR, "admin seed did not persist")
    return seeded
