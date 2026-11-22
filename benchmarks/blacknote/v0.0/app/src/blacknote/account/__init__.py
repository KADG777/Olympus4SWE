"""``mod-account`` — accounts, sessions and identity profiles.

Wave 1 deliverable of BlackNote.  Public surface (everything ``mod-web`` and,
transitively, the other business layers need):

* :class:`AccountService` / :func:`create_account_service` — the
  ``IF-account-auth`` + ``IF-account-profile`` implementation;
* :func:`require_session` / :func:`require_admin_session` — the
  ``INV-auth-session`` and ``INV-role-check`` guards other layers call;
* :class:`Settings`-driven SQLite storage (:mod:`blacknote.account.db`,
  :mod:`blacknote.account.repository`) so content survives a restart;
* :func:`seed_admin_account` — the pre-set administrator path.

Import direction stays strictly ``contracts <- account``: nothing below this
package imports the note, interaction, moderation or web layers.
"""

from __future__ import annotations

from blacknote.contracts.auth_guard import Principal
from blacknote.contracts.interfaces import ProfilePatch, ProfileView, SessionToken, UserSummary
from blacknote.contracts.roles import Role

from .db import ACCOUNT_SCHEMA, Database, parse_database_url
from .guards import (
    AUDIT_UNAUTHORIZED_ADMIN_ATTEMPT,
    is_admin_session,
    require_admin_session,
    require_owner_or_admin_session,
    require_session,
    resolve_optional,
)
from .models import ProfileDetails, SessionGrant, UserRecord, profile_stats
from .patch import ProfilePatchPayload, normalise_profile_patch
from .repository import AccountRepository
from .security import (
    derive_username,
    generate_session_token,
    hash_password,
    verify_password,
)
from .seed import (
    ADMIN_DISPLAY_NAME,
    DEV_ADMIN_PASSWORD,
    SeedOutcome,
    resolve_admin_password,
    seed_admin_account,
)
from .service import (
    AccountService,
    ProfileStatsProvider,
    create_account_service,
    envelope_boundary,
)

__all__ = [
    # service / wiring
    "AccountService",
    "ProfileStatsProvider",
    "create_account_service",
    "envelope_boundary",
    # guards
    "AUDIT_UNAUTHORIZED_ADMIN_ATTEMPT",
    "is_admin_session",
    "require_admin_session",
    "require_owner_or_admin_session",
    "require_session",
    "resolve_optional",
    # storage
    "ACCOUNT_SCHEMA",
    "AccountRepository",
    "Database",
    "parse_database_url",
    # models
    "ProfileDetails",
    "ProfilePatch",
    "ProfilePatchPayload",
    "ProfileView",
    "SessionGrant",
    "SessionToken",
    "UserRecord",
    "UserSummary",
    "normalise_profile_patch",
    "profile_stats",
    # security
    "derive_username",
    "generate_session_token",
    "hash_password",
    "verify_password",
    # seeding
    "ADMIN_DISPLAY_NAME",
    "DEV_ADMIN_PASSWORD",
    "SeedOutcome",
    "resolve_admin_password",
    "seed_admin_account",
    # re-exported contract vocabularies
    "Principal",
    "Role",
]
