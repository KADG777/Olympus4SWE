"""``mod-moderation`` — the content governance layer of BlackNote.

Public surface (signatures frozen in :mod:`blacknote.contracts.interfaces`):

* :class:`ModerationService` — ``IF-moderation-actions``
  (``unlist_note`` / ``block_comment``), admin only.
* :class:`~blacknote.moderation.models.ModerationAuditLog` — the append-only
  record of takedowns and of ``privilege_denied`` violations.
* :class:`~blacknote.moderation.models.ModeratableNote` /
  :class:`~blacknote.moderation.models.ModeratableComment` — flag-only row
  shapes for stores that do not reuse ``mod-note`` / ``mod-interaction`` rows.
* :class:`~blacknote.moderation.stores.InMemoryNoteStore`,
  :class:`~blacknote.moderation.stores.InMemoryCommentStore`,
  :class:`~blacknote.moderation.stores.InMemoryAuditLogStore`,
  :class:`~blacknote.moderation.stores.SQLiteAuditLogStore` — ready-made
  storage for dev wiring, tests and the persisted audit trail.

Scope of this layer (MVP, clarification A3)
-------------------------------------------
Only the manual admin takedown workflow exists: it writes the state flags and
nothing else.  It never filters a read path, never computes visibility, never
deletes content and never re-implements a role or error format — those all
belong to :mod:`blacknote.contracts`:

``contracts.roles.require_admin``      who may act (INV-role-check)
``contracts.auth_guard.require_auth``  is the caller logged in (INV-auth-session)
``contracts.visibility``               ``ModerationState`` + the single predicate
``contracts.errors``                   the only error shape (INV-error-envelope)

Like the other Wave-1 modules this package imports ``blacknote.contracts`` and
the standard library only; sibling layers are wired by ``mod-web`` through the
duck-typed ports in :mod:`blacknote.moderation.ports`.
"""

from __future__ import annotations

from .models import (
    ACTION_TARGET_KIND,
    AUDIT_COLUMNS,
    FLAG_FIELD,
    TARGET_STATE,
    ModeratableComment,
    ModeratableNote,
    ModerationAction,
    ModerationAuditLog,
    ModerationTargetKind,
    flag_field_for,
    flag_of,
    moderation_of,
    moderation_state_for,
    parse_action,
    parse_target_kind,
)
from .ports import (
    AuditLogStore,
    ModerationTargetStore,
    adapt_audit_store,
    adapt_sessions,
    adapt_target_store,
    canonical_target_id,
    target_state_for,
    with_moderation,
)
from .service import AUDIT_INTERFACE_ID, ModerationService
from .stores import (
    InMemoryAuditLogStore,
    InMemoryCommentStore,
    InMemoryModerationStore,
    InMemoryNoteStore,
    SQLiteAuditLogStore,
    WrappingAuditLogStore,
    open_audit_store,
)

__all__ = [
    # service
    "ModerationService",
    "AUDIT_INTERFACE_ID",
    # models
    "ModerationAction",
    "ModerationAuditLog",
    "ModerationTargetKind",
    "ModeratableNote",
    "ModeratableComment",
    "TARGET_STATE",
    "FLAG_FIELD",
    "ACTION_TARGET_KIND",
    "AUDIT_COLUMNS",
    "parse_action",
    "parse_target_kind",
    "moderation_state_for",
    "target_state_for",
    "flag_field_for",
    "flag_of",
    "moderation_of",
    # ports
    "ModerationTargetStore",
    "AuditLogStore",
    "adapt_target_store",
    "adapt_audit_store",
    "adapt_sessions",
    "canonical_target_id",
    "with_moderation",
    # stores
    "InMemoryModerationStore",
    "InMemoryNoteStore",
    "InMemoryCommentStore",
    "InMemoryAuditLogStore",
    "SQLiteAuditLogStore",
    "WrappingAuditLogStore",
    "open_audit_store",
]
