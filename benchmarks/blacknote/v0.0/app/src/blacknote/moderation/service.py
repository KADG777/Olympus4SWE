"""Content governance service — ``mod-moderation`` (``IF-moderation-actions``).

:class:`ModerationService` implements the two frozen admin actions of the
governance layer:

``unlist_note(admin_token, note_id)``
    admin 下架: writes ``Note.moderation = "unlisted"`` (boolean view
    ``Note.moderated = true``).
``block_comment(admin_token, comment_id)``
    admin 屏蔽: writes ``Comment.moderation = "blocked"`` (boolean view
    ``Comment.blocked = true``).  Replies are **not** touched: their visibility
    is derived by the shared predicate from the ancestor chain on the read side
    (``INV-visibility-predicate``), so a single flag column stays the only
    source of truth and no denormalised copy can drift.

Authorisation discipline (``INV-role-check`` / ``INV-auth-session``)
--------------------------------------------------------------------
Every entry point runs the guards **in this order, before any storage call**:

1. :func:`blacknote.contracts.auth_guard.require_auth` — an anonymous, unknown
   or expired session answers ``AUTH_REQUIRED`` (401) and nothing at all is
   written (not even an audit row: a failed authentication is not an
   authorisation decision).
2. :func:`blacknote.contracts.roles.require_admin` — an authenticated caller
   whose role is not ``admin`` answers ``PERMISSION_DENIED`` (403), appends
   exactly one ``ModerationAuditLog(action="privilege_denied")`` record and
   leaves the target untouched.  Escalation success rate is therefore 0 %: the
   flag write is unreachable without an admin role.
3. only then the target lookup (``NOTE_NOT_FOUND`` / ``COMMENT_NOT_FOUND``,
   which also covers tombstones and malformed ids so existence never leaks).

This module never compares roles, never formats an error payload and never
decides visibility — the guards and the envelope come from ``contracts``, the
flag semantics come from ``contracts.visibility.ModerationState``, and reading
"may this actor see it" is always ``contracts.visibility.is_visible_to``.

Error convention
----------------
Failures are raised as :class:`~blacknote.contracts.errors.ContractError`,
whose ``to_envelope()`` *is* the frozen ``ErrorEnvelope`` (the convention the
account and note layers already follow).  The non-raising
``*_result`` wrappers return the envelope union verbatim for callers that
prefer it.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from typing import Any, Final

from blacknote.contracts.auth_guard import Principal, require_auth
from blacknote.contracts.config import Settings
from blacknote.contracts.errors import ContractError, ErrorCode, ErrorEnvelope
from blacknote.contracts.roles import Role, require_admin
from blacknote.contracts.timeutil import now_iso
from blacknote.contracts.visibility import (
    ModerationState,
    VisibilityScope,
    is_visible_to,
)

from .models import (
    AUDIT_INTERFACE_ID,
    ModerationAction,
    ModerationAuditLog,
    ModerationTargetKind,
    flag_field_for,
    flag_of,
    moderation_of,
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
)
from .stores import InMemoryCommentStore, InMemoryNoteStore

__all__ = ["ModerationService", "AUDIT_INTERFACE_ID"]

#: ``operation → (target kind, audit action, not-found code, id field)``
_OPERATIONS: Final[Mapping[str, tuple[str, ModerationAction, ErrorCode, str]]] = {
    "unlist_note": (
        ModerationTargetKind.NOTE,
        ModerationAction.NOTE_UNLISTED,
        ErrorCode.NOTE_NOT_FOUND,
        "note_id",
    ),
    "block_comment": (
        ModerationTargetKind.COMMENT,
        ModerationAction.COMMENT_BLOCKED,
        ErrorCode.COMMENT_NOT_FOUND,
        "comment_id",
    ),
}

SECURITY_LOGGER = logging.getLogger("blacknote.security")
MODERATION_LOGGER = logging.getLogger("blacknote.moderation")


class ModerationService:
    """Concrete implementation of the frozen ``ModerationApi`` protocol."""

    def __init__(
        self,
        notes: Any = None,
        comments: Any = None,
        *,
        audit: Any = None,
        sessions: Any = None,
        settings: Settings | None = None,
        clock: Callable[[], str] | None = None,
    ) -> None:
        """:param notes: note flag store (native store, note repository, note service).
        :param comments: comment flag store (same shapes, keyed by comment id).
        :param audit: audit sink exposing ``append(entry)`` (defaults to memory).
        :param sessions: session resolver port for ``require_auth``.
        :param settings: frozen configuration (page size of the audit listing).
        :param clock: injectable UTC clock for deterministic timestamps.
        """
        self._settings = settings or Settings()
        self._clock: Callable[[], str] = clock or now_iso
        self._notes: ModerationTargetStore = adapt_target_store(
            notes if notes is not None else InMemoryNoteStore(),
            kind=ModerationTargetKind.NOTE,
            clock=self._clock,
        )
        self._comments: ModerationTargetStore = adapt_target_store(
            comments if comments is not None else InMemoryCommentStore(),
            kind=ModerationTargetKind.COMMENT,
            clock=self._clock,
        )
        self._audit: AuditLogStore = adapt_audit_store(audit)
        self._resolver = adapt_sessions(sessions)

    # ------------------------------------------------------------------ #
    # wiring helpers (used by mod-web / tests)
    # ------------------------------------------------------------------ #
    @property
    def audit_store(self) -> AuditLogStore:
        return self._audit

    @property
    def settings(self) -> Settings:
        return self._settings

    def bind_sessions(self, sessions: Any) -> ModerationService:
        """Wire (or re-wire) the session resolver port."""
        self._resolver = adapt_sessions(sessions)
        return self

    def bind_notes(self, notes: Any) -> ModerationService:
        """Wire the note flag store (``NoteRepository`` / ``NoteService``)."""
        self._notes = adapt_target_store(notes, kind=ModerationTargetKind.NOTE,
                                         clock=self._clock)
        return self

    def bind_comments(self, comments: Any) -> ModerationService:
        """Wire the comment flag store (the interaction layer's repository)."""
        self._comments = adapt_target_store(comments, kind=ModerationTargetKind.COMMENT,
                                            clock=self._clock)
        return self

    # ------------------------------------------------------------------ #
    # IF-moderation-actions
    # ------------------------------------------------------------------ #
    def unlist_note(self, admin_token: str, note_id: str) -> None:
        """Take a note down from every public surface (admin only).

        Raises ``AUTH_REQUIRED`` / ``PERMISSION_DENIED`` / ``NOTE_NOT_FOUND``;
        on success the note row carries ``moderation=unlisted`` (``moderated``
        is ``True``) and one ``note_unlisted`` audit record exists.
        """
        return self._execute("unlist_note", admin_token, note_id)

    def block_comment(self, admin_token: str, comment_id: str) -> None:
        """Block a comment (admin only); replies are filtered by the predicate.

        Raises ``AUTH_REQUIRED`` / ``PERMISSION_DENIED`` / ``COMMENT_NOT_FOUND``.
        """
        return self._execute("block_comment", admin_token, comment_id)

    # -- non-raising variants (the literal ``None | ErrorEnvelope`` union) -----
    def unlist_note_result(self, admin_token: str, note_id: str) -> None | ErrorEnvelope:
        return self._result(self.unlist_note, admin_token, note_id)

    def block_comment_result(self, admin_token: str, comment_id: str) -> None | ErrorEnvelope:
        return self._result(self.block_comment, admin_token, comment_id)

    # ------------------------------------------------------------------ #
    # read seams for mod-web (never a visibility decision)
    # ------------------------------------------------------------------ #
    def list_audit_log(
        self,
        *,
        action: ModerationAction | str | None = None,
        user_id: str | None = None,
        note_id: str | None = None,
        comment_id: str | None = None,
        limit: int | None = None,
    ) -> list[ModerationAuditLog]:
        """Newest-first audit listing (admin console), filtered by the model."""
        effective = self._settings.default_page_size if limit is None else max(0, int(limit))
        return list(
            self._audit.list(action=action, user_id=user_id, note_id=note_id,
                             comment_id=comment_id, limit=effective)
        )

    def audit_count(self, *, action: ModerationAction | str | None = None) -> int:
        return int(self._audit.count(action=action))

    def moderation_flag(self, kind: ModerationTargetKind | str, target_id: str) -> bool | None:
        """Current boolean flag (``Note.moderated`` / ``Comment.blocked``).

        A storage probe for admin tooling and tests — it reports the state of a
        row, it does **not** say whether anybody may read it.  ``None`` means
        "no such row".
        """
        store, _ = self._store_for(_kind_of(kind))
        record = store.load(canonical_target_id(target_id) or "")
        if record is None:
            return None
        return _flag_value(record, kind, default=False)

    def is_admin_token(self, admin_token: str | None) -> bool:
        """Non-raising role probe (admin nav entry on the web layer)."""
        try:
            actor = require_auth(admin_token, self._resolver)
        except ContractError:
            return False
        return bool(actor.role is Role.ADMIN)

    def can_moderate_target(self, admin_token: str | None,
                            kind: ModerationTargetKind | str, target_id: str) -> bool:
        """Would ``unlist_note`` / ``block_comment`` succeed for this caller?

        Composes the two contract guards and the contract predicate; no rule is
        re-implemented here.
        """
        kind_name = _kind_of(kind)
        store, _ = self._store_for(kind_name)
        try:
            actor = self._authorize(admin_token, operation=_operation_name(kind_name),
                                    kind=kind_name, target_id=target_id, audit=False)
        except ContractError:
            return False
        record = store.load(canonical_target_id(target_id) or "")
        return record is not None and is_visible_to(
            actor, record, scope=VisibilityScope.MODERATION
        )

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #
    def _execute(self, operation: str, admin_token: str, target_id: str) -> None:
        kind, action, not_found, id_field = _OPERATIONS[operation]
        actor = self._authorize(admin_token, operation=operation, kind=kind,
                                target_id=target_id)
        canonical = canonical_target_id(target_id)
        if canonical is None:
            # a malformed id is indistinguishable from a missing row (404)
            raise ContractError(not_found, None, {id_field: _safe(target_id)})
        store, _ = self._store_for(kind)
        record = self._require_target(store, canonical, actor=actor,
                                      not_found=not_found, id_field=id_field)
        previous = self._previous_state(record, canonical, not_found=not_found,
                                        id_field=id_field)
        state = target_state_for(kind)
        updated = store.apply_moderation(canonical, state)
        if updated is None:  # pragma: no cover - row vanished mid-request
            raise ContractError(not_found, None, {id_field: canonical})
        self._record(
            ModerationAuditLog.create(
                action,
                user_id=actor.user_id,
                **{id_field: canonical},
                reason=_reason_for(kind),
                details={
                    "operation": operation,
                    "flag_field": flag_field_for(kind),
                    "flag_value": _flag_value(updated, kind, default=True),
                    "previous_moderation": str(previous),
                    "moderation": str(state),
                    "idempotent": previous is state,
                    "interface": AUDIT_INTERFACE_ID,
                },
                created_at=self._clock(),
            )
        )
        MODERATION_LOGGER.info(
            "event=%s actor_id=%s %s=%s previous=%s current=%s",
            str(action), actor.user_id, id_field, canonical, previous, state,
        )

    def _authorize(
        self,
        admin_token: str | None,
        *,
        operation: str,
        kind: str,
        target_id: str | None,
        audit: bool = True,
    ) -> Principal:
        """``INV-auth-session`` then ``INV-role-check`` — in that order."""
        actor = require_auth(admin_token, self._resolver)
        try:
            require_admin(actor, reason="administrator role required")
        except ContractError as exc:
            if exc.code is ErrorCode.PERMISSION_DENIED and audit:
                self._record_denial(actor, operation=operation, kind=kind,
                                    target_id=target_id, reason=exc.envelope.message)
            raise
        return actor

    def _record_denial(self, actor: Principal, *, operation: str, kind: str,
                       target_id: str | None, reason: str) -> None:
        """One ``privilege_denied`` record per rejected attempt (never skipped)."""
        id_field = _OPERATIONS[operation][3]
        canonical = canonical_target_id(target_id)
        entry = ModerationAuditLog.denied(
            user_id=actor.user_id,
            operation=operation,
            **({id_field: canonical} if canonical else {}),
            reason=reason or "administrator role required",
            actual_role=str(actor.role),
            created_at=self._clock(),
        )
        SECURITY_LOGGER.warning(
            "event=privilege_denied actor_id=%s operation=%s role=%s target=%s",
            actor.user_id, operation, actor.role, canonical,
        )
        self._record(entry)

    def _record(self, entry: ModerationAuditLog) -> None:
        """Append an audit entry; a broken sink must not mask the outcome."""
        try:
            self._audit.append(entry)
        except Exception:  # pragma: no cover - storage failure is reported, not fatal
            MODERATION_LOGGER.exception("failed to persist moderation audit entry")

    def _require_target(
        self,
        store: ModerationTargetStore,
        target_id: str,
        *,
        actor: Principal,
        not_found: ErrorCode,
        id_field: str,
    ) -> Any:
        """Existence + moderatability, answered by the shared contract predicate.

        ``VisibilityScope.MODERATION`` is exactly the admin queue semantics:
        missing rows, tombstones and corrupt lifecycle flags are all invisible
        there, hence they all collapse into one 404 (no existence leak).
        """
        record = store.load(target_id)
        if record is None or not is_visible_to(
            actor, record, scope=VisibilityScope.MODERATION
        ):
            raise ContractError(not_found, None, {id_field: target_id})
        return record

    @staticmethod
    def _previous_state(
        record: Any,
        target_id: str,
        *,
        not_found: ErrorCode,
        id_field: str,
    ) -> ModerationState:
        """Read the stored flag; a corrupt row is answered as 404, never 500."""
        try:
            return moderation_of(record)
        except ContractError as exc:
            if exc.code is ErrorCode.DATABASE_ERROR:
                raise ContractError(not_found, None, {id_field: target_id}) from exc
            raise

    def _store_for(self, kind: str) -> tuple[ModerationTargetStore, str]:
        if kind == str(ModerationTargetKind.NOTE):
            return self._notes, "note"
        return self._comments, "comment"

    @staticmethod
    def _result(func: Callable[..., None], *args: Any) -> None | ErrorEnvelope:
        try:
            return func(*args)
        except ContractError as exc:
            return exc.to_envelope()


# --------------------------------------------------------------------------- #
# module helpers
# --------------------------------------------------------------------------- #
def _kind_of(value: ModerationTargetKind | str) -> str:
    return parse_target_kind(value)


def _operation_name(kind: str) -> str:
    return "unlist_note" if kind == str(ModerationTargetKind.NOTE) else "block_comment"


def _reason_for(kind: str) -> str:
    return (
        "note unlisted by administrator"
        if kind == str(ModerationTargetKind.NOTE)
        else "comment blocked by administrator"
    )


def _safe(value: Any) -> str:
    text = value if isinstance(value, str) else repr(value)
    return text[:64]


def _flag_value(record: Any, kind: str, *, default: bool) -> bool:
    """Boolean view of the freshly written row (falls back to *default*)."""
    try:
        return flag_of(record, kind)
    except ContractError:
        return default
