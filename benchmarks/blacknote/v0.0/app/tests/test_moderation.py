"""Acceptance tests for ``mod-moderation`` (T1-C1 behaviour + T1-C2 review gates).

Everything is exercised through the frozen surface of ``IF-moderation-actions``:

* ``unlist_note(admin_token, note_id) -> None | ErrorEnvelope``
* ``block_comment(admin_token, comment_id) -> None | ErrorEnvelope``

No sibling business module is imported: the account dependency is a stub
session resolver, and storage is either this layer's own stores or the fake
``get``/``save`` repositories defined below (which also exercises the
duck-typed ports of :mod:`blacknote.moderation.ports`).

Error convention: failures are raised as ``ContractError`` carrying the frozen
``ErrorEnvelope`` (the convention of ``mod-note`` / ``mod-account``), so
assertions read ``excinfo.value.envelope.code``.
"""

from __future__ import annotations

import ast
import inspect
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from blacknote.contracts.auth_guard import Principal
from blacknote.contracts.config import Settings
from blacknote.contracts.errors import ENVELOPE_KEYS, ContractError, ErrorCode
from blacknote.contracts.ids import is_valid_id, new_id
from blacknote.contracts.interfaces import (
    ModerationApi,
    interface_errors,
    interface_functions,
)
from blacknote.contracts.roles import Role
from blacknote.contracts.timeutil import ISO8601_UTC_PATTERN, now_iso, to_iso
from blacknote.contracts.visibility import (
    ContentStatus,
    ModerationState,
    Visibility,
    VisibilityScope,
    is_visible_to,
)
from blacknote.moderation import (
    AUDIT_COLUMNS,
    FLAG_FIELD,
    TARGET_STATE,
    InMemoryAuditLogStore,
    InMemoryCommentStore,
    InMemoryModerationStore,
    InMemoryNoteStore,
    ModeratableComment,
    ModeratableNote,
    ModerationAction,
    ModerationAuditLog,
    ModerationService,
    SQLiteAuditLogStore,
    flag_of,
    moderation_of,
)

MODERATION_DIR = Path(__file__).resolve().parents[1] / "src" / "blacknote" / "moderation"
BASE_MOMENT = datetime(2026, 3, 1, 12, 0, 0, tzinfo=timezone.utc)
FROZEN_ERRORS: tuple[str, ...] = interface_errors("IF-moderation-actions")
CREATED = "2026-03-01T10:00:00.000000Z"


# --------------------------------------------------------------------------- #
# fakes owned by this test module (自有 fake repositories)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class NoteRow:
    """The shape ``mod-note`` persists; moderation may only touch the flags."""

    id: str
    author_id: str
    title: str = "note"
    body: str = "body"
    status: str = str(ContentStatus.PUBLISHED)
    visibility: str = str(Visibility.PUBLIC)
    moderation: str = str(ModerationState.VISIBLE)
    moderated: bool = False
    created_at: str = CREATED
    updated_at: str = CREATED


@dataclass(frozen=True)
class CommentRow:
    """The shape ``mod-interaction`` persists."""

    id: str
    note_id: str
    author_id: str
    content: str = "reply"
    parent_comment_id: str | None = None
    status: str = str(ContentStatus.PUBLISHED)
    moderation: str = str(ModerationState.VISIBLE)
    blocked: bool = False
    created_at: str = CREATED
    updated_at: str = CREATED


class StubSessions:
    """Stand-in for the ``mod-account`` session resolver port."""

    def __init__(self) -> None:
        self.by_token: dict[str, Principal] = {}

    def login(
        self,
        user_id: str | None = None,
        *,
        role: Role = Role.USER,
        expires_at: str | None = None,
    ) -> str:
        token = new_id()
        self.by_token[token] = Principal(
            user_id=user_id or new_id(), token=token, role=role, expires_at=expires_at
        )
        return token

    def resolve(self, token: str) -> Principal | None:
        return self.by_token.get(token)

    def logout(self, token: str) -> None:
        self.by_token.pop(token, None)

    def demote(self, token: str) -> None:
        principal = self.by_token[token]
        self.by_token[token] = Principal(
            user_id=principal.user_id, token=token, role=Role.USER
        )


class FakeNoteRepository:
    """A ``get``/``save`` repository over :class:`NoteRow` (no validation)."""

    record_type = NoteRow

    def __init__(self) -> None:
        self.rows: dict[str, NoteRow] = {}
        self.saves: list[str] = []

    def put(self, row: NoteRow) -> NoteRow:
        self.rows[row.id] = row
        return row

    insert = put

    def get(self, note_id: str) -> NoteRow | None:
        return self.rows.get(note_id)

    def save(self, row: NoteRow) -> NoteRow:
        self.saves.append(row.id)
        return self.put(row)


class FakeCommentRepository(FakeNoteRepository):
    """Same shape for comments; ``saves`` proves which rows were written."""

    record_type = CommentRow

    def put(self, row: CommentRow) -> CommentRow:  # type: ignore[override]
        self.rows[row.id] = row
        return row

    insert = put

    def get(self, comment_id: str) -> CommentRow | None:  # type: ignore[override]
        return self.rows.get(comment_id)

    def save(self, row: CommentRow) -> CommentRow:  # type: ignore[override]
        self.saves.append(row.id)
        return self.put(row)


class RecordingAuditSink:
    """An audit sink that only knows ``append`` (covers the wrapping adapter)."""

    def __init__(self) -> None:
        self.entries: list[ModerationAuditLog] = []

    def append(self, entry: ModerationAuditLog) -> ModerationAuditLog:
        self.entries.append(entry)
        return entry


class ExplodingAuditSink:
    def __init__(self) -> None:
        self.attempts = 0

    def append(self, entry: ModerationAuditLog) -> None:
        self.attempts += 1
        raise RuntimeError("audit storage is down")


class Clock:
    """Deterministic, strictly increasing UTC clock."""

    def __init__(self, start: datetime = BASE_MOMENT) -> None:
        self._moment = start

    def __call__(self) -> str:
        self._moment += timedelta(seconds=5)
        return to_iso(self._moment)


class Harness:
    """Wires one service against stub sessions + fake stores."""

    def __init__(
        self,
        *,
        notes: Any = "native",
        comments: Any = "native",
        audit: Any = None,
        sessions: Any = "stub",
        settings: Settings | None = None,
    ) -> None:
        self.sessions = StubSessions() if sessions == "stub" else sessions
        self.notes = _make_store(notes, "note")
        self.comments = _make_store(comments, "comment")
        self.audit = InMemoryAuditLogStore() if audit is None else audit
        self.clock = Clock()
        self.service = ModerationService(
            self.notes,
            self.comments,
            audit=self.audit,
            sessions=self.sessions,
            settings=settings or Settings(),
            clock=self.clock,
        )
        self.author_id = new_id()
        self.admin_id = new_id()
        self.admin_token = ""
        self.user_token = ""
        if isinstance(self.sessions, StubSessions):
            self.admin_token = self.sessions.login(self.admin_id, role=Role.ADMIN)
            self.user_token = self.sessions.login(self.author_id, role=Role.USER)

    # -- population -----------------------------------------------------------
    def make_note(self, *, author_id: str | None = None, **fields: Any) -> str:
        note_id = fields.pop("id", None) or new_id()
        record_type = getattr(self.notes, "record_type", NoteRow)
        fields.setdefault("created_at", CREATED)
        fields.setdefault("updated_at", CREATED)
        row = record_type(id=note_id, author_id=author_id or self.author_id, **fields)
        self.notes.insert(row)
        return note_id

    def make_comment(
        self, note_id: str | None = None, *, author_id: str | None = None, **fields: Any
    ) -> str:
        comment_id = fields.pop("id", None) or new_id()
        record_type = getattr(self.comments, "record_type", CommentRow)
        fields.setdefault("created_at", CREATED)
        fields.setdefault("updated_at", CREATED)
        row = record_type(
            id=comment_id,
            note_id=note_id or self.make_note(),
            author_id=author_id or self.author_id,
            **fields,
        )
        self.comments.insert(row)
        return comment_id

    # -- inspection -----------------------------------------------------------
    def note_row(self, note_id: str) -> Any:
        return self.notes.get(note_id)

    def comment_row(self, comment_id: str) -> Any:
        return self.comments.get(comment_id)

    def entries(self, action: str | None = None) -> list[ModerationAuditLog]:
        return list(self.service.audit_store.list(action=action))

    def saves(self) -> list[str]:
        return list(getattr(self.notes, "saves", [])) + list(
            getattr(self.comments, "saves", [])
        )


def _make_store(kind: Any, which: str) -> Any:
    if kind == "native":
        return InMemoryNoteStore() if which == "note" else InMemoryCommentStore()
    if kind == "repo":
        return FakeNoteRepository() if which == "note" else FakeCommentRepository()
    return kind


@pytest.fixture
def h() -> Harness:
    return Harness()


@pytest.fixture
def hr() -> Harness:
    """Same service wired over the generic ``get``/``save`` repository shape."""
    return Harness(notes="repo", comments="repo")


# --------------------------------------------------------------------------- #
# frozen surface
# --------------------------------------------------------------------------- #
class TestFrozenSurface:
    def test_service_satisfies_the_frozen_protocol(self, h: Harness) -> None:
        assert isinstance(h.service, ModerationApi)
        assert tuple(interface_functions("IF-moderation-actions")) == (
            "unlist_note",
            "block_comment",
        )
        def shape(function: Any) -> tuple[str, ...]:
            parameters = inspect.signature(function).parameters
            return tuple(
                f"{name}:{parameters[name].kind.description}"
                for name in parameters
            ) + (f"return:{inspect.signature(function).return_annotation}",)

        assert shape(h.service.unlist_note) == (
            "admin_token:positional or keyword",
            "note_id:positional or keyword",
            "return:None",
        )
        assert shape(h.service.block_comment) == (
            "admin_token:positional or keyword",
            "comment_id:positional or keyword",
            "return:None",
        )

    def test_success_returns_none(self, h: Harness) -> None:
        note_id = h.make_note()
        comment_id = h.make_comment(note_id)
        assert h.service.unlist_note(h.admin_token, note_id) is None
        assert h.service.block_comment(h.admin_token, comment_id) is None

    def test_non_raising_variants_return_the_envelope_union(self, h: Harness) -> None:
        note_id = h.make_note()
        assert h.service.unlist_note_result(h.admin_token, note_id) is None
        envelope = h.service.unlist_note_result(h.user_token, note_id)
        assert envelope is not None
        payload = envelope.to_dict()
        assert set(payload) <= set(ENVELOPE_KEYS)
        assert payload["code"] == str(ErrorCode.PERMISSION_DENIED)

    def test_layer_exposes_no_read_filtering(self, h: Harness) -> None:
        """This layer writes flags; read paths belong to the other modules."""
        public = {name for name in dir(h.service) if not name.startswith("_")}
        assert {"unlist_note", "block_comment"} <= public
        allowed_reads = {"list_audit_log"}  # the audit trail is this layer's own data
        offenders = [
            name
            for name in sorted(public)
            if name not in allowed_reads
            and name.startswith(("list", "get", "read", "search", "feed", "filter",
                                 "visible", "iter", "fetch"))
        ]
        assert offenders == [], offenders


# --------------------------------------------------------------------------- #
# unlist_note → Note.moderated = true
# --------------------------------------------------------------------------- #
class TestUnlistNote:
    @pytest.mark.parametrize("shape", ["native", "repo"])
    def test_sets_the_moderated_flag(self, shape: str) -> None:
        h = Harness(notes=shape, comments=shape)
        note_id = h.make_note()
        assert h.service.unlist_note(h.admin_token, note_id) is None
        row = h.note_row(note_id)
        assert row.moderated is True
        assert row.moderation == str(ModerationState.UNLISTED)
        assert moderation_of(row) is ModerationState.UNLISTED
        assert flag_of(row, "note") is True

    def test_written_state_comes_from_the_contract(self) -> None:
        assert TARGET_STATE["note"] is ModerationState.UNLISTED
        assert TARGET_STATE["comment"] is ModerationState.BLOCKED

    def test_content_is_untouched(self, hr: Harness) -> None:
        note_id = hr.make_note(title="keep me", body="keep the body")
        before = hr.note_row(note_id)
        hr.service.unlist_note(hr.admin_token, note_id)
        after = hr.note_row(note_id)
        assert after.author_id == before.author_id
        assert after.title == "keep me" and after.body == "keep the body"
        assert after.status == before.status and after.visibility == before.visibility

    def test_updated_at_is_refreshed_in_utc_iso8601(self, h: Harness) -> None:
        note_id = h.make_note()
        h.service.unlist_note(h.admin_token, note_id)
        row = h.note_row(note_id)
        assert ISO8601_UTC_PATTERN.match(row.updated_at), row.updated_at
        assert row.created_at == CREATED
        assert row.updated_at > row.created_at

    def test_shared_predicate_hides_the_note_after_the_flag_write(self, h: Harness) -> None:
        note_id = h.make_note()
        h.service.unlist_note(h.admin_token, note_id)
        row = h.note_row(note_id)
        assert is_visible_to(None, row) is False
        assert is_visible_to(Principal(user_id=h.author_id), row,
                             scope=VisibilityScope.SELF) is True
        admin = Principal(user_id=h.admin_id, role=Role.ADMIN)
        assert is_visible_to(admin, row, scope=VisibilityScope.MODERATION) is True

    def test_repeating_the_takedown_is_idempotent(self, h: Harness) -> None:
        note_id = h.make_note()
        h.service.unlist_note(h.admin_token, note_id)
        first = h.note_row(note_id)
        h.service.unlist_note(h.admin_token, note_id)
        second = h.note_row(note_id)
        assert second.moderated is True and str(second.moderation) == str(first.moderation)
        # the audit trail is newest-first: the second takedown knows it changed nothing
        assert [entry.details["idempotent"]
                for entry in reversed(h.entries("note_unlisted"))] == [False, True]

    def test_works_through_the_note_service_seam(self, h: Harness) -> None:
        """``apply_moderation_state``/``inspect_note_record`` are used when present."""

        class NoteServiceDouble:
            def __init__(self, author_id: str) -> None:
                self.record = ModeratableNote(id=new_id(), author_id=author_id, title="t")
                self.applied: list[str] = []

            def inspect_note_record(self, note_id: str) -> Any | None:
                return self.record if note_id == self.record.id else None

            def apply_moderation_state(self, note_id: str, state: ModerationState) -> Any:
                self.applied.append(str(state))
                self.record = replace(self.record, moderation=state)
                return self.record

        double = NoteServiceDouble(h.author_id)
        service = ModerationService(double, audit=h.audit, sessions=h.sessions, clock=h.clock)
        service.unlist_note(h.admin_token, double.record.id)
        assert double.applied == [str(ModerationState.UNLISTED)]
        assert double.record.moderated is True

    def test_works_over_plain_dict_rows(self, h: Harness) -> None:
        notes: dict[str, Any] = {}
        service = ModerationService(notes, audit=h.audit, sessions=h.sessions, clock=h.clock)
        note_id = new_id()
        notes[note_id] = {
            "id": note_id,
            "author_id": h.author_id,
            "status": str(ContentStatus.PUBLISHED),
            "visibility": str(Visibility.PUBLIC),
            "moderation": str(ModerationState.VISIBLE),
            "moderated": False,
            "created_at": CREATED,
            "updated_at": CREATED,
        }
        service.unlist_note(h.admin_token, note_id)
        assert notes[note_id]["moderated"] is True
        assert notes[note_id]["moderation"] == str(ModerationState.UNLISTED)

    def test_only_the_target_row_is_saved(self, hr: Harness) -> None:
        other = hr.make_note()
        target = hr.make_note()
        hr.service.unlist_note(hr.admin_token, target)
        assert hr.saves() == [target]
        assert hr.note_row(other).moderated is False


# --------------------------------------------------------------------------- #
# block_comment → Comment.blocked = true
# --------------------------------------------------------------------------- #
class TestBlockComment:
    @pytest.mark.parametrize("shape", ["native", "repo"])
    def test_sets_the_blocked_flag(self, shape: str) -> None:
        h = Harness(notes=shape, comments=shape)
        note_id = h.make_note()
        comment_id = h.make_comment(note_id)
        assert h.service.block_comment(h.admin_token, comment_id) is None
        row = h.comment_row(comment_id)
        assert row.blocked is True
        assert row.moderation == str(ModerationState.BLOCKED)
        assert flag_of(row, "comment") is True

    def test_blocked_comment_is_hidden_by_the_shared_predicate(self, h: Harness) -> None:
        comment_id = h.make_comment()
        h.service.block_comment(h.admin_token, comment_id)
        assert is_visible_to(None, h.comment_row(comment_id)) is False
        admin = Principal(user_id=h.admin_id, role=Role.ADMIN)
        assert is_visible_to(admin, h.comment_row(comment_id),
                             scope=VisibilityScope.MODERATION) is True

    def test_replies_keep_their_own_flags(self, hr: Harness) -> None:
        """Ancestor filtering belongs to the predicate: only the target is written."""
        note_id = hr.make_note()
        parent = hr.make_comment(note_id)
        child = hr.make_comment(note_id, parent_comment_id=parent)
        hr.service.block_comment(hr.admin_token, parent)
        assert hr.comment_row(parent).blocked is True
        assert hr.comment_row(child).blocked is False
        assert str(hr.comment_row(child).moderation) == str(ModerationState.VISIBLE)
        assert hr.comment_row(child).updated_at == CREATED
        assert hr.saves() == [parent]

    def test_flags_never_bleed_between_kinds(self, h: Harness) -> None:
        note_id = h.make_note()
        comment_id = h.make_comment(note_id)
        h.service.block_comment(h.admin_token, comment_id)
        assert h.note_row(note_id).moderated is False
        h.service.unlist_note(h.admin_token, note_id)
        assert h.comment_row(comment_id).blocked is True
        assert str(h.comment_row(comment_id).moderation) == str(ModerationState.BLOCKED)

    def test_the_comment_store_can_be_bound_later(self) -> None:
        """``mod-web`` may inject the interaction repository after construction."""
        sessions = StubSessions()
        admin = sessions.login(new_id(), role=Role.ADMIN)
        comments = FakeCommentRepository()
        comment_id = new_id()
        comments.rows[comment_id] = CommentRow(id=comment_id, note_id=new_id(),
                                               author_id=new_id())
        service = ModerationService(sessions=sessions, clock=Clock())
        service.bind_comments(comments)
        service.block_comment(admin, comment_id)
        assert comments.saves == [comment_id]
        assert comments.rows[comment_id].blocked is True


# --------------------------------------------------------------------------- #
# audit trail
# --------------------------------------------------------------------------- #
class TestAuditTrail:
    def test_unlist_writes_a_note_unlisted_record(self, h: Harness) -> None:
        note_id = h.make_note()
        h.service.unlist_note(h.admin_token, note_id)
        (entry,) = h.entries("note_unlisted")
        assert entry.action == "note_unlisted"
        assert entry.user_id == h.admin_id
        assert entry.note_id == note_id
        assert entry.comment_id is None
        assert is_valid_id(entry.id)
        assert ISO8601_UTC_PATTERN.match(entry.created_at)
        assert entry.details["interface"] == "IF-moderation-actions"
        assert entry.details["flag_field"] == "moderated"
        assert entry.details["flag_value"] is True
        assert entry.details["moderation"] == str(ModerationState.UNLISTED)
        assert entry.target_kind == "note"
        assert entry.is_denial is False

    def test_block_writes_a_comment_blocked_record(self, h: Harness) -> None:
        comment_id = h.make_comment()
        h.service.block_comment(h.admin_token, comment_id)
        (entry,) = h.entries("comment_blocked")
        assert entry.action == "comment_blocked"
        assert entry.comment_id == comment_id
        assert entry.note_id is None
        assert entry.details["flag_field"] == "blocked"
        assert entry.target_kind == "comment"

    def test_actions_are_the_frozen_vocabulary(self, h: Harness) -> None:
        assert {str(item) for item in ModerationAction} == {
            "note_unlisted",
            "comment_blocked",
            "privilege_denied",
        }
        note_id = h.make_note()
        comment_id = h.make_comment(note_id)
        h.service.unlist_note(h.admin_token, note_id)
        h.service.block_comment(h.admin_token, comment_id)
        with pytest.raises(ContractError):
            h.service.unlist_note(h.user_token, note_id)
        assert {entry.action for entry in h.entries()} == {
            "note_unlisted",
            "comment_blocked",
            "privilege_denied",
        }
        assert set(ModerationAction) == {
            ModerationAction.NOTE_UNLISTED,
            ModerationAction.COMMENT_BLOCKED,
            ModerationAction.PRIVILEGE_DENIED,
        }

    def test_record_fields_match_the_model_and_columns(self, h: Harness) -> None:
        """Audit fields stay consistent with ``ModerationAuditLog``."""
        note_id = h.make_note()
        comment_id = h.make_comment(note_id)
        h.service.unlist_note(h.admin_token, note_id)
        h.service.block_comment(h.admin_token, comment_id)
        with pytest.raises(ContractError):
            h.service.block_comment(h.user_token, comment_id)
        assert tuple(ModerationAuditLog.__dataclass_fields__) == AUDIT_COLUMNS
        recorded = h.entries()
        assert len(recorded) == 3
        for entry in recorded:
            assert isinstance(entry, ModerationAuditLog)
            assert tuple(entry.to_dict()) == AUDIT_COLUMNS
            assert tuple(entry.to_row()) == AUDIT_COLUMNS
            assert ModerationAuditLog.from_row(entry.to_row()) == entry
            assert entry.details.get("interface") == "IF-moderation-actions"

    def test_sqlite_schema_matches_the_model(self, tmp_path: Path) -> None:
        path = tmp_path / "audit.db"
        with SQLiteAuditLogStore(f"sqlite:///{path}") as store:
            store.append(
                ModerationAuditLog.create(
                    ModerationAction.PRIVILEGE_DENIED,
                    user_id=new_id(),
                    reason="administrator role required",
                    details={"operation": "unlist_note"},
                )
            )
        conn = sqlite3.connect(path)
        try:
            columns = [row[1] for row in conn.execute("PRAGMA table_info(moderation_audit_log)")]
        finally:
            conn.close()
        assert columns == list(AUDIT_COLUMNS)

    def test_audit_trail_survives_a_restart(self, tmp_path: Path) -> None:
        path = tmp_path / "audit-persist.db"
        sessions = StubSessions()
        admin_id = new_id()
        admin = sessions.login(admin_id, role=Role.ADMIN)
        intruder = sessions.login(new_id(), role=Role.USER)
        notes = InMemoryNoteStore()
        note_id = notes.add(id=new_id(), author_id=new_id()).id
        first = SQLiteAuditLogStore(f"sqlite:///{path}")
        service = ModerationService(notes, audit=first, sessions=sessions, clock=Clock())
        service.unlist_note(admin, note_id)
        with pytest.raises(ContractError):
            service.unlist_note(intruder, note_id)
        first.close()

        with SQLiteAuditLogStore(f"sqlite:///{path}") as reopened:
            assert reopened.count() == 2
            assert reopened.count(action="privilege_denied") == 1
            (denial,) = reopened.list(action="privilege_denied")
            assert denial.user_id != admin_id
            assert denial.details["operation"] == "unlist_note"
            assert denial.details["actual_role"] == str(Role.USER)
            assert denial.note_id == note_id
            (takedown,) = reopened.list(action="note_unlisted")
            assert takedown.user_id == admin_id
        assert path.exists()

    def test_listing_and_counting_the_trail(self, h: Harness) -> None:
        note_id = h.make_note()
        comment_id = h.make_comment(note_id)
        h.service.unlist_note(h.admin_token, note_id)
        h.service.block_comment(h.admin_token, comment_id)
        assert [entry.action for entry in h.service.list_audit_log()] == [
            "comment_blocked",
            "note_unlisted",
        ]
        assert h.service.audit_count(action="note_unlisted") == 1
        assert h.service.audit_count() == 2
        assert [e.action for e in h.service.list_audit_log(note_id=note_id)] == [
            "note_unlisted"
        ]
        assert [e.action for e in h.service.list_audit_log(comment_id=comment_id)] == [
            "comment_blocked"
        ]
        assert h.service.list_audit_log(limit=1)[0].action == "comment_blocked"

    def test_an_append_only_sink_is_accepted(self) -> None:
        sink = RecordingAuditSink()
        sessions = StubSessions()
        admin = sessions.login(new_id(), role=Role.ADMIN)
        notes = InMemoryNoteStore()
        note_id = notes.add(id=new_id(), author_id=new_id()).id
        service = ModerationService(notes, audit=sink, sessions=sessions, clock=Clock())
        service.unlist_note(admin, note_id)
        assert [entry.action for entry in sink.entries] == ["note_unlisted"]
        assert service.audit_count(action="note_unlisted") == 1

    def test_a_broken_audit_sink_does_not_change_the_outcome(self) -> None:
        sink = ExplodingAuditSink()
        sessions = StubSessions()
        admin = sessions.login(new_id(), role=Role.ADMIN)
        intruder = sessions.login(new_id(), role=Role.USER)
        notes = InMemoryNoteStore()
        note_id = notes.add(id=new_id(), author_id=new_id()).id
        service = ModerationService(notes, audit=sink, sessions=sessions, clock=Clock())
        assert service.unlist_note(admin, note_id) is None
        with pytest.raises(ContractError) as excinfo:
            service.unlist_note(intruder, note_id)
        assert excinfo.value.code is ErrorCode.PERMISSION_DENIED
        assert sink.attempts == 2
        assert notes.get(note_id).moderated is True


# --------------------------------------------------------------------------- #
# privilege escalation must always fail (INV-role-check)
# --------------------------------------------------------------------------- #
class TestPrivilegeDenial:
    @pytest.mark.parametrize("operation", ["unlist_note", "block_comment"])
    def test_non_admin_gets_permission_denied(self, h: Harness, operation: str) -> None:
        note_id = h.make_note()
        comment_id = h.make_comment(note_id)
        target = note_id if operation == "unlist_note" else comment_id
        with pytest.raises(ContractError) as excinfo:
            getattr(h.service, operation)(h.user_token, target)
        error = excinfo.value
        assert error.code is ErrorCode.PERMISSION_DENIED
        assert error.http_status == 403
        assert error.envelope.to_dict()["details"] == {
            "required_roles": [str(Role.ADMIN)],
            "actual_role": str(Role.USER),
        }

    @pytest.mark.parametrize("operation", ["unlist_note", "block_comment"])
    def test_target_state_is_untouched(self, hr: Harness, operation: str) -> None:
        note_id = hr.make_note()
        comment_id = hr.make_comment(note_id)
        before = (hr.note_row(note_id), hr.comment_row(comment_id))
        target = note_id if operation == "unlist_note" else comment_id
        with pytest.raises(ContractError):
            getattr(hr.service, operation)(hr.user_token, target)
        after = (hr.note_row(note_id), hr.comment_row(comment_id))
        assert after == before
        assert after[0].moderated is False and after[1].blocked is False
        assert hr.saves() == []
        assert hr.entries("note_unlisted") == [] and hr.entries("comment_blocked") == []

    @pytest.mark.parametrize("operation", ["unlist_note", "block_comment"])
    def test_every_denied_attempt_is_recorded(self, h: Harness, operation: str) -> None:
        note_id = h.make_note()
        comment_id = h.make_comment(note_id)
        target = note_id if operation == "unlist_note" else comment_id
        for _ in range(3):
            with pytest.raises(ContractError):
                getattr(h.service, operation)(h.user_token, target)
        denials = h.entries("privilege_denied")
        assert len(denials) == 3
        assert {entry.action for entry in denials} == {"privilege_denied"}
        assert {entry.user_id for entry in denials} == {h.author_id}
        assert {entry.target_id for entry in denials} == {target}
        assert all(entry.details["operation"] == operation for entry in denials)
        assert all(entry.details["required_roles"] == [str(Role.ADMIN)] for entry in denials)
        assert all(entry.details["actual_role"] == str(Role.USER) for entry in denials)
        assert all(ISO8601_UTC_PATTERN.match(entry.created_at) for entry in denials)
        assert all(entry.is_denial and entry.target_kind is None for entry in denials)
        assert all(entry.reason for entry in denials)

    def test_escalation_success_rate_is_zero(self, h: Harness) -> None:
        note_id = h.make_note()
        comment_id = h.make_comment(note_id)
        attempts = successes = 0
        for _ in range(10):
            for operation, target in (
                ("unlist_note", note_id),
                ("block_comment", comment_id),
            ):
                attempts += 1
                try:
                    getattr(h.service, operation)(h.user_token, target)
                except ContractError as exc:
                    assert exc.code is ErrorCode.PERMISSION_DENIED
                else:
                    successes += 1
        assert attempts == 20 and successes == 0
        assert h.note_row(note_id).moderated is False
        assert h.comment_row(comment_id).blocked is False
        assert h.service.audit_count(action="privilege_denied") == attempts
        assert h.service.audit_count(action="note_unlisted") == 0
        assert h.service.audit_count(action="comment_blocked") == 0

    def test_denial_is_decided_before_the_target_lookup(self, h: Harness) -> None:
        """A non-admin never learns whether an id exists (no 404 before 403)."""
        with pytest.raises(ContractError) as excinfo:
            h.service.unlist_note(h.user_token, new_id())
        assert excinfo.value.code is ErrorCode.PERMISSION_DENIED
        assert h.service.audit_count(action="privilege_denied") == 1

    def test_role_probe_does_not_pollute_the_trail(self, h: Harness) -> None:
        note_id = h.make_note()
        assert h.service.is_admin_token(h.user_token) is False
        assert h.service.can_moderate_target(h.user_token, "note", note_id) is False
        assert h.service.can_moderate_target(h.admin_token, "note", note_id) is True
        assert h.service.audit_count(action="privilege_denied") == 0

    def test_a_demoted_admin_is_denied_immediately(self) -> None:
        """The role comes from the session, never from a cached admin token."""
        sessions = StubSessions()
        token = sessions.login(new_id(), role=Role.ADMIN)
        comments = InMemoryCommentStore()
        comment_id = comments.add(id=new_id(), note_id=new_id(), author_id=new_id()).id
        service = ModerationService(audit=InMemoryAuditLogStore(), sessions=sessions,
                                    clock=Clock())
        service.bind_comments(comments)
        assert service.block_comment(token, comment_id) is None
        sessions.demote(token)
        with pytest.raises(ContractError) as excinfo:
            service.block_comment(token, comment_id)
        assert excinfo.value.code is ErrorCode.PERMISSION_DENIED
        assert excinfo.value.envelope.to_dict()["details"]["actual_role"] == str(Role.USER)


# --------------------------------------------------------------------------- #
# unauthenticated callers (INV-auth-session)
# --------------------------------------------------------------------------- #
class TestAuthentication:
    @pytest.mark.parametrize(
        "token", [None, "", "   ", "not-a-uuid", "0" * 8, 12345],
        ids=["none", "empty", "blank", "malformed", "short", "non-string"],
    )
    def test_unauthenticated_attempt_is_auth_required(self, h: Harness, token: Any) -> None:
        note_id = h.make_note()
        comment_id = h.make_comment(note_id)
        for operation, target in (("unlist_note", note_id), ("block_comment", comment_id)):
            with pytest.raises(ContractError) as excinfo:
                getattr(h.service, operation)(token, target)
            assert excinfo.value.code is ErrorCode.AUTH_REQUIRED
            assert excinfo.value.http_status == 401

    def test_nothing_at_all_is_written_when_unauthenticated(self, hr: Harness) -> None:
        note_id = hr.make_note()
        comment_id = hr.make_comment(note_id)
        with pytest.raises(ContractError):
            hr.service.unlist_note(new_id(), note_id)  # unknown but well-formed token
        with pytest.raises(ContractError):
            hr.service.block_comment("nope", comment_id)
        assert hr.note_row(note_id).moderated is False
        assert hr.comment_row(comment_id).blocked is False
        assert hr.saves() == []
        assert hr.service.audit_count() == 0  # an authN failure is not an authZ record

    def test_revoked_token_is_auth_required(self, h: Harness) -> None:
        note_id = h.make_note()
        h.sessions.logout(h.admin_token)
        with pytest.raises(ContractError) as excinfo:
            h.service.unlist_note(h.admin_token, note_id)
        assert excinfo.value.code is ErrorCode.AUTH_REQUIRED
        assert h.note_row(note_id).moderated is False

    def test_expired_session_is_rejected(self) -> None:
        sessions = StubSessions()
        expired = sessions.login(
            new_id(), role=Role.ADMIN, expires_at=to_iso(BASE_MOMENT - timedelta(minutes=1))
        )
        notes = InMemoryNoteStore()
        note_id = notes.add(id=new_id(), author_id=new_id()).id
        audit = InMemoryAuditLogStore()
        service = ModerationService(notes, audit=audit, sessions=sessions, clock=Clock())
        with pytest.raises(ContractError) as excinfo:
            service.unlist_note(expired, note_id)
        assert excinfo.value.code is ErrorCode.SESSION_EXPIRED
        assert notes.get(note_id).moderated is False
        assert audit.count() == 0

    def test_a_miswired_resolver_fails_closed(self) -> None:
        """Without a session port nothing can be authorised, and nothing writes."""
        notes = InMemoryNoteStore()
        note_id = notes.add(id=new_id(), author_id=new_id()).id
        audit = InMemoryAuditLogStore()
        service = ModerationService(notes, audit=audit, sessions=None, clock=Clock())
        with pytest.raises(ContractError) as excinfo:
            service.unlist_note(new_id(), note_id)
        assert excinfo.value.code is ErrorCode.INTERNAL_ERROR
        assert notes.get(note_id).moderated is False
        assert audit.count() == 0


# --------------------------------------------------------------------------- #
# missing / unreachable targets
# --------------------------------------------------------------------------- #
class TestMissingTargets:
    def test_unknown_note_id_returns_note_not_found(self, h: Harness) -> None:
        with pytest.raises(ContractError) as excinfo:
            h.service.unlist_note(h.admin_token, new_id())
        error = excinfo.value
        assert error.code is ErrorCode.NOTE_NOT_FOUND
        assert error.http_status == 404
        assert is_valid_id(error.envelope.details["note_id"])

    def test_unknown_comment_id_returns_comment_not_found(self, h: Harness) -> None:
        with pytest.raises(ContractError) as excinfo:
            h.service.block_comment(h.admin_token, new_id())
        assert excinfo.value.code is ErrorCode.COMMENT_NOT_FOUND
        assert excinfo.value.http_status == 404
        assert is_valid_id(excinfo.value.envelope.details["comment_id"])

    @pytest.mark.parametrize(
        "bad_id", ["", "   ", "note-1", "0" * 8, "nope-nope-nope", 42, None],
        ids=["empty", "blank", "slug", "short", "word", "int", "none"],
    )
    def test_malformed_ids_are_answered_as_missing(self, h: Harness, bad_id: Any) -> None:
        with pytest.raises(ContractError) as note_exc:
            h.service.unlist_note(h.admin_token, bad_id)
        assert note_exc.value.code is ErrorCode.NOTE_NOT_FOUND
        with pytest.raises(ContractError) as comment_exc:
            h.service.block_comment(h.admin_token, bad_id)
        assert comment_exc.value.code is ErrorCode.COMMENT_NOT_FOUND
        assert h.saves() == []

    def test_deleted_targets_are_reported_as_missing(self, h: Harness) -> None:
        note_id = h.make_note(status=str(ContentStatus.DELETED))
        comment_id = h.make_comment(status=str(ContentStatus.DELETED))
        with pytest.raises(ContractError) as note_exc:
            h.service.unlist_note(h.admin_token, note_id)
        assert note_exc.value.code is ErrorCode.NOTE_NOT_FOUND
        with pytest.raises(ContractError) as comment_exc:
            h.service.block_comment(h.admin_token, comment_id)
        assert comment_exc.value.code is ErrorCode.COMMENT_NOT_FOUND
        assert h.note_row(note_id).moderated is False
        assert h.comment_row(comment_id).blocked is False
        assert h.service.audit_count() == 0

    def test_a_private_note_is_still_moderatable_by_admin(self, h: Harness) -> None:
        """Moderation consults the admin scope of the predicate, not the public one."""
        note_id = h.make_note(visibility=str(Visibility.PRIVATE))
        assert h.service.unlist_note(h.admin_token, note_id) is None
        assert h.note_row(note_id).moderated is True

    def test_a_corrupt_lifecycle_flag_is_answered_as_missing(self) -> None:
        h = Harness(notes="repo", comments="repo")
        note_id = h.make_note(moderation="not-a-state")
        comment_id = h.make_comment(moderation="not-a-state")
        with pytest.raises(ContractError) as note_exc:
            h.service.unlist_note(h.admin_token, note_id)
        assert note_exc.value.code is ErrorCode.NOTE_NOT_FOUND
        with pytest.raises(ContractError) as comment_exc:
            h.service.block_comment(h.admin_token, comment_id)
        assert comment_exc.value.code is ErrorCode.COMMENT_NOT_FOUND
        assert h.note_row(note_id).moderation == "not-a-state"
        assert h.service.audit_count() == 0

    def test_every_business_code_stays_inside_the_frozen_interface(self, h: Harness) -> None:
        note_id = h.make_note()
        comment_id = h.make_comment(note_id)
        calls = (
            lambda: h.service.unlist_note(None, note_id),
            lambda: h.service.block_comment("", comment_id),
            lambda: h.service.unlist_note(h.user_token, note_id),
            lambda: h.service.block_comment(h.user_token, comment_id),
            lambda: h.service.unlist_note(h.admin_token, new_id()),
            lambda: h.service.block_comment(h.admin_token, new_id()),
            lambda: h.service.unlist_note(h.admin_token, "nonsense"),
            lambda: h.service.block_comment(h.admin_token, "nonsense"),
        )
        seen: set[str] = set()
        for call in calls:
            try:
                call()
            except ContractError as exc:
                seen.add(str(exc.code))
        assert seen <= set(FROZEN_ERRORS), seen - set(FROZEN_ERRORS)
        assert seen == {
            "AUTH_REQUIRED",
            "PERMISSION_DENIED",
            "NOTE_NOT_FOUND",
            "COMMENT_NOT_FOUND",
        }


# --------------------------------------------------------------------------- #
# T1-C2 review gates: no duplicated visibility / role / error logic
# --------------------------------------------------------------------------- #
def _package_trees() -> dict[str, ast.Module]:
    return {
        path.name: ast.parse(path.read_text(encoding="utf-8"))
        for path in sorted(MODERATION_DIR.rglob("*.py"))
    }


class TestNoDuplicatedLogic:
    def test_only_contracts_are_imported_from_the_platform(self) -> None:
        offenders: list[str] = []
        for name, tree in _package_trees().items():
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0:
                    modules = [node.module or ""]
                else:
                    continue  # relative imports stay inside the package
                offenders += [
                    f"{name}: {module}"
                    for module in modules
                    if module.startswith("blacknote")
                    and not module.startswith("blacknote.contracts")
                ]
        assert offenders == [], offenders

    def test_the_shared_predicate_is_the_contract_object(self) -> None:
        import blacknote.contracts.visibility as contract_visibility
        import blacknote.moderation.service as moderation_service

        assert moderation_service.is_visible_to is contract_visibility.is_visible_to
        assert moderation_service.VisibilityScope is contract_visibility.VisibilityScope
        assert moderation_service.ModerationState is contract_visibility.ModerationState

    def test_the_guards_are_the_contract_objects(self) -> None:
        import blacknote.contracts.auth_guard as contract_auth
        import blacknote.contracts.roles as contract_roles
        import blacknote.moderation.service as moderation_service

        assert moderation_service.require_admin is contract_roles.require_admin
        assert moderation_service.require_auth is contract_auth.require_auth

    def test_no_local_predicate_or_guard_is_redefined(self) -> None:
        offenders: list[str] = []
        for name, tree in _package_trees().items():
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name in {
                    "is_visible_to", "is_interactable", "is_admin", "is_anonymous",
                    "require_admin", "require_auth", "require_role", "can_moderate",
                }:
                    offenders.append(f"{name}:{node.lineno} {node.name}")
        assert offenders == [], offenders

    def test_no_role_or_lifecycle_vocabulary_is_hardcoded(self) -> None:
        """Roles and flags always come from the contract enums, never literals."""
        banned = (
            {str(role) for role in Role}
            | {str(state) for state in ModerationState}
            | {str(visibility) for visibility in Visibility}
        ) - set(FLAG_FIELD.values())  # the flag *column* names legitimately exist
        offenders: list[str] = []
        for name, tree in _package_trees().items():
            for node in ast.walk(tree):
                if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                        and node.value in banned):
                    offenders.append(f"{name}:{node.lineno} {node.value!r}")
                if isinstance(node, ast.Compare):
                    for comparator in node.comparators:
                        if isinstance(comparator, ast.Attribute) and comparator.attr in {
                            "role", "user_id",
                        }:
                            offenders.append(f"{name}:{node.lineno} identity comparison")
        assert offenders == [], offenders

    def test_flag_writes_come_from_the_contract_state_table(self) -> None:
        import blacknote.moderation.ports as moderation_ports
        import blacknote.moderation.service as moderation_service

        assert moderation_ports.target_state_for("note") is ModerationState.UNLISTED
        assert moderation_ports.target_state_for("comment") is ModerationState.BLOCKED
        source = inspect.getsource(moderation_service.ModerationService._execute)
        assert "target_state_for(" in source
        assert "ModerationState." not in source  # the service picks no state itself

    def test_the_layer_never_builds_a_read_path(self) -> None:
        readers: list[str] = []
        for name, tree in _package_trees().items():
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and (
                    node.name.startswith(("search", "feed", "list_public", "read_"))
                    or node.name in {"visible_items", "filter_visible"}
                ):
                    readers.append(f"{name}:{node.name}")
                if isinstance(node, ast.Attribute) and node.attr in {
                    "list_feed", "search_notes", "get_note", "list_comments",
                }:
                    readers.append(f"{name}:{node.attr}")
        assert readers == [], readers

    def test_error_payloads_are_built_by_the_contract_only(self) -> None:
        import blacknote.contracts.errors as contract_errors
        import blacknote.moderation.service as moderation_service

        assert moderation_service.ContractError is contract_errors.ContractError
        assert moderation_service.ErrorCode is contract_errors.ErrorCode
        offenders: list[str] = []
        for name, tree in _package_trees().items():
            for node in ast.walk(tree):
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                        and node.func.id in {
                            "make_error", "permission_error", "auth_error",
                            "not_found_error", "validation_error", "ErrorEnvelope",
                        }):
                    offenders.append(f"{name}:{node.lineno} {node.func.id}")
        assert offenders == [], offenders

    def test_written_states_are_contract_members(self, h: Harness) -> None:
        note_id = h.make_note()
        comment_id = h.make_comment(note_id)
        h.service.unlist_note(h.admin_token, note_id)
        h.service.block_comment(h.admin_token, comment_id)
        assert moderation_of(h.note_row(note_id)) in set(ModerationState)
        assert moderation_of(h.comment_row(comment_id)) in set(ModerationState)
        assert h.note_row(note_id).moderated is True
        assert h.comment_row(comment_id).blocked is True

    def test_configuration_and_identity_come_from_the_contracts(self, h: Harness) -> None:
        import blacknote.contracts.config as contract_config
        import blacknote.moderation.service as moderation_service

        assert moderation_service.Settings is contract_config.Settings
        assert h.service.settings.default_page_size == Settings().default_page_size
        note_id = h.make_note()
        h.service.unlist_note(h.admin_token, note_id)
        entry = h.entries("note_unlisted")[0]
        assert is_valid_id(entry.id) and is_valid_id(entry.user_id)
        assert "token" not in entry.to_dict()  # the bearer token is never recorded


# --------------------------------------------------------------------------- #
# store & model level invariants
# --------------------------------------------------------------------------- #
class TestStoresAndModels:
    def test_default_wiring_needs_no_arguments(self) -> None:
        sessions = StubSessions()
        admin = sessions.login(new_id(), role=Role.ADMIN)
        service = ModerationService(sessions=sessions)
        assert service.settings is not None
        with pytest.raises(ContractError) as excinfo:
            service.unlist_note(admin, new_id())
        assert excinfo.value.code is ErrorCode.NOTE_NOT_FOUND
        assert service.moderation_flag("note", new_id()) is None

    def test_moderatable_records_keep_flag_and_state_in_agreement(self) -> None:
        note = ModeratableNote(id=new_id(), author_id=new_id(), moderated=True)
        assert note.moderation is ModerationState.VISIBLE  # the enum is the source
        assert replace(note, moderation=ModerationState.UNLISTED).moderated is True
        comment = ModeratableComment(
            id=new_id(), note_id=new_id(), author_id=new_id(),
            moderation=ModerationState.BLOCKED,
        )
        assert comment.blocked is True
        assert comment.changed(blocked=False).blocked is True  # still a projection
        assert comment.is_reply is False and comment.is_tombstone is False
        assert ModeratableNote.from_row(note.to_row()) == note
        assert ModeratableComment.from_row(comment.to_row()) == comment
        assert isinstance(note.to_row()["moderation"], str)

    def test_native_stores_speak_the_port(self) -> None:
        notes, comments = InMemoryNoteStore(), InMemoryCommentStore()
        note_id = notes.add(id=new_id(), author_id=new_id()).id
        comment_id = comments.add(id=new_id(), note_id=note_id, author_id=new_id()).id
        assert notes.load(note_id) is notes.get(note_id)
        assert isinstance(notes, InMemoryModerationStore)
        assert notes.apply_moderation(note_id, ModerationState.UNLISTED).moderated is True
        assert comments.apply_moderation(comment_id, ModerationState.BLOCKED).blocked is True
        assert notes.apply_moderation(new_id(), ModerationState.UNLISTED) is None
        assert notes.get("garbage") is None

    def test_audit_store_rejects_duplicate_ids(self) -> None:
        store = InMemoryAuditLogStore()
        entry = ModerationAuditLog.create(
            ModerationAction.NOTE_UNLISTED, user_id=new_id(), note_id=new_id()
        )
        store.append(entry)
        with pytest.raises(ContractError) as excinfo:
            store.append(entry)
        assert excinfo.value.code is ErrorCode.CONFLICT

    def test_audit_entry_validates_ids_and_timestamps(self) -> None:
        with pytest.raises(ContractError) as bad_user:
            ModerationAuditLog.create(ModerationAction.NOTE_UNLISTED, user_id="nope")
        assert bad_user.value.code is ErrorCode.VALIDATION_ERROR
        with pytest.raises(ContractError) as bad_action:
            ModerationAuditLog.create("delete_everything")
        assert bad_action.value.code is ErrorCode.VALIDATION_ERROR
        with pytest.raises(ContractError) as bad_kind:
            flag_of({}, "unknown-kind")  # type: ignore[arg-type]
        assert bad_kind.value.code is ErrorCode.VALIDATION_ERROR
        stamp = ModerationAuditLog.create(
            ModerationAction.COMMENT_BLOCKED, user_id=new_id(), comment_id=new_id(),
            created_at="2026-03-01T12:00:00+00:00",
        )
        assert stamp.created_at == "2026-03-01T12:00:00.000000Z"
        assert ModerationAuditLog.create(ModerationAction.NOTE_UNLISTED).created_at <= now_iso()
        assert isinstance(ModerationAuditLog.create(
            ModerationAction.NOTE_UNLISTED).details, Mapping)

    def test_sqlite_audit_store_is_append_only_and_filterable(self, tmp_path: Path) -> None:
        user = new_id()
        note_id = new_id()
        with SQLiteAuditLogStore(tmp_path / "filter.db") as store:
            for index in range(3):
                store.append(
                    ModerationAuditLog.create(
                        ModerationAction.NOTE_UNLISTED if index < 2
                        else ModerationAction.PRIVILEGE_DENIED,
                        user_id=user,
                        note_id=note_id,
                        created_at=to_iso(BASE_MOMENT + timedelta(minutes=index)),
                    )
                )
            assert store.count() == 3
            assert store.count(action="note_unlisted") == 2
            assert [e.created_at for e in store.list()] == sorted(
                [e.created_at for e in store.list()], reverse=True
            )
            assert len(store.list(action="comment_blocked")) == 0
            assert len(store.list(user_id=user, limit=1)) == 1
            assert len(store.list(target_id=note_id)) == 3
            with pytest.raises(ContractError) as excinfo:
                store.append(
                    ModerationAuditLog.create(
                        ModerationAction.NOTE_UNLISTED, user_id=user, note_id=note_id,
                        id=store.list()[0].id,
                    )
                )
            assert excinfo.value.code is ErrorCode.CONFLICT
