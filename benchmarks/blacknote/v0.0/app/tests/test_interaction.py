"""Acceptance tests for ``mod-interaction`` (TaskNode T1, Wave 1).

Everything is exercised through the frozen ``IF-interaction-actions`` surface:

* ``like(token, note_id) -> Counters``
* ``unlike(token, note_id) -> Counters``
* ``favorite(token, note_id) -> Counters``
* ``unfavorite(token, note_id) -> Counters``
* ``comment(token, note_id, parent_comment_id, body) -> CommentView``
* ``list_comments(note_id) -> list[CommentThread]``
* ``list_favorites(token, query) -> Page[NoteSummary]``

as well as through the frozen note dependency
``IF-note-exists-for-interaction`` (``is_publicly_interactable`` /
``get_note_author_id``).

No sibling business module is imported.  The note layer is replaced by the two
fakes defined in this file (:class:`FakeNoteRepository` — a probe-only port, and
:class:`LedgerNoteRepository` — a probe **plus** cached counter columns, which
is what ``INV-concurrency-counters`` is really about), the session port is a
plain ``token → Principal`` mapping, and persistence uses this layer's own
stores.  Every temporary database is created under pytest's ``tmp_path`` — the
repository is never touched, and the layer refuses to invent a path of its own
(see :func:`test_sqlite_store_needs_an_explicit_location`).

Beyond the frozen seven, the two seams the composition root needs are covered as
well: moderation flags comments through the store (:func:`_flag`) while the
author may remove a comment again (:meth:`InteractionService.delete_comment`).

Error convention: failures are raised as ``ContractError`` carrying the frozen
``ErrorEnvelope`` (the convention of ``mod-note`` / ``mod-account``), so
assertions read ``excinfo.value.envelope.code``; the non-raising
:class:`InteractionResult` view renders exactly the same envelope.

Covered in order:

1. the frozen surface (types, signatures, error names) and the dependency rule;
2. like / unlike / favorite / unfavorite — idempotency, non-negative counters;
3. authentication ordering (``AUTH_REQUIRED`` writes nothing at all);
4. note existence / visibility (``NOTE_NOT_FOUND``);
5. comments — validation against the contract limit, threading, timestamps;
6. ``list_comments`` — chronology plus blocked/deleted subtree pruning;
7. ``list_favorites`` — keyset paging, projection, counter overlay;
8. storage engines — the same store contract on the in-memory dict and SQLite,
   transactional counter cache (``INV-concurrency-counters``), uniqueness;
9. comment removal (author-only seam) with identical semantics on both engines;
10. the wiring pieces: result envelope, ports, models, read shaping;
11. side-effect freedom: import and default wiring never create a database file.
"""

from __future__ import annotations

import ast
import inspect
import os
import sqlite3
import subprocess
import sys
import threading
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from blacknote.contracts import interfaces as contract_interfaces
from blacknote.contracts.config import Settings
from blacknote.contracts.errors import ENVELOPE_KEYS, ContractError, ErrorCode, status_for
from blacknote.contracts.ids import is_valid_id, new_id
from blacknote.contracts.interfaces import (
    CommentView,
    Counters,
    InteractionApi,
    NoteExistenceApi,
    Page,
    PageQuery,
    Principal,
    interface_errors,
    interface_functions,
)
from blacknote.contracts.config import Settings
from blacknote.contracts.timeutil import from_iso, is_iso8601, to_epoch_ms
from blacknote.contracts.validation import LIMITS
from blacknote.contracts.visibility import (
    ContentStatus,
    ModerationState,
    Visibility,
    VisibilityScope,
    is_interactable,
    is_visible_to,
)
from blacknote.contracts.validation import DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE
from blacknote.interaction.queries import clamp_limit
from blacknote.interaction import (
    ANONYMOUS_NICKNAME,
    MAX_COMMENT_BODY_LENGTH,
    CommentRecord,
    DictCounterSink,
    FavoriteRecord,
    InMemoryIdentityDirectory,
    InMemoryInteractionStore,
    InMemoryNoteDirectory,
    InteractionResult,
    InteractionService,
    LikeRecord,
    NullIdentityDirectory,
    NullNoteDirectory,
    NoteRecordCounterSink,
    SQLiteInteractionStore,
    adapt_directory,
    adapt_sessions,
    assemble_threads,
    authenticate,
    decode_cursor,
    encode_cursor,
    open_store,
    page_of,
    relation_key,
    visible_rows,
)

INTERACTION_DIR = Path(__file__).resolve().parents[1] / "src" / "blacknote" / "interaction"
SANDBOX_ROOT = Path(__file__).resolve().parents[1]
FROZEN_ACTIONS: tuple[str, ...] = interface_errors("IF-interaction-actions")
FROZEN_NAMES: tuple[str, ...] = interface_functions("IF-interaction-actions")
BASE_MOMENT = datetime(2026, 3, 1, 12, 0, 0, tzinfo=timezone.utc)
NOTE_CREATED = "2026-03-01T10:00:00.000000Z"


def stamp(step: int) -> str:
    """Deterministic ISO-8601 UTC stamp, ``step`` seconds after the base moment."""
    return (BASE_MOMENT + timedelta(seconds=step)).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"


# --------------------------------------------------------------------------- #
# fakes owned by this test module (no sibling module is imported)
# --------------------------------------------------------------------------- #
class NotePortFake:
    """Shared machinery of the note fakes.

    Only the two frozen methods are guaranteed; everything else is fixture
    bookkeeping.  Visibility decisions are *not* re-implemented here beyond the
    lifecycle flags the real note layer would carry — the interaction layer must
    ask, never assume.
    """

    def __init__(self) -> None:
        self.rows: dict[str, dict[str, Any]] = {}
        self.probe_calls: list[str] = []
        self.summary_calls: list[str] = []

    # -- fixture helpers ------------------------------------------------------
    def add(
        self,
        *,
        note_id: str | None = None,
        author_id: str,
        author_nickname: str = "",
        title: str = "note",
        body: str = "body",
        tags: tuple[str, ...] = (),
        image_urls: tuple[str, ...] = (),
        visibility: Any = Visibility.PUBLIC,
        status: Any = ContentStatus.PUBLISHED,
        moderation: Any = ModerationState.VISIBLE,
        created_at: str = NOTE_CREATED,
        updated_at: str | None = None,
    ) -> str:
        note_id = note_id or new_id()
        self.rows[note_id] = {
            "id": note_id,
            "author_id": author_id,
            "author_nickname": author_nickname,
            "title": title,
            "body": body,
            "tags": tuple(tags),
            "image_urls": tuple(image_urls),
            "visibility": str(visibility),
            "status": str(status),
            "moderation": str(moderation),
            "created_at": created_at,
            "updated_at": updated_at or created_at,
        }
        return note_id

    def publish(self, note_id: str, **changes: Any) -> None:
        """Flip lifecycle flags (simulates delete / takedown / visibility change)."""
        self.rows.setdefault(note_id, {"id": note_id, "author_id": new_id()}).update(changes)

    def set_flags(self, note_id: str, **changes: Any) -> None:
        self.publish(note_id, **changes)

    def nickname(self, user_id: str, value: str) -> None:
        for row in self.rows.values():
            if row["author_id"] == user_id:
                row["author_nickname"] = value

    # -- IF-note-exists-for-interaction --------------------------------------
    def is_publicly_interactable(self, note_id: str) -> bool:
        """Exactly what ``mod-note`` answers (same contract, both sides)."""
        self.probe_calls.append(str(note_id))
        row = self.rows.get(str(note_id))
        return row is not None and is_interactable(None, row)

    def get_note_author_id(self, note_id: str) -> str | None:
        row = self.rows.get(str(note_id))
        if row is None or not is_visible_to(None, row, scope=VisibilityScope.PUBLIC):
            return None
        return row["author_id"]


class FakeNoteRepository(NotePortFake):
    """The minimal port: existence + visibility + author, nothing else.

    ``IF-note-exists-for-interaction`` is *all* this layer may ask the note
    module — this fake makes that boundary un-fakeable.
    """

    def __init__(self, rows: Any = None) -> None:
        super().__init__()
        for row in rows or ():
            self.ingest(row)

    def ingest(self, row: Any) -> str:
        """Clone a row of another fake (``id`` → ``note_id``)."""
        if not isinstance(row, Mapping):
            return self.add(note_id=str(row), author_id=new_id())
        payload = dict(row)
        note_id = payload.pop("id", None) or payload.pop("note_id", None) or new_id()
        author_id = payload.pop("author_id", None) or new_id()
        return self.add(note_id=note_id, author_id=author_id, **payload)


class LedgerNoteRepository(FakeNoteRepository):
    """Same port, plus the note's own cached counter columns.

    The interaction store is wired to this repository, so any relation write and
    the matching counter delta must land in the same transaction.  ``get`` /
    ``save`` are the seams :class:`NoteRecordCounterSink` looks for, i.e. the
    shape ``mod-note``'s repository has; ``calls`` records every cache write.
    """

    COUNTER_COLUMNS = {
        "likes": "like_count",
        "favorites": "favorite_count",
        "comments": "comment_count",
    }

    def __init__(self, rows: Any = None) -> None:
        super().__init__(rows)
        self.calls: list[str] = []

    def get(self, note_id: str) -> dict[str, Any] | None:
        """Read seam: the note record including its cached counter columns."""
        return self.rows.get(str(note_id))

    def save(self, record: Mapping[str, Any]) -> dict[str, Any]:
        """Write seam: the note layer's own ``UPDATE notes SET ..._count``."""
        self.calls.append("save")
        row = dict(record)
        self.rows[str(row["id"])] = row
        return row

    def counters(self, note_id: str) -> Counters:
        """The note's cached view of the interaction counters."""
        row = self.rows.get(str(note_id), {})
        return Counters(
            likes=int(row.get("like_count", 0) or 0),
            favorites=int(row.get("favorite_count", 0) or 0),
            comments=int(row.get("comment_count", 0) or 0),
        )

    def apply_delta(self, note_id: str, **changes: int) -> Counters:
        """Counter sink write-back (the note layer's own cache update statement)."""
        row = self.rows.setdefault(str(note_id), {"id": str(note_id), "author_id": new_id()})
        current = self.counters(str(note_id))
        values = {
            "likes": current.likes,
            "favorites": current.favorites,
            "comments": current.comments,
        }
        for key, delta in changes.items():
            column = self.COUNTER_COLUMNS.get(key)
            if column is None:
                continue
            values[key] = max(0, values[key] + int(delta or 0))
            row[column] = values[key]
        return Counters(**values)


class CountingNoteRepository(FakeNoteRepository):
    """Records the note port calls so ordering (auth → probe → write) is visible."""

    def __init__(self, rows: Any = None) -> None:
        super().__init__(rows)
        self.events: list[str] = []

    def is_publicly_interactable(self, note_id: str) -> bool:
        self.events.append(f"probe:{note_id}")
        return super().is_publicly_interactable(note_id)


class UnreachableNotePort:
    """A note dependency whose probe raises — wiring faults must not be swallowed."""

    def __init__(self) -> None:
        self.note_id = new_id()
        self.author_id = new_id()
        self.error = RuntimeError("note layer unavailable")

    def is_publicly_interactable(self, note_id: str) -> bool:
        raise self.error

    def get_note_author_id(self, note_id: str) -> str | None:
        return self.author_id


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def users() -> dict[str, str]:
    return {"author": new_id(), "alice": new_id(), "bob": new_id(), "carol": new_id()}


@pytest.fixture
def session(users: Mapping[str, str]) -> tuple[dict[str, str], dict[str, Principal]]:
    """The injected session port: ``(tokens by name, token → Principal)``.

    One expired principal is part of the table so ``SESSION_EXPIRED`` is
    reachable without a real account module.
    """
    tokens: dict[str, str] = {}
    table: dict[str, Principal] = {}
    for name, user_id in users.items():
        token = new_id()
        tokens[name] = token
        table[token] = Principal(user_id=user_id, token=token, nickname=name)
    stale = new_id()
    table[stale] = Principal(
        user_id=users["alice"],
        token=stale,
        issued_at=stamp(-7200),
        expires_at=stamp(-60),
    )
    tokens["expired"] = stale
    return tokens, table


@pytest.fixture
def tokens(session) -> dict[str, str]:
    return session[0]


@pytest.fixture
def principals(session) -> dict[str, Principal]:
    return session[1]


@pytest.fixture
def notes() -> FakeNoteRepository:
    return FakeNoteRepository()


@pytest.fixture
def store() -> InMemoryInteractionStore:
    return InMemoryInteractionStore()


@pytest.fixture
def service(notes, store, principals) -> InteractionService:
    return InteractionService(notes, store=store, sessions=principals)


@pytest.fixture
def note(notes: FakeNoteRepository, users: Mapping[str, str]) -> str:
    return notes.add(author_id=users["author"], title="A note", body="body text")


@pytest.fixture
def note_id() -> str:
    """A well-formed note id (the store trusts the service's existence check)."""
    return new_id()


@pytest.fixture
def ledger() -> LedgerNoteRepository:
    return LedgerNoteRepository()


@pytest.fixture
def sink(ledger: LedgerNoteRepository) -> NoteRecordCounterSink:
    """The note layer's counter-cache update statement, as the note service owns it."""
    return NoteRecordCounterSink(ledger)


@pytest.fixture
def ledger_service(ledger: LedgerNoteRepository, sink, principals) -> InteractionService:
    store = InMemoryInteractionStore(counters=sink)
    return InteractionService(ledger, store=store, sessions=principals)


@pytest.fixture
def ledger_note(ledger: LedgerNoteRepository, users: Mapping[str, str]) -> str:
    return ledger.add(author_id=users["author"], title="Cached")


def cached(ledger: LedgerNoteRepository, note_id: str, column: str) -> int:
    """Read one counter column of the fake note row (the note's own cache)."""
    return int(ledger.rows.get(note_id, {}).get(column, 0) or 0)


class _Catalog:
    """Note directory + service for the tests that need a *projection*."""

    def __init__(self, service: InteractionService, directory: InMemoryNoteDirectory,
                 ids: Mapping[str, str]) -> None:
        self.service = service
        self.directory = directory
        self.ids = dict(ids)

    def __getattr__(self, item: str) -> Any:
        if item in self.ids:
            return self.ids[item]
        raise AttributeError(item)


@pytest.fixture
def catalog(store, principals, users) -> _Catalog:
    """``list_favorites`` projects ``NoteSummary`` rows, so it needs a directory."""
    directory = InMemoryNoteDirectory()
    ids = {}
    for name, visibility, status in (
        ("public_id", Visibility.PUBLIC, ContentStatus.PUBLISHED),
        ("other_id", Visibility.PUBLIC, ContentStatus.PUBLISHED),
        ("third_id", Visibility.PUBLIC, ContentStatus.PUBLISHED),
        ("private_id", Visibility.PRIVATE, ContentStatus.PUBLISHED),
        ("deleted_id", Visibility.PUBLIC, ContentStatus.DELETED),
    ):
        note_id = new_id()
        ids[name] = note_id
        directory.add(
            {
                "id": note_id,
                "author_id": users["author"],
                "author_nickname": "author",
                "title": name.replace("_id", "").title(),
                "body": "body text",
                "visibility": visibility,
                "status": status,
                "created_at": NOTE_CREATED,
            }
        )
    service = InteractionService(
        notes=directory,
        store=store,
        sessions=principals,
        identity={user_id: name for name, user_id in users.items()},
        clock=stamp(0),
    )
    return _Catalog(service, directory, ids)


# --------------------------------------------------------------------------- #
# surface: the frozen interface is implemented, nothing more is required
# --------------------------------------------------------------------------- #
def test_service_satisfies_frozen_interaction_protocol(service, note, tokens, principals):
    assert isinstance(service, InteractionApi)
    for name in FROZEN_NAMES:
        assert callable(getattr(service, name)), name


def test_frozen_signatures_match_the_contract(service):
    declared = dict(inspect.getmembers(contract_interfaces.InteractionApi, inspect.isfunction))
    for name, frozen in declared.items():
        if name.startswith("__"):
            continue
        params = list(inspect.signature(getattr(service, name)).parameters)
        if name == "list_favorites":
            # the contract declares (token, query); defaults for *query* are an
            # addition, the positional order must stay identical
            assert params == ["token", "query"], name
            continue
        if name == "comment":
            assert params == ["token", "note_id", "parent_comment_id", "body"], name
            continue
        assert params == [
            p for p in inspect.signature(frozen).parameters if p != "self"
        ], name


def test_only_contracts_and_own_package_are_imported():
    """``IF-deps``: interaction → contracts + note query port, never a sibling module."""
    forbidden = {"blacknote.note", "blacknote.account", "blacknote.moderation", "blacknote.web"}
    pattern_roots = ("note", "account", "moderation", "web")
    for path in sorted(INTERACTION_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:  # relative import → own package
                    continue
                names = [node.module or ""]
            for name in names:
                parts = name.split(".")
                assert name not in forbidden, f"{path.name} imports {name}"
                if parts[0] == "blacknote" and len(parts) > 1:
                    assert parts[1] in {"contracts", "interaction"}, f"{path.name}: {name}"
                assert not name.startswith(pattern_roots), f"{path.name}: {name}"


def test_injected_note_dependency_satisfies_the_frozen_port():
    """The note fakes are usable *as* ``NoteExistenceApi`` (structural protocol)."""
    fake = FakeNoteRepository()
    assert isinstance(fake, NoteExistenceApi)
    directory = InMemoryNoteDirectory()
    assert isinstance(directory, NoteExistenceApi)
    directory.add({"id": new_id(), "author_id": new_id()})
    assert isinstance(directory, NoteExistenceApi)


def test_service_requires_a_note_dependency(principals):
    """Fail closed: a composition root without the note port is a wiring bug."""
    with pytest.raises(ContractError) as excinfo:
        InteractionService(sessions=principals)
    assert excinfo.value.envelope.code == ErrorCode.CONFIG_ERROR


def test_unavailable_note_probe_is_not_swallowed(principals, tokens):
    """A broken dependency surfaces as itself — no silent ``NOTE_NOT_FOUND``."""
    dependency = UnreachableNotePort()
    service = InteractionService(dependency, sessions=principals)
    with pytest.raises(RuntimeError):
        service.comment(tokens["alice"], dependency.note_id, None, "hello")
    with pytest.raises(RuntimeError):
        service.like(tokens["alice"], dependency.note_id)


# --------------------------------------------------------------------------- #
# likes / favourites
# --------------------------------------------------------------------------- #
def test_like_then_unlike(service, note, tokens):
    token = tokens["alice"]
    assert service.like(token, note) == Counters(likes=1)
    assert service.unlike(token, note) == Counters(likes=0)


def test_like_is_idempotent_per_user(service, note, tokens):
    alice, bob = tokens["alice"], tokens["bob"]
    assert service.like(alice, note) == Counters(likes=1)
    assert service.like(alice, note) == Counters(likes=1)
    assert service.like(alice, note) == Counters(likes=1)
    assert service.like(bob, note) == Counters(likes=2)
    assert service.unlike(bob, note) == Counters(likes=1)
    assert service.unlike(bob, note) == Counters(likes=1)


def test_favorite_is_idempotent_per_user(service, note, tokens):
    alice, bob = tokens["alice"], tokens["bob"]
    assert service.favorite(alice, note) == Counters(favorites=1)
    assert service.favorite(alice, note) == Counters(favorites=1)
    assert service.favorite(bob, note) == Counters(favorites=2)
    assert service.unfavorite(alice, note) == Counters(favorites=1)
    assert service.unfavorite(alice, note) == Counters(favorites=1)


def test_like_and_favorite_counters_are_independent(service, note, tokens):
    token = tokens["alice"]
    service.like(token, note)
    service.favorite(token, note)
    counters = service.like(token, note)
    assert counters.likes == 1 and counters.favorites == 1 and counters.comments == 0


def test_unlike_never_goes_negative(service, note, tokens, users):
    alice = tokens["alice"]
    # nobody liked: cancelling is a no-op and the cache stays exactly at zero
    assert service.unlike(alice, note) == Counters(likes=0)
    service.like(alice, note)
    assert service.unlike(alice, note) == Counters(likes=0)
    assert service.unlike(alice, note) == Counters(likes=0)
    assert service.unlike(alice, note) == Counters(likes=0)
    assert service.liked_notes(users["alice"]) == ()


def test_counter_type_refuses_negative_values():
    """The frozen ``Counters`` cannot even be constructed below zero."""
    with pytest.raises(ValueError):
        Counters(likes=-1)


def test_sink_arithmetic_is_clamped_at_zero(users):
    """The clamp lives in the counter sink, i.e. in the write path itself."""
    note_id = new_id()
    sink = DictCounterSink()
    sink.seed(note_id, Counters())
    assert sink.apply_delta(note_id, likes=-1) == Counters()
    assert sink.read(note_id) == Counters()
    sink.seed(note_id, {"likes": 3, "favorites": 1, "comments": 0})
    assert sink.apply_delta(note_id, likes=-5) == Counters(likes=0, favorites=1)


def test_cache_drift_is_detectable_and_repairable(users):
    """``derive_counters`` is the truth; ``sync_counters`` repairs the cache."""
    note_id = new_id()
    sink = DictCounterSink()
    store = InMemoryInteractionStore(counters=sink)
    store.add_like(users["alice"], note_id)
    store.add_like(users["bob"], note_id)
    assert store.counters(note_id) == store.derive_counters(note_id) == Counters(likes=2)
    sink.seed(note_id, Counters(likes=99))  # simulate a corrupted cache row
    assert store.counters(note_id) != store.derive_counters(note_id)
    assert store.sync_counters(note_id) == Counters(likes=2)
    assert store.counters(note_id) == store.derive_counters(note_id)


def test_repeated_cancellation_stays_at_zero(service, note, tokens, store):
    alice = tokens["alice"]
    service.like(alice, note)
    service.unlike(alice, note)
    for _ in range(3):
        assert service.unlike(alice, note) == Counters(likes=0)
    assert store.counters(note) == Counters(likes=0)
    assert store.derive_counters(note) == store.counters(note)  # cache and truth agree


def test_unfavorite_never_goes_negative(service, note, tokens):
    alice, bob = tokens["alice"], tokens["bob"]
    assert service.unfavorite(alice, note) == Counters(favorites=0)
    service.favorite(bob, note)
    assert service.unfavorite(alice, note) == Counters(favorites=1)  # no-op for alice
    assert service.unfavorite(bob, note) == Counters(favorites=0)
    assert service.unfavorite(bob, note) == Counters(favorites=0)


def test_duplicate_relation_writes_report_changed_false(service, note, tokens):
    token = tokens["alice"]
    first = service.like_result(token, note)
    again = service.like_result(token, note)
    assert first.ok and first.changed is True
    assert again.ok and again.changed is False and again.value == first.value
    dropped = service.unlike_result(token, note)
    assert dropped.changed is True
    assert service.unlike_result(token, note).changed is False


def test_several_users_on_one_note_and_one_user_on_several_notes(
    service, note, notes, tokens, users
):
    second = notes.add(author_id=users["author"])
    first = note
    alice, bob = tokens["alice"], tokens["bob"]
    service.like(alice, first)
    service.like(bob, first)
    service.like(alice, second)
    assert service.like(alice, first) == Counters(likes=2)
    assert service.counters(first) == Counters(likes=2)
    assert service.counters(second) == Counters(likes=1)
    assert set(service.liked_notes(users["alice"])) == {first, second}
    assert service.liked_notes(users["bob"]) == (first,)


# --------------------------------------------------------------------------- #
# R4-AC3 / INV-auth-session: unauthenticated writes are refused and write nothing
# --------------------------------------------------------------------------- #
def test_frozen_error_set_is_the_documented_one():
    assert set(FROZEN_ACTIONS) == {
        "AUTH_REQUIRED",
        "VALIDATION_ERROR",
        "NOTE_NOT_FOUND",
        "COMMENT_NOT_FOUND",
    }
    for name in FROZEN_ACTIONS:
        assert getattr(ErrorCode, name)


@pytest.mark.parametrize(
    "call",
    [
        lambda s, note, anon: s.like(anon, note),
        lambda s, note, anon: s.unlike(anon, note),
        lambda s, note, anon: s.favorite(anon, note),
        lambda s, note, anon: s.unfavorite(anon, note),
        lambda s, note, anon: s.comment(anon, note, None, "hi"),
        lambda s, note, anon: s.list_favorites(anon, PageQuery(limit=10)),
    ],
    ids=["like", "unlike", "favorite", "unfavorite", "comment", "list_favorites"],
)
def test_unauthenticated_writes_require_auth_and_write_nothing(
    call, notes, store, principals, users
):
    note_id = notes.add(author_id=users["author"])
    repository = CountingNoteRepository(notes.rows.values())
    service = InteractionService(repository, store=store, sessions=principals)
    anonymous = new_id()  # unknown token
    with pytest.raises(ContractError) as excinfo:
        call(service, note_id, anonymous)
    error = excinfo.value
    assert error.envelope.code == ErrorCode.AUTH_REQUIRED
    assert error.http_status == 401
    payload = error.envelope.to_dict()
    assert {"code", "message", "details"} <= set(payload) <= set(ENVELOPE_KEYS)
    assert payload["code"] == "AUTH_REQUIRED"
    # nothing was written ...
    assert store.counters(note_id) == Counters()
    assert store.derive_counters(note_id) == Counters()
    assert store.comments_for_note(note_id) == []
    assert store.favorite_page(users["alice"], limit=10).total == 0
    # ... and the note port was not even consulted (authentication happens first)
    assert repository.events == []


@pytest.mark.parametrize("bad", [None, "", "   ", "not-a-uuid", 42, "  " + new_id() + "  "])
def test_malformed_tokens_are_also_auth_required(bad, service, note, store):
    with pytest.raises(ContractError) as excinfo:
        service.like(bad, note)
    assert excinfo.value.envelope.code == ErrorCode.AUTH_REQUIRED
    assert store.counters(note) == Counters()


def test_authenticated_reads_are_not_required_for_list_comments(service, note, tokens):
    """``list_comments`` is an anonymous read: no token exists in its signature."""
    service.comment(tokens["alice"], note, None, "public view")
    assert service.list_comments(note)[0].comment.body == "public view"


# --------------------------------------------------------------------------- #
# NOTE_NOT_FOUND / visibility via the note port
# --------------------------------------------------------------------------- #
def test_like_on_missing_note_raises_note_not_found(service, tokens):
    with pytest.raises(ContractError) as excinfo:
        service.like(tokens["alice"], new_id())
    assert excinfo.value.envelope.code == ErrorCode.NOTE_NOT_FOUND
    assert excinfo.value.http_status == 404


@pytest.mark.parametrize(
    "changes",
    [
        {"visibility": str(Visibility.PRIVATE)},
        {"status": str(ContentStatus.DELETED)},
        {"status": str(ContentStatus.DRAFT)},
        {"moderation": str(ModerationState.UNLISTED)},
        {"moderation": str(ModerationState.BLOCKED)},
    ],
    ids=["private", "deleted", "draft", "unlisted", "blocked"],
)
def test_non_public_note_is_not_interactable(changes, notes, store, principals, users, tokens):
    """The *note port* decides; the interaction layer only asks. (INV-visibility)"""
    note_id = notes.add(author_id=users["author"])
    notes.publish(note_id, **changes)
    service = InteractionService(notes, store=store, sessions=principals)
    token = tokens["alice"]
    with pytest.raises(ContractError) as excinfo:
        service.like(token, note_id)
    assert excinfo.value.envelope.code == ErrorCode.NOTE_NOT_FOUND
    assert store.counters(note_id) == Counters()
    with pytest.raises(ContractError) as excinfo:
        service.comment(token, note_id, None, "nope")
    assert excinfo.value.envelope.code == ErrorCode.NOTE_NOT_FOUND
    assert notes.probe_calls.count(note_id) == 2


def test_port_flips_back_to_interactable(notes, store, principals, users, tokens):
    note_id = notes.add(author_id=users["author"])
    service = InteractionService(notes, store=store, sessions=principals)
    token = tokens["alice"]
    notes.publish(note_id, status=str(ContentStatus.DELETED))
    with pytest.raises(ContractError):
        service.favorite(token, note_id)
    notes.publish(note_id, status=str(ContentStatus.PUBLISHED))
    assert service.favorite(token, note_id) == Counters(favorites=1)


def test_list_comments_on_missing_note_raises(notes, store, principals):
    service = InteractionService(notes, store=store, sessions=principals)
    with pytest.raises(ContractError) as excinfo:
        service.list_comments(new_id())
    assert excinfo.value.envelope.code == ErrorCode.NOTE_NOT_FOUND


def test_invalid_note_id_shape_is_reported_as_absent(service, tokens):
    with pytest.raises(ContractError) as excinfo:
        service.like(tokens["alice"], "abc")
    assert excinfo.value.envelope.code == ErrorCode.NOTE_NOT_FOUND
    assert excinfo.value.envelope.details["note_id"] == ["invalid_or_missing_note_id"]


def test_exists_port_delegates_to_the_note_dependency(service, note, tokens, users):
    assert service.is_publicly_interactable(note) is True
    assert service.note_author(note) == users["author"]
    assert service.is_publicly_interactable(new_id()) is False
    assert service.note_author(new_id()) is None




# --------------------------------------------------------------------------- #
# comments: validation, storage, threading, filtering                         #
# --------------------------------------------------------------------------- #
def test_comment_persists_a_row_and_bumps_the_comment_counter(
    service, note, tokens, users, store
):
    view = service.comment(tokens["alice"], note, None, "first!")

    assert isinstance(view, CommentView)
    assert view.note_id == note
    assert view.author_id == users["alice"]
    assert view.body == "first!"
    assert view.parent_comment_id is None
    assert view.status is ContentStatus.PUBLISHED
    assert is_iso8601(view.created_at)
    assert view.to_dict() == {
        "id": view.id,
        "note_id": note,
        "author_id": users["alice"],
        "author_nickname": view.author_nickname,
        "body": "first!",
        "parent_comment_id": None,
        "status": "published",
        "created_at": view.created_at,
    }

    stored = store.get_comment(view.id)
    assert stored is not None and stored.id == view.id
    assert stored.note_id == note and stored.author_id == view.author_id
    assert stored.parent_comment_id is None
    # the counter is the number of comment rows, not an independent guess
    assert store.derive_counters(note).comments == len(store.comments_for_note(note)) == 1
    assert store.counters(note) == Counters(comments=1)
    assert service.comment_count(note) == 1


@pytest.mark.parametrize("body", ["", "   ", "\n\t ", None, 123, "a" * (MAX_COMMENT_BODY_LENGTH + 1)])
def test_invalid_comment_bodies_are_validation_errors(service, note, tokens, store, body):
    with pytest.raises(ContractError) as excinfo:
        service.comment(tokens["alice"], note, None, body)
    error = excinfo.value
    assert error.envelope.code is ErrorCode.VALIDATION_ERROR
    assert error.http_status == 400
    assert invalid_fields(error) == {"body"}
    assert store.comments == {}
    assert store.counters(note) == Counters()


def test_comment_body_limit_is_read_from_the_contract(service, note, tokens, store):
    """The bound is ``LIMITS['comment_body']`` — never a second hard-coded number.

    The task prose mentions 2000 characters; ``mod-contracts`` freezes the
    product limit (currently 500).  The layer follows the frozen contract, so a
    contract change automatically changes the accepted length.
    """
    limit = MAX_COMMENT_BODY_LENGTH == int(LIMITS["comment_body"]["max"])
    assert limit is True

    accepted = service.comment(tokens["alice"], note, None, "x" * int(LIMITS["comment_body"]["max"]))
    assert len(accepted.body) == int(LIMITS["comment_body"]["max"])
    assert store.counters(note).comments == 1


def test_comment_body_is_cleaned_before_it_is_stored(service, note, tokens, store):
    view = service.comment(tokens["alice"], note, None, "   hello   ")
    assert view.body == "hello"
    assert store.get_comment(view.id).body == "hello"


def test_comment_timestamps_and_nickname_follow_the_wiring(users, store, principals, tokens):
    """Injected clock → created_at; injected identity directory → nickname."""
    directory = InMemoryNoteDirectory()
    note_id = directory.add(
        {"id": new_id(), "author_id": users["author"], "title": "t", "created_at": NOTE_CREATED}
    ).id
    service = InteractionService(
        notes=directory,
        store=store,
        sessions=principals,
        identity={user_id: name for name, user_id in users.items()},
        clock=stamp(5),
    )
    view = service.comment(tokens["alice"], note_id, None, "hi")

    assert view.created_at == stamp(5)
    assert view.author_nickname == "alice"
    stored = store.get_comment(view.id)
    assert stored.created_at == stamp(5) and stored.updated_at == stamp(5)
    # the service canonicalises a custom clock, so ordering stays comparable
    assert is_iso8601(stored.created_at)


def tokens_for(principals):
    return list(principals)


def test_comment_on_missing_note_raises_note_not_found(service, tokens, store):
    with pytest.raises(ContractError) as excinfo:
        service.comment(tokens["alice"], new_id(), None, "hello")
    assert excinfo.value.envelope.code is ErrorCode.NOTE_NOT_FOUND
    assert store.comments == {}


def test_comment_on_a_note_the_port_does_not_know_raises(service, tokens, store):
    with pytest.raises(ContractError) as excinfo:
        service.comment(tokens["alice"], new_id(), None, "hello")
    assert excinfo.value.envelope.code is ErrorCode.NOTE_NOT_FOUND
    assert store.comments == {}


def test_reply_to_missing_comment_raises_comment_not_found(service, note, tokens, store):
    with pytest.raises(ContractError) as excinfo:
        service.comment(tokens["alice"], note, new_id(), "hello")
    assert excinfo.value.envelope.code is ErrorCode.COMMENT_NOT_FOUND
    assert store.comments == {}


def test_reply_to_a_comment_of_another_note_is_rejected(service, notes, users, tokens, store):
    first = notes.add(author_id=users["author"], title="first")
    second = notes.add(author_id=users["author"], title="second")
    root = service.comment(tokens["alice"], first, None, "root")
    with pytest.raises(ContractError) as excinfo:
        service.comment(tokens["bob"], second, root.id, "cross note reply")
    assert excinfo.value.envelope.code is ErrorCode.COMMENT_NOT_FOUND
    assert list(store.comments) == [root.id]


def test_malformed_parent_comment_id_is_a_validation_error(service, note, tokens):
    with pytest.raises(ContractError) as excinfo:
        service.comment(tokens["alice"], note, "not-a-uuid", "hello")
    assert excinfo.value.envelope.code is ErrorCode.VALIDATION_ERROR
    assert invalid_fields(excinfo.value) == {"parent_comment_id"}


def test_comment_needs_auth_and_writes_nothing(service, note, store):
    with pytest.raises(ContractError) as excinfo:
        service.comment(None, note, None, "hello")
    assert excinfo.value.envelope.code is ErrorCode.AUTH_REQUIRED
    assert store.comments == {} and store.counters(note) == Counters()


def test_threaded_replies_are_grouped_under_the_root(service, note, tokens, store):
    root = service.comment(tokens["alice"], note, None, "root")
    first = service.comment(tokens["bob"], note, root.id, "first reply")
    second = service.comment(tokens["carol"], note, root.id, "second reply")
    nested = service.comment(tokens["alice"], note, first.id, "nested")

    assert store.get_comment(first.id).parent_comment_id == root.id
    assert store.get_comment(nested.id).parent_comment_id == first.id

    threads = service.list_comments(note)
    assert [thread.comment.id for thread in threads] == [root.id]
    assert [reply.id for reply in threads[0].replies] == [first.id, second.id, nested.id]
    assert len(threads[0]) == 4
    assert store.counters(note) == store.derive_counters(note) == Counters(comments=4)


def test_list_comments_is_chronological(service, notes, tokens, users, store):
    first = notes.add(author_id=users["author"])
    oldest = CommentRecord(
        id=new_id(), note_id=first, author_id=users["alice"], body="oldest",
        created_at=stamp(-100),
    )
    store.append_comment(oldest)
    newest = service.comment(tokens["bob"], first, None, "newest")

    threads = service.list_comments(first)
    assert [thread.comment.body for thread in threads] == ["oldest", "newest"]
    assert [thread.comment.created_at for thread in threads] == sorted(
        thread.comment.created_at for thread in threads
    )
    assert newest.body == "newest"


def invalid_fields(error: ContractError) -> set[str]:
    """Field names reported by ``check_fields`` (``details["fields"]``)."""
    payload = error.envelope.details.get("fields", ())
    return {entry["field"] for entry in payload}


def _flag(service, comment_id, **changes):
    """Moderate a comment through the storage seam (mod-moderation's job)."""
    record = service.store.get_comment(comment_id)
    service.store.save_comment(record.changed(**changes))
    return record


def test_blocked_root_prunes_its_whole_subtree(service, note, tokens):
    blocked = service.comment(tokens["alice"], note, None, "spam")
    service.comment(tokens["bob"], note, blocked.id, "reply to spam")
    clean = service.comment(tokens["carol"], note, None, "fine")

    _flag(service, blocked.id, moderation=ModerationState.BLOCKED)

    threads = service.list_comments(note)
    assert [thread.comment.id for thread in threads] == [clean.id]
    assert service.comment_count(note) == 3          # blocking is not deletion


def test_blocked_reply_is_dropped_from_its_thread(service, note, tokens):
    root = service.comment(tokens["alice"], note, None, "root")
    child = service.comment(tokens["bob"], note, root.id, "reply")
    _flag(service, child.id, moderation=ModerationState.BLOCKED)

    threads = service.list_comments(note)
    assert [thread.comment.id for thread in threads] == [root.id]
    assert [reply.id for reply in threads[0].replies] == []


def test_deleted_comment_subtree_is_hidden(service, note, tokens):
    root = service.comment(tokens["alice"], note, None, "root")
    child = service.comment(tokens["bob"], note, root.id, "child")
    _flag(service, child.id, status=ContentStatus.DELETED)
    assert [reply.id for reply in service.list_comments(note)[0].replies] == []


def test_replying_to_a_hidden_comment_is_comment_not_found(service, note, tokens, store):
    blocked = service.comment(tokens["alice"], note, None, "spam")
    _flag(service, blocked.id, moderation=ModerationState.BLOCKED)
    before = dict(store.comments)
    with pytest.raises(ContractError) as excinfo:
        service.comment(tokens["bob"], note, blocked.id, "reply")
    assert excinfo.value.envelope.code is ErrorCode.COMMENT_NOT_FOUND
    assert store.comments == before


def test_admin_view_may_keep_hidden_comments(service, note, tokens):
    root = service.comment(tokens["alice"], note, None, "root")
    blocked = service.comment(tokens["bob"], note, root.id, "spam")
    _flag(service, blocked.id, moderation=ModerationState.BLOCKED)

    assert service.comment_threads(note) == service.list_comments(note)
    assert [t.comment.id for t in service.list_comments(note)] == [root.id]
    back_office = service.comment_threads(note, include_hidden=True)
    assert [reply.id for reply in back_office[0].replies] == [blocked.id]


def test_list_comments_needs_no_token_but_favorites_do(service, note, tokens):
    service.comment(tokens["alice"], note, None, "public read")
    assert [t.comment.body for t in service.list_comments(note)] == ["public read"]
    with pytest.raises(ContractError) as excinfo:
        service.list_favorites(None)
    assert excinfo.value.envelope.code is ErrorCode.AUTH_REQUIRED


def test_list_comments_of_a_missing_note_raises(service):
    with pytest.raises(ContractError) as excinfo:
        service.list_comments(new_id())
    assert excinfo.value.envelope.code is ErrorCode.NOTE_NOT_FOUND


def test_private_note_comments_are_not_readable(catalog, store, principals, users):
    """Even with rows present, a non-public note has no public comment read."""
    seeded = InMemoryInteractionStore()
    seeded.append_comment(
        CommentRecord(
            id=new_id(), note_id=catalog.private_id, author_id=users["alice"],
            body="secret talk", created_at=stamp(0),
        )
    )
    quiet = InteractionService(notes=catalog.directory, store=seeded, sessions=principals)
    assert seeded.comments_for_note(catalog.private_id)
    with pytest.raises(ContractError) as excinfo:
        quiet.list_comments(catalog.private_id)
    assert excinfo.value.envelope.code is ErrorCode.NOTE_NOT_FOUND


# --------------------------------------------------------------------------- #
# favorites                                                                   #
# --------------------------------------------------------------------------- #
def test_favorite_and_unfavorite_round_trip(catalog, tokens):
    service, note_id = catalog.service, catalog.public_id
    assert service.favorite(tokens["alice"], note_id).favorites == 1
    assert service.favorite(tokens["bob"], note_id).favorites == 2
    assert service.unfavorite(tokens["alice"], note_id).favorites == 1
    assert service.unfavorite(tokens["alice"], note_id).favorites == 1  # idempotent
    assert service.is_favorited(tokens["bob"], note_id) is True


def test_list_favorites_returns_the_callers_own_collection(catalog, tokens):
    service = catalog.service
    service.favorite(tokens["alice"], catalog.public_id)
    service.favorite(tokens["bob"], catalog.other_id)

    alice = service.list_favorites(tokens["alice"])
    assert isinstance(alice, Page)                       # the frozen return shape
    assert [summary.id for summary in alice.items] == [catalog.public_id]
    assert alice.total == 1
    bob = service.list_favorites(tokens["bob"])
    assert [summary.id for summary in bob.items] == [catalog.other_id]


def test_list_favorites_is_keyset_paginated_newest_first(catalog, tokens, users, store):
    service = catalog.service
    for offset, note_id in enumerate((catalog.public_id, catalog.other_id, catalog.third_id)):
        service.favorite(tokens["alice"], note_id)
        record = store.favorites[(users["alice"], note_id)]
        store.favorites[(users["alice"], note_id)] = FavoriteRecord(
            id=record.id, user_id=record.user_id, note_id=record.note_id,
            created_at=stamp(offset),
        )

    first = service.list_favorites(tokens["alice"], PageQuery(limit=2))
    assert [summary.title for summary in first.items] == ["Third", "Other"]
    assert first.total == 3
    assert first.has_more is True and first.next_cursor
    assert first.limit == 2

    second = service.list_favorites(tokens["alice"], PageQuery(limit=2, cursor=first.next_cursor))
    assert [summary.title for summary in second.items] == ["Public"]
    assert second.next_cursor is None and second.has_more is False

    # a cursor never repeats or drops an entry
    assert {s.id for s in first.items}.isdisjoint({s.id for s in second.items})


def test_favorites_are_re_derived_from_the_relations(catalog, tokens, users, store):
    """The relation table is the source of truth, the cache is a projection."""
    service = catalog.service
    service.favorite(tokens["alice"], catalog.public_id)
    service.favorite(tokens["bob"], catalog.public_id)
    assert store.counters(catalog.public_id).favorites == 2
    assert store.derive_counters(catalog.public_id).favorites == 2

    store.remove_favorite(users["alice"], catalog.public_id)
    assert store.counters(catalog.public_id).favorites == 1


def test_favorites_projection_overlays_the_interaction_counters(catalog, tokens, store):
    service = catalog.service
    service.favorite(tokens["alice"], catalog.public_id)
    service.like(tokens["bob"], catalog.public_id)
    service.like(tokens["carol"], catalog.public_id)
    service.comment(tokens["bob"], catalog.public_id, None, "nice")

    page = service.list_favorites(tokens["alice"])
    assert page.items[0].counters == Counters(likes=2, favorites=1, comments=1)
    # the note row itself was never touched by this layer
    assert catalog.directory.rows[catalog.public_id].counters == {}


def test_list_favorites_skips_notes_that_are_no_longer_public(catalog, tokens):
    service = catalog.service
    service.favorite(tokens["alice"], catalog.public_id)
    service.favorite(tokens["alice"], catalog.other_id)
    catalog.directory.publish(catalog.other_id, status=ContentStatus.DELETED)

    page = service.list_favorites(tokens["alice"])
    assert [summary.id for summary in page.items] == [catalog.public_id]
    assert page.total == 2  # the collection still holds both rows


def test_list_favorites_rejects_a_broken_cursor(catalog, tokens):
    with pytest.raises(ContractError) as excinfo:
        catalog.service.list_favorites(tokens["alice"], PageQuery(cursor="nonsense"))
    assert excinfo.value.envelope.code is ErrorCode.VALIDATION_ERROR
    assert excinfo.value.envelope.details == {"cursor": ["invalid_cursor"]}


def test_the_bare_existence_port_cannot_project_favorites(notes, store, principals, tokens, users):
    """With only ``IF-note-exists-for-interaction`` the page is empty, not wrong.

    Relations are still recorded — the ids remain reachable through the internal
    seam, while the summary projection needs the richer note directory.
    """
    service = InteractionService(notes, store=store, sessions=principals)
    note_id = notes.add(author_id=users["author"])
    service.favorite(tokens["alice"], note_id)
    assert service.list_favorites(tokens["alice"]).items == ()
    assert service.favorite_note_ids(tokens["alice"]) == (note_id,)
    assert store.has_favorite(users["alice"], note_id) is True


# --------------------------------------------------------------------------- #
# storage: one contract, two engines                                          #
# --------------------------------------------------------------------------- #
def test_in_memory_store_enforces_uniqueness_directly(users):
    note_id = new_id()
    store = InMemoryInteractionStore()

    first = store.add_like(users["alice"], note_id)
    duplicate = store.add_like(users["alice"], note_id)
    other = store.add_like(users["bob"], note_id)

    assert (first.changed, duplicate.changed, other.changed) == (True, False, True)
    assert store.counters(note_id) == Counters(likes=2)
    assert len({record.id for record in store.likes.values()}) == 2
    assert store.liked_note_ids(users["alice"]) == (note_id,)

    removed = store.remove_like(users["alice"], note_id)
    missing = store.remove_like(users["alice"], note_id)
    assert (removed.changed, missing.changed) == (True, False)
    assert store.counters(note_id) == Counters(likes=1)
    assert store.has_like(users["alice"], note_id) is False


def test_unique_key_is_per_pair_not_per_note_or_user(users):
    first, second = new_id(), new_id()
    store = InMemoryInteractionStore()
    store.add_like(users["alice"], first)
    store.add_like(users["alice"], second)      # same user, other note
    store.add_like(users["bob"], first)         # same note, other user
    assert store.counters(first) == Counters(likes=2)   # alice + bob
    assert store.counters(second) == Counters(likes=1)  # alice only
    assert store.favorite_page(users["alice"], limit=10).total == 0


def test_store_transaction_rolls_back_relations_and_counters_together(users, note_id):
    store = InMemoryInteractionStore()

    class Boom(RuntimeError):
        pass

    with pytest.raises(Boom):
        with store.transaction():
            store.add_like(users["alice"], note_id)
            raise Boom()
    assert store.likes == {}
    assert store.counters(note_id) == Counters()

    with store.transaction():
        store.add_like(users["alice"], note_id)
    assert store.counters(note_id) == Counters(likes=1)


def test_counter_cache_and_relations_cannot_disagree(service, note, tokens, store):
    """``INV-concurrency-counters`` after every operation of the frozen surface."""
    calls = [
        lambda: service.like(tokens["alice"], note),
        lambda: service.like(tokens["bob"], note),
        lambda: service.unlike(tokens["alice"], note),
        lambda: service.unlike(tokens["alice"], note),
        lambda: service.favorite(tokens["carol"], note),
        lambda: service.comment(tokens["alice"], note, None, "hi"),
    ]
    for index, call in enumerate(calls, start=1):
        call()
        assert store.counters(note) == store.derive_counters(note), index


def test_concurrent_writes_keep_the_counter_equal_to_the_relation_table(users, note_id):
    """Parallel like/unlike of many users must never corrupt the cache.

    The writers line up on a barrier so they really do hit the store at the same
    instant: every relation row is unique per ``(user_id, note_id)`` and the
    counter cache is moved in the same transaction (``INV-concurrency-counters``),
    so the cache can never drift away from the relation table.
    """
    store = InMemoryInteractionStore()
    crowd = [new_id() for _ in range(24)]
    start = threading.Barrier(len(crowd))

    def hammer(user_id: str) -> None:
        start.wait()
        for _ in range(6):
            store.add_like(user_id, note_id)
        for _ in range(3):
            store.remove_like(user_id, note_id)

    with ThreadPoolExecutor(max_workers=len(crowd)) as pool:
        list(pool.map(hammer, crowd))

    # every user cancelled their own like again: rows and cache agree on zero
    assert [key for key in store.likes if key[0] in set(crowd)] == []
    assert store.counters(note_id) == Counters(likes=0)
    assert store.counters(note_id) == store.derive_counters(note_id)

    # and a re-like after the churn still lands exactly once
    store.add_like(crowd[0], note_id)
    store.add_like(crowd[0], note_id)
    assert store.counters(note_id) == store.derive_counters(note_id) == Counters(likes=1)


def test_the_same_user_cannot_double_count_through_two_threads(users, note_id):
    """One racing pair: two threads, one ``(user, note)`` like."""
    store = InMemoryInteractionStore()
    alice = users["alice"]
    start = threading.Barrier(2)

    def racer() -> None:
        start.wait()
        store.add_like(alice, note_id)

    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = pool.submit(racer), pool.submit(racer)
        first.result()
        second.result()

    assert sum(1 for key in store.likes if key == (alice, note_id)) == 1   # exactly one row
    assert store.counters(note_id) == store.derive_counters(note_id) == Counters(likes=1)


def test_sqlite_store_honours_the_same_contract(users, note_id):
    store = SQLiteInteractionStore(":memory:")
    try:
        assert store.add_like(users["alice"], note_id).changed is True
        assert store.add_like(users["alice"], note_id).changed is False   # UNIQUE pair
        assert store.add_like(users["bob"], note_id).counters.likes == 2
        assert store.add_favorite(users["alice"], note_id).changed is True
        assert store.counters(note_id) == Counters(likes=2, favorites=1)
        assert store.derive_counters(note_id) == store.counters(note_id)

        assert store.remove_like(users["alice"], note_id).counters.likes == 1  # bob remains
        assert store.remove_like(new_id(), note_id).changed is False           # unknown user
        assert store.remove_like(users["alice"], note_id).counters.likes == 1  # no second delta
        assert store.remove_like(users["bob"], note_id).counters.likes == 0

        record = CommentRecord(
            id=new_id(), note_id=note_id, author_id=users["alice"], body="hi",
            created_at=stamp(0),
        )
        assert store.append_comment(record).counters.comments == 1
        assert store.get_comment(record.id).body == "hi"
        assert [row.id for row in store.comments_for_note(note_id)] == [record.id]
        assert store.save_comment(record.changed(body="edited")).body == "edited"
        assert store.counters(note_id) == store.derive_counters(note_id)
    finally:
        store.close()


def test_sqlite_store_pages_favorites_newest_first(users):
    store = SQLiteInteractionStore(":memory:")
    try:
        owner, ids = users["alice"], [new_id() for _ in range(3)]
        for offset, note_id in enumerate(ids):
            store.add_favorite(owner, note_id, created_at=stamp(offset))
        window = store.favorite_page(owner, limit=2)
        assert [row.note_id for row in window.rows] == [ids[2], ids[1]]
        assert (window.total, window.has_more) == (3, True)
        nxt = store.favorite_page(owner, limit=2, cursor=(window.rows[-1].created_at, window.rows[-1].id))
        assert [row.note_id for row in nxt.rows] == [ids[0]]
        assert nxt.has_more is False
    finally:
        store.close()


def test_sqlite_transaction_is_atomic_over_relation_and_counter(users, note_id):
    store = SQLiteInteractionStore(":memory:")
    try:
        store.add_like(users["alice"], note_id)
        store.add_favorite(users["alice"], note_id)
        before = store.counters(note_id)
        assert before == Counters(likes=1, favorites=1)

        class Boom(RuntimeError):
            pass

        with pytest.raises(Boom):
            with store.transaction():
                store.remove_like(users["alice"], note_id)
                store.remove_favorite(users["alice"], note_id)
                raise Boom()
        assert store.has_like(users["alice"], note_id) is True
        assert store.counters(note_id) == before
        assert store.derive_counters(note_id) == before
    finally:
        store.close()


def test_sqlite_counter_update_is_clamped_in_sql(users, note_id):
    """``MAX(0, col + d)`` — even a damaged cache cannot render negative."""
    store = SQLiteInteractionStore(":memory:")
    try:
        store.add_like(users["alice"], note_id)
        table = store.counter_sink.table
        key = store.counter_sink.key_column
        with store.transaction():
            store.connection.execute(
                f"UPDATE {table} SET like_count = 0 WHERE {key} = ?", (note_id,)
            )
        assert store.counters(note_id).likes == 0
        assert store.remove_like(users["alice"], note_id).counters.likes == 0
        assert all(value >= 0 for value in store.counters(note_id).to_dict().values())
    finally:
        store.close()


def test_sqlite_uniqueness_is_enforced_by_the_schema(users, note_id):
    store = SQLiteInteractionStore(":memory:")
    try:
        store.add_like(users["alice"], note_id)
        with pytest.raises(sqlite3.IntegrityError):
            store.connection.execute(
                "INSERT INTO note_likes (id, user_id, note_id, created_at) VALUES (?, ?, ?, ?)",
                (new_id(), users["alice"], note_id, stamp(0)),
            )
    finally:
        store.close()


def test_open_store_wiring(prepare_tmp_db):
    assert isinstance(open_store(prefer="memory"), InMemoryInteractionStore)
    sqlite = open_store(prepare_tmp_db, prefer="sqlite")
    assert isinstance(sqlite, SQLiteInteractionStore)
    sqlite.close()
    memory = open_store(prefer="memory", notes=LedgerNoteRepository())
    assert isinstance(memory.counter_sink, NoteRecordCounterSink)


def test_sqlite_store_needs_an_explicit_location():
    """The layer never invents ``data/blacknote.db`` next to the repository."""
    with pytest.raises(ContractError) as excinfo:
        open_store(prefer="sqlite")
    assert excinfo.value.envelope.code is ErrorCode.CONFIG_ERROR
    assert not (SANDBOX_ROOT / "data").exists()
    assert not list(SANDBOX_ROOT.glob("*.db"))
    assert not list(SANDBOX_ROOT.glob("**/blacknote.db"))


@pytest.fixture
def prepare_tmp_db(tmp_path):
    """An explicitly located database file (never inside the repository)."""
    return tmp_path / "interaction.db"


def test_file_backed_store_writes_only_where_asked(prepare_tmp_db, users, note_id):
    assert not prepare_tmp_db.exists()
    store = open_store(prepare_tmp_db, prefer="sqlite")
    store.add_like(users["alice"], note_id)
    store.close()
    assert prepare_tmp_db.exists()
    assert not (SANDBOX_ROOT / "data").exists()


def test_note_cache_columns_move_inside_the_interaction_transaction(ledger_service, ledger, ledger_note, tokens, users):
    """``INV-concurrency-counters``: relation write + note cache column, one unit."""
    service = ledger_service
    service.like(tokens["alice"], ledger_note)
    assert ledger.counters(ledger_note).likes == 1
    assert cached(ledger, ledger_note, "like_count") == 1
    assert ledger.rows[ledger_note]["favorite_count"] == 0

    service.favorite(tokens["bob"], ledger_note)
    assert ledger.counters(ledger_note) == Counters(likes=1, favorites=1)

    service.unlike(tokens["alice"], ledger_note)
    service.unlike(tokens["alice"], ledger_note)      # idempotent → no second delta
    assert ledger.counters(ledger_note) == Counters(likes=0, favorites=1)
    assert ledger.calls.count("save") == 3


def test_note_counter_sink_rolls_the_cache_back_with_the_relation(ledger, sink, users, ledger_note):
    store = InMemoryInteractionStore(counters=sink)
    store.add_like(users["alice"], ledger_note)
    assert ledger.counters(ledger_note) == Counters(likes=1)

    class Boom(RuntimeError):
        pass

    with pytest.raises(Boom):
        with store.transaction():
            store.remove_like(users["alice"], ledger_note)
            raise Boom()
    assert ledger.counters(ledger_note) == Counters(likes=1)   # cache restored
    assert store.derive_counters(ledger_note) == Counters(likes=1)


def test_note_counter_sink_moves_only_the_requested_column(ledger, sink, ledger_note, users):
    sink.apply_delta(ledger_note, comments=3)
    assert ledger.counters(ledger_note) == Counters(comments=3)
    assert sink.apply_delta(ledger_note, likes=-5) == Counters(likes=0, comments=3)
    assert cached(ledger, ledger_note, "comment_count") == 3


def test_counter_sink_requires_a_real_note_store():
    with pytest.raises(ContractError) as excinfo:
        NoteRecordCounterSink(object())
    assert excinfo.value.envelope.code is ErrorCode.CONFIG_ERROR


def test_store_rejects_unknown_relation_partners(users):
    store = InMemoryInteractionStore()
    with pytest.raises(ContractError) as excinfo:
        store.add_like(users["alice"], "not-a-uuid")
    assert excinfo.value.envelope.code is ErrorCode.NOTE_NOT_FOUND


# --------------------------------------------------------------------------- #
# composition seams (moderation / web templates)                              #
# --------------------------------------------------------------------------- #
def test_moderation_seams_flag_without_deleting(service, note, tokens):
    view = service.comment(tokens["alice"], note, None, "report me")

    assert service.get_comment(view.id).id == view.id
    assert service.comment_note_id(view.id) == note
    assert service.get_comment("nope") is None
    assert service.comment_note_id("nope") is None

    record = service.store.get_comment(view.id)
    service.store.save_comment(record.changed(moderation=ModerationState.BLOCKED))

    assert service.get_comment(view.id).moderation is ModerationState.BLOCKED
    assert service.store.comments_for_note(note)          # no cascade delete
    assert service.comment_count(note) == 1
    assert service.list_comments(note) == []


def test_thread_projection_of_a_single_comment(service, note, tokens):
    root = service.comment(tokens["alice"], note, None, "root")
    child = service.comment(tokens["bob"], note, root.id, "child")

    thread = service.thread_of(child.id)
    assert thread.comment.id == root.id
    assert [reply.id for reply in thread.replies] == [child.id]
    assert service.thread_of(root.id).comment.id == root.id

    # an unknown comment is COMMENT_NOT_FOUND, a hidden one has no thread
    with pytest.raises(ContractError) as excinfo:
        service.thread_of(new_id())
    assert excinfo.value.envelope.code is ErrorCode.COMMENT_NOT_FOUND
    _flag(service, child.id, moderation=ModerationState.BLOCKED)
    assert service.thread_of(child.id) is None


def test_counter_seams_read_through_the_note_port(service, notes, note, tokens, users):
    service.like(tokens["alice"], note)
    assert service.is_liked(tokens["alice"], note) is True
    assert service.is_favorited(tokens["alice"], note) is False
    assert service.liked_notes(users["alice"]) == (note,)
    assert service.counters(note).likes == 1
    assert service.derived_counters(note).likes == 1
    assert service.sync_counters(note) == service.counters(note)
    # This layer never writes note rows: the fake note record still holds no
    # counter columns, and the only contact is the read probe (INV-dependency).
    assert "like_count" not in notes.rows[note]
    assert notes.probe_calls, "the note port must be consulted"
    assert service.note_port is not notes  # adapted, so only the port is reachable


def test_cache_drift_is_repaired_explicitly(service, note, tokens, store):
    """``sync_counters`` is the ops repair; reads never trigger it implicitly."""
    service.like(tokens["alice"], note)
    service.comment(tokens["bob"], note, None, "hi")
    store.seed_counters(note, Counters(likes=42, comments=42))

    assert service.counters(note) == Counters(likes=42, comments=42)   # stale cache
    assert service.derived_counters(note) == Counters(likes=1, comments=1)
    assert service.sync_counters(note) == Counters(likes=1, comments=1)
    assert store.counters(note) == store.derive_counters(note)


# --------------------------------------------------------------------------- #
# comment removal (author-only seam above the frozen surface)                  #
# --------------------------------------------------------------------------- #
@pytest.fixture
def thread(service, note, tokens) -> dict[str, CommentView]:
    """root ← reply ← grandchild, plus one unrelated comment by another author."""
    root = service.comment(tokens["alice"], note, None, "root")
    reply = service.comment(tokens["bob"], note, root.id, "reply")
    grand = service.comment(tokens["alice"], note, reply.id, "grandchild")
    other = service.comment(tokens["bob"], note, None, "unrelated")
    return {"root": root, "reply": reply, "grand": grand, "other": other, "note": note}


def test_removing_a_comment_removes_its_whole_thread(thread, service, store, tokens):
    """The counter follows the row state, so cache and relations stay aligned."""
    assert service.counters(thread["note"]).comments == 4
    result = service.delete_comment_result(tokens["alice"], thread["root"].id)
    assert result.changed is True
    assert result.value == Counters(comments=1)          # only "unrelated" is left
    assert store.counters(thread["note"]) == store.derive_counters(thread["note"])
    assert [[c.comment.body] + [r.body for r in c.replies]
            for c in service.list_comments(thread["note"])] == [["unrelated"]]


def test_hidden_comments_survive_for_the_back_office(thread, service, tokens):
    service.delete_comment(tokens["alice"], thread["root"].id)
    rows = service.comment_threads(thread["note"], include_hidden=True)
    bodies = {row.comment.body for row in rows} | {r.body for row in rows for r in row.replies}
    assert {"root", "reply", "grandchild"} <= bodies            # not erased
    assert all(row.comment.status is not None for row in rows)
    assert service.store.get_comment(thread["root"].id).status is ContentStatus.DELETED


def test_deleting_twice_moves_the_counter_once(thread, service, tokens):
    author = tokens["alice"]
    first = service.delete_comment_result(author, thread["grand"].id)
    second = service.delete_comment_result(author, thread["grand"].id)
    assert (first.changed, second.changed) == (True, False)
    assert first.value == second.value


def test_hard_delete_purges_the_row(thread, service, store, tokens):
    author = tokens["alice"]
    result = service.delete_comment_result(author, thread["root"].id, hard=True)
    assert result.changed is True and result.value.comments == 1
    assert store.get_comment(thread["root"].id) is None          # gone, not hidden
    assert store.get_comment(thread["reply"].id) is None         # subtree purged
    assert len(store.comments_for_note(thread["note"])) == 1
    assert store.counters(thread["note"]) == store.derive_counters(thread["note"])


def test_only_the_author_may_delete(thread, service, store, tokens):
    """Foreign comments answer exactly like missing ones — no id probing."""
    strangers = tokens["bob"]
    before = service.counters(thread["note"])
    for comment_id in (thread["root"].id, new_id(), "not-a-uuid"):
        with pytest.raises(ContractError) as excinfo:
            service.delete_comment(strangers, comment_id)
        assert excinfo.value.envelope.code is ErrorCode.COMMENT_NOT_FOUND
    assert service.counters(thread["note"]) == before
    assert service.get_comment(thread["root"].id) is not None


@pytest.mark.parametrize("token", [None, "", "bogus", "expired-alice"], ids=["missing", "empty", "bogus", "expired"])
def test_deleting_a_comment_without_a_session_writes_nothing(thread, service, store, token):
    before = service.counters(thread["note"])
    with pytest.raises(ContractError) as excinfo:
        service.delete_comment(token, thread["root"].id)
    assert excinfo.value.envelope.code is ErrorCode.AUTH_REQUIRED
    assert service.counters(thread["note"]) == before
    assert store.get_comment(thread["root"].id).status is not ContentStatus.DELETED


def test_comment_removal_contract_holds_on_sqlite(users, tokens, principals):
    """Both engines implement one store contract, removal semantics included."""
    directory = InMemoryNoteDirectory()
    service = InteractionService(directory, store=SQLiteInteractionStore(":memory:"),
                                 sessions=principals)
    note_id = new_id()
    directory.add(
        {
            "id": note_id,
            "author_id": users["author"],
            "author_nickname": "author",
            "title": "Removal",
            "body": "body",
            "visibility": Visibility.PUBLIC,
            "status": ContentStatus.PUBLISHED,
            "created_at": NOTE_CREATED,
        }
    )
    root = service.comment(tokens["alice"], note_id, None, "root")
    reply = service.comment(tokens["bob"], note_id, root.id, "reply")
    unrelated = service.comment(tokens["carol"], note_id, None, "unrelated")

    assert service.counters(note_id).comments == 3
    assert service.delete_comment(tokens["alice"], root.id) == Counters(comments=1)
    hidden = service.store.get_comment(reply.id)
    assert hidden is not None and hidden.status is ContentStatus.DELETED
    assert [row.comment.body for row in service.list_comments(note_id)] == ["unrelated"]
    assert service.store.counters(note_id) == service.store.derive_counters(note_id)

    assert service.delete_comment(tokens["carol"], unrelated.id, hard=True) == Counters()
    # hard purge removed that row; the hidden subtree is still stored (raw reads
    # include hidden rows, the read model is what prunes them)
    assert [row.id for row in service.store.comments_for_note(note_id)] == [root.id, reply.id]
    assert all(row.status is ContentStatus.DELETED
               for row in service.store.comments_for_note(note_id))
    assert service.store.get_comment(unrelated.id) is None
    assert service.store.counters(note_id) == service.store.derive_counters(note_id)


# --------------------------------------------------------------------------- #
# wiring pieces: result envelope, ports, models, read shaping                  #
# --------------------------------------------------------------------------- #
def test_rich_results_describe_the_write(service, note, tokens, users):
    """``*_result`` is the non-raising view the web layer renders directly."""
    result = service.like_result(tokens["alice"], note)
    assert result.ok is True and result.is_failure is False
    assert result.value == Counters(likes=1) == result.counters
    assert result.changed is True
    assert result.actor_id == users["alice"]
    assert result.entity_id == note
    assert result.status == 200 and result.envelope is None and result.to_error() is None
    assert result.to_dict()["counters"] == {"likes": 1, "favorites": 0, "comments": 0}

    again = service.like_result(tokens["alice"], note)
    assert again.changed is False and again.value.likes == 1


def test_failure_results_render_the_envelope_the_exception_would(users, note_id):
    raised = ContractError(ErrorCode.AUTH_REQUIRED, None, {"token": "missing"})
    failure = InteractionResult.from_error(raised)
    assert failure.is_failure and failure.code is ErrorCode.AUTH_REQUIRED
    assert failure.status == status_for("AUTH_REQUIRED") == 401
    assert failure.envelope == raised.envelope           # one error, two shapes
    assert set(failure.to_dict()) >= {"ok", "value", "error", "message", "details", "counters"}
    assert failure.unwrap is not None
    with pytest.raises(ContractError) as excinfo:
        failure.unwrap()
    assert excinfo.value.envelope == raised.envelope
    assert failure.value is None and failure.counters == Counters()

    explicit = InteractionResult.failure(
        "VALIDATION_ERROR", "too long", {"body": ["too_long"]}, request_id="req-1"
    )
    assert (explicit.code, explicit.status, explicit.request_id) == (
        ErrorCode.VALIDATION_ERROR, 400, "req-1",
    )

    unexpected = InteractionResult.from_error(RuntimeError("boom"))
    assert unexpected.code is ErrorCode.INTERNAL_ERROR
    assert unexpected.status == 500 and "boom" in str(unexpected.message)


def test_a_service_without_a_note_directory_fails_closed(principals, tokens, store):
    """``NullNoteDirectory``: nothing is interactable, so nothing is written."""
    directory = NullNoteDirectory()
    assert directory.is_publicly_interactable(new_id()) is False
    assert directory.get_note_author_id(new_id()) is None
    assert directory.summary(new_id()) is None

    service = InteractionService(directory, store=store, sessions=principals)
    with pytest.raises(ContractError) as excinfo:
        service.like(tokens["alice"], new_id())
    assert excinfo.value.envelope.code is ErrorCode.NOTE_NOT_FOUND
    assert store.likes == {} and store.counters(new_id()) == Counters()


def test_the_session_port_is_accepted_in_every_shape_the_platform_provides(users, principals, tokens):
    """``adapt_sessions``: mapping / callable / facade / fixed principal / nothing."""
    alice = principals[tokens["alice"]]

    class Resolver:
        def resolve(self, token):
            return principals.get(token)

    class CurrentUser:
        def current_user(self, token):
            return principals.get(token)

    resolvers = [
        adapt_sessions(principals),
        adapt_sessions(principals.get),
        adapt_sessions(Resolver()),
        adapt_sessions(CurrentUser()),
        adapt_sessions(alice),                     # fixed principal (dev wiring)
    ]
    for resolver in resolvers:
        assert authenticate(tokens["alice"], resolver).user_id == users["alice"]
    assert adapt_sessions(None) is None            # fail closed inside auth_guard
    with pytest.raises(ContractError) as excinfo:
        authenticate(None, adapt_sessions(principals))
    assert excinfo.value.envelope.code is ErrorCode.AUTH_REQUIRED
    with pytest.raises(ContractError):
        authenticate(tokens["expired"], adapt_sessions(principals))


def test_nicknames_come_from_the_injected_identity_directory(users, principals, tokens, notes, store):
    """Comment rendering needs a display name, never a raw user id."""
    directory = InMemoryIdentityDirectory({users["alice"]: "Alice"})
    service = InteractionService(notes, store=store, sessions=principals, identity=directory)
    note_id = notes.add(author_id=users["author"])
    assert service.comment(tokens["alice"], note_id, None, "hi").author_nickname == "Alice"

    directory.set_nickname(users["alice"], "   ")          # blank → anonymous mask
    second = notes.add(author_id=users["author"])
    assert service.comment(tokens["alice"], second, None, "hi").author_nickname == ANONYMOUS_NICKNAME
    assert NullIdentityDirectory().nickname_of(users["alice"]) == ANONYMOUS_NICKNAME
    assert adapt_directory({users["bob"]: "Bob"}).nickname_of(users["bob"]) == "Bob"


def test_relation_models_canonicalise_and_round_trip(users, note_id):
    """Rows survive the trip through the storage shape untouched."""
    like = LikeRecord(id=new_id(), user_id=users["alice"], note_id=note_id)
    assert is_iso8601(like.created_at)
    assert like.key == (users["alice"], note_id)
    assert LikeRecord.from_row(like.to_row()) == like

    comment = CommentRecord(id=new_id(), note_id=note_id, author_id=users["alice"], body="hi")
    assert comment.status is ContentStatus.PUBLISHED
    assert comment.moderation is ModerationState.VISIBLE
    assert comment.is_reply is False and comment.sort_key == (comment.created_at, comment.id)
    assert CommentRecord.from_row(comment.to_row()) == comment

    edited = comment.changed(body="hello")
    assert edited.id == comment.id and edited.body == "hello"
    assert edited.created_at == comment.created_at        # the stamp never moves
    assert comment.body == "hi"                           # frozen: copy-on-write

    view = comment.to_view("alice")
    assert isinstance(view, CommentView) and view.author_nickname == "alice"
    assert view.note_id == note_id and view.parent_comment_id is None

    assert relation_key("", note_id) is None
    assert relation_key(users["alice"], "not-a-uuid") is None
    assert relation_key(users["alice"], note_id) == (users["alice"], note_id)


def test_pagination_helpers_keep_the_platform_shape():
    assert clamp_limit(None) == DEFAULT_PAGE_SIZE
    assert clamp_limit(999) == MAX_PAGE_SIZE == int(LIMITS["page_size"]["max"])
    assert clamp_limit(3) == 3
    for junk in ("many", 0, -1):
        with pytest.raises(ContractError) as excinfo:
            clamp_limit(junk)
        assert excinfo.value.envelope.code is ErrorCode.VALIDATION_ERROR

    cursor = encode_cursor(stamp(1), new_id())
    assert decode_cursor(cursor) == (stamp(1), decode_cursor(cursor)[1])
    assert decode_cursor(None) is None and decode_cursor("") is None
    with pytest.raises(ContractError) as excinfo:
        decode_cursor("garbage")
    assert excinfo.value.envelope.details == {"cursor": ["invalid_cursor"]}

    page = page_of(("a", "b", "c"), limit=2, cursor=cursor, total=5)
    assert page.items == ("a", "b", "c") and page.total == 5 and page.limit == 2
    assert page.has_more is True                     # has_more follows next_cursor
    assert page_of(("a",), limit=2, cursor=None).has_more is False


def test_thread_assembly_is_a_pure_function(store, service, note, tokens, users):
    """The read model is reusable outside the service (templates, exports)."""
    root = service.comment(tokens["alice"], note, None, "root")
    reply = service.comment(tokens["bob"], note, root.id, "re")
    leaf = service.comment(tokens["carol"], note, reply.id, "leaf")
    sibling = service.comment(tokens["bob"], note, None, "sibling")
    rows = store.comments_for_note(note)

    threads = assemble_threads(rows)
    assert [entry[0].id for entry in threads] == [root.id, sibling.id]
    assert [row.id for row in threads[0][1]] == [reply.id, leaf.id]
    assert visible_rows(rows) == {row.id: row for row in rows}

    store.save_comment(store.get_comment(reply.id).changed(moderation=ModerationState.BLOCKED))
    pruned = assemble_threads(store.comments_for_note(note))
    assert [entry[0].id for entry in pruned] == [root.id, sibling.id]
    assert list(pruned[0][1]) == []    # the blocked reply *and its subtree* disappear
    assert leaf.id not in {row.id for _, branch in pruned for row in branch}

    # blocking is a flag, not a deletion: the rows are all still stored
    raw = assemble_threads(store.comments_for_note(note), prune=False)
    assert {row.id for _, branch in raw for row in branch} >= {reply.id, leaf.id}
    assert service.comment_count(note) == 4


def test_default_settings_never_open_the_contracts_default_database(users, principals, tokens, notes):
    """Regression guard: this layer must not create ``data/blacknote.db``.

    ``Settings.database_url`` defaults to ``sqlite:///data/blacknote.db``. The
    interaction module deliberately never reads storage settings: a service
    wired with plain defaults still writes only through the store it was given,
    so no import, no write call and no configuration can drop a database file
    inside the repository.
    """
    service = InteractionService(notes, sessions=principals, settings=Settings())
    assert service.settings is not None
    note_id = notes.add(author_id=users["author"])
    assert service.like(tokens["alice"], note_id).likes == 1
    assert service.favorite(tokens["bob"], note_id).favorites == 1
    assert service.comment(tokens["carol"], note_id, None, "hi").body == "hi"
    assert list(service.list_comments(note_id))

    assert not (SANDBOX_ROOT / "data").exists()
    assert not list(SANDBOX_ROOT.rglob("blacknote.db"))
    assert not list(SANDBOX_ROOT.rglob("*.sqlite"))


def test_the_layer_is_importable_without_touching_the_filesystem():
    """A fresh interpreter: importing mod-interaction has no side effects."""
    probe = subprocess.run(
        [sys.executable, "-c", "import blacknote.interaction"],
        cwd=SANDBOX_ROOT,
        env={**os.environ, "PYTHONPATH": str(SANDBOX_ROOT / "src")},
        capture_output=True,
        text=True,
    )
    assert probe.returncode == 0, probe.stderr
    assert not (SANDBOX_ROOT / "data").exists()
    assert not list(SANDBOX_ROOT.rglob("blacknote.db"))


def test_every_identifier_and_timestamp_is_canonical(users, principals, tokens, notes, store):
    """``INV-time-encoding`` + UUIDv4 ids on everything the layer emits."""
    service = InteractionService(notes, store=store, sessions=principals)
    note_id = notes.add(author_id=users["author"])
    first = service.comment(tokens["alice"], note_id, None, "first")
    second = service.comment(tokens["bob"], note_id, first.id, "second")
    third = service.comment(tokens["carol"], note_id, None, "third")

    for view in (first, second, third):
        assert is_valid_id(view.id) and is_valid_id(view.note_id) and is_valid_id(view.author_id)
        assert is_iso8601(view.created_at)
        assert store.get_comment(view.id).id == view.id

    rows = store.comments_for_note(note_id)
    stamps = [to_epoch_ms(from_iso(row.created_at)) for row in rows]
    assert stamps == sorted(stamps)                      # storage order == time order
    ordered = [row.id for row in sorted(rows, key=lambda row: row.sort_key)]
    threads = service.list_comments(note_id)
    roots = [row.id for row in sorted(rows, key=lambda row: row.sort_key)
             if row.parent_comment_id is None]
    assert [thread.comment.id for thread in threads] == roots        # threads, chronologically
    assert [node.id for thread in threads for node in (thread.comment, *thread.replies)] == ordered
    assert second.parent_comment_id == first.id


def test_result_variants_fold_business_failures_into_the_envelope(service, note, tokens, store):
    """``*_result`` never raises for a *user* error, but says the same thing.

    The frozen methods raise the platform ``ContractError``; the result view is
    for callers (templates, JSON handlers) that prefer to render
    ``result.to_dict()``.  Both shapes carry the same code / status / details,
    so there is still exactly one error vocabulary.
    """
    anonymous = service.like_result(None, note)
    assert anonymous.ok is False and anonymous.code is ErrorCode.AUTH_REQUIRED
    assert anonymous.status == 401 and anonymous.value is None
    assert "token" in str(anonymous.message)
    assert store.counters(note) == Counters() and store.likes == {}

    invalid = service.comment_result(tokens["alice"], note, None, "   ")
    assert invalid.code is ErrorCode.VALIDATION_ERROR and invalid.status == 400
    assert {entry["field"] for entry in invalid.details["fields"]} == {"body"}

    absent = service.favorite_result(tokens["alice"], new_id())
    assert absent.code is ErrorCode.NOTE_NOT_FOUND and absent.status == 404
    assert absent.details == {"note_id": absent.details["note_id"]}

    foreign = service.delete_comment_result(tokens["bob"], note, hard=False)
    assert foreign.code is ErrorCode.COMMENT_NOT_FOUND

    # the raised form and the returned form describe one and the same error
    with pytest.raises(ContractError) as excinfo:
        service.like(None, note)
    assert excinfo.value.envelope == anonymous.envelope
    payload = invalid.to_dict()
    assert payload["ok"] is False and payload["error"] == "VALIDATION_ERROR"
    assert payload["status"] == 400


def test_a_broken_dependency_is_never_folded_into_a_user_error(principals, tokens, store):
    """Only contract errors become results; a server bug must still explode.

    ``TypeError`` / ``ValueError`` / ``LookupError`` while probing the optional
    note seams mean "this dependency cannot answer" and fail closed into
    ``NOTE_NOT_FOUND`` (see :mod:`blacknote.interaction.ports`); anything else
    is a bug and travels through the result wrapper untouched.
    """
    dependency = UnreachableNotePort()
    service = InteractionService(dependency, store=store, sessions=principals)
    with pytest.raises(RuntimeError):
        service.comment_result(tokens["alice"], dependency.note_id, None, "hello")
    with pytest.raises(RuntimeError):
        service.like_result(tokens["alice"], dependency.note_id)
    assert store.counters(dependency.note_id) == Counters()
