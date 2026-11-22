"""Interaction persistence: relations **and** note counters in one transaction.

What this module guarantees (``INV-concurrency-counters``)
---------------------------------------------------------
Every mutation of a like / favorite / comment relation and the corresponding
update of the note-side counter cache happen inside a *single* transaction:

``store.add_like(user, note)``
    ``BEGIN`` → ``INSERT`` the edge (unique ``(user_id, note_id)``) → delta the
    ``like_count`` cache → read the cache back → ``COMMIT``.

Consequences that a reviewer can check mechanically:

* **Idempotency comes from the unique constraint.**  A second like for the same
  ``(user_id, note_id)`` pair hits ``UNIQUE`` (SQLite) / the dict key (memory),
  the store reports ``changed=False`` and *no* counter delta is applied — there
  is no "read then write" pre-check anywhere in the write path.
* **Cancellations decrement exactly once.**  ``DELETE`` reports the affected row
  count; ``0`` means "already un-liked" and no delta is applied.
* **Counters can never go negative.**  Both sinks clamp with ``MAX(0, col + d)``
  / :meth:`Counters.delta`, and the delta is only ever issued for a relation
  write that actually happened.
* **Two views differ by 0.**  The value returned to the caller is read back from
  the cache inside the same transaction (``store.counters``), and
  ``store.derive_counters`` recomputes the truth from the relation tables, so
  drift is detectable (and repairable via ``store.sync_counters``).

Two storage engines are provided behind one port:

:class:`InMemoryInteractionStore`
    thread safe dicts, used by tests and dev wiring; its transaction is the
    store lock plus an explicit begin/commit/rollback journal.
:class:`SQLiteInteractionStore`
    file backed storage (``INV-config-db``); the counter update is a second
    statement of the same ``BEGIN IMMEDIATE`` transaction.  When the database
    also holds the ``notes`` table written by ``mod-note``, the cache columns
    ``like_count`` / ``favorite_count`` / ``comment_count`` are updated in place;
    otherwise the store maintains its own ``note_counters`` table with the same
    columns, so the atomicity property does not depend on the wiring.

This module imports ``blacknote.contracts`` and the standard library only.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Protocol, runtime_checkable

from blacknote.contracts.errors import ContractError, ErrorCode
from blacknote.contracts.ids import is_valid_id, new_id
from blacknote.contracts.interfaces import Counters
from blacknote.contracts.timeutil import now_iso
from blacknote.contracts.visibility import ContentStatus

from .models import CommentRecord, FavoriteRecord, LikeRecord, relation_key

__all__ = [
    "Transition",
    "FavoriteWindow",
    "CounterSink",
    "DictCounterSink",
    "NoteRecordCounterSink",
    "InteractionStore",
    "InMemoryInteractionStore",
    "SQLiteInteractionStore",
    "open_store",
    "LIKE",
    "FAVORITE",
    "COMMENT",
]

#: Counter field names (single source of truth for the deltas below).
LIKE: Final[str] = "likes"
FAVORITE: Final[str] = "favorites"
COMMENT: Final[str] = "comments"

_COLUMNS: Final[Mapping[str, str]] = {
    LIKE: "like_count",
    FAVORITE: "favorite_count",
    COMMENT: "comment_count",
}


def _stamp(value: str | None = None) -> str:
    """Caller-supplied timestamp or "now" (``INV-time-encoding``).

    Idempotency and keyset pagination both rely on comparable canonical strings,
    so a relation row never carries ``NULL`` / an empty timestamp.
    """
    return value or now_iso()


@dataclass(frozen=True)
class Transition:
    """Outcome of one transactional relation write.

    ``changed`` answers "did the relation actually move?" — the flag that keeps
    duplicate likes and duplicate un-likes from touching a counter.
    """

    changed: bool
    counters: Counters

    def to_dict(self) -> dict[str, Any]:
        return {"changed": self.changed, "counters": self.counters.to_dict()}


@dataclass(frozen=True)
class FavoriteWindow:
    """One keyset page of favorites (``created_at DESC``) plus its bookkeeping."""

    rows: tuple[FavoriteRecord, ...] = ()
    total: int = 0
    has_more: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "rows", tuple(self.rows))
        if self.total < 0:
            raise ValueError("FavoriteWindow.total must be >= 0")


# --------------------------------------------------------------------------- #
# Counter cache sinks
# --------------------------------------------------------------------------- #
@runtime_checkable
class CounterSink(Protocol):
    """Where the note-side counter cache lives (always inside the store txn)."""

    def begin(self) -> None:  # pragma: no cover - protocol
        ...

    def apply_delta(
        self, note_id: str, *, likes: int = 0, favorites: int = 0, comments: int = 0
    ) -> Counters:  # pragma: no cover - protocol
        ...

    def read(self, note_id: str) -> Counters:  # pragma: no cover - protocol
        ...

    def commit(self) -> None:  # pragma: no cover - protocol
        ...

    def rollback(self) -> None:  # pragma: no cover - protocol
        ...


class DictCounterSink:
    """Cache held in a dict (the default for :class:`InMemoryInteractionStore`)."""

    def __init__(self, initial: Mapping[str, Counters] | None = None) -> None:
        self._lock = threading.RLock()
        self._committed: dict[str, Counters] = dict(initial or {})
        self._pending: dict[str, Counters] = {}

    def begin(self) -> None:
        with self._lock:
            self._pending = {}

    def seed(self, note_id: str, counters: Counters | Mapping[str, int]) -> None:
        with self._lock:
            payload = counters if isinstance(counters, Counters) else Counters(**dict(counters))
            self._committed[note_id] = payload
            self._pending.pop(note_id, None)

    def apply_delta(
        self, note_id: str, *, likes: int = 0, favorites: int = 0, comments: int = 0
    ) -> Counters:
        with self._lock:
            current = self.read(note_id)
            updated = current.delta(likes=likes, favorites=favorites, comments=comments)
            self._pending[note_id] = updated
            return updated

    def read(self, note_id: str) -> Counters:
        with self._lock:
            if note_id in self._pending:
                return self._pending[note_id]
            return self._committed.get(note_id, Counters())

    def commit(self) -> None:
        with self._lock:
            self._committed.update(self._pending)
            self._pending = {}

    def rollback(self) -> None:
        with self._lock:
            self._pending = {}


class NoteRecordCounterSink:
    """Write the counter cache into the note layer's own rows.

    Used when the composition root wires the interaction store onto a note
    repository/service pair (``inspect_note_record``/``get`` + ``save``).  The
    old record of every touched note is journaled so the store's transaction can
    roll the cache back together with the relation tables.
    """

    def __init__(self, notes: Any) -> None:
        self._reader = _pick_reader(notes)
        self._writer = _pick_writer(notes)
        if self._reader is None or self._writer is None:
            raise ContractError(
                ErrorCode.CONFIG_ERROR,
                "a note counter sink needs a note store exposing a read seam "
                "(get/inspect_note_record) and a write seam (save)",
                {"got": type(notes).__name__},
            )
        self._lock = threading.RLock()
        self._journal: list[Any] = []

    def begin(self) -> None:
        with self._lock:
            self._journal = []

    def apply_delta(
        self, note_id: str, *, likes: int = 0, favorites: int = 0, comments: int = 0
    ) -> Counters:
        with self._lock:
            record = self._load(note_id)
            if record is None:
                raise ContractError(ErrorCode.NOTE_NOT_FOUND, None, {"note_id": note_id})
            previous = _counters_of(record)
            updated = previous.delta(likes=likes, favorites=favorites, comments=comments)
            self._journal.append((record, previous))
            self._writer(_with_counters(record, updated))
            return updated

    def read(self, note_id: str) -> Counters:
        record = self._load(note_id)
        return _counters_of(record) if record is not None else Counters()

    def commit(self) -> None:
        with self._lock:
            self._journal = []

    def rollback(self) -> None:
        with self._lock:
            while self._journal:
                record, previous = self._journal.pop()
                try:
                    self._writer(_with_counters(record, previous))
                except ContractError:  # pragma: no cover - best effort repair
                    pass

    def _load(self, note_id: str) -> Any:
        if not is_valid_id(note_id):
            return None
        try:
            return self._reader(note_id)
        except (ContractError, LookupError):
            return None


def _pick_reader(target: Any) -> Any:
    for name in ("inspect_note_record", "get_record", "get", "load", "find"):
        candidate = getattr(target, name, None)
        if callable(candidate):
            return candidate
    repository = getattr(target, "repository", None)
    getter = getattr(repository, "get", None)
    return getter if callable(getter) else None


def _pick_writer(target: Any) -> Any:
    for name in ("save_note_record", "save", "update", "put", "upsert", "store"):
        candidate = getattr(target, name, None)
        if callable(candidate):
            return candidate
    repository = getattr(target, "repository", None)
    saver = getattr(repository, "save", None)
    return saver if callable(saver) else None


def _counters_of(record: Any) -> Counters:
    if record is None:
        return Counters()
    if isinstance(record, Mapping):
        payload = dict(record)
        counters = payload.get("counters")
        if isinstance(counters, Counters):
            return counters
        if isinstance(counters, Mapping):
            return _counters_from_mapping(counters)
        return _counters_from_mapping(payload)
    counters = getattr(record, "counters", None)
    if isinstance(counters, Counters):
        return counters
    if isinstance(counters, Mapping):
        return _counters_from_mapping(counters)
    return _counters_from_mapping(record)


def _counters_from_mapping(source: Any) -> Counters:
    def pick(*names: str) -> int:
        for name in names:
            if isinstance(source, Mapping):
                value = source.get(name)
            else:
                value = getattr(source, name, None)
            if value is not None:
                try:
                    return max(0, int(value))
                except (TypeError, ValueError):
                    return 0
        return 0

    return Counters(
        likes=pick("likes", "like_count"),
        favorites=pick("favorites", "favorite_count"),
        comments=pick("comments", "comment_count"),
    )


def _with_counters(record: Any, counters: Counters) -> Any:
    """Return a copy of *record* carrying the new cache (never mutates a row)."""
    changed = getattr(record, "changed", None)
    if callable(changed):
        try:
            return changed(counters=counters)
        except TypeError:  # pragma: no cover - stricter changed() signature
            pass
    if isinstance(record, Mapping):
        payload = dict(record)
        payload["counters"] = counters
        payload.update(
            {"like_count": counters.likes, "favorite_count": counters.favorites,
             "comment_count": counters.comments}
        )
        return payload
    import dataclasses

    if dataclasses.is_dataclass(record) and not isinstance(record, type):
        names = {meta.name for meta in dataclasses.fields(record)}
        if "counters" in names:
            return dataclasses.replace(record, counters=counters)
        updates: dict[str, Any] = {}
        for field_name, column in (("like_count", LIKE), ("favorite_count", FAVORITE),
                                   ("comment_count", COMMENT)):
            if field_name in names:
                updates[field_name] = getattr(counters, column)
        if updates:
            return dataclasses.replace(record, **updates)
    raise ContractError(
        ErrorCode.INTERNAL_ERROR,
        "note store returned an unsupported record shape",
        {"record_type": type(record).__name__},
    )


# --------------------------------------------------------------------------- #
# Store port
# --------------------------------------------------------------------------- #
@runtime_checkable
class InteractionStore(Protocol):
    """Everything ``mod-interaction`` persists (relations + counter cache).

    Both engines expose the same unit of work: :meth:`transaction` groups the
    relation write and the counter cache delta so they commit together
    (``INV-concurrency-counters``).
    """

    @contextmanager
    def transaction(self) -> Iterator[None]:  # pragma: no cover - protocol stub
        ...

    # -- likes ----------------------------------------------------------------
    def add_like(self, user_id: str, note_id: str, *, created_at: str | None = None) -> Transition: ...

    def remove_like(self, user_id: str, note_id: str) -> Transition: ...

    # -- favorites ------------------------------------------------------------
    def add_favorite(self, user_id: str, note_id: str, *, created_at: str | None = None) -> Transition: ...

    def remove_favorite(self, user_id: str, note_id: str) -> Transition: ...

    # -- comments -------------------------------------------------------------
    def append_comment(self, record: CommentRecord) -> Transition: ...

    def get_comment(self, comment_id: str) -> CommentRecord | None: ...

    def save_comment(self, record: CommentRecord) -> CommentRecord: ...

    def comments_for_note(self, note_id: str) -> list[CommentRecord]: ...

    def delete_comment(self, comment_id: str, *, hard: bool = False) -> Transition: ...

    # -- reads ----------------------------------------------------------------
    def counters(self, note_id: str) -> Counters: ...

    def derive_counters(self, note_id: str) -> Counters: ...

    def sync_counters(self, note_id: str) -> Counters: ...

    def has_like(self, user_id: str, note_id: str) -> bool: ...

    def has_favorite(self, user_id: str, note_id: str) -> bool: ...

    def liked_note_ids(self, user_id: str) -> tuple[str, ...]: ...

    def favorite_page(
        self, user_id: str, *, limit: int, cursor: tuple[str, str] | None = None
    ) -> FavoriteWindow: ...


class InMemoryInteractionStore:
    """Dict backed store (thread safe).  Transaction = lock + rollback journal."""

    def __init__(
        self,
        *,
        counters: CounterSink | None = None,
        records: Mapping[str, Any] | None = None,
    ) -> None:
        self._lock = threading.RLock()
        self._likes: dict[tuple[str, str], LikeRecord] = {}
        self._favorites: dict[tuple[str, str], FavoriteRecord] = {}
        self._comments: dict[str, CommentRecord] = {}
        self._counters: CounterSink = counters if counters is not None else DictCounterSink()
        self._tx_depth = 0
        for record in (records or {}).values() if records else ():
            self._adopt(record)

    # -- wiring ---------------------------------------------------------------
    @property
    def counter_sink(self) -> CounterSink:
        return self._counters

    @property
    def likes(self) -> Mapping[tuple[str, str], LikeRecord]:
        return self._likes

    @property
    def favorites(self) -> Mapping[tuple[str, str], FavoriteRecord]:
        return self._favorites

    @property
    def comments(self) -> Mapping[str, CommentRecord]:
        return self._comments

    @property
    def rows(self) -> Mapping[str, CommentRecord]:
        """Row map alias (dict-backed fixture seam used by ``mod-moderation``)."""
        return self._comments

    def _adopt(self, record: Any) -> None:
        if isinstance(record, LikeRecord):
            self._likes[record.key] = record
        elif isinstance(record, FavoriteRecord):
            self._favorites[record.key] = record
        elif isinstance(record, CommentRecord):
            self._comments[record.id] = record

    # -- transaction ----------------------------------------------------------
    @contextmanager
    def _txn(self) -> Iterator[None]:
        """Serialise writers and give every sink an explicit rollback boundary."""
        with self._lock:
            outer = self._tx_depth == 0
            likes = dict(self._likes)
            favorites = dict(self._favorites)
            comments = dict(self._comments)
            if outer:
                self._counters.begin()
            self._tx_depth += 1
            try:
                yield
            except BaseException:
                if outer:
                    self._likes, self._favorites, self._comments = likes, favorites, comments
                    self._counters.rollback()
                raise
            else:
                self._tx_depth -= 1
                if outer:
                    self._counters.commit()

    # -- likes ----------------------------------------------------------------
    # -- transaction ----------------------------------------------------------
    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Public unit-of-work handle (parity with :class:`SQLiteInteractionStore`).

        Composite operations — e.g. importing an existing relation table while
        rebuilding its counter cache — must commit or roll back as one block.
        """
        with self._txn():
            yield

    def add_like(self, user_id: str, note_id: str, *, created_at: str | None = None) -> Transition:
        key = relation_key(user_id, note_id)
        if key is None:
            raise ContractError(ErrorCode.NOTE_NOT_FOUND, None, {"note_id": note_id})
        with self._txn():
            candidate = LikeRecord(id=new_id(), user_id=user_id, note_id=note_id,
                                    created_at=_stamp(created_at))
            if not insert_unique(self._likes, key, candidate):
                # the (user_id, note_id) key is taken → UNIQUE violation, so the
                # duplicate like is a no-op and no counter delta is applied.
                return Transition(False, self._counters.read(note_id))
            counters = self._counters.apply_delta(note_id, likes=+1)
            return Transition(True, counters)

    def remove_like(self, user_id: str, note_id: str) -> Transition:
        key = relation_key(user_id, note_id)
        if key is None:
            raise ContractError(ErrorCode.NOTE_NOT_FOUND, None, {"note_id": note_id})
        with self._txn():
            if delete_unique(self._likes, key) == 0:
                return Transition(False, self._counters.read(note_id))
            counters = self._counters.apply_delta(note_id, likes=-1)
            return Transition(True, counters)

    # -- favorites ------------------------------------------------------------
    def add_favorite(self, user_id: str, note_id: str, *, created_at: str | None = None) -> Transition:
        key = relation_key(user_id, note_id)
        if key is None:
            raise ContractError(ErrorCode.NOTE_NOT_FOUND, None, {"note_id": note_id})
        with self._txn():
            candidate = FavoriteRecord(id=new_id(), user_id=user_id, note_id=note_id,
                                        created_at=_stamp(created_at))
            if not insert_unique(self._favorites, key, candidate):
                return Transition(False, self._counters.read(note_id))
            counters = self._counters.apply_delta(note_id, favorites=+1)
            return Transition(True, counters)

    def remove_favorite(self, user_id: str, note_id: str) -> Transition:
        key = relation_key(user_id, note_id)
        if key is None:
            raise ContractError(ErrorCode.NOTE_NOT_FOUND, None, {"note_id": note_id})
        with self._txn():
            if delete_unique(self._favorites, key) == 0:
                return Transition(False, self._counters.read(note_id))
            counters = self._counters.apply_delta(note_id, favorites=-1)
            return Transition(True, counters)

    # -- comments -------------------------------------------------------------
    def append_comment(self, record: CommentRecord) -> Transition:
        if not is_valid_id(record.note_id):
            raise ContractError(ErrorCode.NOTE_NOT_FOUND, None, {"note_id": record.note_id})
        with self._txn():
            if not insert_unique(self._comments, record.id, record):
                # unreachable in practice (the service always mints a fresh id);
                # kept as a storage-level integrity assertion
                raise ContractError(
                    ErrorCode.INTERNAL_ERROR, "comment id collision",
                    {"comment_id": [record.id]},
                )
            counters = self._counters.apply_delta(record.note_id, comments=+1)
            return Transition(True, counters)

    def get_comment(self, comment_id: str) -> CommentRecord | None:
        if not is_valid_id(comment_id):
            return None
        with self._lock:
            return self._comments.get(comment_id)

    def save_comment(self, record: CommentRecord) -> CommentRecord:
        """Whole-row write seam (also used by ``mod-moderation`` for flags)."""
        with self._txn():
            self._comments[record.id] = record
            return record

    #: ``mod-moderation`` adapts a ``load()`` + ``save()`` repository pair.
    save = save_comment

    def comments_for_note(self, note_id: str) -> list[CommentRecord]:
        if not is_valid_id(note_id):
            return []
        with self._lock:
            rows = [row for row in self._comments.values() if row.note_id == note_id]
        return sorted(rows, key=lambda row: row.sort_key)

    def delete_comment(self, comment_id: str, *, hard: bool = False) -> Transition:
        """Take a comment out of circulation — hidden (default) or purged.

        The counter follows :meth:`derive_counters`: only *counted* rows (not
        ``deleted`` ones) contribute, so hiding and purging both decrement by
        exactly one, while a repeated cancellation changes nothing.  Replying to
        a removed comment is impossible, so the whole subtree is removed too —
        otherwise the thread would keep showing children whose parent is gone.
        Soft removal keeps the rows for the moderation back office.
        """
        record = self.get_comment(comment_id)
        if record is None:
            return Transition(False, Counters())
        with self._txn():
            victims = self._subtree(record)
            counted = [row for row in victims if row.status is not ContentStatus.DELETED]
            if hard:
                for victim in victims:
                    self._comments.pop(victim.id, None)
            else:
                for victim in counted:
                    self._comments[victim.id] = victim.changed(
                        status=ContentStatus.DELETED, updated_at=now_iso()
                    )
            if not counted:
                return Transition(False, self._counters.read(record.note_id))
            counters = self._counters.apply_delta(
                record.note_id, comments=-len(counted)
            )
            return Transition(True, counters)

    def _subtree(self, record: CommentRecord) -> list[CommentRecord]:
        """*record* plus every descendant (children are keyed by parent id)."""
        with self._lock:
            rows = list(self._comments.values())
        by_parent: dict[str, list[CommentRecord]] = {}
        for row in rows:
            if row.parent_comment_id:
                by_parent.setdefault(row.parent_comment_id, []).append(row)
        out: list[CommentRecord] = [record]
        queue: list[CommentRecord] = [record]
        while queue:
            children = by_parent.get(queue.pop(0).id, ())
            out.extend(children)
            queue.extend(children)
        return out

    def find_comment(self, comment_id: str) -> CommentRecord | None:
        """Alias of :meth:`get_comment` (read seam naming parity with moderation)."""
        return self.get_comment(comment_id)

    def load(self, comment_id: str) -> CommentRecord | None:
        """Alias of :meth:`get_comment` (``ModerationTargetStore`` read shape)."""
        return self.get_comment(comment_id)

    # -- reads ----------------------------------------------------------------
    def counters(self, note_id: str) -> Counters:
        with self._lock:
            return self._counters.read(note_id)

    def derive_counters(self, note_id: str) -> Counters:
        with self._lock:
            likes = sum(1 for (_user, note) in self._likes if note == note_id)
            favorites = sum(1 for (_user, note) in self._favorites if note == note_id)
            comments = sum(
                1
                for row in self._comments.values()
                if row.note_id == note_id and row.status is not ContentStatus.DELETED
            )
        return Counters(likes=likes, favorites=favorites, comments=comments)

    def sync_counters(self, note_id: str) -> Counters:
        """Repair the cache from the relations (derivation, ``INV-concurrency``)."""
        with self._txn():
            derived = self.derive_counters(note_id)
            current = self._counters.read(note_id)
            return self._counters.apply_delta(
                note_id,
                likes=derived.likes - current.likes,
                favorites=derived.favorites - current.favorites,
                comments=derived.comments - current.comments,
            )

    def has_like(self, user_id: str, note_id: str) -> bool:
        key = relation_key(user_id, note_id)
        with self._lock:
            return key is not None and key in self._likes

    def has_favorite(self, user_id: str, note_id: str) -> bool:
        key = relation_key(user_id, note_id)
        with self._lock:
            return key is not None and key in self._favorites

    def liked_note_ids(self, user_id: str) -> tuple[str, ...]:
        if not is_valid_id(user_id):
            return ()
        with self._lock:
            return tuple(note for (user, note) in self._likes if user == user_id)

    def favorite_page(
        self, user_id: str, *, limit: int, cursor: tuple[str, str] | None = None
    ) -> FavoriteWindow:
        if not is_valid_id(user_id):
            return FavoriteWindow()
        size = max(1, int(limit))
        with self._lock:
            rows = sorted(
                (record for key, record in self._favorites.items() if key[0] == user_id),
                key=lambda record: (record.created_at, record.id),
                reverse=True,
            )
            total = len(rows)
            if cursor is not None:
                rows = [row for row in rows if (row.created_at, row.id) < cursor]
        window = rows[: size + 1]
        return FavoriteWindow(rows=window[:size], total=total, has_more=len(window) > size)

    # -- test / dev helpers ---------------------------------------------------
    def seed_counters(self, note_id: str, counters: Counters | Mapping[str, int]) -> None:
        seed = getattr(self._counters, "seed", None)
        if callable(seed):
            seed(note_id, counters)


# --------------------------------------------------------------------------- #
# SQLite storage
# --------------------------------------------------------------------------- #
_INTERACTION_SCHEMA: Final[tuple[str, ...]] = (
    """
    CREATE TABLE IF NOT EXISTS note_likes (
        id          TEXT PRIMARY KEY,
        user_id     TEXT NOT NULL,
        note_id     TEXT NOT NULL,
        created_at  TEXT NOT NULL,
        UNIQUE (user_id, note_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS note_favorites (
        id          TEXT PRIMARY KEY,
        user_id     TEXT NOT NULL,
        note_id     TEXT NOT NULL,
        created_at  TEXT NOT NULL,
        UNIQUE (user_id, note_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS comments (
        id                  TEXT PRIMARY KEY,
        note_id             TEXT NOT NULL,
        author_id           TEXT NOT NULL,
        body                TEXT NOT NULL,
        parent_comment_id   TEXT,
        status              TEXT NOT NULL DEFAULT 'published',
        moderation          TEXT NOT NULL DEFAULT 'visible',
        created_at          TEXT NOT NULL,
        updated_at          TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_comments_note ON comments(created_at, id)",
    "CREATE INDEX IF NOT EXISTS idx_comments_parent ON comments(parent_comment_id)",
    "CREATE INDEX IF NOT EXISTS idx_note_likes_note ON note_likes(note_id)",
    "CREATE INDEX IF NOT EXISTS idx_note_likes_user ON note_likes(user_id)",
    "CREATE INDEX IF NOT EXISTS idx_note_favorites_note ON note_favorites(note_id)",
    "CREATE INDEX IF NOT EXISTS idx_note_favorites_user "
    "ON note_favorites(user_id, created_at DESC, id DESC)",
)

_COUNTER_TABLE: Final[str] = "note_counters"
_NOTES_TABLE: Final[str] = "notes"


class SqlNoteColumnSink:
    """Counter cache = the ``notes`` table columns written by ``mod-note``.

    ``table`` / ``key_column`` are the only difference to the fallback cache
    table below, so the SQL is generated from them and both engines stay in
    sync by construction.
    """

    table = _NOTES_TABLE
    key_column = "id"

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def begin(self) -> None:  # the enclosing SQL transaction owns the boundary
        return None

    def apply_delta(
        self, note_id: str, *, likes: int = 0, favorites: int = 0, comments: int = 0
    ) -> Counters:
        assignments = ", ".join(
            f"{column} = MAX(0, {column} + ?)"
            for column in (_COLUMNS[LIKE], _COLUMNS[FAVORITE], _COLUMNS[COMMENT])
        )
        cursor = self._conn.execute(
            f"UPDATE {self.table} SET {assignments} WHERE {self.key_column} = ?",
            (likes, favorites, comments, note_id),
        )
        if cursor.rowcount == 0:
            raise ContractError(ErrorCode.NOTE_NOT_FOUND, None, {"note_id": note_id})
        return self.read(note_id)

    def read(self, note_id: str) -> Counters:
        row = self._conn.execute(
            f"SELECT like_count, favorite_count, comment_count FROM {self.table} "
            f"WHERE {self.key_column} = ?",
            (note_id,),
        ).fetchone()
        if row is None:
            return Counters()
        values = dict(row) if isinstance(row, sqlite3.Row) else dict(zip(_COLUMNS.values(), row))
        return Counters(
            likes=max(0, int(values.get("like_count") or 0)),
            favorites=max(0, int(values.get("favorite_count") or 0)),
            comments=max(0, int(values.get("comment_count") or 0)),
        )

    def commit(self) -> None:
        return None

    def rollback(self) -> None:
        return None


class SqlCounterTableSink(SqlNoteColumnSink):
    """Fallback cache table used when no ``notes`` table lives in this database.

    Rows are created lazily: the cache is a projection of the relation tables,
    so a note that never received an interaction simply has no cache row yet.
    """

    table = _COUNTER_TABLE
    key_column = "note_id"

    def apply_delta(
        self, note_id: str, *, likes: int = 0, favorites: int = 0, comments: int = 0
    ) -> Counters:
        self._conn.execute(
            f"INSERT INTO {self.table} ({self.key_column}, like_count, favorite_count, "
            "comment_count) VALUES (?, 0, 0, 0) ON CONFLICT DO NOTHING",
            (note_id,),
        )
        return super().apply_delta(note_id, likes=likes, favorites=favorites, comments=comments)


class SQLiteInteractionStore:
    """File backed store; relations and counters share one SQL transaction."""

    def __init__(
        self,
        database: str | Path | sqlite3.Connection = ":memory:",
        *,
        counter_sink: CounterSink | None = None,
    ) -> None:
        self._external_conn = database if isinstance(database, sqlite3.Connection) else None
        if self._external_conn is not None:
            self._conn = database
            self._path: Path | None = None
        else:
            self._path = _resolve_path(database)
            self._conn = sqlite3.connect(
                str(self._path) if self._path is not None else ":memory:",
                check_same_thread=False,
                isolation_level=None,  # explicit transactions
            )
        if self._conn.row_factory is None:
            self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._depth = 0
        self._bootstrap()
        self._counters: CounterSink = counter_sink or self._default_sink()

    # -- lifecycle ------------------------------------------------------------
    def _bootstrap(self) -> None:
        with self._lock, self._conn:
            for statement in _INTERACTION_SCHEMA:
                self._conn.execute(statement)
            if not self._has_table(_NOTES_TABLE):
                self._conn.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {_COUNTER_TABLE} (
                        note_id        TEXT PRIMARY KEY,
                        like_count     INTEGER NOT NULL DEFAULT 0,
                        favorite_count INTEGER NOT NULL DEFAULT 0,
                        comment_count  INTEGER NOT NULL DEFAULT 0
                    )
                    """
                )

    def _default_sink(self) -> CounterSink:
        return (
            SqlNoteColumnSink(self._conn)
            if self._has_table(_NOTES_TABLE)
            else SqlCounterTableSink(self._conn)
        )

    def _has_table(self, name: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)
        ).fetchone()
        return row is not None

    @property
    def counter_sink(self) -> CounterSink:
        return self._counters

    @property
    def connection(self) -> sqlite3.Connection:
        return self._conn

    @property
    def database_path(self) -> Path | None:
        return self._path

    def close(self) -> None:
        if self._external_conn is None:
            with self._lock:
                self._conn.close()

    def __enter__(self) -> "SQLiteInteractionStore":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # -- transaction ----------------------------------------------------------
    @contextmanager
    def _txn(self) -> Iterator[None]:
        """``BEGIN IMMEDIATE`` … ``COMMIT`` with rollback on any failure.

        Re-entrant: only the outermost frame opens the SQL transaction, so a
        composite operation (relation write + counter delta + cache read-back)
        is one atomic unit even though it issues several statements.
        """
        with self._lock:
            outermost = self._depth == 0
            if outermost:
                self._conn.execute("BEGIN IMMEDIATE")
                self._counters.begin()
            self._depth += 1
            try:
                yield
            except BaseException:
                self._depth -= 1
                if outermost:
                    self._counters.rollback()
                    self._conn.execute("ROLLBACK")
                raise
            else:
                self._depth -= 1
                if outermost:
                    self._counters.commit()
                    self._conn.execute("COMMIT")

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Public unit-of-work handle (same contract as the in-memory engine)."""
        with self._txn():
            yield

    # -- likes ----------------------------------------------------------------
    def add_like(self, user_id: str, note_id: str, *, created_at: str | None = None) -> Transition:
        key = relation_key(user_id, note_id)
        if key is None:
            raise ContractError(ErrorCode.NOTE_NOT_FOUND, None, {"note_id": note_id})
        with self._txn():
            try:
                self._conn.execute(
                    "INSERT INTO note_likes (id, user_id, note_id, created_at) VALUES (?, ?, ?, ?)",
                    (new_id(), user_id, note_id, _stamp(created_at)),
                )
            except sqlite3.IntegrityError:
                # UNIQUE (user_id, note_id): the second like is a no-op, so the
                # counter is never incremented twice for the same pair.
                return Transition(False, self._counters.read(note_id))
            return Transition(True, self._counters.apply_delta(note_id, likes=+1))

    def remove_like(self, user_id: str, note_id: str) -> Transition:
        key = relation_key(user_id, note_id)
        if key is None:
            raise ContractError(ErrorCode.NOTE_NOT_FOUND, None, {"note_id": note_id})
        with self._txn():
            cursor = self._conn.execute(
                "DELETE FROM note_likes WHERE user_id = ? AND note_id = ?", (user_id, note_id)
            )
            if cursor.rowcount == 0:
                return Transition(False, self._counters.read(note_id))
            return Transition(True, self._counters.apply_delta(note_id, likes=-1))

    # -- favorites ------------------------------------------------------------
    def add_favorite(self, user_id: str, note_id: str, *, created_at: str | None = None) -> Transition:
        key = relation_key(user_id, note_id)
        if key is None:
            raise ContractError(ErrorCode.NOTE_NOT_FOUND, None, {"note_id": note_id})
        with self._txn():
            try:
                self._conn.execute(
                    "INSERT INTO note_favorites (id, user_id, note_id, created_at) "
                    "VALUES (?, ?, ?, ?)",
                    (new_id(), user_id, note_id, _stamp(created_at)),
                )
            except sqlite3.IntegrityError:
                return Transition(False, self._counters.read(note_id))
            return Transition(True, self._counters.apply_delta(note_id, favorites=+1))

    def remove_favorite(self, user_id: str, note_id: str) -> Transition:
        key = relation_key(user_id, note_id)
        if key is None:
            raise ContractError(ErrorCode.NOTE_NOT_FOUND, None, {"note_id": note_id})
        with self._txn():
            cursor = self._conn.execute(
                "DELETE FROM note_favorites WHERE user_id = ? AND note_id = ?",
                (user_id, note_id),
            )
            if cursor.rowcount == 0:
                return Transition(False, self._counters.read(note_id))
            return Transition(True, self._counters.apply_delta(note_id, favorites=-1))

    # -- comments -------------------------------------------------------------
    def append_comment(self, record: CommentRecord) -> Transition:
        if not is_valid_id(record.note_id):
            raise ContractError(ErrorCode.NOTE_NOT_FOUND, None, {"note_id": record.note_id})
        row = record.to_row()
        with self._txn():
            try:
                self._conn.execute(
                    "INSERT INTO comments (id, note_id, author_id, body, parent_comment_id, "
                    "status, moderation, created_at, updated_at) "
                    "VALUES (:id, :note_id, :author_id, :body, :parent_comment_id, :status, "
                    ":moderation, :created_at, :updated_at)",
                    row,
                )
            except sqlite3.IntegrityError as exc:
                raise ContractError(
                    ErrorCode.INTERNAL_ERROR, "comment id collision", {"comment_id": [record.id]}
                ) from exc
            return Transition(True, self._counters.apply_delta(record.note_id, comments=+1))

    def get_comment(self, comment_id: str) -> CommentRecord | None:
        if not is_valid_id(comment_id):
            return None
        with self._lock:
            row = self._conn.execute("SELECT * FROM comments WHERE id = ?", (comment_id,)).fetchone()
        return CommentRecord.from_row(row) if row is not None else None

    load = get_comment

    def find_comment(self, comment_id: str) -> CommentRecord | None:
        return self.get_comment(comment_id)

    def save_comment(self, record: CommentRecord) -> CommentRecord:
        """Whole-row write seam (``mod-moderation`` flips ``moderation`` through it)."""
        row = record.to_row()
        with self._txn():
            self._conn.execute(
                "INSERT INTO comments (id, note_id, author_id, body, parent_comment_id, "
                "status, moderation, created_at, updated_at) "
                "VALUES (:id, :note_id, :author_id, :body, :parent_comment_id, :status, "
                ":moderation, :created_at, :updated_at) "
                "ON CONFLICT(id) DO UPDATE SET "
                "body = excluded.body, parent_comment_id = excluded.parent_comment_id, "
                "status = excluded.status, moderation = excluded.moderation, "
                "updated_at = excluded.updated_at",
                row,
            )
        return record

    save = save_comment

    def comments_for_note(self, note_id: str) -> list[CommentRecord]:
        if not is_valid_id(note_id):
            return []
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM comments WHERE note_id = ? ORDER BY created_at ASC, id ASC",
                (note_id,),
            ).fetchall()
        return [CommentRecord.from_row(row) for row in rows]

    def delete_comment(self, comment_id: str, *, hard: bool = False) -> Transition:
        """Hide (default) or purge a comment subtree, counter included.

        Same semantics as :meth:`InMemoryInteractionStore.delete_comment`: the
        delta only covers rows that are still *counted*, so hiding twice — or
        purging an already hidden row — moves nothing.  The DELETE/UPDATE and the
        counter delta share one SQL transaction (``INV-concurrency-counters``).
        """
        record = self.get_comment(comment_id)
        if record is None:
            return Transition(False, Counters())
        with self._txn():
            victims = self._subtree_ids(record)
            marks = ", ".join("?" for _ in victims)
            rows = self._conn.execute(
                f"SELECT id, status FROM comments WHERE id IN ({marks})", tuple(victims)
            ).fetchall()
            counted = [
                row["id"]
                for row in rows
                if str(row["status"]) != str(ContentStatus.DELETED.value)
            ]
            if not counted:
                return Transition(False, self._counters.read(record.note_id))
            if hard:
                self._conn.execute(f"DELETE FROM comments WHERE id IN ({marks})", tuple(victims))
            else:
                stamps = ", ".join("?" for _ in counted)
                self._conn.execute(
                    f"UPDATE comments SET status = ?, updated_at = ? WHERE id IN ({stamps})",
                    (str(ContentStatus.DELETED.value), now_iso(), *counted),
                )
            return Transition(
                True, self._counters.apply_delta(record.note_id, comments=-len(counted))
            )

    def _subtree_ids(self, record: CommentRecord) -> list[str]:
        """Ids of *record* and all of its descendants (single read, BFS walk)."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, parent_comment_id FROM comments WHERE note_id = ?",
                (record.note_id,),
            ).fetchall()
        by_parent: dict[str, list[str]] = {}
        for row in rows:
            if row["parent_comment_id"]:
                by_parent.setdefault(row["parent_comment_id"], []).append(row["id"])
        out: list[str] = [record.id]
        queue: list[str] = [record.id]
        while queue:
            children = by_parent.get(queue.pop(0), ())
            out.extend(children)
            queue.extend(children)
        return out

    # -- reads ----------------------------------------------------------------
    def counters(self, note_id: str) -> Counters:
        with self._lock:
            return self._counters.read(note_id)

    def derive_counters(self, note_id: str) -> Counters:
        if not is_valid_id(note_id):
            return Counters()
        with self._lock:
            likes = self._scalar(
                "SELECT COUNT(*) FROM note_likes WHERE note_id = ?", (note_id,)
            )
            favorites = self._scalar(
                "SELECT COUNT(*) FROM note_favorites WHERE note_id = ?", (note_id,)
            )
            comments = self._scalar(
                "SELECT COUNT(*) FROM comments WHERE note_id = ? AND status <> ?",
                (note_id, str(ContentStatus.DELETED)),
            )
        return Counters(likes=likes, favorites=favorites, comments=comments)

    def _scalar(self, sql: str, params: tuple[Any, ...]) -> int:
        row = self._conn.execute(sql, params).fetchone()
        if row is None:
            return 0
        return int(row[0] if not isinstance(row, sqlite3.Row) else row[-1])

    def sync_counters(self, note_id: str) -> Counters:
        with self._txn():
            derived = self.derive_counters(note_id)
            current = self._counters.read(note_id)
            return self._counters.apply_delta(
                note_id,
                likes=derived.likes - current.likes,
                favorites=derived.favorites - current.favorites,
                comments=derived.comments - current.comments,
            )

    def has_like(self, user_id: str, note_id: str) -> bool:
        key = relation_key(user_id, note_id)
        if key is None:
            return False
        with self._lock:
            return self._conn.execute(
                "SELECT 1 FROM note_likes WHERE user_id = ? AND note_id = ?", key
            ).fetchone() is not None

    def has_favorite(self, user_id: str, note_id: str) -> bool:
        key = relation_key(user_id, note_id)
        if key is None:
            return False
        with self._lock:
            return self._conn.execute(
                "SELECT 1 FROM note_favorites WHERE user_id = ? AND note_id = ?", key
            ).fetchone() is not None

    def liked_note_ids(self, user_id: str) -> tuple[str, ...]:
        if not is_valid_id(user_id):
            return ()
        with self._lock:
            rows = self._conn.execute(
                "SELECT note_id FROM note_likes WHERE user_id = ? ORDER BY created_at DESC",
                (user_id,),
            ).fetchall()
        return tuple(str(row["note_id"] if isinstance(row, sqlite3.Row) else row[0])
                     for row in rows)

    def favorite_page(
        self, user_id: str, *, limit: int, cursor: tuple[str, str] | None = None
    ) -> FavoriteWindow:
        if not is_valid_id(user_id):
            return FavoriteWindow()
        size = max(1, int(limit))
        params: list[Any] = [user_id]
        window = ""
        if cursor is not None:
            window = " AND (created_at < ? OR (created_at = ? AND id < ?))"
            params.extend([cursor[0], cursor[0], cursor[1]])
        with self._lock:
            total_row = self._conn.execute(
                "SELECT COUNT(*) FROM note_favorites WHERE user_id = ?", (user_id,)
            ).fetchone()
            rows = self._conn.execute(
                "SELECT * FROM note_favorites WHERE user_id = ?" + window
                + " ORDER BY created_at DESC, id DESC LIMIT ?",
                (*params, size + 1),
            ).fetchall()
        total = int(total_row[-1]) if total_row is not None else 0
        fetched = [FavoriteRecord.from_row(row) for row in rows]
        return FavoriteWindow(rows=fetched[:size], total=total, has_more=len(fetched) > size)


def insert_unique(table: dict[Any, Any], key: Any, value: Any) -> bool:
    """Insert *key* or report failure — the in-memory twin of a UNIQUE index.

    Returning ``False`` means "the pair already exists", exactly what SQLite
    signals with :class:`sqlite3.IntegrityError`.  Callers must treat that as an
    idempotent no-op; there is deliberately no "read the row, then decide to
    write it" pre-check in the write path.
    """
    if key in table:
        return False
    table[key] = value
    return True


def delete_unique(table: dict[Any, Any], key: Any) -> int:
    """Delete *key*; the return value mirrors ``cursor.rowcount`` of a DELETE."""
    return 1 if table.pop(key, None) is not None else 0


def _resolve_path(target: str | Path | sqlite3.Connection) -> Path | None:
    """Translate a ``sqlite://`` URL (or plain path) into a filesystem path.

    ``:memory:`` (and the empty body) intentionally returns ``None``: no file is
    created, so importing or instantiating this module never writes outside the
    caller's own directory.
    """
    if isinstance(target, Path):
        target.parent.mkdir(parents=True, exist_ok=True)
        return target
    text = str(target).strip()
    if text.startswith("sqlite:///"):
        body = text[len("sqlite:///"):].split("?", 1)[0]
    elif text.startswith("sqlite://"):
        body = text[len("sqlite://"):].lstrip("/").split("?", 1)[0]
    else:
        body = text
    if body in ("", ":memory:"):
        return None
    path = Path(body).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def open_store(
    database: str | Path | sqlite3.Connection | None = None,
    *,
    notes: Any = None,
    prefer: str = "sqlite",
) -> InteractionStore:
    """Wiring helper used by ``mod-web``.

    ``prefer="memory"`` builds the in-process store; its counter cache is pushed
    into the note rows when a note store is supplied, otherwise it is kept
    locally.  The default opens a file backed store (``INV-config-db``) and must
    be given an explicit ``database`` location — there is no implicit ``data/``
    default here, so a mis-wired call fails loudly instead of writing files in
    an unexpected place.
    """
    if prefer == "memory":
        sink: CounterSink | None = None
        if notes is not None and _pick_reader(notes) is not None and _pick_writer(notes) is not None:
            sink = NoteRecordCounterSink(notes)
        return InMemoryInteractionStore(counters=sink)
    if database is None:
        raise ContractError(
            ErrorCode.CONFIG_ERROR,
            "SQLiteInteractionStore needs an explicit database location "
            "(Settings.database_url), there is no implicit data/ default",
        )
    return SQLiteInteractionStore(database)
