"""Note persistence (``mod-note``).

Two implementations of the same tiny port are provided:

:class:`InMemoryNoteRepository`
    thread safe dict store — the fake used by the test suite and by dev/demo
    wiring.
:class:`SQLiteNoteRepository`
    file backed storage honouring ``INV-config-db`` (content survives a
    restart); tags live in their own table so ``topic`` / ``tag`` filtering is
    exact rather than substring guessing.

Both are *dumb* on purpose: they filter by topic / tag / author / keyword and by
the keyset cursor, and they know nothing about who may see what.  Visibility is
applied by the service through ``contracts.visibility.is_visible_to``
(``INV-visibility-predicate``), which is why ``stream`` also yields drafts and
tombstones.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Final, Protocol, runtime_checkable

from blacknote.contracts.config import Settings
from blacknote.contracts.errors import ContractError, ErrorCode
from blacknote.contracts.ids import is_valid_id

from .models import NoteRecord
from .queries import NoteFilters

__all__ = ["NoteRepository", "InMemoryNoteRepository", "SQLiteNoteRepository", "open_repository"]

_SORT_REVERSE: Final[bool] = True  # created_at DESC (newest first)


@runtime_checkable
class NoteRepository(Protocol):
    """Storage port for notes."""

    def insert(self, record: NoteRecord) -> NoteRecord:  # pragma: no cover - protocol
        ...

    def save(self, record: NoteRecord) -> NoteRecord:  # pragma: no cover - protocol
        ...

    def get(self, note_id: str) -> NoteRecord | None:  # pragma: no cover - protocol
        ...

    def stream(self, filters: NoteFilters | None = None) -> Iterator[NoteRecord]:
        """Yield matching records newest first (``created_at DESC, id DESC``)."""

    def count(self) -> int:  # pragma: no cover - protocol
        ...


def _sort_key(record: NoteRecord) -> tuple[str, str]:
    return record.sort_key


def _matches(record: NoteRecord, filters: NoteFilters) -> bool:
    """Non-visibility selection shared by the in-memory store and the tests."""
    if filters.is_empty_selector:
        return False
    if filters.topic and record.topic != filters.topic:
        return False
    if filters.tag and not record.has_tag(filters.tag):
        return False
    if filters.author_id and record.author_id != filters.author_id:
        return False
    if filters.author_ids is not None and record.author_id not in set(filters.author_ids):
        return False
    if filters.tokens:
        haystack = f"{record.title}\n{record.body}".casefold()
        if not all(token in haystack for token in filters.tokens):
            return False
    if filters.before_created_at is not None and filters.before_id is not None:
        if _strictly_newer_window(record, filters):
            return False
    return True


def _strictly_newer_window(record: NoteRecord, filters: NoteFilters) -> bool:
    """True when *record* is NOT strictly older than the cursor position."""
    window = (str(filters.before_created_at), str(filters.before_id))
    # rows must sort strictly *after* the cursor in DESC order:
    return record.sort_key >= window


class InMemoryNoteRepository:
    """Dictionary backed repository (thread safe, process local)."""

    def __init__(self, records: Iterator[NoteRecord] | None = None) -> None:
        self._lock = threading.RLock()
        self._rows: dict[str, NoteRecord] = {}
        for record in records or ():
            self.insert(record)

    # -- writes ---------------------------------------------------------------
    def insert(self, record: NoteRecord) -> NoteRecord:
        with self._lock:
            if record.id in self._rows:
                raise ContractError(
                    ErrorCode.CONFLICT, "note already exists", {"note_id": record.id}
                )
            self._rows[record.id] = record
            return record

    def save(self, record: NoteRecord) -> NoteRecord:
        with self._lock:
            self._rows[record.id] = record
            return record

    # -- reads ----------------------------------------------------------------
    def get(self, note_id: str) -> NoteRecord | None:
        if not is_valid_id(note_id):
            return None
        with self._lock:
            return self._rows.get(note_id)

    def stream(self, filters: NoteFilters | None = None) -> Iterator[NoteRecord]:
        selected = filters or NoteFilters()
        with self._lock:
            snapshot = list(self._rows.values())
        ordered = sorted((r for r in snapshot if _matches(r, selected)),
                         key=_sort_key, reverse=_SORT_REVERSE)
        yield from ordered

    def count(self) -> int:
        with self._lock:
            return len(self._rows)

    def clear(self) -> None:
        with self._lock:
            self._rows.clear()


_SCHEMA: Final[tuple[str, ...]] = (
    """
    CREATE TABLE IF NOT EXISTS notes (
        id              TEXT PRIMARY KEY,
        author_id       TEXT NOT NULL,
        title           TEXT NOT NULL,
        body            TEXT NOT NULL,
        topic           TEXT,
        image_urls      TEXT NOT NULL DEFAULT '[]',
        visibility      TEXT NOT NULL DEFAULT 'public',
        status          TEXT NOT NULL DEFAULT 'published',
        moderation      TEXT NOT NULL DEFAULT 'visible',
        like_count      INTEGER NOT NULL DEFAULT 0,
        favorite_count  INTEGER NOT NULL DEFAULT 0,
        comment_count   INTEGER NOT NULL DEFAULT 0,
        created_at      TEXT NOT NULL,
        updated_at      TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS note_tags (
        note_id TEXT NOT NULL REFERENCES notes(id) ON DELETE CASCADE,
        tag     TEXT NOT NULL,
        position INTEGER NOT NULL DEFAULT 0,
        PRIMARY KEY (note_id, tag)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_notes_created ON notes(created_at DESC, id DESC)",
    "CREATE INDEX IF NOT EXISTS idx_notes_author ON notes(author_id, created_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_notes_topic ON notes(topic, created_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_note_tags_tag ON note_tags(tag)",
)


class SQLiteNoteRepository:
    """File backed repository (server side persistence, ``INV-config-db``)."""

    def __init__(
        self,
        database_url_or_path: str | Path = "sqlite:///data/blacknote_notes.db",
        *,
        settings: Settings | None = None,
        batch_size: int = 200,
    ) -> None:
        self._path = _resolve_path(database_url_or_path)
        self._settings = settings or Settings()
        self._batch_size = max(1, int(batch_size))
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            str(self._path) if self._path else ":memory:",
            check_same_thread=False,
            isolation_level=None,  # explicit transactions
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._bootstrap()

    # -- lifecycle ------------------------------------------------------------
    def _bootstrap(self) -> None:
        with self._lock:
            self._conn.executescript(";\n".join(_SCHEMA) + ";\n")

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "SQLiteNoteRepository":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # -- writes ---------------------------------------------------------------
    def insert(self, record: NoteRecord) -> NoteRecord:
        """Add a brand new note; ``CONFLICT`` when the id is already taken."""
        if self.get(record.id) is not None:
            raise ContractError(
                ErrorCode.CONFLICT, "note already exists", {"note_id": record.id}
            )
        return self.save(record)

    def save(self, record: NoteRecord) -> NoteRecord:
        """Upsert a note row plus its tags inside a single transaction."""
        row = record.to_row()
        params = {key: value for key, value in row.items() if key not in ("tags", "image_urls")}
        params["image_urls"] = json.dumps(row["image_urls"], ensure_ascii=False)
        with self._lock, self._conn:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._conn.execute(
                    """
                    INSERT INTO notes (id, author_id, title, body, topic, image_urls, visibility,
                                       status, moderation, like_count, favorite_count,
                                       comment_count, created_at, updated_at)
                    VALUES (:id, :author_id, :title, :body, :topic, :image_urls, :visibility,
                            :status, :moderation, :like_count, :favorite_count, :comment_count,
                            :created_at, :updated_at)
                    ON CONFLICT(id) DO UPDATE SET
                        author_id = excluded.author_id,
                        title = excluded.title,
                        body = excluded.body,
                        topic = excluded.topic,
                        image_urls = excluded.image_urls,
                        visibility = excluded.visibility,
                        status = excluded.status,
                        moderation = excluded.moderation,
                        like_count = excluded.like_count,
                        favorite_count = excluded.favorite_count,
                        comment_count = excluded.comment_count,
                        created_at = excluded.created_at,
                        updated_at = excluded.updated_at
                    """,
                    params,
                )
                self._conn.execute("DELETE FROM note_tags WHERE note_id = ?", (record.id,))
                self._conn.executemany(
                    "INSERT INTO note_tags (note_id, tag, position) VALUES (?, ?, ?)",
                    [(record.id, tag, index) for index, tag in enumerate(record.tags)],
                )
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise
        return record

    # -- reads ----------------------------------------------------------------
    def get(self, note_id: str) -> NoteRecord | None:
        if not is_valid_id(note_id):
            return None
        with self._lock:
            cursor = self._conn.execute("SELECT * FROM notes WHERE id = ?", (note_id,))
            row = cursor.fetchone()
            cursor.close()
        if row is None:
            return None
        return self._hydrate(row)

    def stream(self, filters: NoteFilters | None = None) -> Iterator[NoteRecord]:
        selected = filters or NoteFilters()
        if selected.is_empty_selector:
            return
        sql, params = self._build_stream(selected)
        with self._lock:
            cursor = self._conn.execute(sql, params)
            while True:
                rows = cursor.fetchmany(self._batch_size)
                if not rows:
                    break
                batch = [self._hydrate(row) for row in rows]
                cursor_ids = tuple(record.id for record in batch)
                tag_map = self._tag_map(cursor_ids)
                for record in batch:
                    yield record.changed(tags=tag_map.get(record.id, record.tags))
            cursor.close()

    def count(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) AS n FROM notes").fetchone()
        return int(row["n"]) if row else 0

    # -- internals ------------------------------------------------------------
    def _build_stream(self, filters: NoteFilters) -> tuple[str, list[Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if filters.topic:
            clauses.append("n.topic = ?")
            params.append(filters.topic)
        if filters.tag:
            clauses.append(
                "EXISTS (SELECT 1 FROM note_tags t WHERE t.note_id = n.id "
                "AND lower(t.tag) = lower(?))"
            )
            params.append(filters.tag)
        if filters.author_id:
            clauses.append("n.author_id = ?")
            params.append(filters.author_id)
        if filters.author_ids is not None:
            placeholders = ", ".join("?" for _ in filters.author_ids) or "NULL"
            clauses.append(f"n.author_id IN ({placeholders})")
            params.extend(filters.author_ids)
        tokens = filters.tokens
        if tokens:
            terms: list[str] = []
            for token in tokens:
                escaped = _like_escape(token)
                terms.append("(n.title LIKE ? ESCAPE '\\' OR n.body LIKE ? ESCAPE '\\')")
                params.extend([escaped, escaped])
            clauses.append(" AND ".join(terms))
        if filters.before_created_at and filters.before_id:
            clauses.append(
                "(n.created_at < ? OR (n.created_at = ? AND n.id < ?))"
            )
            params.extend([filters.before_created_at, filters.before_created_at,
                           filters.before_id])
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT n.* FROM notes n {where} ORDER BY n.created_at DESC, n.id DESC"
        return sql, params

    def _hydrate(self, row: sqlite3.Row) -> NoteRecord:
        data = dict(row)
        data["image_urls"] = json.loads(data.get("image_urls") or "[]")
        tags = self._tag_map((data["id"],)).get(data["id"], ())
        return NoteRecord.from_row(data, tags)

    def _tag_map(self, note_ids: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
        if not note_ids:
            return {}
        placeholders = ", ".join("?" for _ in note_ids)
        with self._lock:
            cursor = self._conn.execute(
                "SELECT note_id, tag FROM note_tags WHERE note_id IN ({}) "
                "ORDER BY position".format(placeholders),
                note_ids,
            )
            rows = cursor.fetchall()
            cursor.close()
        mapping: dict[str, list[str]] = {}
        for row in rows:
            mapping.setdefault(row["note_id"], []).append(row["tag"])
        return {key: tuple(value) for key, value in mapping.items()}


def _like_escape(keyword: str) -> str:
    """Quote LIKE wildcards so a search term matches literally."""
    escaped = keyword.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def _resolve_path(target: str | Path) -> Path | None:
    """Translate a ``sqlite://`` URL (or plain path) into a filesystem path."""
    if isinstance(target, Path):
        target.mkdir(parents=True, exist_ok=True)
        return target
    text = str(target).strip()
    if text.startswith("sqlite:///"):
        body = text[len("sqlite:///"):].split("?", 1)[0]
    elif text.startswith("sqlite://"):
        body = text[len("sqlite://"):].lstrip("/").split("?", 1)[0]
    else:
        body = text
    if body in ("", ":memory:"):
        return None  # shared in-process database
    path = Path(body).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def open_repository(
    settings: Settings | None = None,
    *,
    prefer: str = "sqlite",
) -> NoteRepository:
    """Wiring helper used by ``mod-web``: build the configured repository.

    ``prefer="memory"`` returns the in-process store (tests / demos); the
    default opens the database addressed by ``Settings.database_url`` so content
    survives restarts.
    """
    if prefer == "memory":
        return InMemoryNoteRepository()
    return SQLiteNoteRepository((settings or Settings()).database_url, settings=settings)
