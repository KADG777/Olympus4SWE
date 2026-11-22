"""Concrete stores shipped with ``mod-moderation``.

Two responsibilities, kept apart from the rest of the platform on purpose:

Content flags
    :class:`InMemoryNoteStore` / :class:`InMemoryCommentStore` implement the
    moderation port directly, which makes them the natural fake for tests and
    the zero-configuration default for dev wiring.  In a real deployment the
    web layer injects the ``mod-note`` repository / service instead (see
    :func:`blacknote.moderation.ports.adapt_target_store`); this layer never
    owns a notes or comments table, it only flips flags.

Audit trail
    :class:`InMemoryAuditLogStore` (process local) and
    :class:`SQLiteAuditLogStore` (file backed, ``INV-config-db``: takedowns and
    privilege violations survive a restart).  Rows are append-only, the column
    set is exactly :data:`blacknote.moderation.models.AUDIT_COLUMNS`, i.e. the
    fields of :class:`~blacknote.moderation.models.ModerationAuditLog`.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any, Final, Self

from blacknote.contracts.errors import ContractError, ErrorCode
from blacknote.contracts.ids import coerce_id, is_valid_id
from blacknote.contracts.timeutil import now_iso
from blacknote.contracts.visibility import ModerationState

from .models import (
    AUDIT_COLUMNS,
    ModeratableComment,
    ModeratableNote,
    ModerationAction,
    ModerationAuditLog,
    parse_action,
)
from .ports import ModerationTargetStore, with_moderation

__all__ = [
    "InMemoryModerationStore",
    "InMemoryNoteStore",
    "InMemoryCommentStore",
    "InMemoryAuditLogStore",
    "SQLiteAuditLogStore",
    "WrappingAuditLogStore",
    "open_audit_store",
]


def _row_id(record: Any) -> str | None:
    if isinstance(record, Mapping):
        return coerce_id(record.get("id"))
    return coerce_id(getattr(record, "id", None))


class InMemoryModerationStore(ModerationTargetStore):
    """Thread-safe dict store of flag bearing rows (test / dev default)."""

    record_type: type[Any]

    def __init__(self, *, kind: str, records: Iterable[Any] | None = None,
                 clock: Callable[[], str] | None = None) -> None:
        self._kind = kind
        self._clock = clock or now_iso
        self._lock = threading.RLock()
        self._rows: dict[str, Any] = {}
        for record in records or ():
            self.insert(record)

    # -- population helpers (wiring + tests) ----------------------------------
    def insert(self, record: Any) -> Any:
        """Add a brand new row (mappings are hydrated into the record type)."""
        stored = self._coerce(record)
        key = _row_id(stored)
        if key is None:
            raise ContractError(
                ErrorCode.VALIDATION_ERROR, "row requires a canonical UUIDv4 id"
            )
        with self._lock:
            if key in self._rows:
                raise ContractError(ErrorCode.CONFLICT, "row already exists", {"id": key})
            self._rows[key] = stored
        return stored

    def save(self, record: Any) -> Any:
        """Upsert a row (the generic repository contract)."""
        stored = self._coerce(record)
        key = _row_id(stored)
        if key is None:
            raise ContractError(
                ErrorCode.VALIDATION_ERROR, "record requires a canonical UUIDv4 id"
            )
        with self._lock:
            self._rows[key] = stored
        return stored

    def add(self, **fields: Any) -> Any:
        """Convenience: build a record from keyword fields and store it."""
        return self.insert(self.record_type(**fields))

    # -- reads ----------------------------------------------------------------
    def get(self, target_id: str) -> Any | None:
        if not is_valid_id(target_id):
            return None
        with self._lock:
            return self._rows.get(target_id)

    def count(self) -> int:
        with self._lock:
            return len(self._rows)

    @property
    def rows(self) -> Mapping[str, Any]:
        """Read-only snapshot (fixtures inspect flags through it)."""
        with self._lock:
            return dict(self._rows)

    # -- the moderation port --------------------------------------------------
    def load(self, target_id: str) -> Any | None:
        return self.get(target_id)

    def apply_moderation(self, target_id: str, state: ModerationState) -> Any | None:
        with self._lock:
            record = self.get(target_id)
            if record is None:
                return None
            updated = with_moderation(record, self._kind, state, updated_at=self._clock())
            self._rows[str(_row_id(updated) or target_id)] = updated
            return updated

    # -- internals ------------------------------------------------------------
    def _coerce(self, record: Any) -> Any:
        if isinstance(record, Mapping):
            return type(self).record_type.from_row(record)
        return record


class InMemoryNoteStore(InMemoryModerationStore):
    """``note`` flag store (``unlist_note`` writes ``moderated``/``unlisted``)."""

    record_type = ModeratableNote

    def __init__(self, records: Iterable[Any] | None = None, *,
                 clock: Callable[[], str] | None = None) -> None:
        super().__init__(kind="note", records=records, clock=clock)


class InMemoryCommentStore(InMemoryModerationStore):
    """``comment`` flag store (``block_comment`` writes ``blocked``/``blocked``)."""

    record_type = ModeratableComment

    def __init__(self, records: Iterable[Any] | None = None, *,
                 clock: Callable[[], str] | None = None) -> None:
        super().__init__(kind="comment", records=records, clock=clock)


# --------------------------------------------------------------------------- #
# Audit log stores
# --------------------------------------------------------------------------- #
class InMemoryAuditLogStore:
    """Append-only list backed audit store (process local, thread safe)."""

    def __init__(self, entries: Iterable[ModerationAuditLog] | None = None) -> None:
        self._lock = threading.RLock()
        self._entries: list[ModerationAuditLog] = []
        self._ids: set[str] = set()
        for entry in entries or ():
            self.append(entry)

    def append(self, entry: ModerationAuditLog | Mapping[str, Any]) -> ModerationAuditLog:
        record = entry if isinstance(entry, ModerationAuditLog) else ModerationAuditLog.from_row(entry)
        if not isinstance(record, ModerationAuditLog):  # pragma: no cover - defensive
            raise ContractError(ErrorCode.INTERNAL_ERROR, "expected a ModerationAuditLog")
        with self._lock:
            if record.id in self._ids:
                raise ContractError(
                    ErrorCode.CONFLICT, "audit entry already recorded", {"id": record.id}
                )
            self._ids.add(record.id)
            self._entries.append(record)
        return record

    def list(
        self,
        *,
        action: ModerationAction | str | None = None,
        user_id: str | None = None,
        note_id: str | None = None,
        comment_id: str | None = None,
        target_id: str | None = None,
        limit: int | None = None,
    ) -> list[ModerationAuditLog]:
        with self._lock:
            snapshot = list(self._entries)
        selected = [
            entry
            for entry in snapshot
            if entry.matches(action=action, user_id=user_id, note_id=note_id,
                             comment_id=comment_id, target_id=target_id)
        ]
        selected.sort(key=lambda entry: (entry.created_at, entry.id), reverse=True)
        if limit is not None:
            selected = selected[: max(0, int(limit))]
        return selected

    def count(self, *, action: ModerationAction | str | None = None) -> int:
        return len(self.list(action=action))

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._ids.clear()


class WrappingAuditLogStore:
    """Adapter over an injected object exposing ``append`` (+ optional reads).

    Keeps the moderation service usable with any audit sink the web layer
    provides (a logging handler, the account layer's security recorder, a test
    spy, ...) without this layer pretending to own its storage.
    """

    def __init__(self, target: Any) -> None:
        if not callable(getattr(target, "append", None)):
            raise ContractError(
                ErrorCode.CONFIG_ERROR, "audit sink must expose append(entry)"
            )
        self._target = target
        self._mirror = InMemoryAuditLogStore()

    def append(self, entry: ModerationAuditLog) -> ModerationAuditLog:
        self._target.append(entry)
        self._mirror.append(entry)
        return entry

    def _sink_entries(self) -> list[Any] | None:
        for name in ("entries", "records", "items", "_entries"):
            value = getattr(self._target, name, None)
            if isinstance(value, list):
                return list(value)
        return None

    def list(self, *, action: Any = None, user_id: str | None = None,
             note_id: str | None = None, comment_id: str | None = None,
             target_id: str | None = None, limit: int | None = None) -> list[Any]:
        lister = getattr(self._target, "list", None)
        if callable(lister):
            return list(
                lister(action=action, user_id=user_id, note_id=note_id,
                       comment_id=comment_id, target_id=target_id, limit=limit)
            )
        entries = self._sink_entries()
        source: list[Any] = entries if entries is not None else self._mirror.list()
        out: list[Any] = []
        for entry in source:
            if isinstance(entry, ModerationAuditLog) and entry.matches(
                action=action, user_id=user_id, note_id=note_id,
                comment_id=comment_id, target_id=target_id,
            ):
                out.append(entry)
        out.sort(key=lambda entry: (entry.created_at, entry.id), reverse=True)
        return out[: max(0, int(limit))] if limit is not None else out

    def count(self, *, action: Any = None) -> int:
        entries = self._sink_entries()
        if entries is not None and action is None:
            return len(entries)
        return len(self.list(action=action))


_SCHEMA: Final[tuple[str, ...]] = (
    """
    CREATE TABLE IF NOT EXISTS moderation_audit_log (
        id          TEXT PRIMARY KEY,
        action      TEXT NOT NULL,
        user_id     TEXT,
        note_id     TEXT,
        comment_id  TEXT,
        reason      TEXT,
        details     TEXT NOT NULL DEFAULT '{}',
        created_at  TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_mal_action ON moderation_audit_log(action, created_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_mal_user ON moderation_audit_log(user_id, created_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_mal_note ON moderation_audit_log(note_id)",
    "CREATE INDEX IF NOT EXISTS idx_mal_comment ON moderation_audit_log(comment_id)",
)

_COLUMNS: Final[str] = ", ".join(AUDIT_COLUMNS)


class SQLiteAuditLogStore:
    """File backed audit store (server side persistence, ``INV-config-db``)."""

    def __init__(self, database_url_or_path: str | Path = "sqlite:///data/blacknote_moderation.db"
                 ) -> None:
        self._path = _resolve_path(database_url_or_path)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            str(self._path) if self._path else ":memory:",
            check_same_thread=False,
            isolation_level=None,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode = WAL")
        with self._lock:
            self._conn.executescript(";\n".join(_SCHEMA) + ";\n")

    # -- lifecycle ------------------------------------------------------------
    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # -- writes ---------------------------------------------------------------
    def append(self, entry: ModerationAuditLog | Mapping[str, Any]) -> ModerationAuditLog:
        record = entry if isinstance(entry, ModerationAuditLog) else ModerationAuditLog.from_row(entry)
        row = record.to_row()
        with self._lock, self._conn:
            try:
                self._conn.execute(
                    f"INSERT INTO moderation_audit_log ({_COLUMNS}) "
                    f"VALUES ({', '.join(':' + name for name in AUDIT_COLUMNS)})",
                    row,
                )
            except sqlite3.IntegrityError:
                raise ContractError(
                    ErrorCode.CONFLICT, "audit entry already recorded", {"id": record.id}
                ) from None
        return record

    # -- reads ----------------------------------------------------------------
    def list(
        self,
        *,
        action: ModerationAction | str | None = None,
        user_id: str | None = None,
        note_id: str | None = None,
        comment_id: str | None = None,
        target_id: str | None = None,
        limit: int | None = None,
    ) -> list[ModerationAuditLog]:
        clauses: list[str] = []
        params: list[Any] = []
        if action is not None:
            clauses.append("action = ?")
            params.append(str(parse_action(action)))
        for column, value in (
            ("user_id", coerce_id(user_id) if user_id is not None else None),
            ("note_id", coerce_id(note_id) if note_id is not None else None),
            ("comment_id", coerce_id(comment_id) if comment_id is not None else None),
        ):
            if value:
                clauses.append(f"{column} = ?")
                params.append(value)
        if target_id is not None:
            wanted = coerce_id(target_id)
            if wanted:
                clauses.append("(note_id = ? OR comment_id = ?)")
                params.extend([wanted, wanted])
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT {_COLUMNS} FROM moderation_audit_log {where} " \
              "ORDER BY created_at DESC, id DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(max(0, int(limit)))
        with self._lock:
            cursor = self._conn.execute(sql, params)
            rows = cursor.fetchall()
            cursor.close()
        return [ModerationAuditLog.from_row(row) for row in rows]

    def count(self, *, action: ModerationAction | str | None = None) -> int:
        if action is None:
            sql, params = "SELECT COUNT(*) AS n FROM moderation_audit_log", []
        else:
            sql = "SELECT COUNT(*) AS n FROM moderation_audit_log WHERE action = ?"
            params = [str(parse_action(action))]
        with self._lock:
            row = self._conn.execute(sql, params).fetchone()
        return int(row["n"]) if row else 0

    def clear(self) -> None:
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM moderation_audit_log")


def _resolve_path(target: str | Path) -> Path | None:
    """Translate a ``sqlite://`` URL (or plain path) into a filesystem path."""
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


def open_audit_store(
    database_url: str | None = None,
    *,
    prefer: str = "sqlite",
) -> Any:
    """Wiring helper for ``mod-web``: persisted store by default, memory on demand."""
    if prefer == "memory":
        return InMemoryAuditLogStore()
    return SQLiteAuditLogStore(database_url or "sqlite:///data/blacknote_moderation.db")
