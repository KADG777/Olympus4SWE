"""SQLite plumbing for the account layer (**INV-config-db**).

Responsibilities of this module, and *only* these:

* turn the frozen :class:`~blacknote.contracts.config.Settings.database_url`
  connection string into a usable SQLite target (file path or ``:memory:``);
* hold one serialized connection per repository instance, so the web layer may
  serve several threads while SQLite stays consistent;
* create the account schema idempotently (``CREATE TABLE IF NOT EXISTS``), so a
  process restart re-attaches to the *same* data instead of losing it.

Everything crossing this module upward uses the contract error vocabulary
(``CONFIG_ERROR`` for an unusable connection string, ``DATABASE_ERROR`` for a
storage failure) — a raw ``sqlite3`` traceback never escapes.
"""

from __future__ import annotations

import re
import sqlite3
import threading
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from types import TracebackType
from typing import Any, Final

from blacknote.contracts.config import Settings, load_settings
from blacknote.contracts.errors import ContractError, ErrorCode
from blacknote.contracts.timeutil import now_iso

__all__ = [
    "ACCOUNT_SCHEMA",
    "MEMORY_TARGET",
    "SCHEMA_VERSION",
    "SQLITE_SCHEMES",
    "Database",
    "parse_database_url",
]

MEMORY_TARGET: Final[str] = ":memory:"
SQLITE_SCHEMES: Final[frozenset[str]] = frozenset({"sqlite", "sqlite+aiosqlite"})
SCHEMA_VERSION: Final[int] = 1

_URL_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^(?P<scheme>[a-z][a-z0-9+.\-]*)://(?P<rest>.*)$", re.IGNORECASE
)


def _config_error(message: str) -> ContractError:
    return ContractError(ErrorCode.CONFIG_ERROR, message, {"invariant": "INV-config-db"})


def parse_database_url(url: str) -> str:
    """Resolve a SQLAlchemy-style *url* into ``":memory:"`` or a file path.

    Supported shapes::

        sqlite://                      → in-memory
        sqlite:///:memory:             → in-memory
        sqlite:///data/blacknote.db    → relative path (the Settings default)
        sqlite:////tmp/bn/blacknote.db → absolute path
        sqlite+aiosqlite:///…          → same rules (served by stdlib sqlite3)
    """
    text = (url or "").strip()
    match = _URL_PATTERN.match(text)
    if match is None:
        raise _config_error(f"database_url is not a full URL: {text!r}")
    scheme = match.group("scheme").lower()
    if scheme not in SQLITE_SCHEMES:
        raise _config_error(
            f"database_url scheme {scheme!r} is not supported by the account layer; "
            "configure a sqlite:/// connection string (INV-config-db)"
        )
    rest = match.group("rest").split("?", 1)[0].split("#", 1)[0]
    slashes = len(rest) - len(rest.lstrip("/"))
    body = rest.lstrip("/")
    if body in ("", MEMORY_TARGET):
        return MEMORY_TARGET
    # ``sqlite://`` + one slash → relative path; a second slash → absolute.
    return f"/{body}" if slashes >= 2 else body


class Database:
    """A serialized SQLite connection plus the account schema lifecycle."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        target: str | None = None,
        schema: str | None = None,
    ) -> None:
        self.settings = settings if settings is not None else load_settings()
        self.target = target if target is not None else parse_database_url(
            self.settings.database_url
        )
        self.schema = schema if schema is not None else ACCOUNT_SCHEMA
        self._lock = threading.RLock()
        self._conn: sqlite3.Connection | None = None
        self._depth = 0
        if self.target != MEMORY_TARGET:
            self._prepare_file()

    # -- connection -----------------------------------------------------------
    @property
    def is_memory(self) -> bool:
        return self.target == MEMORY_TARGET

    @property
    def file_path(self) -> Path | None:
        return None if self.is_memory else Path(self.target)

    @property
    def connection(self) -> sqlite3.Connection:
        with self._lock:
            if self._conn is None:
                self._conn = self._connect()
            return self._conn

    def _prepare_file(self) -> None:
        path = Path(self.target)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise _config_error(f"database file directory is unusable: {exc}") from exc

    def _connect(self) -> sqlite3.Connection:
        timeout = float(self.settings.timeout("db"))
        try:
            conn = sqlite3.connect(
                self.target,
                timeout=timeout,
                isolation_level=None,  # explicit BEGIN through transaction()
                check_same_thread=False,
            )
        except sqlite3.Error as exc:
            raise ContractError(
                ErrorCode.DATABASE_ERROR,
                f"cannot open database: {exc}",
                {"target": self.target},
            ) from exc
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA foreign_keys = ON")
            if not self.is_memory:
                conn.execute("PRAGMA journal_mode = WAL")
            conn.execute(f"PRAGMA busy_timeout = {max(1, int(timeout * 1000))}")
        except sqlite3.Error as exc:  # pragma: no cover - exotic SQLite builds
            conn.close()
            raise ContractError(
                ErrorCode.DATABASE_ERROR, f"cannot configure database: {exc}"
            ) from exc
        return conn

    # -- DDL ------------------------------------------------------------------
    def initialize(self) -> None:
        """Create the schema idempotently (safe on every start-up)."""
        with self._lock:
            try:
                self.connection.executescript(self.schema)
                self.connection.execute(
                    "INSERT INTO schema_meta(id, version, upgraded_at) VALUES (1, ?, ?) "
                    "ON CONFLICT(id) DO NOTHING",
                    (SCHEMA_VERSION, now_iso()),
                )
            except sqlite3.Error as exc:
                raise ContractError(
                    ErrorCode.DATABASE_ERROR, f"cannot initialize schema: {exc}"
                ) from exc

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                finally:
                    self._conn = None

    # -- DML ------------------------------------------------------------------
    def rows(self, sql: str, params: Iterable[Any] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return list(self._cursor(sql, params).fetchall())

    def row(self, sql: str, params: Iterable[Any] = ()) -> sqlite3.Row | None:
        with self._lock:
            return self._cursor(sql, params).fetchone()

    def scalar(self, sql: str, params: Iterable[Any] = ()) -> Any:
        row = self.row(sql, params)
        return row[0] if row is not None else None

    def execute(self, sql: str, params: Iterable[Any] = ()) -> sqlite3.Cursor:
        with self._lock:
            return self._cursor(sql, params)

    def _cursor(self, sql: str, params: Mapping[str, Any] | Sequence[Any]) -> sqlite3.Cursor:
        values = params if isinstance(params, Mapping) else tuple(params)
        try:
            return self.connection.execute(sql, values)
        except sqlite3.IntegrityError:
            # Callers translate constraint violations into CONFLICT / EMAIL_TAKEN.
            raise
        except sqlite3.Error as exc:
            raise ContractError(
                ErrorCode.DATABASE_ERROR, f"storage layer failure: {exc}"
            ) from exc

    # -- transactions ---------------------------------------------------------
    class _Transaction:
        def __init__(self, db: "Database") -> None:
            self._db = db

        def __enter__(self) -> "Database":
            self._db._begin()
            return self._db

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            tb: TracebackType | None,
        ) -> bool:
            self._db._commit_or_rollback(commit=exc_type is None)
            return False

    def transaction(self) -> "_Transaction":
        """Re-entrant ``BEGIN IMMEDIATE`` / ``COMMIT`` / ``ROLLBACK`` block."""
        return Database._Transaction(self)

    def _begin(self) -> None:
        with self._lock:
            if self._depth == 0:
                try:
                    self.connection.execute("BEGIN IMMEDIATE")
                except sqlite3.Error as exc:
                    raise ContractError(ErrorCode.DATABASE_ERROR, str(exc)) from exc
            self._depth += 1

    def _commit_or_rollback(self, *, commit: bool) -> None:
        with self._lock:
            self._depth = max(0, self._depth - 1)
            if self._depth > 0:
                return
            try:
                self.connection.execute("COMMIT" if commit else "ROLLBACK")
            except sqlite3.Error as exc:  # pragma: no cover - defensive
                self._depth = 0
                raise ContractError(ErrorCode.DATABASE_ERROR, str(exc)) from exc

    # -- introspection (tests / ops) ------------------------------------------
    def table_names(self) -> set[str]:
        return {
            str(row["name"])
            for row in self.rows("SELECT name FROM sqlite_master WHERE type = 'table'")
        }

    def schema_version(self) -> int | None:
        row = self.row("SELECT version FROM schema_meta WHERE id = 1")
        return int(row["version"]) if row is not None else None

    def __enter__(self) -> "Database":
        self.initialize()
        return self

    def __exit__(self, *exc_info: Any) -> bool:
        self.close()
        return False

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"Database(target={self.target!r}, memory={self.is_memory})"


# --------------------------------------------------------------------------- #
# Account schema: users + sessions + a minimal security audit trail.
# ``expertise_tags`` is stored as a single delimited text column so the row
# stays readable with plain SQL; the repository owns the encoding.
# --------------------------------------------------------------------------- #
ACCOUNT_SCHEMA: Final[str] = """
CREATE TABLE IF NOT EXISTS schema_meta (
    id          INTEGER PRIMARY KEY CHECK (id = 1),
    version     INTEGER NOT NULL,
    upgraded_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS users (
    id              TEXT PRIMARY KEY CHECK (length(id) = 36),
    email           TEXT NOT NULL UNIQUE,
    password_hash   TEXT NOT NULL,
    role            TEXT NOT NULL DEFAULT 'user' CHECK (role IN ('user', 'admin')),
    username        TEXT NOT NULL,
    bio             TEXT NOT NULL DEFAULT '',
    avatar_url      TEXT,
    specialty       TEXT NOT NULL DEFAULT '',
    affiliation     TEXT NOT NULL DEFAULT '',
    expertise_tags  TEXT NOT NULL DEFAULT '',
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_users_role ON users(role);

CREATE TABLE IF NOT EXISTS sessions (
    token       TEXT PRIMARY KEY CHECK (length(token) = 36),
    user_id     TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at  TEXT NOT NULL,
    expires_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_expires_at ON sessions(expires_at);

CREATE TABLE IF NOT EXISTS auth_audit (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    event      TEXT NOT NULL,
    user_id    TEXT,
    actor_id   TEXT,
    reason     TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_auth_audit_event ON auth_audit(event);
"""
