"""Persistence port for accounts and sessions.

Only the account layer touches the database through this class.  Reads always
happen inside the transaction block owned by :class:`AccountService`, expired
sessions are pruned lazily, and constraint violations are translated into the
frozen contract error vocabulary (``EMAIL_TAKEN``) instead of leaking
``sqlite3.IntegrityError``.

Nothing here knows about HTTP or about the note / interaction / moderation
domains: it stores and returns :class:`~blacknote.account.models.UserRecord`
and :class:`~blacknote.account.models.SessionGrant` rows.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping
from typing import Any, Final

from blacknote.contracts.errors import ContractError, ErrorCode
from blacknote.contracts.timeutil import from_iso, is_expired, now_iso, to_iso
from blacknote.contracts.validation import normalize_email

from .db import Database
from .models import SessionGrant, UserRecord, encode_tags

__all__ = [
    "ACCOUNT_COLUMNS",
    "AccountRepository",
    "AccountRepositoryError",
    "session_is_live",
]

ACCOUNT_COLUMNS: Final[tuple[str, ...]] = (
    "id",
    "email",
    "password_hash",
    "role",
    "username",
    "bio",
    "avatar_url",
    "specialty",
    "affiliation",
    "expertise_tags",
    "created_at",
    "updated_at",
)

_SELECT_USER: Final[str] = f"SELECT {', '.join(ACCOUNT_COLUMNS)} FROM users"


class AccountRepositoryError(ContractError):
    """Repository failures always speak the contract error language."""

    def __init__(self, code: str | ErrorCode, message: str | None = None,
                 details: Mapping[str, Any] | None = None) -> None:
        super().__init__(code, message, details)


class AccountRepository:
    """Row level gateway for ``users`` / ``sessions`` / ``auth_audit``."""

    def __init__(self, database: Database) -> None:
        if not isinstance(database, Database):
            raise ContractError(
                ErrorCode.CONFIG_ERROR, "AccountRepository requires a Database instance"
            )
        self.db = database

    # -- lifecycle ------------------------------------------------------------
    def initialize(self) -> None:
        self.db.initialize()

    def close(self) -> None:
        self.db.close()

    def transaction(self) -> Any:
        return self.db.transaction()

    # -- users ----------------------------------------------------------------
    def email_exists(self, email: str) -> bool:
        row = self.db.row(
            "SELECT 1 AS hit FROM users WHERE email = ? LIMIT 1", (normalize_email(email),)
        )
        return row is not None

    def get_user_by_id(self, user_id: str) -> UserRecord | None:
        row = self.db.row(f"{_SELECT_USER} WHERE id = ?", (user_id,))
        return UserRecord.from_row(row) if row is not None else None

    def get_user_by_email(self, email: str) -> UserRecord | None:
        row = self.db.row(f"{_SELECT_USER} WHERE email = ?", (normalize_email(email),))
        return UserRecord.from_row(row) if row is not None else None

    def insert_user(self, record: UserRecord) -> UserRecord:
        row = record.to_row()
        columns = ", ".join(ACCOUNT_COLUMNS)
        placeholders = ", ".join(f":{name}" for name in ACCOUNT_COLUMNS)
        try:
            with self.db.transaction():
                self.db.execute(
                    f"INSERT INTO users ({columns}) VALUES ({placeholders})", row
                )
        except sqlite3.IntegrityError as exc:
            raise self._conflict(exc) from exc
        return record

    def update_user(self, record: UserRecord, *, changes: Mapping[str, Any]) -> UserRecord:
        """Apply *changes* (already validated) and bump ``updated_at``."""
        payload = {
            name: value
            for name, value in dict(changes).items()
            if name in ACCOUNT_COLUMNS and name not in ("id", "created_at")
        }
        if "expertise_tags" in payload and not isinstance(payload["expertise_tags"], str):
            payload["expertise_tags"] = encode_tags(tuple(payload["expertise_tags"]))
        if "avatar_url" in payload and not payload["avatar_url"]:
            payload["avatar_url"] = ""
        assignments = ", ".join(f"{name} = :{name}" for name in payload)
        if not assignments:
            return record
        bound = dict(payload)
        bound["updated_at"] = now_iso()
        bound["id"] = record.id
        try:
            with self.db.transaction():
                cursor = self.db.execute(
                    f"UPDATE users SET {assignments}, updated_at = :updated_at "
                    "WHERE id = :id",
                    bound,
                )
        except sqlite3.IntegrityError as exc:
            raise self._conflict(exc) from exc
        if cursor.rowcount != 1:
            raise ContractError(ErrorCode.USER_NOT_FOUND, "user row disappeared")
        updated = self.get_user_by_id(record.id)
        if updated is None:  # pragma: no cover - defensive
            raise ContractError(ErrorCode.USER_NOT_FOUND)
        return updated

    def list_users(self, *, limit: int = 20, role: str | None = None) -> list[UserRecord]:
        sql = f"{_SELECT_USER}"
        params: list[Any] = []
        if role is not None:
            sql += " WHERE role = ?"
            params.append(role)
        sql += " ORDER BY created_at ASC, id ASC LIMIT ?"
        params.append(max(1, int(limit)))
        return [UserRecord.from_row(row) for row in self.db.rows(sql, params)]

    def count_users(self) -> int:
        return int(self.db.scalar("SELECT COUNT(*) FROM users") or 0)

    def set_role(self, user_id: str, role: str) -> UserRecord:
        """Used exclusively by the seed path (never by a user facing route)."""
        try:
            with self.db.transaction():
                cursor = self.db.execute(
                    "UPDATE users SET role = ?, updated_at = ? WHERE id = ?",
                    (str(role), now_iso(), user_id),
                )
        except sqlite3.IntegrityError as exc:
            raise self._conflict(exc) from exc
        if cursor.rowcount != 1:
            raise ContractError(ErrorCode.USER_NOT_FOUND)
        return self.get_user_by_id(user_id)  # type: ignore[return-value]

    # -- sessions -------------------------------------------------------------
    def create_session(
        self,
        *,
        token: str,
        user_id: str,
        created_at: str,
        expires_at: str,
    ) -> SessionGrant:
        try:
            with self.db.transaction():
                self.db.execute(
                    "INSERT INTO sessions(token, user_id, created_at, expires_at) "
                    "VALUES (?, ?, ?, ?)",
                    (token, user_id, created_at, expires_at),
                )
        except sqlite3.IntegrityError as exc:  # token collision → retry upstream
            raise ContractError(
                ErrorCode.DATABASE_ERROR, "session token collision", {"reason": str(exc)}
            ) from exc
        return SessionGrant(
            token=token, user_id=user_id, created_at=created_at, expires_at=expires_at
        )

    def get_session(self, token: str) -> SessionGrant | None:
        row = self.db.row(
            "SELECT token, user_id, created_at, expires_at FROM sessions WHERE token = ?",
            (token,),
        )
        return SessionGrant.from_row(row) if row is not None else None

    def delete_session(self, token: str) -> bool:
        with self.db.transaction():
            cursor = self.db.execute("DELETE FROM sessions WHERE token = ?", (token,))
        return cursor.rowcount > 0

    def delete_sessions_for_user(self, user_id: str) -> int:
        with self.db.transaction():
            cursor = self.db.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
        return int(cursor.rowcount)

    def purge_expired_sessions(self, *, now: Any = None) -> int:
        reference = to_iso(now) if now is not None else now_iso()
        with self.db.transaction():
            cursor = self.db.execute("DELETE FROM sessions WHERE expires_at <= ?", (reference,))
        return int(cursor.rowcount)

    def live_session_count(self) -> int:
        return int(self.db.scalar("SELECT COUNT(*) FROM sessions") or 0)

    # -- audit ----------------------------------------------------------------
    def record_audit(
        self,
        event: str,
        *,
        actor_id: str | None = None,
        user_id: str | None = None,
        reason: str | None = None,
    ) -> None:
        """One row per security-relevant refusal (**INV-role-check**)."""
        try:
            with self.db.transaction():
                self.db.execute(
                    "INSERT INTO auth_audit(event, actor_id, user_id, reason, created_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (event, actor_id, user_id, reason, now_iso()),
                )
        except ContractError:  # audit must never break the request path
            pass

    def audit_rows(self, event: str | None = None, *, limit: int = 50) -> list[dict[str, Any]]:
        sql = "SELECT event, actor_id, user_id, reason, created_at FROM auth_audit"
        params: list[Any] = []
        if event is not None:
            sql += " WHERE event = ?"
            params.append(event)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(max(1, int(limit)))
        return [dict(row) for row in self.db.rows(sql, params)]

    # -- helpers --------------------------------------------------------------
    @staticmethod
    def _conflict(exc: BaseException) -> ContractError:
        text = str(exc).lower()
        if "users.email" in text or "email" in text:
            return ContractError(
                ErrorCode.EMAIL_TAKEN,
                "That email address is already registered.",
                {"field": "email"},
            )
        if "users.role" in text or "role" in text:
            return ContractError(
                ErrorCode.VALIDATION_ERROR, "role must be 'user' or 'admin'", {"field": "role"}
            )
        return ContractError(ErrorCode.CONFLICT, "the account already exists",
                             {"reason": str(exc)})


def session_is_live(grant: SessionGrant, *, now: Any = None) -> bool:
    """Expiry check that fails closed on an unparsable timestamp."""
    try:
        from_iso(grant.expires_at)
    except ContractError:
        return False
    return not is_expired(grant.expires_at, now=now)
