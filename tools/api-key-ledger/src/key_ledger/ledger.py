"""SQLite rotation state, spend accounting and immutable attempt records."""

from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from collections import Counter
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

from .config import Budget, Config, Credential, Pool, Price, Slot
from .usage import Usage, cap_units, money

REQUEST_SECONDS = 600
RESERVATION_TTL = REQUEST_SECONDS + 120
GROUP_COLUMNS = {
    "key": "pool_id || '/' || key_alias",
    "model": "COALESCE(model, '(unknown)')",
    "provider": "pool_id",
    "day": "strftime('%Y-%m-%d', finished_at, 'unixepoch')",
    "tag": "COALESCE(tag, '(untagged)')",
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS slot_state (
    pool_id TEXT NOT NULL, key_alias TEXT NOT NULL, fingerprint TEXT,
    disabled INTEGER NOT NULL DEFAULT 0, quarantined INTEGER NOT NULL DEFAULT 0,
    cooldown_until REAL NOT NULL DEFAULT 0, selections INTEGER NOT NULL DEFAULT 0,
    generation INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (pool_id, key_alias)
);
CREATE TABLE IF NOT EXISTS cursor_state (pool_id TEXT PRIMARY KEY, last_alias TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS sticky_scopes (
    pool_id TEXT NOT NULL, sticky_key TEXT NOT NULL, key_alias TEXT NOT NULL,
    PRIMARY KEY (pool_id, sticky_key)
);
CREATE TABLE IF NOT EXISTS spend (
    pool_id TEXT NOT NULL, key_alias TEXT NOT NULL, currency TEXT NOT NULL,
    period TEXT NOT NULL, units TEXT NOT NULL,
    PRIMARY KEY (pool_id, key_alias, currency, period)
);
CREATE TABLE IF NOT EXISTS reservations (
    attempt_id TEXT PRIMARY KEY, pool_id TEXT NOT NULL, key_alias TEXT NOT NULL,
    fingerprint TEXT NOT NULL, currency TEXT NOT NULL, generation INTEGER NOT NULL,
    started_at REAL NOT NULL, model TEXT, sticky_key TEXT, tag TEXT
);
CREATE INDEX IF NOT EXISTS reservations_pool ON reservations(pool_id, key_alias);
CREATE INDEX IF NOT EXISTS reservations_time ON reservations(started_at);
CREATE TABLE IF NOT EXISTS ledger_entries (
    id INTEGER PRIMARY KEY, attempt_id TEXT NOT NULL UNIQUE,
    pool_id TEXT NOT NULL, key_alias TEXT NOT NULL, fingerprint TEXT NOT NULL,
    started_at REAL NOT NULL, finished_at REAL NOT NULL, latency_ms INTEGER NOT NULL,
    model TEXT, sticky_key TEXT, tag TEXT, http_status INTEGER, outcome TEXT NOT NULL,
    input_tokens INTEGER, output_tokens INTEGER,
    cached_input_tokens INTEGER, cache_write_tokens INTEGER,
    currency TEXT NOT NULL, pricing_status TEXT NOT NULL, price_snapshot TEXT, cost_units TEXT
);
CREATE INDEX IF NOT EXISTS ledger_finished ON ledger_entries(finished_at);
CREATE TRIGGER IF NOT EXISTS ledger_no_update BEFORE UPDATE ON ledger_entries
BEGIN SELECT RAISE(ABORT, 'ledger entries are immutable'); END;
CREATE TRIGGER IF NOT EXISTS ledger_no_delete BEFORE DELETE ON ledger_entries
BEGIN SELECT RAISE(ABORT, 'ledger entries are immutable'); END;
"""


@dataclass(frozen=True)
class Attempt:
    attempt_id: str
    key_alias: str
    credential: Credential


@dataclass(frozen=True)
class _SlotState:
    slot: Slot
    row: sqlite3.Row
    status: str
    total: int
    daily: int


def _day(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, UTC).date().isoformat()


def _timestamp(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, UTC).isoformat(timespec="milliseconds")


def _over_budget(budget: Budget, total: int, daily: int) -> bool:
    return (budget.total is not None and total >= cap_units(budget.total)) or (
        budget.daily is not None and daily >= cap_units(budget.daily)
    )


def _capped(budget: Budget) -> bool:
    return budget.daily is not None or budget.total is not None


class Ledger:
    def __init__(
        self, config: Config, credentials: Mapping[tuple[str, str], Credential] | None = None
    ) -> None:
        self.config = config
        self.credentials = credentials or {}
        config.database.parent.mkdir(mode=0o700, exist_ok=True, parents=True)
        try:
            descriptor = os.open(config.database, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            pass
        else:
            os.close(descriptor)
        with self._connection() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript(_SCHEMA)
        with self._transaction() as db:
            for pool_id, pool in config.pools.items():
                for slot in pool.keys:
                    cred = self.credentials.get((pool_id, slot.key_alias))
                    db.execute(
                        """
                        INSERT INTO slot_state(pool_id, key_alias, fingerprint) VALUES (?, ?, ?)
                        ON CONFLICT(pool_id, key_alias) DO UPDATE SET
                          fingerprint = excluded.fingerprint, generation = generation + 1
                        WHERE excluded.fingerprint IS NOT NULL
                          AND slot_state.fingerprint IS NOT excluded.fingerprint
                        """,
                        (pool_id, slot.key_alias, cred.fingerprint if cred else None),
                    )
            self._recover_expired(db, time.time())

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.config.database, timeout=5, isolation_level=None)
        db.row_factory = sqlite3.Row
        try:
            yield db
        finally:
            db.close()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                yield db
            except BaseException:
                db.rollback()
                raise
            else:
                db.commit()

    @staticmethod
    def _recover_expired(db: sqlite3.Connection, now: float) -> None:
        expiry = now - RESERVATION_TTL
        db.execute(
            """
            INSERT INTO ledger_entries(
                attempt_id, pool_id, key_alias, fingerprint, started_at, finished_at,
                latency_ms, model, sticky_key, tag, outcome, currency, pricing_status)
            SELECT attempt_id, pool_id, key_alias, fingerprint, started_at,
                   started_at + ?, ?, model, sticky_key, tag, 'interrupted',
                   currency, 'usage_unavailable'
            FROM reservations WHERE started_at <= ?
            """,
            (REQUEST_SECONDS, REQUEST_SECONDS * 1000, expiry),
        )
        db.execute("DELETE FROM reservations WHERE started_at <= ?", (expiry,))

    @staticmethod
    def _spend(
        db: sqlite3.Connection, pool_id: str, alias: str, currency: str, day: str
    ) -> tuple[int, int]:
        rows = db.execute(
            """
            SELECT period, units FROM spend WHERE pool_id=? AND key_alias=?
              AND currency=? AND period IN ('total', ?)
            """,
            (pool_id, alias, currency, day),
        )
        periods = {row["period"]: int(row["units"]) for row in rows}
        return periods.get("total", 0), periods.get(day, 0)

    def _states(
        self, db: sqlite3.Connection, pool_id: str, pool: Pool, now: float
    ) -> list[_SlotState]:
        rows = {
            row["key_alias"]: row
            for row in db.execute("SELECT * FROM slot_state WHERE pool_id=?", (pool_id,))
        }
        day = _day(now)
        pool_exceeded = _over_budget(pool.budget, *self._spend(db, pool_id, "", pool.currency, day))
        states = []
        for slot in pool.keys:
            row = rows[slot.key_alias]
            total, daily = self._spend(db, pool_id, slot.key_alias, pool.currency, day)
            if row["disabled"]:
                status = "disabled"
            elif row["quarantined"]:
                status = "quarantined"
            elif pool_exceeded or _over_budget(slot.budget, total, daily):
                status = "budget_exceeded"
            elif row["cooldown_until"] > now:
                status = "cooldown"
            else:
                status = "active"
            states.append(_SlotState(slot, row, status, total, daily))
        return states

    def reserve(
        self,
        pool_id: str,
        *,
        model: str | None,
        sticky_key: str | None,
        tag: str | None,
        tried: set[str],
    ) -> tuple[Attempt | None, dict[str, int]]:
        pool = self.config.pools[pool_id]
        with self._transaction() as db:
            now = time.time()
            self._recover_expired(db, now)
            states = self._states(db, pool_id, pool, now)
            busy = {
                row["key_alias"]
                for row in db.execute(
                    "SELECT key_alias FROM reservations WHERE pool_id=?", (pool_id,)
                )
            }
            health: Counter[str] = Counter()
            available: dict[str, _SlotState] = {}
            pinned = None
            pinned_busy = False
            if sticky_key:
                pinned = db.execute(
                    "SELECT key_alias FROM sticky_scopes WHERE pool_id=? AND sticky_key=?",
                    (pool_id, sticky_key),
                ).fetchone()
            for state in states:
                alias = state.slot.key_alias
                cred = self.credentials[pool_id, alias]
                status = state.status
                if status == "active":
                    if state.row["fingerprint"] != cred.fingerprint:
                        status = "configuration_changed"
                    elif alias in tried:
                        status = "attempted"
                    elif (_capped(pool.budget) and busy) or (
                        _capped(state.slot.budget) and alias in busy
                    ):
                        status = "busy"
                    else:
                        available[alias] = state
                health[status] += 1
                if pinned and alias == pinned["key_alias"] and status == "busy":
                    # In-flight spend can make a pin ineligible. Keep it until settlement.
                    pinned_busy = True
            if not available or pinned_busy:
                return None, dict(health)

            chosen = available.get(pinned["key_alias"]) if pinned else None
            if chosen is None:
                chosen = self._choose(db, pool_id, pool, available)
            alias = chosen.slot.key_alias
            credential = self.credentials[pool_id, alias]
            attempt_id = uuid.uuid4().hex
            db.execute(
                """
                INSERT INTO reservations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    attempt_id,
                    pool_id,
                    alias,
                    credential.fingerprint,
                    pool.currency,
                    chosen.row["generation"],
                    now,
                    model,
                    sticky_key,
                    tag,
                ),
            )
            db.execute(
                "UPDATE slot_state SET selections=selections+1 WHERE pool_id=? AND key_alias=?",
                (pool_id, alias),
            )
            if not pinned or pinned["key_alias"] != alias:
                db.execute(
                    """
                    INSERT INTO cursor_state VALUES (?, ?)
                    ON CONFLICT(pool_id) DO UPDATE SET last_alias=excluded.last_alias
                    """,
                    (pool_id, alias),
                )
                if sticky_key:
                    db.execute(
                        """
                        INSERT INTO sticky_scopes VALUES (?, ?, ?)
                        ON CONFLICT(pool_id, sticky_key) DO UPDATE SET key_alias=excluded.key_alias
                        """,
                        (pool_id, sticky_key, alias),
                    )
            return Attempt(attempt_id, alias, credential), dict(health)

    @staticmethod
    def _choose(
        db: sqlite3.Connection, pool_id: str, pool: Pool, available: dict[str, _SlotState]
    ) -> _SlotState:
        if pool.rotation == "least_used":
            return min(available.values(), key=lambda state: int(state.row["selections"]))
        if pool.rotation == "least_cost":
            return min(available.values(), key=lambda state: (state.total, state.row["selections"]))
        aliases = [slot.key_alias for slot in pool.keys]
        last = db.execute(
            "SELECT last_alias FROM cursor_state WHERE pool_id=?", (pool_id,)
        ).fetchone()
        index = (
            aliases.index(last["last_alias"]) + 1 if last and last["last_alias"] in aliases else 0
        )
        return next(
            available[alias] for alias in aliases[index:] + aliases[:index] if alias in available
        )

    def finish(
        self,
        attempt: Attempt,
        *,
        usage: Usage,
        price: Price | None,
        outcome: str,
        http_status: int | None = None,
        cooldown: float = 0,
    ) -> None:
        now = time.time()
        pricing_status, cost = usage.charge(price)
        if outcome != "success" and pricing_status == "priced":
            pricing_status = "partial"
        snapshot = price.model_dump_json() if price is not None else None
        with self._transaction() as db:
            row = db.execute(
                "SELECT * FROM reservations WHERE attempt_id=?", (attempt.attempt_id,)
            ).fetchone()
            if row is None:
                return  # Stream cleanup may be called twice after a client disconnect.
            pool, alias, currency = row["pool_id"], row["key_alias"], row["currency"]
            db.execute(
                """
                INSERT INTO ledger_entries(
                    attempt_id, pool_id, key_alias, fingerprint, started_at, finished_at,
                    latency_ms, model, sticky_key, tag, http_status, outcome, input_tokens,
                    output_tokens, cached_input_tokens, cache_write_tokens, currency,
                    pricing_status, price_snapshot, cost_units
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    attempt.attempt_id,
                    pool,
                    alias,
                    row["fingerprint"],
                    row["started_at"],
                    now,
                    max(0, int((now - row["started_at"]) * 1000)),
                    usage.model,
                    row["sticky_key"],
                    row["tag"],
                    http_status,
                    outcome,
                    usage.input,
                    usage.output,
                    usage.cache_read,
                    usage.cache_write,
                    currency,
                    pricing_status,
                    snapshot,
                    str(cost) if cost is not None else None,
                ),
            )
            if cost:
                day = _day(now)
                for owner in (alias, ""):
                    total, daily = self._spend(db, pool, owner, currency, day)
                    db.executemany(
                        """
                        INSERT INTO spend VALUES (?, ?, ?, ?, ?)
                        ON CONFLICT(pool_id, key_alias, currency, period)
                        DO UPDATE SET units=excluded.units
                        """,
                        [
                            (pool, owner, currency, "total", str(total + cost)),
                            (pool, owner, currency, day, str(daily + cost)),
                        ],
                    )
            db.execute(
                """
                UPDATE slot_state SET quarantined=MAX(quarantined, ?),
                    cooldown_until=MAX(cooldown_until, ?)
                WHERE pool_id=? AND key_alias=? AND generation=?
                """,
                (
                    int(outcome == "quarantined"),
                    now + cooldown if cooldown else 0,
                    pool,
                    alias,
                    row["generation"],
                ),
            )
            db.execute("DELETE FROM reservations WHERE attempt_id=?", (attempt.attempt_id,))

    def change_key(
        self, pool_id: str, alias: str, action: Literal["enable", "disable", "reset"]
    ) -> None:
        pool = self.config.pools.get(pool_id)
        if pool is None or not any(slot.key_alias == alias for slot in pool.keys):
            raise ValueError("unknown pool or key alias")
        with self._transaction() as db:
            db.execute(
                """
                UPDATE slot_state SET
                    disabled=CASE ? WHEN 'enable' THEN 0 WHEN 'disable' THEN 1 ELSE disabled END,
                    quarantined=CASE WHEN ?='disable' THEN quarantined ELSE 0 END,
                    cooldown_until=CASE WHEN ?='disable' THEN cooldown_until ELSE 0 END,
                    generation=generation+1
                WHERE pool_id=? AND key_alias=?
                """,
                (action, action, action, pool_id, alias),
            )

    def key_statuses(self) -> list[dict[str, Any]]:
        now = time.time()
        result = []
        with self._connection() as db:
            db.execute("BEGIN")
            for pool_id, pool in self.config.pools.items():
                pool_total, pool_daily = self._spend(db, pool_id, "", pool.currency, _day(now))
                for state in self._states(db, pool_id, pool, now):
                    end = state.row["cooldown_until"]
                    result.append(
                        {
                            "pool_id": pool_id,
                            "key_alias": state.slot.key_alias,
                            "fingerprint": state.row["fingerprint"],
                            "status": state.status,
                            "selections": state.row["selections"],
                            "cooldown_until": _timestamp(end) if end > now else None,
                            "currency": pool.currency,
                            "total_spend": money(state.total),
                            "daily_spend": money(state.daily),
                            "budget": state.slot.budget.model_dump(mode="json", exclude_none=True),
                            "pool_total_spend": money(pool_total),
                            "pool_daily_spend": money(pool_daily),
                            "pool_budget": pool.budget.model_dump(mode="json", exclude_none=True),
                        }
                    )
        return result

    def events(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._connection() as db:
            rows = db.execute(
                "SELECT * FROM ledger_entries ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        result = []
        for row in rows:
            event = dict(row)
            units = event.pop("cost_units")
            event["cost"] = money(int(units)) if units is not None else None
            snapshot = event["price_snapshot"]
            event["price_snapshot"] = json.loads(snapshot) if snapshot is not None else None
            for name in ("started_at", "finished_at"):
                event[name] = _timestamp(event[name])
            result.append(event)
        return result

    def summary(self, by: str, since: float = 0) -> list[dict[str, Any]]:
        group = GROUP_COLUMNS.get(by)
        if group is None:
            raise ValueError("invalid summary grouping")
        groups: dict[tuple[str, str], Counter[str]] = {}
        with self._connection() as db:
            rows = db.execute(
                f"""
                SELECT {group} AS label, currency, outcome, input_tokens, output_tokens,
                    cached_input_tokens, cache_write_tokens, cost_units, pricing_status
                FROM ledger_entries WHERE finished_at>=?
                """,  # group is selected from fixed SQL expressions above.
                (since,),
            )
            for row in rows:
                stats = groups.setdefault((row["label"], row["currency"]), Counter())
                stats["attempts"] += 1
                stats["successes"] += row["outcome"] == "success"
                for name in (
                    "input_tokens",
                    "output_tokens",
                    "cached_input_tokens",
                    "cache_write_tokens",
                ):
                    stats[name] += row[name] or 0
                stats["cost"] += int(row["cost_units"] or "0")
                stats["unpriced"] += row["pricing_status"] == "unpriced"
                stats["unknown_usage"] += (
                    row["input_tokens"] is None or row["output_tokens"] is None
                )
        return [
            {"group": label, "currency": currency, **stats, "cost": money(stats["cost"])}
            for (label, currency), stats in sorted(groups.items())
        ]
