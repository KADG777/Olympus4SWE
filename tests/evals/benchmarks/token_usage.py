"""Canonical token-usage accounting shared by the Harbor benchmark adapters.

Olympus reports per-session ``token_usage`` inside its JSON ``SessionEnvelope``:
``reasoning_tokens`` is a non-additive subset of ``output_tokens`` (never
double-counted) and ``total_tokens`` is
``input + output + cache_read + cache_write``. When an envelope fails or omits
the field, the adapters fall back to the cumulative ``budget.settled`` events in
the container's ``<state-dir>/olympus.db``.

The module is stdlib-only: ``interact_bridge.py`` imports it after being uploaded
into a task container.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Any

_TOKEN_FIELDS = (
    "input_tokens",
    "output_tokens",
    "reasoning_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
)

BUDGET_SETTLED_DB_MARKER = "# OLYMPUS_BUDGET_SETTLED_FALLBACK"


def _as_non_negative_int(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _counters(source: Mapping[str, Any]) -> dict[str, int]:
    return {name: _as_non_negative_int(source.get(name)) for name in _TOKEN_FIELDS}


def _total(counters: Mapping[str, int]) -> int:
    return (
        counters["input_tokens"]
        + counters["output_tokens"]
        + counters["cache_read_tokens"]
        + counters["cache_write_tokens"]
    )


def _cache_hit_rate(counters: Mapping[str, int]) -> float:
    denominator = counters["input_tokens"] + counters["cache_read_tokens"]
    if denominator <= 0:
        return 0.0
    return round(counters["cache_read_tokens"] / denominator, 4)


def zero_token_usage() -> dict[str, Any]:
    """A canonical all-zero record for runs that reported no usage at all."""
    return {
        "llm_calls": 0,
        **{name: 0 for name in _TOKEN_FIELDS},
        "total_tokens": 0,
        "cache_hit_rate": 0.0,
        "by_actor": {},
    }


def normalize_token_usage(payload: Any) -> dict[str, Any] | None:
    """Return a canonical token-usage record, or ``None`` when no usage is present.

    Totals and the cache-hit rate are recomputed from the primitive counters so
    inconsistent upstream values can never double-count reasoning tokens.
    """
    if not isinstance(payload, Mapping):
        return None
    counters = _counters(payload)
    by_actor: dict[str, dict[str, Any]] = {}
    raw_actors = payload.get("by_actor")
    if isinstance(raw_actors, Mapping):
        for actor, entry in raw_actors.items():
            if not isinstance(entry, Mapping):
                continue
            actor_counters = _counters(entry)
            by_actor[str(actor)] = {
                "calls": _as_non_negative_int(entry.get("calls")),
                **actor_counters,
                "total_tokens": _total(actor_counters),
            }
    llm_calls = _as_non_negative_int(payload.get("llm_calls"))
    if not llm_calls and by_actor:
        llm_calls = sum(entry["calls"] for entry in by_actor.values())
    empty = llm_calls == 0 and _total(counters) == 0 and not by_actor
    if empty:
        return None
    return {
        "llm_calls": llm_calls,
        **counters,
        "total_tokens": _total(counters),
        "cache_hit_rate": _cache_hit_rate(counters),
        "by_actor": by_actor,
    }


def merge_token_usage(
    current: Mapping[str, Any] | None,
    incoming: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Keep the cumulative snapshot with the larger token total.

    Session envelopes carry cumulative usage for a run, so a later reply's
    snapshot supersedes an earlier one; the max guard tolerates adapters that
    report per-turn deltas by preferring the richer record.
    """
    current_usage = normalize_token_usage(current) if current is not None else None
    incoming_usage = normalize_token_usage(incoming) if incoming is not None else None
    if current_usage is None:
        return incoming_usage
    if incoming_usage is None:
        return current_usage
    if incoming_usage["total_tokens"] >= current_usage["total_tokens"]:
        return incoming_usage
    return current_usage


def token_usage_context_fields(usage: Mapping[str, Any] | None) -> dict[str, int]:
    """Map Olympus counters onto Harbor ``AgentContext`` token fields.

    Harbor defines ``n_input_tokens`` as *including* cached tokens, so the
    cache-read counter is added to the input counter rather than double-counted.
    """
    if not usage:
        return {}
    return {
        "n_input_tokens": _as_non_negative_int(usage.get("input_tokens"))
        + _as_non_negative_int(usage.get("cache_read_tokens")),
        "n_output_tokens": _as_non_negative_int(usage.get("output_tokens")),
        "n_cache_tokens": _as_non_negative_int(usage.get("cache_read_tokens")),
    }


def usage_from_context_fields(
    *,
    n_input_tokens: Any = None,
    n_output_tokens: Any = None,
    n_cache_tokens: Any = None,
) -> dict[str, Any] | None:
    """Reconstruct canonical usage from Harbor's ``n_*`` context fields."""
    if n_input_tokens is None and n_output_tokens is None and n_cache_tokens is None:
        return None
    cache_tokens = _as_non_negative_int(n_cache_tokens)
    input_tokens = max(0, _as_non_negative_int(n_input_tokens) - cache_tokens)
    return normalize_token_usage(
        {
            "input_tokens": input_tokens,
            "output_tokens": _as_non_negative_int(n_output_tokens),
            "cache_read_tokens": cache_tokens,
        }
    )


def aggregate_budget_settled(rows: Iterable[Any]) -> dict[str, Any] | None:
    """Sum ``budget.settled`` event rows into canonical token usage.

    ``rows`` is an iterable of ``(actor, payload)`` pairs, matching the fallback
    query against the container's event store. ``llm_calls`` takes the largest
    cumulative ``used`` counter, which is monotonic across both settlement paths.
    """
    counters = {name: 0 for name in _TOKEN_FIELDS}
    actors: dict[str, dict[str, Any]] = {}
    settled = 0
    max_used = 0
    for row in rows:
        if isinstance(row, Mapping):
            actor = str(row.get("actor") or "")
            payload = row.get("payload")
        else:
            try:
                actor, payload = row
            except (TypeError, ValueError):
                continue
            actor = str(actor or "")
        if not isinstance(payload, Mapping):
            payload = {}
        settled += 1
        max_used = max(max_used, _as_non_negative_int(payload.get("used")))
        actor_entry = actors.setdefault(
            actor, {"calls": 0, **{name: 0 for name in _TOKEN_FIELDS}}
        )
        actor_entry["calls"] += 1
        for name in _TOKEN_FIELDS:
            value = _as_non_negative_int(payload.get(name))
            counters[name] += value
            actor_entry[name] += value
    if not settled:
        return None
    by_actor = {
        actor: {**entry, "total_tokens": _total(entry)}
        for actor, entry in actors.items()
        if actor
    }
    return normalize_token_usage(
        {
            "llm_calls": max_used or settled,
            **counters,
            "by_actor": by_actor,
        }
    )


def parse_budget_settled_stdout(stdout: str | None) -> dict[str, Any] | None:
    """Parse the JSON row list printed by :func:`build_budget_settled_command`."""
    for line in reversed((stdout or "").strip().splitlines()):
        line = line.strip()
        if not line.startswith("["):
            continue
        try:
            rows = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(rows, list):
            return aggregate_budget_settled(rows)
    return None


def build_budget_settled_command(db_path: str, run_id: str | None = None) -> str:
    """Shell command that dumps cumulative ``budget.settled`` usage as JSON.

    The script is embedded via a quoted heredoc so it runs with any POSIX shell
    and prints ``[]`` instead of failing when the event store is missing.
    """
    script = "\n".join(
        [
            BUDGET_SETTLED_DB_MARKER,
            "import json, sqlite3",
            f"db_path = {db_path!r}",
            f"run_id = {run_id!r}",
            "rows = []",
            "try:",
            "    conn = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)",
            "    query = (",
            "        \"SELECT actor, payload_json FROM events \"",
            "        \"WHERE event_type = 'budget.settled'\"",
            "    )",
            "    params = ()",
            "    if run_id:",
            "        query += ' AND run_id = ?'",
            "        params = (run_id,)",
            "    for actor, payload_json in conn.execute(query, params).fetchall():",
            "        try:",
            "            payload = json.loads(payload_json or '{}')",
            "        except (TypeError, ValueError):",
            "            payload = {}",
            "        rows.append([str(actor or ''), payload])",
            "    conn.close()",
            "except Exception:",
            "    rows = []",
            "print(json.dumps(rows))",
        ]
    )
    return "python3 - <<'OLYMPUS_BUDGET_SETTLED'\n" + script + "\nOLYMPUS_BUDGET_SETTLED"


__all__ = [
    "BUDGET_SETTLED_DB_MARKER",
    "aggregate_budget_settled",
    "build_budget_settled_command",
    "merge_token_usage",
    "normalize_token_usage",
    "parse_budget_settled_stdout",
    "token_usage_context_fields",
    "usage_from_context_fields",
    "zero_token_usage",
]
