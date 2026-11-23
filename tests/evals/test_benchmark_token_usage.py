"""Unit tests for the benchmark token-usage accounting helpers."""

from __future__ import annotations

import json
import sqlite3
import subprocess
from pathlib import Path

from tests.evals.benchmarks.token_usage import (
    BUDGET_SETTLED_DB_MARKER,
    aggregate_budget_settled,
    build_budget_settled_command,
    merge_token_usage,
    normalize_token_usage,
    parse_budget_settled_stdout,
    token_usage_context_fields,
    usage_from_context_fields,
    zero_token_usage,
)


def test_normalize_recomputes_totals_and_never_double_counts_reasoning() -> None:
    usage = normalize_token_usage(
        {
            "llm_calls": 3,
            "input_tokens": 1000,
            "output_tokens": 200,
            "reasoning_tokens": 150,
            "cache_read_tokens": 500,
            "cache_write_tokens": 10,
            # A buggy upstream total that double-counts reasoning must be ignored.
            "total_tokens": 999_999,
            "cache_hit_rate": 0.1,
        }
    )
    assert usage is not None
    assert usage["total_tokens"] == 1000 + 200 + 500 + 10
    assert usage["reasoning_tokens"] == 150
    assert usage["cache_hit_rate"] == round(500 / (1000 + 500), 4)
    assert usage["by_actor"] == {}


def test_normalize_derives_llm_calls_from_actor_breakdown() -> None:
    usage = normalize_token_usage(
        {
            "input_tokens": 10,
            "by_actor": {
                "align:proposer": {
                    "calls": 2,
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "cache_read_tokens": 30,
                }
            },
        }
    )
    assert usage is not None
    assert usage["llm_calls"] == 2
    actor = usage["by_actor"]["align:proposer"]
    assert actor["total_tokens"] == 10 + 5 + 30
    assert actor["calls"] == 2


def test_normalize_returns_none_for_empty_or_invalid_payloads() -> None:
    assert normalize_token_usage(None) is None
    assert normalize_token_usage({}) is None
    assert normalize_token_usage({"llm_calls": 0, "input_tokens": 0}) is None
    assert normalize_token_usage(["not", "a", "dict"]) is None


def test_zero_token_usage_is_canonical() -> None:
    zero = zero_token_usage()
    assert zero["total_tokens"] == 0
    assert zero["cache_hit_rate"] == 0.0
    assert zero["by_actor"] == {}
    assert set(zero) == {
        "llm_calls",
        "input_tokens",
        "output_tokens",
        "reasoning_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
        "total_tokens",
        "cache_hit_rate",
        "by_actor",
    }


def test_zero_cache_readers_report_zero_hit_rate() -> None:
    usage = normalize_token_usage({"input_tokens": 100, "output_tokens": 10})
    assert usage is not None
    assert usage["cache_hit_rate"] == 0.0


def test_merge_keeps_the_richer_cumulative_snapshot() -> None:
    small = {"input_tokens": 10, "output_tokens": 1}
    large = {"input_tokens": 100, "output_tokens": 10}
    assert merge_token_usage(small, large) == normalize_token_usage(large)
    assert merge_token_usage(large, small) == normalize_token_usage(large)
    assert merge_token_usage(None, small) == normalize_token_usage(small)
    assert merge_token_usage(small, None) == normalize_token_usage(small)
    assert merge_token_usage(None, None) is None


def test_token_usage_context_fields_add_cache_to_input() -> None:
    usage = normalize_token_usage(
        {"input_tokens": 1000, "output_tokens": 200, "cache_read_tokens": 500}
    )
    assert token_usage_context_fields(usage) == {
        "n_input_tokens": 1500,
        "n_output_tokens": 200,
        "n_cache_tokens": 500,
    }
    assert token_usage_context_fields(None) == {}


def test_usage_from_context_fields_inverts_harbor_semantics() -> None:
    usage = usage_from_context_fields(
        n_input_tokens=1500, n_output_tokens=200, n_cache_tokens=500
    )
    assert usage is not None
    assert usage["input_tokens"] == 1000
    assert usage["cache_read_tokens"] == 500
    assert usage["total_tokens"] == 1000 + 200 + 500
    assert usage_from_context_fields() is None


def test_aggregate_budget_settled_sums_actors_and_uses_max_calls() -> None:
    usage = aggregate_budget_settled(
        [
            ("align:proposer", {"used": 1, "input_tokens": 100, "cache_read_tokens": 10}),
            ("align:proposer", {"used": 2, "input_tokens": 50, "output_tokens": 5}),
            ("build:worker", {"used": 3, "input_tokens": 25, "cache_write_tokens": 7}),
        ]
    )
    assert usage is not None
    assert usage["llm_calls"] == 3
    assert usage["input_tokens"] == 175
    assert usage["output_tokens"] == 5
    assert usage["cache_read_tokens"] == 10
    assert usage["cache_write_tokens"] == 7
    assert usage["total_tokens"] == 175 + 5 + 10 + 7
    assert usage["by_actor"]["align:proposer"]["calls"] == 2
    assert usage["by_actor"]["build:worker"]["total_tokens"] == 32
    assert aggregate_budget_settled([]) is None


def test_budget_settled_command_queries_a_real_sqlite_store(tmp_path: Path) -> None:
    db_path = tmp_path / "olympus.db"
    connection = sqlite3.connect(db_path)
    connection.execute(
        "CREATE TABLE events (run_id TEXT, event_type TEXT, actor TEXT, payload_json TEXT)"
    )
    connection.executemany(
        "INSERT INTO events VALUES (?, ?, ?, ?)",
        [
            ("run-1", "budget.settled", "align:proposer",
             json.dumps({"used": 1, "input_tokens": 100, "cache_read_tokens": 40})),
            ("run-1", "budget.settled", "build:worker",
             json.dumps({"used": 2, "input_tokens": 50, "output_tokens": 10})),
            ("run-2", "budget.settled", "align:proposer",
             json.dumps({"used": 9, "input_tokens": 9999})),
            ("run-1", "run.created", "api", "{}"),
        ],
    )
    connection.commit()
    connection.close()

    command = build_budget_settled_command(str(db_path), "run-1")
    assert BUDGET_SETTLED_DB_MARKER in command
    result = subprocess.run(
        ["/bin/sh", "-c", command], capture_output=True, text=True, check=True
    )
    usage = parse_budget_settled_stdout(result.stdout)
    assert usage is not None
    assert usage["llm_calls"] == 2
    assert usage["input_tokens"] == 150
    assert usage["cache_read_tokens"] == 40
    assert usage["total_tokens"] == 150 + 10 + 40
    assert usage["by_actor"]["build:worker"]["calls"] == 1


def test_budget_settled_command_is_empty_when_store_missing(tmp_path: Path) -> None:
    command = build_budget_settled_command(str(tmp_path / "missing.db"), None)
    result = subprocess.run(
        ["/bin/sh", "-c", command], capture_output=True, text=True, check=True
    )
    assert parse_budget_settled_stdout(result.stdout) is None
