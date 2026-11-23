"""Unit tests for SQLite RunStore invariants: events, leases, approvals, idempotency."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from olympus.core.contracts import (
    AcceptanceCheck,
    ApprovalDecision,
    ApprovalRecord,
    ApprovalTarget,
    BudgetLedger,
    BudgetPolicy,
    CheckKind,
    GoalSpec,
    RequirementItem,
    RunMode,
    RunStatus,
    TaskNode,
    TaskPlan,
    TaskStatus,
)
from olympus.core.store import ConcurrencyConflictError, RunStore


def _make_plan(spec: GoalSpec) -> TaskPlan:
    return TaskPlan(
        spec_id=spec.spec_id,
        spec_hash=spec.content_hash,
        tasks=[
            TaskNode(
                task_id="T1",
                title="Implement",
                objective="Write module",
                requirement_refs=["R1"],
                write_scope=["src/mod.py"],
                acceptance_checks=[
                    AcceptanceCheck(
                        id="C1",
                        kind=CheckKind.COMMAND,
                        description="pytest",
                        command="pytest -q",
                    )
                ],
            )
        ],
    )


def test_store_events_optimistic_revision_and_replay(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "olympus.db")
    rid = store.create_run(title="Demo", mode=RunMode.SIMULATION)

    rev = store.set_run_status(rid, RunStatus.ALIGNING, expected_revision=1)
    assert rev == 2

    with pytest.raises(ConcurrencyConflictError):
        store.set_run_status(rid, RunStatus.RUNNING, expected_revision=1)

    events = store.list_events(rid)
    assert [e.event_seq for e in events] == [1, 2]
    assert [e.event_type for e in events] == ["run.created", "run.status_changed"]
    assert [e.event_seq for e in store.list_events(rid, after_seq=1)] == [2]


def test_store_lease_generation_rejects_stale_updates(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "olympus.db")
    rid = store.create_run(title="Lease test")
    spec = GoalSpec(
        goal="Goal",
        requirements=[RequirementItem(id="R1", description="Do X", acceptance_criteria=["OK"])],
    )
    store.save_spec(rid, spec)
    plan = _make_plan(spec)
    store.save_plan(rid, plan)

    leased1 = store.acquire_task_lease(rid, "T1")
    assert leased1.lease_generation == 1
    assert leased1.status == TaskStatus.RUNNING

    # Simulate reset to READY and re-lease by another attempt
    store.update_task_with_lease(
        rid,
        "T1",
        expected_lease_generation=1,
        mutator=lambda t: t.model_copy(update={"status": TaskStatus.READY}),
        actor="chronos",
        event_type="task.requeued",
    )
    leased2 = store.acquire_task_lease(rid, "T1")
    assert leased2.lease_generation == 2

    # Stale worker with lease_generation=1 must be rejected
    with pytest.raises(ConcurrencyConflictError):
        store.update_task_with_lease(
            rid,
            "T1",
            expected_lease_generation=1,
            mutator=lambda t: t.model_copy(update={"status": TaskStatus.ACCEPTED}),
            actor="stale-worker",
            event_type="task.accepted",
        )


def test_store_approval_invalidated_on_hash_or_commit_change(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "olympus.db")
    rid = store.create_run(title="Approval test")
    spec_v1 = GoalSpec(
        spec_id="spec-a",
        version=1,
        goal="Goal v1",
        requirements=[RequirementItem(id="R1", description="Req", acceptance_criteria=["AC"])],
    )
    store.save_spec(rid, spec_v1)

    store.record_approval(
        ApprovalRecord(
            run_id=rid,
            target=ApprovalTarget.GOAL_SPEC,
            target_id=spec_v1.spec_id,
            target_hash=spec_v1.content_hash,
            commit_sha="deadbeef",
            approver="zeus",
            decision=ApprovalDecision.APPROVED,
        )
    )
    assert store.is_approved(
        run_id=rid,
        target=ApprovalTarget.GOAL_SPEC,
        target_id=spec_v1.spec_id,
        target_hash=spec_v1.content_hash,
        commit_sha="deadbeef",
    )
    # New commit invalidates commit-bound approval
    assert not store.is_approved(
        run_id=rid,
        target=ApprovalTarget.GOAL_SPEC,
        target_id=spec_v1.spec_id,
        target_hash=spec_v1.content_hash,
        commit_sha="cafebabe",
    )
    # Content change invalidates spec approval
    spec_v2 = spec_v1.model_copy(update={"version": 2, "out_of_scope": ["extra"]})
    assert spec_v2.content_hash != spec_v1.content_hash
    assert not store.is_approved(
        run_id=rid,
        target=ApprovalTarget.GOAL_SPEC,
        target_id=spec_v2.spec_id,
        target_hash=spec_v2.content_hash,
    )


def test_store_idempotent_commands_and_monotonic_budget(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "olympus.db")
    rid = store.create_run(title="Cmd test", budget_policy=BudgetPolicy(max_llm_calls=1))

    calls = 0

    def handler(conn, cur_rev):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        new_rev = cur_rev + 1
        conn.execute(
            "UPDATE runs SET status = ?, state_revision = ? WHERE run_id = ?",
            (RunStatus.PAUSED.value, new_rev, rid),
        )
        store.append_event_in_tx(
            conn,
            run_id=rid,
            stage="control",
            event_type="run.paused",
            actor="zeus",
            payload={},
            state_revision=new_rev,
        )
        return {"status": "paused"}

    res1 = store.apply_idempotent_command(
        operation_id="op-1",
        run_id=rid,
        actor="zeus",
        action="pause",
        expected_state_revision=1,
        payload={},
        handler=handler,
    )
    res2 = store.apply_idempotent_command(
        operation_id="op-1",
        run_id=rid,
        actor="zeus",
        action="pause",
        expected_state_revision=1,
        payload={},
        handler=handler,
    )
    assert calls == 1
    assert res1["deduplicated"] is False
    assert res2["deduplicated"] is True
    assert res2["status"] == "paused"

    # Budget exhaustion across reload
    store.reserve_llm_call(rid, actor="align", purpose="probe")
    store.settle_llm_call(rid, actor="align", input_tokens=10, output_tokens=5)
    with pytest.raises(ValueError):
        store.reserve_llm_call(rid, actor="align", purpose="exceed")


def test_store_accounts_cache_tokens_and_reads_legacy_ledger_json(tmp_path: Path) -> None:
    db_path = tmp_path / "olympus.db"
    store = RunStore(db_path)
    rid = store.create_run(
        title="Cache ledger", budget_policy=BudgetPolicy(max_total_tokens=500)
    )
    legacy_ledger = {
        "llm_calls_used": 0,
        "llm_calls_reserved": 0,
        "input_tokens": 10,
        "output_tokens": 5,
        "unknown_outcome_calls": 0,
    }
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE runs SET budget_ledger_json = ? WHERE run_id = ?",
            (json.dumps(legacy_ledger), rid),
        )

    assert store.get_run(rid)["budget_ledger"].total_tokens == 15

    store.reserve_llm_call(rid, actor="align", purpose="cached call")
    ledger = store.settle_llm_call(
        rid,
        actor="align",
        input_tokens=20,
        output_tokens=10,
        cache_read_tokens=300,
        cache_write_tokens=100,
    )
    assert (ledger.cache_read_tokens, ledger.cache_write_tokens) == (300, 100)
    assert ledger.total_tokens == 445
    assert ledger.can_reserve(BudgetPolicy(max_total_tokens=445)) is False

    over_budget = store.record_external_llm_usage(
        rid,
        actor="pi",
        purpose="pi_step_1",
        input_tokens=30,
        output_tokens=10,
        cache_read_tokens=60,
        cache_write_tokens=10,
    )
    assert over_budget is True
    ledger = store.get_run(rid)["budget_ledger"]
    assert ledger.llm_calls_used == 2
    assert (ledger.cache_read_tokens, ledger.cache_write_tokens) == (360, 110)
    assert ledger.total_tokens == 555


def test_workspace_registry_roundtrip_and_unregister_is_metadata_only(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "olympus.db")
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "README.md").write_text("# demo\n", encoding="utf-8")

    first = store.register_workspace(
        name="repo", root_path=repo, is_git_repo=False
    )
    assert first["workspace_id"].startswith("ws-")
    assert first["root_path"] == str(repo.resolve())
    assert first["is_git_repo"] is False
    assert first["default_branch"] is None

    # Registering the same canonical root is idempotent, not a duplicate row.
    again = store.register_workspace(
        name="renamed", root_path=repo, is_git_repo=True, default_branch="main"
    )
    assert again["workspace_id"] == first["workspace_id"]
    assert again["name"] == "repo"
    assert len(store.list_workspaces()) == 1

    assert store.get_workspace(first["workspace_id"]) == first
    assert store.get_workspace_by_root(repo) == first
    assert store.get_workspace_by_root(tmp_path / "elsewhere") is None

    assert store.delete_workspace(first["workspace_id"]) is True
    assert store.delete_workspace(first["workspace_id"]) is False
    assert store.list_workspaces() == []
    assert (repo / "README.md").read_text(encoding="utf-8") == "# demo\n"


def test_legacy_database_migrates_workspace_scope_columns(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy.db"
    legacy_schema = """
    CREATE TABLE runs (
        run_id TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        mode TEXT NOT NULL,
        status TEXT NOT NULL,
        state_revision INTEGER NOT NULL,
        budget_policy_json TEXT NOT NULL,
        budget_ledger_json TEXT NOT NULL,
        spec_json TEXT,
        plan_json TEXT,
        workflow_name TEXT NOT NULL DEFAULT 'repo_fast_patch',
        provider_config_json TEXT,
        execution_drivers_json TEXT,
        workspace_root TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );
    """
    with sqlite3.connect(db_path) as conn:
        conn.executescript(legacy_schema)
        conn.execute(
            """
            INSERT INTO runs (
                run_id, title, mode, status, state_revision,
                budget_policy_json, budget_ledger_json, workflow_name,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "run-legacy",
                "Legacy run",
                "execution",
                "created",
                1,
                BudgetPolicy().model_dump_json(),
                BudgetLedger().model_dump_json(),
                "product_blueprint",
                "2026-01-01T00:00:00+00:00",
                "2026-01-01T00:00:00+00:00",
            ),
        )

    store = RunStore(db_path)
    snap = store.get_run("run-legacy")
    assert snap["workspace_id"] is None
    assert snap["workspace_mode"] == "brownfield"
    assert snap["task_type"] == "feature"
    assert snap["base_ref"] is None
    assert snap["scope_paths"] == []

    with sqlite3.connect(db_path) as conn:
        run_columns = {row[1] for row in conn.execute("PRAGMA table_info(runs)")}
        workspace_columns = {row[1] for row in conn.execute("PRAGMA table_info(workspaces)")}
    assert {
        "workspace_id",
        "workspace_mode",
        "task_type",
        "base_ref",
        "scope_paths_json",
    } <= run_columns
    assert {
        "workspace_id",
        "name",
        "root_path",
        "is_git_repo",
        "default_branch",
        "created_at",
        "updated_at",
    } == workspace_columns

    rid = store.create_run(
        title="Scoped",
        workspace_id="ws-1",
        workspace_mode="greenfield",
        task_type="fix",
        base_ref="main",
        scope_paths=["src/app.py"],
    )
    created = store.get_run(rid)
    assert created["workspace_id"] == "ws-1"
    assert created["workspace_mode"] == "greenfield"
    assert created["task_type"] == "fix"
    assert created["base_ref"] == "main"
    assert created["scope_paths"] == ["src/app.py"]
