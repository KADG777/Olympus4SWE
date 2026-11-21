"""Unit tests for the provider-free RunControl HITL control plane."""

from __future__ import annotations

from pathlib import Path

import pytest

from olympus.core.contracts import (
    AcceptanceCheck,
    AmbiguityItem,
    ApprovalDecision,
    ApprovalTarget,
    BudgetPolicy,
    CheckKind,
    CritiqueNote,
    CritiquePerspective,
    CritiqueSeverity,
    DesignModule,
    DesignSpec,
    GoalSpec,
    RequirementItem,
    RunStatus,
    SpecStatus,
    TaskNode,
    TaskPlan,
    TaskStatus,
)
from olympus.core.control import RunControl
from olympus.core.store import ConcurrencyConflictError, RunStore


def _spec(*, with_blocker: bool = False) -> GoalSpec:
    return GoalSpec(
        spec_id="spec-control",
        goal="Ship a deterministic control plane",
        requirements=[
            RequirementItem(id="R1", description="Do X", acceptance_criteria=["X works"])
        ],
        open_ambiguities=[
            AmbiguityItem(
                id="AMB-1",
                question="Which storage?",
                impact="Choices differ",
                blocking=True,
            )
        ]
        if with_blocker
        else [],
        minority_notes=[
            CritiqueNote(
                id="CRIT-1",
                perspective=CritiquePerspective.QA,
                severity=CritiqueSeverity.BLOCKING,
                summary="Missing rollback path",
            )
        ]
        if with_blocker
        else [],
        status=SpecStatus.AWAITING_HUMAN,
    )


def _design(spec: GoalSpec) -> DesignSpec:
    return DesignSpec(
        design_id="design-control",
        spec_id=spec.spec_id,
        spec_hash=spec.content_hash,
        architecture_summary="One module, one seam.",
        modules=[
            DesignModule(
                id="MOD-1",
                name="Core",
                responsibility="Do the work",
                requirement_refs=["R1"],
                target_paths=["src/core.py"],
            )
        ],
    )


def _plan(spec: GoalSpec, *, task_status: TaskStatus = TaskStatus.WAITING) -> TaskPlan:
    return TaskPlan(
        plan_id="plan-control",
        spec_id=spec.spec_id,
        spec_hash=spec.content_hash,
        tasks=[
            TaskNode(
                task_id="T1",
                title="Implement",
                objective="Write module",
                requirement_refs=["R1"],
                write_scope=["src/core.py"],
                acceptance_checks=[
                    AcceptanceCheck(
                        id="C1",
                        kind=CheckKind.COMMAND,
                        description="pytest",
                        command="pytest -q",
                    )
                ],
                status=task_status,
            )
        ],
    )


def _seeded(
    tmp_path: Path,
    *,
    with_blocker: bool = False,
    task_status: TaskStatus = TaskStatus.WAITING,
    budget_policy: BudgetPolicy | None = None,
) -> tuple[RunStore, RunControl, str, GoalSpec, TaskPlan]:
    store = RunStore(tmp_path / "control.db")
    run_id = store.create_run(title="Control", budget_policy=budget_policy)
    spec = _spec(with_blocker=with_blocker)
    store.save_spec(run_id, spec, actor="align")
    design = _design(spec)
    store.save_design(run_id, design, actor="design")
    plan = _plan(spec, task_status=task_status)
    store.save_plan(run_id, plan, actor="orchestrate")
    return store, RunControl(store), run_id, spec, plan


def test_set_paused_is_idempotent_and_enforces_revision(tmp_path: Path) -> None:
    store, control, run_id, _, _ = _seeded(tmp_path)
    revision = store.get_run(run_id)["state_revision"]

    paused = control.set_paused(
        run_id,
        paused=True,
        operation_id="op-pause-1",
        expected_state_revision=revision,
    )
    assert paused == {
        "run_id": run_id,
        "status": RunStatus.PAUSED.value,
        "state_revision": revision + 1,
        "deduplicated": False,
    }
    snap = store.get_run(run_id)
    assert snap["status"] == RunStatus.PAUSED
    assert snap["state_revision"] == revision + 1

    replay = control.set_paused(
        run_id,
        paused=True,
        operation_id="op-pause-1",
        expected_state_revision=revision,
    )
    assert replay["deduplicated"] is True
    assert replay["state_revision"] == revision + 1
    assert store.get_run(run_id)["state_revision"] == revision + 1

    with pytest.raises(ConcurrencyConflictError):
        control.set_paused(
            run_id,
            paused=False,
            operation_id="op-resume-stale",
            expected_state_revision=revision,
        )

    resumed = control.set_paused(
        run_id,
        paused=False,
        operation_id="op-resume-1",
        expected_state_revision=revision + 1,
    )
    assert resumed["status"] == RunStatus.RUNNING.value
    assert resumed["state_revision"] == revision + 2

    event_types = [event.event_type for event in store.list_events(run_id)]
    assert event_types[-2:] == ["run.paused", "run.resumed"]
    pause_event = next(e for e in store.list_events(run_id) if e.event_type == "run.paused")
    assert pause_event.stage == "orchestrate"
    assert pause_event.payload == {"status": RunStatus.PAUSED.value}


def test_clarify_bumps_version_and_invalidates_approval(tmp_path: Path) -> None:
    store, control, run_id, spec, _ = _seeded(tmp_path, with_blocker=True)
    original_hash = spec.content_hash

    confirmed = control.confirm_spec(
        run_id,
        approver="zeus",
        accept_risk=True,
        reason="ship with known ambiguity",
    )
    assert store.is_approved(
        run_id=run_id,
        target=ApprovalTarget.GOAL_SPEC,
        target_id=confirmed.spec_id,
        target_hash=confirmed.content_hash,
    )

    revision = store.get_run(run_id)["state_revision"]
    clarified = control.clarify_spec(
        run_id,
        answers={"AMB-1": "SQLite"},
        resolutions={"CRIT-1": "Added rollback note"},
        extra_out_of_scope=["No cloud sync"],
        expected_revision=revision,
    )
    assert clarified.version == 2
    assert clarified.content_hash != original_hash
    assert clarified.status == SpecStatus.AWAITING_HUMAN
    assert clarified.open_ambiguities[0].answer == "SQLite"
    assert clarified.minority_notes[0].resolved is True
    assert clarified.out_of_scope == ["No cloud sync"]
    assert clarified.unresolved_blockers() == []

    assert not store.is_approved(
        run_id=run_id,
        target=ApprovalTarget.GOAL_SPEC,
        target_id=clarified.spec_id,
        target_hash=clarified.content_hash,
    )
    clarified_events = [e for e in store.list_events(run_id) if e.event_type == "spec.clarified"]
    assert clarified_events[-1].payload["version"] == 2

    with pytest.raises(ConcurrencyConflictError):
        control.clarify_spec(
            run_id,
            answers={},
            expected_revision=revision,
        )


def test_confirm_spec_enforces_blockers_hash_and_revision(tmp_path: Path) -> None:
    store, control, run_id, spec, _ = _seeded(tmp_path, with_blocker=True)

    with pytest.raises(ValueError, match="unresolved blockers"):
        control.confirm_spec(run_id)

    with pytest.raises(ValueError):
        control.confirm_spec(run_id, accept_risk=True, reason="   ")

    revision = store.get_run(run_id)["state_revision"]
    with pytest.raises(ConcurrencyConflictError, match="target hash no longer matches"):
        control.confirm_spec(
            run_id,
            accept_risk=True,
            reason="risk accepted",
            expected_target_hash="sha256:stale",
        )
    with pytest.raises(ConcurrencyConflictError):
        control.confirm_spec(run_id, accept_risk=True, reason="risk accepted", expected_revision=1)

    confirmed = control.confirm_spec(
        run_id,
        approver="athena",
        accept_risk=True,
        reason="risk accepted",
        expected_revision=revision,
        expected_target_hash=spec.content_hash,
    )
    assert confirmed.status == SpecStatus.CONFIRMED
    assert store.get_run(run_id)["spec"].status == SpecStatus.CONFIRMED
    assert store.is_approved(
        run_id=run_id,
        target=ApprovalTarget.GOAL_SPEC,
        target_id=confirmed.spec_id,
        target_hash=confirmed.content_hash,
    )


def test_approve_design_is_atomic_and_hash_bound(tmp_path: Path) -> None:
    store, control, run_id, _, _ = _seeded(tmp_path)
    design = store.get_run(run_id)["design"]
    assert design is not None
    revision = store.get_run(run_id)["state_revision"]

    with pytest.raises(ConcurrencyConflictError, match="target hash no longer matches"):
        control.approve_design(run_id, expected_target_hash="sha256:stale")
    with pytest.raises(ConcurrencyConflictError):
        control.approve_design(run_id, expected_revision=revision - 1)

    record = control.approve_design(
        run_id,
        approver="athena",
        reason="looks right",
        expected_revision=revision,
        expected_target_hash=design.content_hash,
    )
    assert record.target == ApprovalTarget.DESIGN_SPEC
    assert record.target_id == design.design_id
    assert record.decision == ApprovalDecision.APPROVED
    snap = store.get_run(run_id)
    assert snap["design"].status == SpecStatus.CONFIRMED
    assert snap["state_revision"] == revision + 2
    assert store.is_approved(
        run_id=run_id,
        target=ApprovalTarget.DESIGN_SPEC,
        target_id=design.design_id,
        target_hash=design.content_hash,
    )


def test_approve_plan_requires_approved_upstream_binding(tmp_path: Path) -> None:
    store, control, run_id, spec, plan = _seeded(tmp_path)

    with pytest.raises(ValueError, match="does not match any approved upstream artifact"):
        control.approve_plan(run_id)

    control.confirm_spec(run_id)
    revision = store.get_run(run_id)["state_revision"]

    with pytest.raises(ConcurrencyConflictError, match="target hash no longer matches"):
        control.approve_plan(run_id, expected_target_hash="sha256:stale")
    with pytest.raises(ConcurrencyConflictError):
        control.approve_plan(run_id, expected_revision=revision - 1)

    record = control.approve_plan(
        run_id,
        approver="athena",
        reason="plan ok",
        expected_revision=revision,
        expected_target_hash=plan.content_hash,
    )
    assert record.target == ApprovalTarget.TASK_PLAN
    assert record.target_id == plan.plan_id
    assert store.is_approved(
        run_id=run_id,
        target=ApprovalTarget.TASK_PLAN,
        target_id=plan.plan_id,
        target_hash=plan.content_hash,
    )

    rebound = plan.model_copy(update={"spec_id": spec.spec_id, "spec_hash": "sha256:bogus"})
    store.save_plan(run_id, rebound)
    with pytest.raises(ValueError, match="does not match any approved upstream artifact"):
        control.approve_plan(run_id)


def test_retry_task_is_idempotent_and_state_guarded(tmp_path: Path) -> None:
    store, control, run_id, _, _ = _seeded(tmp_path, task_status=TaskStatus.BLOCKED)
    revision = store.get_run(run_id)["state_revision"]

    result = control.retry_task(
        run_id,
        "T1",
        operation_id="op-retry-1",
        expected_state_revision=revision,
        hint="try the simpler path",
    )
    assert result == {
        "task_id": "T1",
        "status": TaskStatus.READY.value,
        "state_revision": revision + 1,
        "deduplicated": False,
    }
    task = store.get_run(run_id)["plan"].tasks[0]
    assert task.status == TaskStatus.READY
    assert task.review_cycle == 0
    assert task.last_error == "[Human hint]: try the simpler path"

    replay = control.retry_task(
        run_id,
        "T1",
        operation_id="op-retry-1",
        expected_state_revision=revision,
        hint="try the simpler path",
    )
    assert replay["deduplicated"] is True
    assert store.get_run(run_id)["state_revision"] == revision + 1

    with pytest.raises(ConcurrencyConflictError):
        control.retry_task(
            run_id,
            "T1",
            operation_id="op-retry-stale",
            expected_state_revision=revision,
        )
    with pytest.raises(ValueError, match="not blocked/failed"):
        control.retry_task(
            run_id,
            "T1",
            operation_id="op-retry-invalid",
            expected_state_revision=revision + 1,
        )

    event = next(e for e in store.list_events(run_id) if e.event_type == "task.retried_by_human")
    assert event.stage == "control"


def test_accept_task_risk_requires_reason_and_is_idempotent(tmp_path: Path) -> None:
    store, control, run_id, _, _ = _seeded(tmp_path, task_status=TaskStatus.BLOCKED)
    revision = store.get_run(run_id)["state_revision"]

    with pytest.raises(ValueError, match="non-empty reason"):
        control.accept_task_risk(
            run_id,
            "T1",
            operation_id="op-risk-empty",
            expected_state_revision=revision,
            reason="   ",
        )

    result = control.accept_task_risk(
        run_id,
        "T1",
        operation_id="op-risk-1",
        expected_state_revision=revision,
        reason="single admin is acceptable for v1",
    )
    assert result["status"] == "accepted_with_risk"
    task = store.get_run(run_id)["plan"].tasks[0]
    assert task.status == TaskStatus.ACCEPTED
    assert task.accepted_risk_reason == "single admin is acceptable for v1"

    replay = control.accept_task_risk(
        run_id,
        "T1",
        operation_id="op-risk-1",
        expected_state_revision=revision,
        reason="single admin is acceptable for v1",
    )
    assert replay["deduplicated"] is True
    event = next(
        e for e in store.list_events(run_id) if e.event_type == "task.accepted_with_risk"
    )
    assert event.payload["reason"] == "single admin is acceptable for v1"


def test_reassign_task_human_uses_default_note_and_guards_status(tmp_path: Path) -> None:
    store, control, run_id, _, _ = _seeded(tmp_path, task_status=TaskStatus.BLOCKED)
    revision = store.get_run(run_id)["state_revision"]

    result = control.reassign_task_human(
        run_id,
        "T1",
        operation_id="op-reassign-1",
        expected_state_revision=revision,
    )
    assert result["status"] == TaskStatus.READY.value
    task = store.get_run(run_id)["plan"].tasks[0]
    assert task.status == TaskStatus.READY
    assert task.last_error == "[HUMAN_ASSIGNED] 已改派给人工工程师处理（提交后仍须经独立验收）"

    replay = control.reassign_task_human(
        run_id,
        "T1",
        operation_id="op-reassign-1",
        expected_state_revision=revision,
    )
    assert replay["deduplicated"] is True
    event = next(e for e in store.list_events(run_id) if e.event_type == "task.reassigned_human")
    assert event.stage == "control"

    current = store.get_run(run_id)["plan"]
    assert current is not None
    store.save_plan(
        run_id,
        current.model_copy(
            update={
                "tasks": [
                    current.tasks[0].model_copy(update={"status": TaskStatus.ACCEPTED})
                ]
            }
        ),
    )
    with pytest.raises(ValueError, match="cannot be reassigned"):
        control.reassign_task_human(
            run_id,
            "T1",
            operation_id="op-reassign-invalid",
            expected_state_revision=store.get_run(run_id)["state_revision"],
        )


def test_promote_to_execute_uplifts_budget_and_gates_approvals(tmp_path: Path) -> None:
    store, control, run_id, spec, plan = _seeded(
        tmp_path,
        budget_policy=BudgetPolicy(max_llm_calls=50, max_total_tokens=10_000),
    )

    with pytest.raises(ValueError, match="must be approved before execution"):
        control.promote_to_execute(
            run_id,
            target_workflow="full_sdlc_tdd",
            current_stop_after="plan",
            operation_id="op-promote-unapproved",
            expected_state_revision=store.get_run(run_id)["state_revision"],
        )

    control.confirm_spec(run_id)
    control.approve_plan(run_id)
    revision = store.get_run(run_id)["state_revision"]

    promoted = control.promote_to_execute(
        run_id,
        target_workflow="full_sdlc_tdd",
        current_stop_after="plan",
        operation_id="op-promote-1",
        expected_state_revision=revision,
        max_llm_calls=250,
        auto_advance=False,
    )
    assert promoted["status"] == RunStatus.AWAITING_HUMAN.value
    assert promoted["workflow"] == "full_sdlc_tdd"
    assert promoted["from_workflow"] == "product_blueprint"
    assert promoted["budget_max_calls"] == 250
    assert promoted["state_revision"] == revision + 1
    snap = store.get_run(run_id)
    assert snap["status"] == RunStatus.AWAITING_HUMAN
    assert snap["workflow_name"] == "full_sdlc_tdd"
    assert snap["budget_policy"].max_llm_calls == 250
    assert snap["budget_policy"].max_total_tokens == max(len(plan.tasks) * 250_000, 500_000)

    replay = control.promote_to_execute(
        run_id,
        target_workflow="full_sdlc_tdd",
        current_stop_after="plan",
        operation_id="op-promote-1",
        expected_state_revision=revision,
        max_llm_calls=250,
        auto_advance=False,
    )
    assert replay["deduplicated"] is True
    assert store.get_run(run_id)["state_revision"] == revision + 1
    event = next(
        e for e in store.list_events(run_id) if e.event_type == "run.workflow_promoted"
    )
    assert event.payload["to_workflow"] == "full_sdlc_tdd"

    assert spec.content_hash == plan.spec_hash


def test_promote_to_execute_rejects_completed_and_active_runs(tmp_path: Path) -> None:
    store, control, run_id, _, _ = _seeded(tmp_path)
    control.confirm_spec(run_id)
    control.approve_plan(run_id)

    store.set_run_status(run_id, RunStatus.RUNNING, actor="chronos", stage="graph")
    revision = store.get_run(run_id)["state_revision"]
    with pytest.raises(ValueError, match="currently active"):
        control.promote_to_execute(
            run_id,
            target_workflow="full_sdlc_tdd",
            current_stop_after="plan",
            operation_id="op-promote-active",
            expected_state_revision=revision,
        )

    store.set_run_status(run_id, RunStatus.COMPLETED, actor="chronos", stage="graph")
    revision = store.get_run(run_id)["state_revision"]
    with pytest.raises(ValueError, match="already completed execution"):
        control.promote_to_execute(
            run_id,
            target_workflow="full_sdlc_tdd",
            current_stop_after="execute",
            operation_id="op-promote-done",
            expected_state_revision=revision,
        )


def test_promote_to_execute_auto_advances_to_running(tmp_path: Path) -> None:
    store, control, run_id, _, _ = _seeded(tmp_path)
    control.confirm_spec(run_id)
    control.approve_plan(run_id)

    promoted = control.promote_to_execute(
        run_id,
        target_workflow="full_sdlc_tdd",
        current_stop_after="plan",
        operation_id="op-promote-auto",
        expected_state_revision=store.get_run(run_id)["state_revision"],
        auto_advance=True,
    )
    assert promoted["status"] == RunStatus.RUNNING.value
    assert store.get_run(run_id)["status"] == RunStatus.RUNNING


def test_prepare_workflow_retry_only_transitions_failed_runs(tmp_path: Path) -> None:
    store, control, run_id, _, _ = _seeded(tmp_path)
    revision = store.get_run(run_id)["state_revision"]

    control.prepare_workflow_retry(run_id, expected_revision=revision)
    assert store.get_run(run_id)["status"] != RunStatus.RUNNING
    assert store.get_run(run_id)["state_revision"] == revision

    store.set_run_status(run_id, RunStatus.FAILED, actor="chronos", stage="graph")
    revision = store.get_run(run_id)["state_revision"]
    with pytest.raises(ConcurrencyConflictError):
        control.prepare_workflow_retry(run_id, expected_revision=revision - 1)

    control.prepare_workflow_retry(run_id, expected_revision=revision, actor="athena")
    snap = store.get_run(run_id)
    assert snap["status"] == RunStatus.RUNNING
    event = store.list_events(run_id)[-1]
    assert event.event_type == "run.status_changed"
    assert event.stage == "graph"
    assert event.actor == "athena"
    assert event.payload["action"] == "retry_workflow"
