"""Seam tests for the canonical workflow gate/resume/completion module."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from olympus.core.contracts import (
    AcceptanceCheck,
    AmbiguityItem,
    ApprovalDecision,
    ApprovalRecord,
    ApprovalTarget,
    CheckKind,
    DesignModule,
    DesignSpec,
    GoalSpec,
    RequirementItem,
    RunEvent,
    RunStatus,
    TaskNode,
    TaskPlan,
    TaskStatus,
)
from olympus.core.store import RunStore
from olympus.graph.engine import GraphPaused
from olympus.graph.loader import (
    WorkflowDefinition,
    WorkflowNodeDef,
    WorkflowRuntime,
    _node_human_gate,
    list_builtin_workflows,
    load_workflow,
)
from olympus.graph.progress import (
    GATE_SPECS,
    build_workflow_view,
    is_artifact_approved,
    is_blueprint_completed,
    resolve_resume_node,
)
from olympus.providers.base import MockLLMProvider

RUN_ID = "run-progress"


@pytest.fixture()
def store(tmp_path: Path) -> RunStore:
    return RunStore(tmp_path / "olympus.db")


def _new_run(store: RunStore, workflow_name: str = "product_blueprint") -> str:
    return store.create_run(title="Progress seam run", workflow_name=workflow_name, run_id=RUN_ID)


def _make_spec(*, blocking: bool = False) -> GoalSpec:
    ambiguities = (
        [
            AmbiguityItem(
                id="A1",
                question="Which gate wins when an artifact is already approved?",
                impact="Resume entry point",
                options=["approved artifact", "paused gate"],
                blocking=True,
            )
        ]
        if blocking
        else []
    )
    return GoalSpec(
        spec_id="spec-progress",
        goal="Ship the canonical workflow progress model",
        requirements=[
            RequirementItem(
                id="R1",
                description="Gates reconcile with artifact approvals",
                acceptance_criteria=["An approved gate is reported as completed"],
            )
        ],
        open_ambiguities=ambiguities,
    )


def _make_design(spec: GoalSpec) -> DesignSpec:
    return DesignSpec(
        design_id="design-progress",
        spec_id=spec.spec_id,
        spec_hash=spec.content_hash,
        architecture_summary="A single progress seam owns gate decisions.",
        modules=[
            DesignModule(
                id="M1",
                name="Progress",
                responsibility="Resolve gates and resume entry points",
                requirement_refs=["R1"],
                target_paths=["src/olympus/graph/progress.py"],
            )
        ],
    )


def _make_plan(spec: GoalSpec, *, status: TaskStatus = TaskStatus.WAITING) -> TaskPlan:
    return TaskPlan(
        plan_id="plan-progress",
        spec_id=spec.spec_id,
        spec_hash=spec.content_hash,
        tasks=[
            TaskNode(
                task_id="T1",
                title="Implement progress",
                objective="Wire the canonical gate table",
                requirement_refs=["R1"],
                write_scope=["src/olympus/graph/progress.py"],
                acceptance_checks=[
                    AcceptanceCheck(
                        id="T1-c",
                        kind=CheckKind.COMMAND,
                        description="import progress",
                        command="python -c 'pass'",
                    )
                ],
                status=status,
            )
        ],
    )


def _approve(store: RunStore, run_id: str, gate: str, artifact: Any) -> None:
    gate_spec = GATE_SPECS[gate]
    store.record_approval(
        ApprovalRecord(
            run_id=run_id,
            target=gate_spec["target"],
            target_id=getattr(artifact, gate_spec["id_attr"]),
            target_hash=artifact.content_hash,
            approver="zeus",
            decision=ApprovalDecision.APPROVED,
        )
    )


def _event(seq: int, event_type: str, payload: dict[str, Any]) -> RunEvent:
    return RunEvent(
        event_seq=seq,
        run_id=RUN_ID,
        stage="graph",
        event_type=event_type,
        actor="supervisor",
        payload=payload,
        state_revision=seq,
    )


# --------------------------------------------------------------- canonical table


def test_gate_specs_are_the_canonical_table() -> None:
    assert list(GATE_SPECS) == ["spec", "design", "plan"]
    assert [GATE_SPECS[gate]["node"] for gate in GATE_SPECS] == [
        "gate_spec",
        "gate_design",
        "gate_plan",
    ]
    assert [GATE_SPECS[gate]["snapshot_key"] for gate in GATE_SPECS] == [
        "spec",
        "design",
        "plan",
    ]
    assert [GATE_SPECS[gate]["target"] for gate in GATE_SPECS] == [
        ApprovalTarget.GOAL_SPEC,
        ApprovalTarget.DESIGN_SPEC,
        ApprovalTarget.TASK_PLAN,
    ]
    assert [GATE_SPECS[gate]["id_attr"] for gate in GATE_SPECS] == [
        "spec_id",
        "design_id",
        "plan_id",
    ]
    assert [GATE_SPECS[gate]["label"] for gate in GATE_SPECS] == [
        "GoalSpec",
        "DesignSpec",
        "TaskPlan",
    ]


def test_gate_specs_match_builtin_workflow_nodes() -> None:
    node_names = {
        node.name
        for workflow in list_builtin_workflows()
        for node in load_workflow(workflow).nodes
    }
    assert {gate_spec["node"] for gate_spec in GATE_SPECS.values()} <= node_names


def test_is_artifact_approved_is_hash_bound(store: RunStore) -> None:
    run_id = _new_run(store)
    assert is_artifact_approved(store, run_id, "spec", store.get_run(run_id)) is False

    spec = _make_spec()
    store.save_spec(run_id, spec)
    assert is_artifact_approved(store, run_id, "spec", store.get_run(run_id)) is False

    _approve(store, run_id, "spec", spec)
    assert is_artifact_approved(store, run_id, "spec", store.get_run(run_id)) is True

    revised = spec.model_copy(update={"version": 2, "goal": "Ship the revised progress model"})
    store.save_spec(run_id, revised)
    assert is_artifact_approved(store, run_id, "spec", store.get_run(run_id)) is False


# ------------------------------------------------------------------- resume node


def test_resolve_resume_node_starts_at_intake(store: RunStore) -> None:
    run_id = _new_run(store)
    definition = load_workflow("product_blueprint")
    assert resolve_resume_node(definition, store.get_run(run_id), store=store) == "intake"


def test_resolve_resume_node_pauses_on_unapproved_spec(store: RunStore) -> None:
    run_id = _new_run(store)
    store.save_spec(run_id, _make_spec())
    definition = load_workflow("product_blueprint")
    assert resolve_resume_node(definition, store.get_run(run_id), store=store) == "gate_spec"


def test_resolve_resume_node_enters_design_after_spec_approval(store: RunStore) -> None:
    run_id = _new_run(store)
    spec = _make_spec()
    store.save_spec(run_id, spec)
    _approve(store, run_id, "spec", spec)
    definition = load_workflow("product_blueprint")
    assert resolve_resume_node(definition, store.get_run(run_id), store=store) == "design"


def test_resolve_resume_node_pauses_on_unapproved_design(store: RunStore) -> None:
    run_id = _new_run(store)
    spec = _make_spec()
    store.save_spec(run_id, spec)
    _approve(store, run_id, "spec", spec)
    store.save_design(run_id, _make_design(spec))
    definition = load_workflow("product_blueprint")
    assert resolve_resume_node(definition, store.get_run(run_id), store=store) == "gate_design"


def test_resolve_resume_node_enters_plan_after_design_approval(store: RunStore) -> None:
    run_id = _new_run(store)
    spec = _make_spec()
    design = _make_design(spec)
    store.save_spec(run_id, spec)
    _approve(store, run_id, "spec", spec)
    store.save_design(run_id, design)
    _approve(store, run_id, "design", design)
    definition = load_workflow("product_blueprint")
    assert resolve_resume_node(definition, store.get_run(run_id), store=store) == "plan"


def test_resolve_resume_node_reenters_terminal_gate_for_blueprint(store: RunStore) -> None:
    run_id = _new_run(store)
    spec = _make_spec()
    design = _make_design(spec)
    plan = _make_plan(spec)
    store.save_spec(run_id, spec)
    _approve(store, run_id, "spec", spec)
    store.save_design(run_id, design)
    _approve(store, run_id, "design", design)
    store.save_plan(run_id, plan)
    _approve(store, run_id, "plan", plan)
    definition = load_workflow("product_blueprint")
    assert resolve_resume_node(definition, store.get_run(run_id), store=store) == "gate_plan"


def test_resolve_resume_node_enters_execute_when_workflow_supports_it(store: RunStore) -> None:
    run_id = _new_run(store)
    spec = _make_spec()
    design = _make_design(spec)
    plan = _make_plan(spec)
    store.save_spec(run_id, spec)
    _approve(store, run_id, "spec", spec)
    store.save_design(run_id, design)
    _approve(store, run_id, "design", design)
    store.save_plan(run_id, plan)
    _approve(store, run_id, "plan", plan)
    definition = load_workflow("full_sdlc_tdd")
    assert resolve_resume_node(definition, store.get_run(run_id), store=store) == "execute"


def test_resolve_resume_node_skips_missing_design_nodes(store: RunStore) -> None:
    run_id = _new_run(store)
    spec = _make_spec()
    store.save_spec(run_id, spec)
    _approve(store, run_id, "spec", spec)
    definition = load_workflow("repo_fast_patch")
    assert resolve_resume_node(definition, store.get_run(run_id), store=store) == "plan"


def test_resolve_resume_node_returns_none_for_unroutable_design(store: RunStore) -> None:
    run_id = _new_run(store)
    spec = _make_spec()
    store.save_spec(run_id, spec)
    _approve(store, run_id, "spec", spec)
    store.save_design(run_id, _make_design(spec))
    definition = load_workflow("repo_fast_patch")
    assert resolve_resume_node(definition, store.get_run(run_id), store=store) is None


def test_resolve_resume_node_returns_none_without_entry_nodes(store: RunStore) -> None:
    run_id = _new_run(store)
    definition = WorkflowDefinition(
        name="bare",
        nodes=[WorkflowNodeDef(name="only", type="align_goal")],
    )
    assert resolve_resume_node(definition, store.get_run(run_id), store=store) is None


def test_resolve_resume_node_falls_back_to_align(store: RunStore) -> None:
    run_id = _new_run(store)
    store.save_spec(run_id, _make_spec())
    definition = WorkflowDefinition(
        name="align_only",
        nodes=[WorkflowNodeDef(name="align", type="align_goal")],
    )
    assert resolve_resume_node(definition, store.get_run(run_id), store=store) == "align"


def test_resolve_resume_node_returns_none_without_terminal_gate(store: RunStore) -> None:
    run_id = _new_run(store)
    spec = _make_spec()
    design = _make_design(spec)
    plan = _make_plan(spec)
    store.save_spec(run_id, spec)
    _approve(store, run_id, "spec", spec)
    store.save_design(run_id, design)
    _approve(store, run_id, "design", design)
    store.save_plan(run_id, plan)
    _approve(store, run_id, "plan", plan)
    definition = WorkflowDefinition(
        name="no_gates",
        nodes=[WorkflowNodeDef(name="execute", type="execute_task_dag")],
    )
    assert resolve_resume_node(definition, store.get_run(run_id), store=store) is None


# ------------------------------------------------------------------- workflow view


@pytest.mark.parametrize("gate", ["spec", "design", "plan"])
def test_build_workflow_view_reconciles_approved_gates(gate: str) -> None:
    node = GATE_SPECS[gate]["node"]
    events = [
        _event(1, "graph.node_started", {"node": node}),
        _event(2, "graph.paused", {"node": node, "reason": "awaiting"}),
    ]
    view = build_workflow_view(
        "product_blueprint",
        events,
        RunStatus.AWAITING_HUMAN,
        **{f"{gate}_approved": True},
    )
    assert view["node_statuses"][node] == "completed"
    assert view["paused"] == {"node": node, "reason": "awaiting"}


def test_build_workflow_view_replays_a_pause_without_approval() -> None:
    events = [
        _event(1, "graph.node_started", {"node": "intake"}),
        _event(2, "graph.node_finished", {"node": "intake"}),
        _event(3, "graph.node_started", {"node": "gate_spec"}),
        _event(
            4,
            "graph.paused",
            {"node": "gate_spec", "reason": "GoalSpec spec-progress is awaiting human approval"},
        ),
    ]
    view = build_workflow_view("product_blueprint", events, RunStatus.AWAITING_HUMAN)

    assert view["name"] == "product_blueprint"
    assert view["topology"] == "modular_product"
    assert view["stop_after"] == "plan"
    assert view["graph"] is not None
    assert view["node_statuses"]["intake"] == "completed"
    assert view["node_statuses"]["gate_spec"] == "paused"
    assert view["paused"] == {
        "node": "gate_spec",
        "reason": "GoalSpec spec-progress is awaiting human approval",
    }
    assert view["current_node"] is None
    assert view["failed_node"] is None
    assert view["finished"] is False


def test_build_workflow_view_completed_clears_paused_gates() -> None:
    events = [
        _event(1, "graph.node_started", {"node": "gate_plan"}),
        _event(2, "graph.paused", {"node": "gate_plan", "reason": "awaiting"}),
    ]
    view = build_workflow_view("product_blueprint", events, RunStatus.COMPLETED)

    assert view["node_statuses"]["gate_plan"] == "completed"
    assert view["paused"] is None
    assert view["finished"] is True


def test_build_workflow_view_marks_running_nodes_failed() -> None:
    events = [_event(1, "graph.node_started", {"node": "design"})]
    view = build_workflow_view("product_blueprint", events, RunStatus.FAILED)

    assert view["node_statuses"]["design"] == "failed"
    assert view["failed_node"] == "design"
    assert view["last_error"] == "工作流执行异常中断"
    assert view["finished"] is False


def test_build_workflow_view_uses_recorded_failure_error() -> None:
    events = [
        _event(1, "graph.node_started", {"node": "align"}),
        _event(2, "graph.failed", {"error": "provider timeout"}),
    ]
    view = build_workflow_view("product_blueprint", events, RunStatus.FAILED)

    assert view["node_statuses"]["align"] == "failed"
    assert view["failed_node"] == "align"
    assert view["last_error"] == "provider timeout"


def test_build_workflow_view_unknown_workflow_stays_renderable() -> None:
    view = build_workflow_view("does_not_exist", [], RunStatus.CREATED)

    assert view["graph"] is None
    assert view["topology"] is None
    assert view["stop_after"] is None
    assert view["nodes"] == []
    assert view["node_statuses"] == {}
    assert view["traversed_edges"] == []
    assert view["finished"] is False


# ------------------------------------------------------------- workflow trajectory


def test_build_workflow_view_derives_traversed_edges_and_skips_bypassed_nodes() -> None:
    events = [
        _event(1, "graph.started", {"graph": "full_sdlc_tdd", "start_from": None}),
        _event(2, "graph.node_started", {"node": "intake"}),
        _event(3, "graph.node_finished", {"node": "intake"}),
        _event(4, "graph.routed", {"from": ["intake"], "to": ["align"], "wave": 1}),
        _event(5, "graph.node_started", {"node": "align"}),
        _event(6, "graph.node_finished", {"node": "align"}),
        _event(7, "graph.routed", {"from": ["align"], "to": ["gate_spec"], "wave": 2}),
        _event(8, "graph.node_started", {"node": "gate_spec"}),
        _event(9, "graph.paused", {"node": "gate_spec", "reason": "human_gate"}),
    ]
    view = build_workflow_view("full_sdlc_tdd", events, RunStatus.AWAITING_HUMAN)

    assert view["traversed_edges"] == [
        {"source": "intake", "target": "align"},
        {"source": "align", "target": "gate_spec"},
    ]
    assert view["node_statuses"]["critic"] == "skipped"
    node_entries = {node["name"]: node["status"] for node in view["nodes"]}
    assert node_entries["critic"] == "skipped"
    for downstream in (
        "design",
        "validate_design",
        "gate_design",
        "plan",
        "validate_plan",
        "gate_plan",
        "execute",
    ):
        assert view["node_statuses"][downstream] == "pending"


def test_build_workflow_view_skips_alignment_chain_on_design_ready_triage() -> None:
    events = [
        _event(1, "graph.node_started", {"node": "intake"}),
        _event(2, "graph.node_finished", {"node": "intake"}),
        _event(3, "graph.routed", {"from": ["intake"], "to": ["validate_design"], "wave": 1}),
        _event(4, "graph.node_started", {"node": "validate_design"}),
        _event(5, "graph.node_finished", {"node": "validate_design"}),
        _event(6, "graph.routed", {"from": ["validate_design"], "to": ["gate_design"], "wave": 2}),
        _event(7, "graph.node_started", {"node": "gate_design"}),
        _event(8, "graph.paused", {"node": "gate_design", "reason": "human_gate"}),
    ]
    view = build_workflow_view("full_sdlc_tdd", events, RunStatus.AWAITING_HUMAN)

    assert {"source": "intake", "target": "validate_design"} in view["traversed_edges"]
    assert {"source": "validate_design", "target": "gate_design"} in view["traversed_edges"]
    for bypassed in ("align", "critic", "gate_spec", "design"):
        assert view["node_statuses"][bypassed] == "skipped"
    assert view["node_statuses"]["validate_design"] == "completed"
    assert view["node_statuses"]["gate_design"] == "paused"
    for unreached in ("plan", "validate_plan", "gate_plan", "execute"):
        assert view["node_statuses"][unreached] == "pending"


def test_build_workflow_view_falls_back_to_node_transitions_without_routed_events() -> None:
    events = [
        _event(1, "graph.node_started", {"node": "intake"}),
        _event(2, "graph.node_finished", {"node": "intake"}),
        _event(3, "graph.node_started", {"node": "align"}),
        _event(4, "graph.node_finished", {"node": "align"}),
        _event(5, "graph.node_started", {"node": "critic"}),
        _event(6, "graph.node_finished", {"node": "critic"}),
        _event(7, "graph.node_started", {"node": "gate_spec"}),
        _event(8, "graph.paused", {"node": "gate_spec", "reason": "human_gate"}),
    ]
    view = build_workflow_view("product_blueprint", events, RunStatus.AWAITING_HUMAN)

    assert view["traversed_edges"] == [
        {"source": "intake", "target": "align"},
        {"source": "align", "target": "critic"},
        {"source": "critic", "target": "gate_spec"},
    ]
    assert view["node_statuses"]["critic"] == "completed"
    assert view["node_statuses"]["gate_spec"] == "paused"
    assert "skipped" not in view["node_statuses"].values()


def test_build_workflow_view_dedupes_traversed_edges_and_drops_undeclared_pairs() -> None:
    events = [
        _event(1, "graph.routed", {"from": ["intake"], "to": ["align"]}),
        _event(2, "graph.routed", {"from": ["intake"], "to": ["align"]}),
        _event(3, "graph.routed", {"from": ["align"], "to": ["execute"]}),
    ]
    view = build_workflow_view("product_blueprint", events, RunStatus.RUNNING)

    assert view["traversed_edges"] == [{"source": "intake", "target": "align"}]


def test_build_workflow_view_preserves_skipped_after_execute_completes() -> None:
    events = [
        _event(1, "graph.node_started", {"node": "intake"}),
        _event(2, "graph.node_finished", {"node": "intake"}),
        _event(3, "graph.routed", {"from": ["intake"], "to": ["align"], "wave": 1}),
        _event(4, "graph.node_started", {"node": "align"}),
        _event(5, "graph.node_finished", {"node": "align"}),
        _event(6, "graph.routed", {"from": ["align"], "to": ["gate_spec"], "wave": 2}),
        _event(7, "graph.node_started", {"node": "gate_spec"}),
        _event(8, "graph.node_finished", {"node": "gate_spec"}),
        _event(9, "graph.routed", {"from": ["gate_spec"], "to": ["design"], "wave": 3}),
        _event(10, "graph.node_started", {"node": "design"}),
        _event(11, "graph.node_finished", {"node": "design"}),
        _event(12, "graph.routed", {"from": ["design"], "to": ["validate_design"], "wave": 4}),
        _event(13, "graph.node_started", {"node": "validate_design"}),
        _event(14, "graph.node_finished", {"node": "validate_design"}),
        _event(15, "graph.routed", {"from": ["validate_design"], "to": ["gate_design"], "wave": 5}),
        _event(16, "graph.node_started", {"node": "gate_design"}),
        _event(17, "graph.node_finished", {"node": "gate_design"}),
        _event(18, "graph.routed", {"from": ["gate_design"], "to": ["plan"], "wave": 6}),
        _event(19, "graph.node_started", {"node": "plan"}),
        _event(20, "graph.node_finished", {"node": "plan"}),
        _event(21, "graph.routed", {"from": ["plan"], "to": ["validate_plan"], "wave": 7}),
        _event(22, "graph.node_started", {"node": "validate_plan"}),
        _event(23, "graph.node_finished", {"node": "validate_plan"}),
        _event(24, "graph.routed", {"from": ["validate_plan"], "to": ["gate_plan"], "wave": 8}),
        _event(25, "graph.node_started", {"node": "gate_plan"}),
        _event(26, "graph.node_finished", {"node": "gate_plan"}),
        _event(27, "graph.routed", {"from": ["gate_plan"], "to": ["execute"], "wave": 9}),
        _event(28, "graph.node_started", {"node": "execute"}),
        _event(29, "graph.node_finished", {"node": "execute"}),
        _event(30, "graph.finished", {"graph": "full_sdlc_tdd", "steps": 9}),
    ]
    view = build_workflow_view("full_sdlc_tdd", events, RunStatus.COMPLETED)

    assert view["finished"] is True
    assert view["node_statuses"]["critic"] == "skipped"
    assert view["node_statuses"]["execute"] == "completed"
    assert {"source": "align", "target": "gate_spec"} in view["traversed_edges"]
    assert {"source": "gate_plan", "target": "execute"} in view["traversed_edges"]


def test_build_workflow_view_keeps_future_nodes_pending_after_failed_frontier() -> None:
    events = [
        _event(1, "graph.node_started", {"node": "intake"}),
        _event(2, "graph.node_finished", {"node": "intake"}),
        _event(3, "graph.routed", {"from": ["intake"], "to": ["align"], "wave": 1}),
        _event(4, "graph.node_started", {"node": "align"}),
        _event(5, "graph.node_finished", {"node": "align"}),
        _event(6, "graph.routed", {"from": ["align"], "to": ["critic"], "wave": 2}),
        _event(7, "graph.node_started", {"node": "critic"}),
        _event(8, "graph.node_error", {"node": "critic", "error": "provider timeout"}),
        _event(9, "graph.failed", {"error": "provider timeout"}),
    ]
    view = build_workflow_view("full_sdlc_tdd", events, RunStatus.FAILED)

    assert view["node_statuses"]["critic"] == "failed"
    for downstream in ("gate_spec", "design", "validate_design", "execute"):
        assert view["node_statuses"][downstream] == "pending"


def test_build_workflow_view_records_pause_resume_gate_transition() -> None:
    events = [
        _event(1, "graph.started", {"graph": "full_sdlc_tdd", "start_from": None}),
        _event(2, "graph.node_started", {"node": "intake"}),
        _event(3, "graph.node_finished", {"node": "intake"}),
        _event(4, "graph.routed", {"from": ["intake"], "to": ["align"], "wave": 1}),
        _event(5, "graph.node_started", {"node": "align"}),
        _event(6, "graph.node_finished", {"node": "align"}),
        _event(7, "graph.routed", {"from": ["align"], "to": ["gate_spec"], "wave": 2}),
        _event(8, "graph.node_started", {"node": "gate_spec"}),
        _event(9, "graph.paused", {"node": "gate_spec", "reason": "human_gate"}),
        _event(10, "graph.started", {"graph": "full_sdlc_tdd", "start_from": "design"}),
        _event(11, "graph.node_started", {"node": "design"}),
        _event(12, "graph.node_finished", {"node": "design"}),
    ]
    view = build_workflow_view(
        "full_sdlc_tdd",
        events,
        RunStatus.RUNNING,
        spec_approved=True,
    )

    assert view["node_statuses"]["gate_spec"] == "completed"
    assert view["node_statuses"]["critic"] == "skipped"
    assert {"source": "gate_spec", "target": "design"} in view["traversed_edges"]
    assert view["node_statuses"]["validate_design"] == "pending"


def test_derive_skipped_ignores_back_edges_in_cyclic_graphs() -> None:
    from olympus.graph.progress import _derive_skipped

    statuses = {"a": "completed", "b": "paused", "c": "pending"}
    declared = [("a", "b"), ("b", "c"), ("c", "a")]
    traversed = [{"source": "a", "target": "b"}]

    assert _derive_skipped(statuses, declared, traversed, starts=["a"]) == set()


def test_derive_skipped_is_independent_of_node_declaration_order() -> None:
    from olympus.graph.progress import _derive_skipped

    declared = [("gate_spec", "optional"), ("optional", "terminal"), ("gate_spec", "terminal")]
    traversed = [{"source": "gate_spec", "target": "terminal"}]
    for order in (
        ("gate_spec", "optional", "terminal"),
        ("gate_spec", "terminal", "optional"),
    ):
        statuses = {
            name: ("pending" if name == "optional" else "completed")
            for name in order
        }
        assert _derive_skipped(statuses, declared, traversed, starts=["gate_spec"]) == {"optional"}


def test_derive_traversed_edges_preserves_declared_self_loop() -> None:
    from olympus.graph.progress import _declared_edges, _derive_traversed_edges

    description = {
        "edges": [{"source": "gate_spec", "target": "gate_spec", "conditional": False}],
    }
    declared = _declared_edges(description)
    events = [
        _event(1, "graph.routed", {"from": ["gate_spec"], "to": ["gate_spec"], "wave": 1}),
        _event(2, "graph.routed", {"from": ["gate_spec"], "to": ["gate_spec"], "wave": 2}),
    ]
    assert _derive_traversed_edges(events, declared) == [
        {"source": "gate_spec", "target": "gate_spec"},
    ]


# -------------------------------------------------------------- blueprint completion


@pytest.mark.parametrize(
    ("plan_status", "plan_approved", "run_status", "stop_after", "expected"),
    [
        (TaskStatus.WAITING, True, RunStatus.COMPLETED, "plan", True),
        (TaskStatus.WAITING, True, RunStatus.COMPLETED, "design", True),
        (TaskStatus.ACCEPTED, True, RunStatus.COMPLETED, "plan", False),
        (TaskStatus.CANCELLED, True, RunStatus.COMPLETED, "plan", False),
        (TaskStatus.WAITING, False, RunStatus.COMPLETED, "plan", False),
        (TaskStatus.WAITING, True, RunStatus.RUNNING, "plan", False),
        (TaskStatus.WAITING, True, RunStatus.COMPLETED, "execute", False),
        (TaskStatus.WAITING, True, RunStatus.COMPLETED, None, False),
    ],
)
def test_is_blueprint_completed_matrix(
    plan_status: TaskStatus,
    plan_approved: bool,
    run_status: RunStatus,
    stop_after: str | None,
    expected: bool,
) -> None:
    plan = _make_plan(_make_spec(), status=plan_status)
    assert (
        is_blueprint_completed(
            plan=plan,
            plan_approved=plan_approved,
            run_status=run_status,
            stop_after=stop_after,
        )
        is expected
    )


def test_is_blueprint_completed_requires_a_plan() -> None:
    assert (
        is_blueprint_completed(
            plan=None,
            plan_approved=True,
            run_status=RunStatus.COMPLETED,
            stop_after="plan",
        )
        is False
    )


# ------------------------------------------------------------ human-gate parity


def _runtime(
    store: RunStore,
    tmp_path: Path,
    *,
    auto_approve: bool = False,
) -> WorkflowRuntime:
    return WorkflowRuntime(
        store=store,
        workspace_root=tmp_path,
        artifacts_root=tmp_path / "runs",
        provider=MockLLMProvider(),
        auto_approve=auto_approve,
    )


def _gate_state(runtime: WorkflowRuntime, run_id: str) -> dict[str, Any]:
    return {"run_id": run_id, "_runtime": runtime}


def _seed_gate_artifact(store: RunStore, run_id: str, gate: str) -> Any:
    spec = _make_spec()
    store.save_spec(run_id, spec)
    if gate == "spec":
        return spec
    _approve(store, run_id, "spec", spec)
    design = _make_design(spec)
    store.save_design(run_id, design)
    if gate == "design":
        return design
    _approve(store, run_id, "design", design)
    plan = _make_plan(spec)
    store.save_plan(run_id, plan)
    return plan


@pytest.mark.parametrize("gate", ["spec", "design", "plan"])
def test_human_gate_missing_artifact_uses_canonical_node(
    store: RunStore, tmp_path: Path, gate: str
) -> None:
    run_id = _new_run(store)
    runtime = _runtime(store, tmp_path)
    with pytest.raises(GraphPaused) as excinfo:
        _node_human_gate(_gate_state(runtime, run_id), {"gate": gate})

    gate_spec = GATE_SPECS[gate]
    assert excinfo.value.node_name == gate_spec["node"]
    assert excinfo.value.reason == f"Run has no {gate_spec['label']} to approve"
    assert excinfo.value.state_updates == {}


@pytest.mark.parametrize("gate", ["spec", "design", "plan"])
def test_human_gate_pauses_with_canonical_metadata(
    store: RunStore, tmp_path: Path, gate: str
) -> None:
    run_id = _new_run(store)
    artifact = _seed_gate_artifact(store, run_id, gate)
    runtime = _runtime(store, tmp_path)
    with pytest.raises(GraphPaused) as excinfo:
        _node_human_gate(_gate_state(runtime, run_id), {"gate": gate})

    gate_spec = GATE_SPECS[gate]
    artifact_id = getattr(artifact, gate_spec["id_attr"])
    assert excinfo.value.node_name == gate_spec["node"]
    assert excinfo.value.reason == (
        f"{gate_spec['label']} {artifact_id} is awaiting human approval"
    )
    assert excinfo.value.state_updates == {"route": "awaiting"}


@pytest.mark.parametrize("gate", ["spec", "design", "plan"])
def test_human_gate_auto_approve_matches_progress(
    store: RunStore, tmp_path: Path, gate: str
) -> None:
    run_id = _new_run(store)
    _seed_gate_artifact(store, run_id, gate)
    runtime = _runtime(store, tmp_path, auto_approve=True)

    assert _node_human_gate(_gate_state(runtime, run_id), {"gate": gate}) == {
        "route": "approved"
    }
    assert is_artifact_approved(store, run_id, gate, store.get_run(run_id)) is True


def test_human_gate_disabled_auto_approves_without_flag(store: RunStore, tmp_path: Path) -> None:
    run_id = _new_run(store)
    _seed_gate_artifact(store, run_id, "spec")
    runtime = _runtime(store, tmp_path)

    result = _node_human_gate(_gate_state(runtime, run_id), {"gate": "spec", "enabled": False})

    assert result == {"route": "approved"}
    assert is_artifact_approved(store, run_id, "spec", store.get_run(run_id)) is True


def test_human_gate_auto_approve_accepts_risk_for_blocking_spec(
    store: RunStore, tmp_path: Path
) -> None:
    run_id = _new_run(store)
    store.save_spec(run_id, _make_spec(blocking=True))
    runtime = _runtime(store, tmp_path, auto_approve=True)

    assert _node_human_gate(_gate_state(runtime, run_id), {"gate": "spec"}) == {
        "route": "approved"
    }
    assert is_artifact_approved(store, run_id, "spec", store.get_run(run_id)) is True


def test_human_gate_unknown_gate_keeps_legacy_pause(store: RunStore, tmp_path: Path) -> None:
    run_id = _new_run(store)
    runtime = _runtime(store, tmp_path)
    with pytest.raises(GraphPaused) as excinfo:
        _node_human_gate(_gate_state(runtime, run_id), {"gate": "release"})

    assert excinfo.value.node_name == "gate_release"
    assert excinfo.value.reason == "Unknown human gate 'release'"
    assert excinfo.value.state_updates == {}