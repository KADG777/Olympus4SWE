"""Regression tests for the independent review of the architecture-deepening branch.

Each test pins one Important finding (CAS-bound gate writes, atomic clarify
persistence, operation_id replay identity, strict write scopes, glob delivery
parity, symlink escapes, upstream plan binding, demo seeding) or a review nit.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

import olympus.web.projection as projection_module
from olympus.core.contracts import (
    AcceptanceCheck,
    AmbiguityItem,
    CheckKind,
    DesignModule,
    DesignSpec,
    GoalSpec,
    RequirementItem,
    RunEvent,
    RunStatus,
    TaskNode,
    TaskPlan,
)
from olympus.core.control import RunControl, is_plan_bound_to_upstream
from olympus.core.dag import paths_overlap, scopes_overlap
from olympus.core.store import ConcurrencyConflictError, RunStore
from olympus.graph.loader import load_workflow
from olympus.graph.progress import build_workflow_view, resolve_resume_node
from olympus.providers.base import ChatMessage, LLMResponse, MockLLMProvider, ToolSpec
from olympus.stages.align import AlignStage
from olympus.stages.orchestrate import OrchestrateStage, _copy_matching_scope
from olympus.tools.aci import WorkspacePolicy, WorkspaceToolkit
from olympus.web.demos import DELIVERY_DEMO_ID, seed_demo_runs
from olympus.web.server import AriadneApp

# --------------------------------------------------------------------------- helpers


def _spec(*, with_blocker: bool = False) -> GoalSpec:
    ambiguities = (
        [
            AmbiguityItem(
                id="AMB-1",
                question="Which storage backend?",
                impact="Determines persistence contract",
                options=["sqlite", "postgres"],
                blocking=True,
            )
        ]
        if with_blocker
        else []
    )
    return GoalSpec(
        spec_id="spec-review",
        goal="Ship a deterministic control plane",
        requirements=[
            RequirementItem(id="R1", description="Do X", acceptance_criteria=["X works"])
        ],
        open_ambiguities=ambiguities,
    )


def _design(spec: GoalSpec) -> DesignSpec:
    return DesignSpec(
        design_id="design-review",
        spec_id=spec.spec_id,
        spec_hash=spec.content_hash,
        architecture_summary="One module, one seam.",
        modules=[
            DesignModule(
                id="M1",
                name="Core",
                responsibility="Do the work",
                requirement_refs=["R1"],
                target_paths=["src/core.py"],
            )
        ],
    )


def _plan(spec: GoalSpec) -> TaskPlan:
    return TaskPlan(
        plan_id="plan-review",
        spec_id=spec.spec_id,
        spec_hash=spec.content_hash,
        tasks=[
            TaskNode(
                task_id="T1",
                title="Implement",
                objective="Write the module",
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
            )
        ],
    )


def _seeded(tmp_path: Path) -> tuple[RunStore, RunControl, str, GoalSpec, TaskPlan]:
    store = RunStore(tmp_path / "review.db")
    run_id = store.create_run(title="Review")
    spec = _spec()
    store.save_spec(run_id, spec, actor="align")
    store.save_design(run_id, _design(spec), actor="design")
    plan = _plan(spec)
    store.save_plan(run_id, plan, actor="orchestrate")
    return store, RunControl(store), run_id, spec, plan


def _seeded_spec_only(tmp_path: Path) -> tuple[RunStore, RunControl, str, GoalSpec]:
    store = RunStore(tmp_path / "review-spec-only.db")
    run_id = store.create_run(title="Spec only")
    spec = _spec()
    store.save_spec(run_id, spec, actor="align")
    store.save_plan(run_id, _plan(spec), actor="orchestrate")
    control = RunControl(store)
    control.confirm_spec(run_id)
    control.approve_plan(run_id)
    return store, control, run_id, spec


def _event(event_seq: int, event_type: str, payload: dict[str, object] | None = None) -> RunEvent:
    return RunEvent(
        event_seq=event_seq,
        run_id="run-view",
        stage="graph",
        event_type=event_type,
        actor="supervisor",
        payload=payload or {},
        state_revision=event_seq,
    )


# ------------------------------------------------------- Finding 1: CAS on gates


@pytest.mark.parametrize("gate", ["clarify", "approve_spec", "approve_design", "approve_plan"])
def test_gate_writes_bind_to_the_read_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, gate: str
) -> None:
    store, control, run_id, spec, _ = _seeded(tmp_path)
    if gate == "approve_plan":
        control.confirm_spec(run_id)

    stale = store.get_run(run_id)
    # A concurrent writer lands between the command's read and its write.
    store.save_spec(run_id, spec.model_copy(update={"goal": "concurrent edit"}), actor="align")
    monkeypatch.setattr(store, "get_run", lambda *args, **kwargs: stale)

    # Approvals reproduced the stale hash-only flow: expected_target_hash is set
    # while expected_revision is omitted, so CAS must come from the read snapshot.
    target_hash = {
        "approve_spec": stale["spec"].content_hash,
        "approve_design": stale["design"].content_hash,
        "approve_plan": stale["plan"].content_hash,
    }.get(gate)

    with pytest.raises(ConcurrencyConflictError):
        if gate == "clarify":
            control.clarify_spec(run_id, answers={})
        elif gate == "approve_spec":
            control.confirm_spec(run_id, expected_target_hash=target_hash)
        elif gate == "approve_design":
            control.approve_design(run_id, expected_target_hash=target_hash)
        else:
            control.approve_plan(run_id, expected_target_hash=target_hash)


def test_human_gate_auto_approval_is_cas_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The graph gate's blocker check and approval write share one revision."""
    from olympus.graph import loader

    store, _, run_id, _, _ = _seeded(tmp_path / "gate")
    runtime = loader.WorkflowRuntime(
        store=store,
        workspace_root=tmp_path,
        artifacts_root=tmp_path / ".olympus" / "runs",
        provider=MockLLMProvider(),
        auto_approve=True,
    )
    held = store.get_run(run_id)
    # A concurrent writer lands after the gate read its snapshot.
    store.save_spec(run_id, held["spec"].model_copy(update={"goal": "newer"}), actor="align")
    monkeypatch.setattr(store, "get_run", lambda *args, **kwargs: held)

    with pytest.raises(ConcurrencyConflictError):
        loader._node_human_gate({"run_id": run_id, "_runtime": runtime}, {"gate": "spec"})


# ------------------------------------------------- Finding 2: atomic clarify


def test_clarify_rolls_back_spec_when_audit_event_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, control, run_id, spec, _ = _seeded(tmp_path)
    original_insert = RunStore._insert_event

    def failing_insert(self, conn, **kwargs):  # type: ignore[no-untyped-def]
        if kwargs.get("event_type") == "spec.clarified":
            raise RuntimeError("disk full")
        return original_insert(self, conn, **kwargs)

    monkeypatch.setattr(RunStore, "_insert_event", failing_insert)
    with pytest.raises(RuntimeError, match="disk full"):
        control.clarify_spec(run_id, answers={})

    snap = store.get_run(run_id)
    assert snap["spec"].version == spec.version
    saved_versions = [
        e.payload["version"] for e in store.list_events(run_id) if e.event_type == "spec.saved"
    ]
    assert spec.version in saved_versions
    assert not any(e.event_type == "spec.clarified" for e in store.list_events(run_id))


# ------------------------------------- Finding 3: operation_id replay contract


def test_idempotent_command_rejects_operation_id_identity_changes(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "commands.db")
    run_a = store.create_run(title="A")
    run_b = store.create_run(title="B")

    def handler(conn, cur_rev):  # type: ignore[no-untyped-def]
        return {"ok": True}

    first = store.apply_idempotent_command(
        operation_id="op-fixed",
        run_id=run_a,
        actor="zeus",
        action="pause",
        expected_state_revision=1,
        payload={"flag": 1},
        handler=handler,
    )
    assert first["deduplicated"] is False

    replay = store.apply_idempotent_command(
        operation_id="op-fixed",
        run_id=run_a,
        actor="zeus",
        action="pause",
        expected_state_revision=1,
        payload={"flag": 1},
        handler=handler,
    )
    assert replay == {"ok": True, "deduplicated": True}

    with pytest.raises(ConcurrencyConflictError, match="operation_id"):
        store.apply_idempotent_command(
            operation_id="op-fixed",
            run_id=run_b,
            actor="zeus",
            action="pause",
            expected_state_revision=1,
            payload={"flag": 1},
            handler=handler,
        )
    with pytest.raises(ConcurrencyConflictError, match="operation_id"):
        store.apply_idempotent_command(
            operation_id="op-fixed",
            run_id=run_a,
            actor="zeus",
            action="resume",
            expected_state_revision=1,
            payload={"flag": 1},
            handler=handler,
        )
    with pytest.raises(ConcurrencyConflictError, match="operation_id"):
        store.apply_idempotent_command(
            operation_id="op-fixed",
            run_id=run_a,
            actor="zeus",
            action="pause",
            expected_state_revision=1,
            payload={"flag": 2},
            handler=handler,
        )


@pytest.mark.parametrize("gate", ["clarify", "approve_spec", "approve_design", "approve_plan"])
def test_gate_commands_replay_without_double_write(tmp_path: Path, gate: str) -> None:
    store, control, run_id, _, _ = _seeded(tmp_path / gate)
    if gate == "approve_plan":
        control.confirm_spec(run_id)
    revision = store.get_run(run_id)["state_revision"]

    if gate == "clarify":
        first = control.clarify_spec(
            run_id, answers={}, operation_id="op-1", expected_revision=revision
        )
    elif gate == "approve_spec":
        first = control.confirm_spec(run_id, operation_id="op-1", expected_revision=revision)
    elif gate == "approve_design":
        first = control.approve_design(run_id, operation_id="op-1", expected_revision=revision)
    else:
        first = control.approve_plan(run_id, operation_id="op-1", expected_revision=revision)

    settled_revision = store.get_run(run_id)["state_revision"]
    if gate == "clarify":
        second = control.clarify_spec(
            run_id, answers={}, operation_id="op-1", expected_revision=revision
        )
    elif gate == "approve_spec":
        second = control.confirm_spec(run_id, operation_id="op-1", expected_revision=revision)
    elif gate == "approve_design":
        second = control.approve_design(run_id, operation_id="op-1", expected_revision=revision)
    else:
        second = control.approve_plan(run_id, operation_id="op-1", expected_revision=revision)

    assert isinstance(first, dict) and first["deduplicated"] is False
    assert isinstance(second, dict) and second["deduplicated"] is True
    assert store.get_run(run_id)["state_revision"] == settled_revision

    if gate == "clarify":
        assert store.get_run(run_id)["spec"].version == first["version"]
        events = [e for e in store.list_events(run_id) if e.event_type == "spec.clarified"]
        assert [e.payload["version"] for e in events] == [first["version"]]

    # Reusing the same operation id for a different action is a conflict, not a replay.
    with pytest.raises(ConcurrencyConflictError, match="operation_id"):
        if gate == "clarify":
            control.confirm_spec(run_id, operation_id="op-1")
        else:
            control.clarify_spec(run_id, answers={}, operation_id="op-1")


def test_promote_replay_returns_recorded_result_before_active_guard(tmp_path: Path) -> None:
    store, control, run_id, _, _ = _seeded(tmp_path)
    control.confirm_spec(run_id)
    control.approve_plan(run_id)
    revision = store.get_run(run_id)["state_revision"]

    first = control.promote_to_execute(
        run_id,
        target_workflow="full_sdlc_tdd",
        current_stop_after="plan",
        operation_id="op-promote",
        expected_state_revision=revision,
        max_llm_calls=200,
        auto_advance=True,
    )
    assert first["deduplicated"] is False
    assert first["status"] == RunStatus.RUNNING.value

    # The run is now active; the same operation must still replay its recorded result.
    replay = control.promote_to_execute(
        run_id,
        target_workflow="full_sdlc_tdd",
        current_stop_after="plan",
        operation_id="op-promote",
        expected_state_revision=revision,
        max_llm_calls=200,
        auto_advance=True,
    )
    assert replay["deduplicated"] is True
    assert replay["state_revision"] == first["state_revision"]

    with pytest.raises(ConcurrencyConflictError, match="operation_id"):
        control.promote_to_execute(
            run_id,
            target_workflow="full_sdlc_tdd",
            current_stop_after="plan",
            operation_id="op-promote",
            expected_state_revision=revision,
            max_llm_calls=999,
            auto_advance=True,
        )

    other_run = store.create_run(title="Other")
    with pytest.raises(ConcurrencyConflictError, match="operation_id"):
        control.promote_to_execute(
            other_run,
            target_workflow="full_sdlc_tdd",
            current_stop_after="plan",
            operation_id="op-promote",
            expected_state_revision=1,
        )


def test_retry_workflow_replay_never_applies_twice(tmp_path: Path) -> None:
    store, control, run_id, _, _ = _seeded(tmp_path)
    store.set_run_status(run_id, RunStatus.FAILED, actor="chronos", stage="graph")
    revision = store.get_run(run_id)["state_revision"]

    first = control.prepare_workflow_retry(
        run_id, expected_revision=revision, operation_id="op-retry"
    )
    assert first["deduplicated"] is False
    after_first = store.get_run(run_id)
    assert after_first["status"] == RunStatus.RUNNING
    assert after_first["state_revision"] == first["state_revision"]

    replay = control.prepare_workflow_retry(
        run_id, expected_revision=revision, operation_id="op-retry"
    )
    assert replay["deduplicated"] is True
    assert store.get_run(run_id)["state_revision"] == after_first["state_revision"]


# ------------------------------------------- Finding 4: strict write scopes


def test_workspace_policy_never_grants_ancestors_or_ignores_trailing_space(
    tmp_path: Path,
) -> None:
    workspace = (tmp_path / "ws").resolve()
    workspace.mkdir()
    policy = WorkspacePolicy(
        workspace_root=workspace, allowed_write_scope=["docs/guide.md"]
    )

    assert policy._relative_target("docs/guide.md ") == "docs/guide.md "
    assert policy.authorize_tool("write_file", {"path": "docs/guide.md"}) == (True, "")
    for denied in ("docs", "docs/guide.md "):
        allowed, reason = policy.authorize_tool("write_file", {"path": denied})
        assert allowed is False
        assert "outside authorized write_scope" in reason
    with pytest.raises(PermissionError):
        policy.assert_write_allowed(workspace / "docs")

    dir_policy = WorkspacePolicy(workspace_root=workspace, allowed_write_scope=["docs/"])
    assert dir_policy.authorize_tool("write_file", {"path": "docs/nested/x.md"}) == (True, "")


def test_forbidden_scope_still_blocks_ancestors_and_descendants(tmp_path: Path) -> None:
    workspace = (tmp_path / "ws").resolve()
    workspace.mkdir()
    policy = WorkspacePolicy(
        workspace_root=workspace,
        allowed_write_scope=["docs/"],
        forbidden_write_scope=["docs/locked/contract.md", "docs/generated/"],
    )
    for denied in ("docs/locked", "docs/locked/contract.md", "docs/generated/x.py"):
        allowed, reason = policy.authorize_tool("write_file", {"path": denied})
        assert allowed is False
        assert "read-only shared contract path" in reason


# ------------------------------------------- Finding 5: glob delivery parity


def test_copy_matching_scope_honors_glob_scopes(tmp_path: Path) -> None:
    src = tmp_path / "src"
    (src / "docs").mkdir(parents=True)
    (src / "docs" / "a.md").write_text("a", encoding="utf-8")
    (src / "docs" / "b.txt").write_text("b", encoding="utf-8")
    (src / "pkg").mkdir()
    (src / "pkg" / "x.py").write_text("x", encoding="utf-8")

    glob_dst = tmp_path / "glob_dst"
    _copy_matching_scope(src, glob_dst, ["docs/*.md"])
    assert (glob_dst / "docs" / "a.md").is_file()
    assert not (glob_dst / "docs" / "b.txt").exists()
    assert not (glob_dst / "pkg" / "x.py").exists()

    dir_dst = tmp_path / "dir_dst"
    _copy_matching_scope(src, dir_dst, ["docs/"])
    assert (dir_dst / "docs" / "a.md").is_file()
    assert (dir_dst / "docs" / "b.txt").is_file()
    assert not (dir_dst / "pkg" / "x.py").exists()


def test_glob_write_scopes_detect_overlap(tmp_path: Path) -> None:
    del tmp_path
    assert paths_overlap("docs/*.md", "docs/guide.md")
    assert paths_overlap("docs/*.md", "docs/*.txt")
    assert not paths_overlap("docs/*.md", "src/*.py")
    assert scopes_overlap(["docs/*.md"], ["docs/guide.md"]) == [
        ("docs/*.md", "docs/guide.md")
    ]


# ------------------------------------------------ Finding 6: symlink escapes


def test_grep_and_glob_skip_symlinks_that_escape_the_workspace(tmp_path: Path) -> None:
    workspace = (tmp_path / "ws").resolve()
    workspace.mkdir()
    outside = (tmp_path / "outside").resolve()
    outside.mkdir()
    (outside / "secret.txt").write_text("SECRET_TOKEN\n", encoding="utf-8")
    (workspace / "inside.txt").write_text("SECRET_TOKEN\n", encoding="utf-8")
    (workspace / "leak.txt").symlink_to(outside / "secret.txt")
    (workspace / "linkdir").symlink_to(outside, target_is_directory=True)

    toolkit = WorkspaceToolkit(workspace, read_only=True)
    grep_out = toolkit.execute("grep_files", {"pattern": "SECRET_TOKEN"})
    assert "inside.txt" in grep_out
    assert "leak.txt" not in grep_out
    assert "linkdir" not in grep_out

    glob_out = toolkit.execute("glob_files", {"pattern": "**/*.txt"})
    assert str(workspace / "inside.txt") in glob_out
    assert "leak.txt" not in glob_out


# -------------------------------------- Finding 7: upstream artifact binding


def test_stale_plan_after_goal_spec_v2_is_rejected_everywhere(tmp_path: Path) -> None:
    store, control, run_id, spec = _seeded_spec_only(tmp_path)
    snap = store.get_run(run_id)
    assert is_plan_bound_to_upstream(store, run_id, snap) is True

    spec_v2 = spec.model_copy(update={"version": 2, "goal": "Changed after the plan"})
    store.save_spec(run_id, spec_v2, actor="align")
    control.confirm_spec(run_id)
    stale_snap = store.get_run(run_id)
    assert stale_snap["spec"].content_hash == spec_v2.content_hash
    assert is_plan_bound_to_upstream(store, run_id, stale_snap) is False

    with pytest.raises(ValueError, match="stale"):
        control.promote_to_execute(
            run_id,
            target_workflow="full_sdlc_tdd",
            current_stop_after="plan",
            operation_id="op-promote-stale",
            expected_state_revision=stale_snap["state_revision"],
        )

    definition = load_workflow("repo_fast_patch")
    assert resolve_resume_node(definition, stale_snap, store=store) == "plan"

    stage = OrchestrateStage(
        store=store, provider=MockLLMProvider(), workspace_root=tmp_path
    )
    with pytest.raises(ValueError, match="stale"):
        stage.run_execution(run_id=run_id)


# ---------------------------------------------- Finding 8: demo seeding


def test_projection_reexports_seed_demo_runs_and_seeds_past_100_runs(tmp_path: Path) -> None:
    assert projection_module.seed_demo_runs is seed_demo_runs

    store = RunStore(tmp_path / "demos.db")
    for index in range(105):
        store.create_run(title=f"Filler {index}", run_id=f"filler-{index:03d}")

    seeded = seed_demo_runs(store, tmp_path)
    assert seeded == [DELIVERY_DEMO_ID, "run-campus-waitlist"]
    assert seed_demo_runs(store, tmp_path) == []


# ----------------------------------------------------------- Nit 1: align repair


def test_align_repair_pass_is_always_revalidated(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "align.db")
    run_id = store.create_run(title="Align repair")
    responses = iter(
        [
            {
                "goal": "Build a tool",
                "requirements": [
                    {
                        "id": "R1",
                        "description": "Do the thing",
                        "acceptance_criteria": ["The thing works"],
                    }
                ],
            },
            {
                "goal": "Build a tool",
                "requirements": [
                    {
                        "id": "R1",
                        "description": "Do the thing",
                        "acceptance_criteria": ["The thing works"],
                    }
                ],
                "open_ambiguities": [
                    {
                        "id": "AMB-1",
                        "question": "Which backend?",
                        "impact": "Storage contract",
                        "blocking": True,
                    }
                ],
            },
        ]
    )

    def responder(messages: Sequence[ChatMessage], tools: Sequence[ToolSpec]) -> LLMResponse:
        del messages, tools
        return LLMResponse(content=json.dumps(next(responses), ensure_ascii=False))

    align = AlignStage(
        store=store,
        provider=MockLLMProvider(responder=responder),
        workspace_root=tmp_path,
    )
    spec = align.propose_spec(
        run_id=run_id, raw_goal="Build a tool", strict_canonical=True, max_steps=1
    )
    assert spec.unresolved_blockers() == ["ambiguity:AMB-1"]
    failures = [
        event for event in store.list_events(run_id) if event.event_type == "spec.validation_failed"
    ]
    assert len(failures) == 1
    assert any("deliverable_kind" in violation for violation in failures[0].payload["violations"])


# ------------------------------------------------ Nit 2: workflow view reset


def test_workflow_view_clears_finished_when_a_new_wave_starts() -> None:
    events = [
        _event(1, "graph.node_started", {"node": "execute"}),
        _event(2, "graph.finished"),
        _event(3, "graph.started"),
        _event(4, "graph.node_started", {"node": "execute"}),
    ]
    view = build_workflow_view("full_sdlc_tdd", events, RunStatus.RUNNING)
    assert view["finished"] is False
    assert view["node_statuses"]["execute"] == "running"


# ---------------------------------------------------- server retry idempotency


def test_http_retries_do_not_resume_or_clarify_twice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = RunStore(tmp_path / "server.db")
    run_id = store.create_run(title="Retry once", workflow_name="product_blueprint")
    store.set_run_status(run_id, RunStatus.FAILED, actor="chronos", stage="graph")
    app = AriadneApp(store=store, workspace_root=tmp_path)
    resume_calls: list[str] = []
    monkeypatch.setattr(
        app.supervisor,
        "resume_run",
        lambda rid, sync=False: resume_calls.append(rid) or {"run_id": rid},
    )
    revision = store.get_run(run_id)["state_revision"]
    body = {
        "action": "retry_workflow",
        "operation_id": "op-server-retry",
        "expected_state_revision": revision,
    }
    first = app.execute_command(run_id, body)
    second = app.execute_command(run_id, body)
    assert resume_calls == [run_id]
    assert first["command_result"]["deduplicated"] is False
    assert second["command_result"]["deduplicated"] is True

    # Clarify replays likewise must not resume the supervised graph a second time.
    clarify_store = RunStore(tmp_path / "server-clarify.db")
    clarify_run = clarify_store.create_run(
        title="Clarify once", workflow_name="product_blueprint"
    )
    clarify_store.save_spec(clarify_run, _spec(), actor="align")
    clarify_store.set_run_status(
        clarify_run, RunStatus.AWAITING_HUMAN, actor="supervisor", stage="graph"
    )
    clarify_app = AriadneApp(store=clarify_store, workspace_root=tmp_path)
    clarify_resumes: list[str] = []
    monkeypatch.setattr(clarify_app.supervisor, "is_supervised", lambda rid: True)
    monkeypatch.setattr(
        clarify_app.supervisor,
        "resume_run",
        lambda rid, sync=False: clarify_resumes.append(rid) or {"run_id": rid},
    )
    clarify_body = {
        "action": "clarify",
        "operation_id": "op-server-clarify",
        "answers": {},
        "expected_state_revision": clarify_store.get_run(clarify_run)["state_revision"],
    }
    clarify_app.execute_command(clarify_run, clarify_body)
    clarify_app.execute_command(clarify_run, clarify_body)
    assert clarify_resumes == [clarify_run]
    assert clarify_store.get_run(clarify_run)["spec"].version == 2
