"""Stage 2 Orchestrate & Verify MANIFEST.md integrity artifact tests."""

from __future__ import annotations

import json
import re
import subprocess
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from olympus.core.contracts import (
    ApprovalDecision,
    ApprovalRecord,
    ApprovalTarget,
    CheckKind,
    DesignSpec,
    ExecutionStrategy,
    GoalSpec,
    RunMode,
    RunStatus,
    SpecStatus,
    TaskKind,
    TaskPlan,
    TaskStatus,
)
from olympus.core.store import RunStore
from olympus.providers.base import (
    ChatMessage,
    LLMResponse,
    MockLLMProvider,
    ToolCall,
    ToolSpec,
)
from olympus.stages.align import AlignStage
from olympus.stages.orchestrate import OrchestrateStage
from olympus.web.projection import build_run_projection

Responder = Callable[[Sequence[ChatMessage], Sequence[ToolSpec]], LLMResponse]


def _init_repo(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "README.md").write_text("# Sample\n", encoding="utf-8")
    subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "dev@example.com"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "Dev"], cwd=root, check=True)
    subprocess.run(["git", "add", "README.md"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=root, check=True, capture_output=True)


def _confirmed_run(
    tmp_path: Path, *, criteria: list[str] | None = None
) -> tuple[RunStore, str]:
    store = RunStore(tmp_path / ".olympus" / "olympus.db")
    run_id = store.create_run(title="Manifest run", mode=RunMode.SIMULATION)
    spec = GoalSpec(
        spec_id="spec-manifest",
        goal="Add a feature flag module",
        requirements=[
            {
                "id": "R1",
                "description": "Provide feature.py with FLAG = True",
                "acceptance_criteria": criteria or ["feature.FLAG is True"],
            }
        ],
    )
    store.save_spec(run_id, spec)
    align = AlignStage(store=store, provider=MockLLMProvider(), workspace_root=tmp_path)
    align.confirm_spec(run_id=run_id, approver="zeus")
    return store, run_id


def _plan_payload(
    *,
    spec_id: str,
    spec_hash: str,
    file_name: str,
    checks: list[dict[str, Any]],
    max_cycles: int = 3,
) -> dict[str, Any]:
    return {
        "plan_id": "plan-manifest",
        "spec_id": spec_id,
        "spec_hash": spec_hash,
        "version": 1,
        "tasks": [
            {
                "task_id": "T1",
                "title": f"Implement {file_name}",
                "objective": f"Create {file_name} inside the sandbox",
                "requirement_refs": ["R1"],
                "depends_on": [],
                "write_scope": [file_name],
                "max_cycles": max_cycles,
                "acceptance_checks": checks,
            }
        ],
    }


def _stage_responder(
    *,
    plan: dict[str, Any],
    file_name: str,
    review_passes: list[bool],
) -> Responder:
    review_calls = 0

    def responder(messages: Sequence[ChatMessage], tools: Sequence[ToolSpec]) -> LLMResponse:
        nonlocal review_calls
        del tools
        system_message = messages[0].content
        user_message = messages[1].content if len(messages) > 1 else ""

        if "Task Planner" in system_message:
            return LLMResponse(content=json.dumps(plan))

        if "Execution Engineer" in system_message:
            match = re.search(r"Sandbox workspace root: (.+)", user_message)
            assert match is not None
            sandbox = Path(match.group(1).strip())
            if not any(msg.role == "tool" for msg in messages):
                return LLMResponse(
                    content=f"writing {file_name}",
                    tool_calls=(
                        ToolCall(
                            id="w1",
                            name="write_file",
                            arguments={
                                "path": str(sandbox / file_name),
                                "content": "FLAG = True\n",
                            },
                        ),
                    ),
                )
            return LLMResponse(content=f"implemented {file_name}")

        if "Independent Verifier" in system_message:
            passed = review_passes[min(review_calls, len(review_passes) - 1)]
            review_calls += 1
            findings: list[dict[str, Any]] = []
            if not passed:
                findings.append(
                    {
                        "id": "F1",
                        "severity": "important",
                        "file_path": file_name,
                        "line": 1,
                        "message": "Missing overflow guard",
                    }
                )
            verdict = {
                "task_id": "T1",
                "attempt": 1,
                "passed": passed,
                "summary": "Looks good" if passed else "Missing overflow guard",
                "findings": findings,
            }
            return LLMResponse(content=json.dumps(verdict))

        raise AssertionError(f"Unexpected system prompt: {system_message[:80]}")

    return responder


def test_delivery_acceptance_uses_executed_checks_on_the_composed_output(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    store, run_id = _confirmed_run(
        tmp_path, criteria=["feature.FLAG is True", "feature.AUDIT is True", "UI exposes FLAG"]
    )
    spec = store.get_run(run_id)["spec"]
    checks = [
        {
            "id": "CHK-1",
            "kind": "command",
            "description": "Verify the flag module imports",
            "command": "python3 -c 'import feature; assert feature.FLAG'",
        }
    ]
    plan = _plan_payload(
        spec_id=spec.spec_id, spec_hash=spec.content_hash, file_name="feature.py", checks=checks
    )
    plan["delivery_checks"] = [
        {
            "requirement_id": "R1",
            "criterion_index": 1,
            "command": "python3 -c 'import feature; assert feature.FLAG; print(\"tested FLAG\")'",
        },
        {
            "requirement_id": "R1",
            "criterion_index": 2,
            "command": "python3 -c 'import feature; assert feature.AUDIT'",
        },
    ]
    orch = OrchestrateStage(
        store=store,
        provider=MockLLMProvider(
            responder=_stage_responder(plan=plan, file_name="feature.py", review_passes=[True])
        ),
        workspace_root=tmp_path,
    )
    orch.propose_plan(run_id=run_id, auto_approve=True)
    pending = build_run_projection(store, tmp_path, run_id)["delivery_acceptance"]
    assert pending["counts"] == {"passed": 0, "failed": 0, "unverified": 3}

    orch.run_execution(run_id=run_id)
    review = build_run_projection(store, tmp_path, run_id)["delivery_acceptance"]
    assert review["counts"] == {"passed": 1, "failed": 1, "unverified": 1}
    passed, failed, unverified = review["criteria"]
    assert [row["text"] for row in review["criteria"]] == [
        "feature.FLAG is True", "feature.AUDIT is True", "UI exposes FLAG"
    ]
    assert "tested FLAG" in passed["evidence"]["log"]
    assert failed["evidence"]["delivery_version"] == passed["evidence"]["delivery_version"]
    assert failed["reason"] and unverified["reason"]
    assert unverified["evidence"] is None
    assert not (tmp_path / "feature.py").exists()

    executed = next(
        event
        for event in store.list_events(run_id)
        if event.event_seq == passed["evidence"]["event_seq"]
    )
    wrong_version = {"delivery_version": "sha256:" + "0" * 64}
    store.append_event(
        run_id=run_id,
        stage=executed.stage,
        event_type=executed.event_type,
        actor=executed.actor,
        payload=executed.payload | wrong_version,
    )
    stale = build_run_projection(store, tmp_path, run_id)["delivery_acceptance"]
    assert stale["counts"] == {"passed": 0, "failed": 0, "unverified": 3}
    assert "交付版本" in stale["criteria"][0]["reason"]

    store.append_event(
        run_id=run_id,
        stage=executed.stage,
        event_type=executed.event_type,
        actor=executed.actor,
        payload=executed.payload,
    )
    artifact = tmp_path / ".olympus" / "runs" / run_id / "artifacts"
    check_log = artifact / passed["evidence"]["receipt_id"] / "check-1.log"
    check_log.write_text("different run output", encoding="utf-8")
    tampered = build_run_projection(store, tmp_path, run_id)["delivery_acceptance"]
    assert tampered["counts"] == {"passed": 0, "failed": 1, "unverified": 2}
    assert "指纹不一致" in tampered["criteria"][0]["reason"]

    store.save_plan(run_id, store.get_run(run_id)["plan"])
    replaced = build_run_projection(store, tmp_path, run_id)["delivery_acceptance"]
    assert replaced["counts"] == {"passed": 0, "failed": 0, "unverified": 3}

    revised = spec.model_copy(
        update={
            "version": spec.version + 1,
            "requirements": [
                spec.requirements[0].model_copy(update={"acceptance_criteria": ["FLAG is false"]})
            ],
        }
    )
    store.save_spec(run_id, revised)
    assert build_run_projection(store, tmp_path, run_id)["delivery_acceptance"] is None
    AlignStage(store=store, provider=MockLLMProvider(), workspace_root=tmp_path).confirm_spec(
        run_id=run_id
    )
    new_review = build_run_projection(store, tmp_path, run_id)["delivery_acceptance"]
    assert new_review["spec_hash"] == revised.content_hash
    assert new_review["counts"] == {"passed": 0, "failed": 0, "unverified": 1}
    assert new_review["criteria"][0]["text"] == "FLAG is false"


def test_orchestrate_writes_manifest_for_rejected_then_accepted_attempts(
    tmp_path: Path,
) -> None:
    _init_repo(tmp_path)
    store, run_id = _confirmed_run(tmp_path)
    spec = store.get_run(run_id)["spec"]
    assert spec is not None
    plan = _plan_payload(
        spec_id=spec.spec_id,
        spec_hash=spec.content_hash,
        file_name="feature.py",
        checks=[
            {
                "id": "CHK-1",
                "kind": CheckKind.COMMAND.value,
                "description": "Verify the flag module imports",
                "command": "python3 -c 'import feature; assert feature.FLAG'",
            },
            {
                "id": "CHK-REV-1",
                "kind": CheckKind.INDEPENDENT_REVIEW.value,
                "description": "Cold-start independent review",
            },
        ],
    )
    orch = OrchestrateStage(
        store=store,
        provider=MockLLMProvider(
            responder=_stage_responder(
                plan=plan,
                file_name="feature.py",
                review_passes=[False, True],
            )
        ),
        workspace_root=tmp_path,
    )
    orch.propose_plan(run_id=run_id, auto_approve=True)
    snap = orch.run_execution(run_id=run_id)

    assert snap["status"] == RunStatus.COMPLETED
    task = snap["plan"].tasks[0]
    assert task.status == TaskStatus.ACCEPTED
    assert task.attempt == 2

    attempts_dir = tmp_path / ".olympus" / "runs" / run_id / "artifacts" / "T1"
    rejected_manifest = attempts_dir / "attempt-1" / "MANIFEST.md"
    accepted_manifest = attempts_dir / "attempt-2" / "MANIFEST.md"
    assert rejected_manifest.is_file()
    assert accepted_manifest.is_file()

    rejected_text = rejected_manifest.read_text(encoding="utf-8")
    assert "# MANIFEST — T1 (attempt 1)" in rejected_text
    assert "Outcome: rejected" in rejected_text
    assert "Checks: 1/2 passed" in rejected_text
    assert "Diff: +1/-0 lines" in rejected_text
    assert "Review verdict" in rejected_text
    assert "Missing overflow guard" in rejected_text
    assert "Turn: turn-" in rejected_text
    assert "hephaestus:T1 · 2 calls" in rejected_text

    accepted_text = accepted_manifest.read_text(encoding="utf-8")
    assert "# MANIFEST — T1 (attempt 2)" in accepted_text
    assert "Outcome: accepted" in accepted_text
    assert "Checks: 2/2 passed" in accepted_text
    assert "Looks good" in accepted_text

    # Existing artifacts remain untouched alongside the new manifest
    assert (attempts_dir / "attempt-1" / "diff.patch").is_file()
    assert (attempts_dir / "attempt-1" / "verdict-cycle-1.json").is_file()
    assert (attempts_dir / "attempt-2" / "diff.patch").is_file()
    assert (attempts_dir / "attempt-2" / "verdict-cycle-2.json").is_file()


def test_orchestrate_writes_manifest_when_command_checks_fail(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    store, run_id = _confirmed_run(tmp_path)
    spec = store.get_run(run_id)["spec"]
    assert spec is not None
    plan = _plan_payload(
        spec_id=spec.spec_id,
        spec_hash=spec.content_hash,
        file_name="gate.py",
        max_cycles=2,
        checks=[
            {
                "id": "CHK-1",
                "kind": CheckKind.COMMAND.value,
                "description": "Always fails",
                "command": "python3 -c 'import sys; sys.exit(1)'",
            }
        ],
    )
    orch = OrchestrateStage(
        store=store,
        provider=MockLLMProvider(
            responder=_stage_responder(
                plan=plan,
                file_name="gate.py",
                review_passes=[True],
            )
        ),
        workspace_root=tmp_path,
    )
    orch.propose_plan(run_id=run_id, auto_approve=True)
    snap = orch.run_execution(run_id=run_id)

    assert snap["status"] == RunStatus.AWAITING_HUMAN
    task = snap["plan"].tasks[0]
    assert task.status == TaskStatus.BLOCKED
    assert task.review_cycle == 2

    attempts_dir = tmp_path / ".olympus" / "runs" / run_id / "artifacts" / "T1"
    for attempt in (1, 2):
        manifest = attempts_dir / f"attempt-{attempt}" / "MANIFEST.md"
        assert manifest.is_file()
        text = manifest.read_text(encoding="utf-8")
        assert f"# MANIFEST — T1 (attempt {attempt})" in text
        assert "Outcome: checks_failed" in text
        assert "Checks: 0/1 passed" in text
        assert "CHK-1 (command): not passed — Always fails" in text
        assert (attempts_dir / f"attempt-{attempt}" / "check-1-CHK-1.log").is_file()


def test_design_spec_v2_readonly_contract_hook_and_serial_handoff(tmp_path: Path) -> None:
    from olympus.core.contracts import (
        ApprovalDecision,
        ApprovalRecord,
        ApprovalTarget,
        DesignSpec,
        ExecutionStrategy,
        SpecStatus,
        TaskKind,
        TaskPlan,
    )
    from olympus.harness.hooks import default_write_scope_hook

    # 1. Wave 1 hook rejects edits to shared_contract_paths even if write_scope is broad.
    hook = default_write_scope_hook(
        ["src/"],
        forbidden_globs=["src/contracts/"],
        workspace_root=tmp_path,
    )
    allowed, _ = hook("write_file", {"path": str(tmp_path / "src" / "api" / "app.py")})
    assert allowed is True
    denied, reason = hook("write_file", {"path": str(tmp_path / "src" / "contracts" / "dto.py")})
    assert denied is False
    assert "read-only shared contract path" in reason

    # 2. ExecutionStrategy.SERIAL_HANDOFF forces sequential execution and injects invariants.
    _init_repo(tmp_path)
    store, run_id = _confirmed_run(tmp_path)
    spec = store.get_run(run_id)["spec"]
    assert spec is not None
    design = DesignSpec.model_validate(
        {
            "design_id": "design-serial-1",
            "spec_id": spec.spec_id,
            "spec_hash": spec.content_hash,
            "architecture_summary": "Tightly coupled flag engine sharing single config table.",
            "shared_invariants": [
                {
                    "id": "INV-FLAG-01",
                    "category": "naming_convention",
                    "rule": "Prefix every feature flag key with 'ff_'.",
                    "enforcement_hint": "FlagKey validator",
                }
            ],
            "shared_contract_paths": ["contracts/"],
            "execution_strategy": ExecutionStrategy.SERIAL_HANDOFF.value,
            "strategy_rationale": "Single config ledger mutates shared schema sequentially.",
            "modules": [
                {
                    "id": "M1",
                    "name": "Core",
                    "responsibility": "Flag core",
                    "requirement_refs": ["R1"],
                    "target_paths": ["src/flags/"],
                }
            ],
            "interfaces": [],
            "status": SpecStatus.CONFIRMED.value,
        }
    )
    store.save_design(run_id, design, actor="design")
    store.record_approval(
        ApprovalRecord(
            run_id=run_id,
            target=ApprovalTarget.DESIGN_SPEC,
            target_id=design.design_id,
            target_hash=design.content_hash,
            approver="zeus",
            decision=ApprovalDecision.APPROVED,
        )
    )
    plan = TaskPlan.model_validate(
        {
            "plan_id": "plan-serial-1",
            "spec_id": spec.spec_id,
            "spec_hash": spec.content_hash,
            "design_id": design.design_id,
            "design_hash": design.content_hash,
            "tasks": [
                {
                    "task_id": "T0",
                    "title": "Pin contracts",
                    "objective": "Create contract file",
                    "requirement_refs": ["R1"],
                    "design_refs": ["M1"],
                    "kind": TaskKind.CONTRACT.value,
                    "write_scope": ["contracts/"],
                    "acceptance_checks": [
                        {
                            "id": "T0-C1",
                            "kind": CheckKind.COMMAND.value,
                            "description": "ok",
                            "command": "python -c 'pass'",
                        }
                    ],
                },
                {
                    "task_id": "T1",
                    "title": "Implement core",
                    "objective": "Implement flags",
                    "requirement_refs": ["R1"],
                    "design_refs": ["M1"],
                    "depends_on": ["T0"],
                    "kind": TaskKind.IMPLEMENT.value,
                    "write_scope": ["src/flags/"],
                    "acceptance_checks": [
                        {
                            "id": "T1-C1",
                            "kind": CheckKind.COMMAND.value,
                            "description": "ok",
                            "command": "python -c 'pass'",
                        }
                    ],
                },
            ],
        }
    )
    store.save_plan(run_id, plan, actor="orchestrate")
    store.record_approval(
        ApprovalRecord(
            run_id=run_id,
            target=ApprovalTarget.TASK_PLAN,
            target_id=plan.plan_id,
            target_hash=plan.content_hash,
            approver="zeus",
            decision=ApprovalDecision.APPROVED,
        )
    )

    seen_prompts: list[str] = []

    def _capture_responder(
        messages: Sequence[ChatMessage], tools: Sequence[ToolSpec]
    ) -> LLMResponse:
        del tools
        system = messages[0].content
        user_msg = messages[-1].content
        seen_prompts.append(user_msg)
        if "Independent Verifier (Argus)" in system:
            return LLMResponse(
                content=json.dumps(
                    {
                        "task_id": "T0",
                        "attempt": 1,
                        "passed": True,
                        "summary": "Verified invariants and seams.",
                        "findings": [],
                    }
                )
            )
        return LLMResponse(content="Done.")

    orch = OrchestrateStage(
        store=store,
        provider=MockLLMProvider(responder=_capture_responder),
        workspace_root=tmp_path,
    )
    snap = orch.run_execution(run_id=run_id)
    assert snap["status"] == RunStatus.COMPLETED
    assert any("INV-FLAG-01" in p for p in seen_prompts)
    assert any("READ-ONLY shared contracts" in p for p in seen_prompts)
    assert any("Prior Accepted Handoff Summary (serial_handoff)" in p for p in seen_prompts)
    events = [
        e for e in store.list_events(run_id) if e.event_type == "run.status_changed"
    ]
    assert any(e.payload.get("serial_execution") is True for e in events)


def test_propose_plan_repairs_broken_inline_python_check_via_validation_loop(
    tmp_path: Path,
) -> None:
    store, run_id = _confirmed_run(tmp_path)
    snap = store.get_run(run_id)
    spec = snap["spec"]
    assert spec is not None

    bad_check = {
        "id": "CHK-BAD",
        "kind": CheckKind.COMMAND.value,
        "description": "Frozen check",
        "command": 'python -c "import feature; try: x = 1; except Exception: pass"',
    }
    good_check = {
        "id": "CHK-GOOD",
        "kind": CheckKind.COMMAND.value,
        "description": "Frozen check",
        "command": "python -m pytest -q",
    }
    calls: list[str] = []

    def _repair_responder(
        messages: Sequence[ChatMessage], tools: Sequence[ToolSpec]
    ) -> LLMResponse:
        del tools
        calls.append(messages[-1].content)
        payload = _plan_payload(
            spec_id=spec.spec_id,
            spec_hash=spec.content_hash,
            file_name="feature.py",
            checks=[bad_check if len(calls) == 1 else good_check],
        )
        return LLMResponse(content=json.dumps(payload))

    orch = OrchestrateStage(
        store=store,
        provider=MockLLMProvider(responder=_repair_responder),
        workspace_root=tmp_path,
    )
    plan = orch.propose_plan(run_id=run_id)
    assert len(calls) == 2
    assert "[PLAN VALIDATION FAILED]" in calls[1]
    assert "invalid Python syntax" in calls[1]
    assert plan.tasks[0].acceptance_checks[0].command == "python -m pytest -q"


def test_timeout_retry_is_visible_while_an_independent_task_completes(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    store, run_id = _confirmed_run(tmp_path)
    spec = store.get_run(run_id)["spec"]
    plan = _plan_payload(
        spec_id=spec.spec_id,
        spec_hash=spec.content_hash,
        file_name="feature.py",
        checks=[
            {"id": "review", "kind": CheckKind.INDEPENDENT_REVIEW.value, "description": "Review"}
        ],
    )
    plan["tasks"].append({**plan["tasks"][0], "task_id": "T2", "write_scope": ["other.py"]})
    plan["tasks"].append(
        {**plan["tasks"][0], "task_id": "T3", "depends_on": ["T1"], "write_scope": ["child.py"]}
    )
    feature = _stage_responder(plan=plan, file_name="feature.py", review_passes=[True])
    other = _stage_responder(plan=plan, file_name="other.py", review_passes=[True])
    child = _stage_responder(plan=plan, file_name="child.py", review_passes=[True])
    timed_out = False
    during_retry = None
    after_recovery = None

    def respond(messages: Sequence[ChatMessage], tools: Sequence[ToolSpec]) -> LLMResponse:
        nonlocal timed_out, during_retry, after_recovery
        prompt = messages[1].content
        if "Task T1:" in prompt and not any(msg.role == "tool" for msg in messages):
            if not timed_out:
                timed_out = True
                raise TimeoutError("test call timed out")
            deadline = time.monotonic() + 5
            while not any(
                event.event_type == "task.accepted" and event.payload["task_id"] == "T2"
                for event in store.list_events(run_id)
            ):
                assert time.monotonic() < deadline, "Independent task did not finish during retry"
                time.sleep(0.01)
            during_retry = build_run_projection(store, tmp_path, run_id)
        elif "Task T1:" in prompt:
            after_recovery = build_run_projection(store, tmp_path, run_id)
        if "Task T2:" in prompt or '"task_id":"T2"' in prompt:
            return other(messages, tools)
        if "Task T3:" in prompt or '"task_id":"T3"' in prompt:
            return child(messages, tools)
        return feature(messages, tools)

    orch = OrchestrateStage(
        store=store, provider=MockLLMProvider(responder=respond), workspace_root=tmp_path
    )
    orch.propose_plan(run_id=run_id, auto_approve=True)
    assert orch.run_execution(run_id=run_id)["status"] == RunStatus.COMPLETED

    assert during_retry is not None
    active = {node["task_id"]: node for node in during_retry["dag"]["nodes"]}
    assert active["T1"]["status"] == "running"
    assert active["T1"]["provider_retry"]["phase"] == "started"
    assert active["T2"]["status"] == "accepted"
    assert active["T3"]["status"] == "waiting"
    assert active["T3"]["blocked_by"] == []
    assert during_retry["progress"]["accepted_tasks"] == 1
    assert during_retry["hitl_banner"] is None

    assert after_recovery is not None
    running = after_recovery["dag"]["nodes"][0]
    assert running["status"] == "running"
    assert running["provider_retry"] is None
    assert running["provider_retry_history"][-1]["phase"] == "recovered"

    final = build_run_projection(store, tmp_path, run_id)
    task = final["dag"]["nodes"][0]
    assert task["status"] == "accepted"
    assert task["attempt"] == 1
    assert task["provider_retry"] is None
    history = task["provider_retry_history"]
    assert [event["phase"] for event in history] == ["started", "recovered"]
    assert history[0]["retry_id"] == history[1]["retry_id"]
    assert history[0]["event_seq"] < history[1]["event_seq"]


def test_failed_timeout_retry_blocks_only_its_dependency_chain(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    store, run_id = _confirmed_run(tmp_path)
    spec = store.get_run(run_id)["spec"]
    plan = _plan_payload(
        spec_id=spec.spec_id,
        spec_hash=spec.content_hash,
        file_name="feature.py",
        max_cycles=1,
        checks=[
            {"id": "review", "kind": CheckKind.INDEPENDENT_REVIEW.value, "description": "Review"}
        ],
    )
    root = plan["tasks"][0]
    root["deliverables"] = ["feature.py"]
    plan["tasks"].extend(
        {
            **root,
            "task_id": task_id,
            "depends_on": deps,
            "write_scope": [path],
            "deliverables": [path],
        }
        for task_id, deps, path in [
            ("T2", [], "other.py"),
            ("T3", ["T1", "T2"], "child.py"),
            ("T4", ["T3"], "release.py"),
        ]
    )
    other = _stage_responder(plan=plan, file_name="other.py", review_passes=[True])
    while_blocked = None

    def respond(messages: Sequence[ChatMessage], tools: Sequence[ToolSpec]) -> LLMResponse:
        nonlocal while_blocked
        prompt = messages[1].content
        if "Task T1:" in prompt:
            raise TimeoutError("flag service timed out")
        if "Task T2:" in prompt and not any(msg.role == "tool" for msg in messages):
            deadline = time.monotonic() + 5
            while not any(
                event.event_type == "task.blocked" for event in store.list_events(run_id)
            ):
                assert time.monotonic() < deadline, "Expected T1 to reach its task cycle limit"
                time.sleep(0.01)
            while_blocked = build_run_projection(store, tmp_path, run_id)
        return other(messages, tools)

    orch = OrchestrateStage(
        store=store, provider=MockLLMProvider(responder=respond), workspace_root=tmp_path
    )
    orch.propose_plan(run_id=run_id, auto_approve=True)
    snap = orch.run_execution(run_id=run_id)
    assert snap["status"] == RunStatus.AWAITING_HUMAN

    assert while_blocked is not None
    nodes = {node["task_id"]: node for node in while_blocked["dag"]["nodes"]}
    assert nodes["T1"]["status"] == "blocked"
    assert nodes["T1"]["provider_retry"] is None
    assert [row["phase"] for row in nodes["T1"]["provider_retry_history"]] == ["started", "failed"]
    assert nodes["T2"]["status"] == "running"
    assert nodes["T2"]["blocked_by"] == []
    assert nodes["T3"]["blocked_by"] == ["T1"]
    assert nodes["T4"]["blocked_by"] == ["T1"]
    banner = while_blocked["hitl_banner"]
    assert banner["task_id"] == "T1"
    assert banner["affected_task_ids"] == ["T1", "T3", "T4"]
    assert "flag service timed out" in banner["detail"]
    assert "retry" in banner["actions"]
    assert {s["id"]: s["status"] for s in while_blocked["stage_pipeline"]}["execute"] == "active"
    assert {(e["from"], e["to"]) for e in while_blocked["dag"]["edges"] if e["blocked_edge"]} == {
        ("T1", "T3"),
        ("T3", "T4"),
    }

    final = build_run_projection(store, tmp_path, run_id)
    assert final["progress"]["accepted_tasks"] == 1
    assert {s["id"]: s["status"] for s in final["stage_pipeline"]}["execute"] == "blocked"
    orch.retry_blocked_task(
        operation_id="retry-t1",
        run_id=run_id,
        task_id="T1",
        expected_state_revision=snap["state_revision"],
    )
    after_manual_retry = build_run_projection(store, tmp_path, run_id)
    nodes = {node["task_id"]: node for node in after_manual_retry["dag"]["nodes"]}
    assert nodes["T1"]["status"] == "ready"
    assert "flag service timed out" in nodes["T1"]["last_error"]
    assert nodes["T1"]["diagnostic_output"] == "就绪待调度"
    assert nodes["T1"]["provider_retry"] is None
    assert nodes["T3"]["blocked_by"] == []
    # A ready task on an awaiting run now surfaces the resume banner (Issue #54).
    assert after_manual_retry["hitl_banner"]["kind"] == "ready_to_resume"
    assert after_manual_retry["hitl_banner"]["actions"] == ["retry_workflow"]


def _approve_plan(
    store: RunStore,
    run_id: str,
    plan: TaskPlan,
    *,
    design: DesignSpec | None = None,
) -> None:
    """Persist and approve a hand-built plan (optionally pinning an approved design)."""
    if design is not None:
        store.save_design(run_id, design, actor="design")
        store.record_approval(
            ApprovalRecord(
                run_id=run_id,
                target=ApprovalTarget.DESIGN_SPEC,
                target_id=design.design_id,
                target_hash=design.content_hash,
                approver="zeus",
                decision=ApprovalDecision.APPROVED,
            )
        )
    store.save_plan(run_id, plan, actor="orchestrate")
    store.record_approval(
        ApprovalRecord(
            run_id=run_id,
            target=ApprovalTarget.TASK_PLAN,
            target_id=plan.plan_id,
            target_hash=plan.content_hash,
            approver="zeus",
            decision=ApprovalDecision.APPROVED,
        )
    )


def _contract_design(spec: GoalSpec, *, shared_paths: list[str]) -> DesignSpec:
    return DesignSpec.model_validate(
        {
            "design_id": "design-diff-gate",
            "spec_id": spec.spec_id,
            "spec_hash": spec.content_hash,
            "architecture_summary": "Shared contracts pin cross-task boundaries.",
            "shared_contract_paths": shared_paths,
            "execution_strategy": ExecutionStrategy.CONTRACT_FIRST_PARALLEL.value,
            "strategy_rationale": "Wave 0 contract task locks shared paths before implement.",
            "modules": [
                {
                    "id": "M1",
                    "name": "Core",
                    "responsibility": "Feature flag core",
                    "requirement_refs": ["R1"],
                    "target_paths": ["src/"],
                }
            ],
            "status": SpecStatus.CONFIRMED.value,
        }
    )


def _single_task_plan(
    *,
    spec: GoalSpec,
    write_scope: list[str],
    checks: list[dict[str, Any]],
    max_cycles: int = 3,
    kind: TaskKind = TaskKind.IMPLEMENT,
) -> TaskPlan:
    return TaskPlan.model_validate(
        {
            "plan_id": "plan-diff-gate",
            "spec_id": spec.spec_id,
            "spec_hash": spec.content_hash,
            "tasks": [
                {
                    "task_id": "T1",
                    "title": "Implement the flag module",
                    "objective": "Implement the assigned flag module",
                    "requirement_refs": ["R1"],
                    "kind": kind.value,
                    "write_scope": write_scope,
                    "max_cycles": max_cycles,
                    "acceptance_checks": checks,
                }
            ],
        }
    )


def _marker_command_check() -> dict[str, Any]:
    return {
        "id": "CHK-1",
        "kind": CheckKind.COMMAND.value,
        "description": "Write a marker proving acceptance checks ran",
        "command": (
            "python3 -c 'from pathlib import Path; "
            'Path("check-ran.txt").write_text("ran")\''
        ),
    }


def _bash_action(command: str) -> Callable[[Path], ToolCall]:
    return lambda sandbox: ToolCall(  # noqa: ARG005 - action bound to one sandbox
        id="bash-1",
        name="run_bash",
        arguments={"command": command},
    )


def _write_action(file_name: str, content: str) -> Callable[[Path], ToolCall]:
    return lambda sandbox: ToolCall(
        id="write-1",
        name="write_file",
        arguments={"path": str(sandbox / file_name), "content": content},
    )


def _scope_gate_responder(
    actions: Sequence[Callable[[Path], ToolCall]],
) -> tuple[Responder, dict[str, int]]:
    """Replay one worker action per attempt; the last action repeats if needed.

    Review verdicts always pass so a requeue can only come from the Diff Gate.
    """
    state = {"attempts": 0, "reviews": 0}

    def responder(messages: Sequence[ChatMessage], tools: Sequence[ToolSpec]) -> LLMResponse:
        del tools
        system_message = messages[0].content
        user_message = messages[1].content if len(messages) > 1 else ""
        if "Independent Verifier" in system_message:
            state["reviews"] += 1
            return LLMResponse(
                content=json.dumps(
                    {
                        "task_id": "T1",
                        "attempt": 1,
                        "passed": True,
                        "summary": "Scope-clean implementation verified.",
                        "findings": [],
                    }
                )
            )
        if "Execution Engineer" not in system_message:
            raise AssertionError(f"Unexpected system prompt: {system_message[:80]}")
        match = re.search(r"Sandbox workspace root: (.+)", user_message)
        assert match is not None
        sandbox = Path(match.group(1).strip())
        if not any(msg.role == "tool" for msg in messages):
            action = actions[min(state["attempts"], len(actions) - 1)]
            state["attempts"] += 1
            return LLMResponse(content="acting", tool_calls=(action(sandbox),))
        return LLMResponse(content="finished")

    return responder, state


def _gate_rejections(store: RunStore, run_id: str) -> list[Any]:
    return [e for e in store.list_events(run_id) if e.event_type == "task.diff_gate_rejected"]


def test_diff_gate_rejects_tracked_edit_outside_write_scope_before_checks(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    notes = tmp_path / "docs" / "notes.md"
    notes.parent.mkdir()
    notes.write_text("original\n", encoding="utf-8")
    subprocess.run(["git", "add", "docs/notes.md"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "docs"], cwd=tmp_path, check=True, capture_output=True)

    store, run_id = _confirmed_run(tmp_path)
    spec = store.get_run(run_id)["spec"]
    assert spec is not None
    plan = _single_task_plan(
        spec=spec,
        write_scope=["feature.py"],
        max_cycles=2,
        checks=[
            _marker_command_check(),
            {
                "id": "CHK-REV-1",
                "kind": CheckKind.INDEPENDENT_REVIEW.value,
                "description": "Cold-start independent review",
            },
        ],
    )
    _approve_plan(store, run_id, plan)
    responder, state = _scope_gate_responder(
        [
            _bash_action("printf 'tampered\\n' >> docs/notes.md"),
            _write_action("feature.py", "FLAG = True\n"),
        ]
    )
    orch = OrchestrateStage(
        store=store, provider=MockLLMProvider(responder=responder), workspace_root=tmp_path
    )
    snap = orch.run_execution(run_id=run_id)

    assert snap["status"] == RunStatus.COMPLETED
    task = snap["plan"].tasks[0]
    assert task.status == TaskStatus.ACCEPTED
    assert task.attempt == 2
    assert state["attempts"] == 2
    assert state["reviews"] == 1

    events = store.list_events(run_id)
    rejections = [e for e in events if e.event_type == "task.diff_gate_rejected"]
    assert len(rejections) == 1
    payload = rejections[0].payload
    assert payload["task_id"] == "T1"
    assert payload["attempt"] == 1
    assert payload["violating_paths"] == ["docs/notes.md"]
    assert payload["allowed_write_scope"] == ["feature.py"]
    assert payload["forbidden_globs"] == []

    # Gate rejection precedes every check and review event; the retry is the only review.
    check_events = [e for e in events if e.event_type == "check.command_completed"]
    assert check_events
    assert all(e.event_seq > rejections[0].event_seq for e in check_events)
    review_events = [e for e in events if e.event_type == "review.verdict_recorded"]
    assert [e.payload["attempt"] for e in review_events] == [2]
    assert len([e for e in events if e.event_type == "task.requeued"]) == 1

    attempt_dir = tmp_path / ".olympus" / "runs" / run_id / "artifacts" / "T1" / "attempt-1"
    assert (attempt_dir / "diff.patch").is_file()
    assert (attempt_dir / "MANIFEST.md").is_file()
    assert not list(attempt_dir.glob("check-*.log"))
    assert not list(attempt_dir.glob("verdict-cycle-*.json"))

    overlay = tmp_path / ".olympus" / "runs" / run_id / "accepted_overlay"
    assert (overlay / "feature.py").is_file()
    assert not (overlay / "docs" / "notes.md").exists()
    assert notes.read_text(encoding="utf-8") == "original\n"


def test_diff_gate_rejects_untracked_file_outside_write_scope(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    store, run_id = _confirmed_run(tmp_path)
    spec = store.get_run(run_id)["spec"]
    assert spec is not None
    plan = _single_task_plan(
        spec=spec,
        write_scope=["feature.py"],
        max_cycles=1,
        checks=[_marker_command_check()],
    )
    _approve_plan(store, run_id, plan)
    responder, state = _scope_gate_responder(
        [_bash_action("mkdir -p stray && printf 'x\\n' > stray/notes.txt")]
    )
    orch = OrchestrateStage(
        store=store, provider=MockLLMProvider(responder=responder), workspace_root=tmp_path
    )
    snap = orch.run_execution(run_id=run_id)

    assert snap["status"] == RunStatus.AWAITING_HUMAN
    task = snap["plan"].tasks[0]
    assert task.status == TaskStatus.BLOCKED
    assert task.review_cycle == 1
    assert task.last_error is not None
    assert "Diff Gate" in task.last_error
    assert "stray/notes.txt" in task.last_error
    assert "feature.py" in task.last_error
    assert state["reviews"] == 0

    rejections = _gate_rejections(store, run_id)
    assert len(rejections) == 1
    payload = rejections[0].payload
    assert payload["violating_paths"] == ["stray/notes.txt"]
    assert payload["allowed_write_scope"] == ["feature.py"]

    attempt_dir = tmp_path / ".olympus" / "runs" / run_id / "artifacts" / "T1" / "attempt-1"
    assert not list(attempt_dir.glob("check-*.log"))
    overlay = tmp_path / ".olympus" / "runs" / run_id / "accepted_overlay"
    assert not overlay.exists() or not any(overlay.iterdir())


def test_diff_gate_rejects_forbidden_contract_path_even_when_write_scope_matches(
    tmp_path: Path,
) -> None:
    _init_repo(tmp_path)
    store, run_id = _confirmed_run(tmp_path)
    spec = store.get_run(run_id)["spec"]
    assert spec is not None
    design = _contract_design(spec, shared_paths=["src/contracts/"])
    plan = _single_task_plan(
        spec=spec,
        write_scope=["src/"],
        max_cycles=1,
        checks=[_marker_command_check()],
    )
    _approve_plan(store, run_id, plan, design=design)
    responder, state = _scope_gate_responder(
        [_bash_action("mkdir -p src/contracts && printf 'x\\n' > src/contracts/api.py")]
    )
    orch = OrchestrateStage(
        store=store, provider=MockLLMProvider(responder=responder), workspace_root=tmp_path
    )
    snap = orch.run_execution(run_id=run_id)

    task = snap["plan"].tasks[0]
    assert task.status == TaskStatus.BLOCKED
    assert task.last_error is not None
    assert "forbidden" in task.last_error
    assert "src/contracts/api.py" in task.last_error
    assert "src/contracts/" in task.last_error
    assert state["reviews"] == 0

    rejections = _gate_rejections(store, run_id)
    assert len(rejections) == 1
    payload = rejections[0].payload
    assert payload["violating_paths"] == ["src/contracts/api.py"]
    assert payload["allowed_write_scope"] == ["src/"]
    assert payload["forbidden_globs"] == ["src/contracts/"]
    assert payload["violations"][0]["reason"] == "forbidden"


def test_diff_gate_allows_contract_task_to_edit_shared_contract_paths(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    store, run_id = _confirmed_run(tmp_path)
    spec = store.get_run(run_id)["spec"]
    assert spec is not None
    design = _contract_design(spec, shared_paths=["contracts/"])
    plan = _single_task_plan(
        spec=spec,
        write_scope=["contracts/"],
        kind=TaskKind.CONTRACT,
        checks=[
            {
                "id": "CHK-1",
                "kind": CheckKind.COMMAND.value,
                "description": "Contract file exists with pinned content",
                "command": (
                    "python3 -c 'from pathlib import Path; "
                    'assert Path("contracts/api.py").read_text() == "PINNED = 1\\n"\''
                ),
            }
        ],
    )
    _approve_plan(store, run_id, plan, design=design)
    responder, state = _scope_gate_responder([_write_action("contracts/api.py", "PINNED = 1\n")])
    orch = OrchestrateStage(
        store=store, provider=MockLLMProvider(responder=responder), workspace_root=tmp_path
    )
    snap = orch.run_execution(run_id=run_id)

    assert snap["status"] == RunStatus.COMPLETED
    assert snap["plan"].tasks[0].status == TaskStatus.ACCEPTED
    assert state["reviews"] == 1
    assert _gate_rejections(store, run_id) == []
    overlay = tmp_path / ".olympus" / "runs" / run_id / "accepted_overlay"
    assert (overlay / "contracts" / "api.py").read_text(encoding="utf-8") == "PINNED = 1\n"


def test_diff_gate_passes_tracked_and_untracked_changes_inside_write_scope(
    tmp_path: Path,
) -> None:
    _init_repo(tmp_path)
    app = tmp_path / "app.py"
    app.write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "app.py"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "app"], cwd=tmp_path, check=True, capture_output=True)

    store, run_id = _confirmed_run(tmp_path)
    spec = store.get_run(run_id)["spec"]
    assert spec is not None
    plan = _single_task_plan(
        spec=spec,
        write_scope=["app.py", "pkg/"],
        checks=[
            {
                "id": "CHK-1",
                "kind": CheckKind.COMMAND.value,
                "description": "Tracked edit and new module are both present",
                "command": (
                    "python3 -c 'from pathlib import Path; "
                    'assert "VALUE = 2" in Path("app.py").read_text(); '
                    'assert Path("pkg/util.py").is_file()\''
                ),
            }
        ],
    )
    _approve_plan(store, run_id, plan)
    responder, state = _scope_gate_responder(
        [
            _bash_action(
                "printf 'VALUE = 2\\n' > app.py && mkdir -p pkg && printf 'X = 1\\n' > pkg/util.py"
            )
        ]
    )
    orch = OrchestrateStage(
        store=store, provider=MockLLMProvider(responder=responder), workspace_root=tmp_path
    )
    snap = orch.run_execution(run_id=run_id)

    assert snap["status"] == RunStatus.COMPLETED
    assert snap["plan"].tasks[0].status == TaskStatus.ACCEPTED
    assert state["reviews"] == 1
    assert _gate_rejections(store, run_id) == []
    overlay = tmp_path / ".olympus" / "runs" / run_id / "accepted_overlay"
    assert (overlay / "app.py").read_text(encoding="utf-8") == "VALUE = 2\n"
    assert (overlay / "pkg" / "util.py").read_text(encoding="utf-8") == "X = 1\n"
    # SIMULATION mode with apply_to_workspace=False never touches the host checkout.
    assert app.read_text(encoding="utf-8") == "VALUE = 1\n"


def test_diff_gate_validates_non_git_sandbox_without_crashing(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("# Sample\n", encoding="utf-8")
    store, run_id = _confirmed_run(tmp_path)
    spec = store.get_run(run_id)["spec"]
    assert spec is not None
    plan = _single_task_plan(
        spec=spec,
        write_scope=["feature.py"],
        max_cycles=2,
        checks=[_marker_command_check()],
    )
    _approve_plan(store, run_id, plan)
    responder, state = _scope_gate_responder(
        [
            _bash_action("printf 'x\\n' > stray.py"),
            _write_action("feature.py", "FLAG = True\n"),
        ]
    )
    orch = OrchestrateStage(
        store=store, provider=MockLLMProvider(responder=responder), workspace_root=tmp_path
    )
    snap = orch.run_execution(run_id=run_id)

    assert snap["status"] == RunStatus.COMPLETED
    assert snap["plan"].tasks[0].status == TaskStatus.ACCEPTED
    assert snap["plan"].tasks[0].attempt == 2
    assert state["attempts"] == 2
    assert state["reviews"] == 1

    rejections = _gate_rejections(store, run_id)
    assert len(rejections) == 1
    assert rejections[0].payload["violating_paths"] == ["stray.py"]
    overlay = tmp_path / ".olympus" / "runs" / run_id / "accepted_overlay"
    assert (overlay / "feature.py").is_file()
    assert not (overlay / "stray.py").exists()


def test_diff_gate_rejects_deleted_and_renamed_paths_outside_write_scope(
    tmp_path: Path,
) -> None:
    _init_repo(tmp_path)
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "notes.md").write_text("original\n", encoding="utf-8")
    (docs / "legacy.txt").write_text("legacy\n", encoding="utf-8")
    subprocess.run(["git", "add", "docs/"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "docs"], cwd=tmp_path, check=True, capture_output=True)

    store, run_id = _confirmed_run(tmp_path)
    spec = store.get_run(run_id)["spec"]
    assert spec is not None
    plan = _single_task_plan(
        spec=spec,
        write_scope=["feature.py"],
        max_cycles=1,
        checks=[_marker_command_check()],
    )
    _approve_plan(store, run_id, plan)
    responder, state = _scope_gate_responder(
        [_bash_action("mv docs/notes.md docs/notes-renamed.md && rm docs/legacy.txt")]
    )
    orch = OrchestrateStage(
        store=store, provider=MockLLMProvider(responder=responder), workspace_root=tmp_path
    )
    snap = orch.run_execution(run_id=run_id)

    task = snap["plan"].tasks[0]
    assert task.status == TaskStatus.BLOCKED
    assert task.last_error is not None
    for path in ("docs/legacy.txt", "docs/notes-renamed.md", "docs/notes.md"):
        assert path in task.last_error
    assert state["reviews"] == 0

    rejections = _gate_rejections(store, run_id)
    assert len(rejections) == 1
    assert rejections[0].payload["violating_paths"] == [
        "docs/legacy.txt",
        "docs/notes-renamed.md",
        "docs/notes.md",
    ]
    attempt_dir = tmp_path / ".olympus" / "runs" / run_id / "artifacts" / "T1" / "attempt-1"
    assert not list(attempt_dir.glob("check-*.log"))
    # The host checkout keeps both documents intact: nothing is auto-reverted or promoted.
    assert (docs / "notes.md").is_file()
    assert (docs / "legacy.txt").is_file()
    assert not (docs / "notes-renamed.md").exists()
