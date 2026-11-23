"""Seam 1: offline deterministic tests for ``olympus session start/reply``.

Every test drives the real CLI through Typer's ``CliRunner`` against a real
temporary git repository, with a scripted ``MockLLMProvider`` and an external
state directory, so the observable contract is exactly what a Harbor adapter sees.
"""

from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Sequence
from pathlib import Path

import pytest
from typer.testing import CliRunner

from olympus.cli.app import app
from olympus.core.contracts import ApprovalTarget, RunStatus
from olympus.core.store import RunStore
from olympus.graph.loader import WORKFLOW_DIR
from olympus.providers.base import ChatMessage, LLMResponse, MockLLMProvider, ToolCall, ToolSpec
from olympus.session.contracts import PauseKind
from olympus.session.router import parse_reply
from olympus.session.runner import resolve_state_dir

RUNNER = CliRunner()

_CALC_CODE = (
    "def divide(a, b):\n"
    "    if b == 0:\n"
    "        raise ValueError('zero')\n"
    "    return a / b\n"
)
_CALC_CODE_V2 = _CALC_CODE + "\n\ndef subtract(a, b):\n    return a - b\n"

_DIVIDE_CHECK = (
    "python3 -c \"import calculator; assert calculator.divide(6, 2) == 3\""
)
_SUBTRACT_CHECK = (
    "python3 -c \"import calculator; assert calculator.subtract(5, 2) == 3\""
)


# --------------------------------------------------------------------------- fixtures


def _init_git_repo(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "README.md").write_text("# Calculator\n", encoding="utf-8")
    subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "dev@example.com"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "Dev"], cwd=root, check=True)
    subprocess.run(["git", "add", "README.md"], cwd=root, check=True)
    subprocess.run(
        ["git", "commit", "-m", "init"], cwd=root, check=True, capture_output=True
    )


def _git_status(root: Path) -> str:
    return subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout


class SessionScript:
    """Deterministic provider script keyed by stage system-prompt markers."""

    def __init__(self) -> None:
        self.align: list[str] = []
        self.design: list[str] = []
        self.plan: list[str] = []
        self.router: list[str] = []
        self.workers: list[tuple[str, str]] = []
        self.worker_rounds = 0

    def provider(self) -> MockLLMProvider:
        return MockLLMProvider(responder=self._respond)

    def _respond(
        self, messages: Sequence[ChatMessage], tools: Sequence[ToolSpec]
    ) -> LLMResponse:
        del tools
        system = messages[0].content if messages else ""
        user = messages[1].content if len(messages) > 1 else ""
        if "Session Reply Router" in system:
            assert self.router, "router response queue exhausted"
            return LLMResponse(content=self.router.pop(0))
        if "Requirements Alignment Architect" in system:
            assert self.align, "align response queue exhausted"
            return LLMResponse(content=self.align.pop(0))
        if "White-Box Solution Architect" in system:
            assert self.design, "design response queue exhausted"
            return LLMResponse(content=self.design.pop(0))
        if "Task Planner" in system:
            assert self.plan, "plan response queue exhausted"
            return LLMResponse(content=self.plan.pop(0))
        if "Independent Verifier" in system:
            return LLMResponse(
                content=json.dumps(
                    {
                        "task_id": "T1",
                        "attempt": 1,
                        "passed": True,
                        "summary": "Implementation matches the frozen contract.",
                        "findings": [],
                    }
                )
            )
        if "Execution Engineer" in system:
            if any(message.role == "tool" for message in messages):
                self.worker_rounds += 1
                return LLMResponse(content="Implementation finished.")
            assert self.workers, "worker response queue exhausted"
            relative, content = self.workers.pop(0)
            match = re.search(r"Sandbox workspace root: (.+)", user)
            assert match is not None
            sandbox = Path(match.group(1).strip())
            return LLMResponse(
                content="Writing the calculator implementation.",
                tool_calls=(
                    ToolCall(
                        id=f"write-{self.worker_rounds}",
                        name="write_file",
                        arguments={"path": str(sandbox / relative), "content": content},
                    ),
                ),
            )
        raise AssertionError(f"Unexpected system prompt: {system[:120]}")


def _patch_provider(monkeypatch: pytest.MonkeyPatch, script: SessionScript) -> None:
    instance = script.provider()

    def _builder(
        *, provider: str | None = None, model: str | None = None, base_url: str | None = None
    ) -> MockLLMProvider:
        del provider, model, base_url
        return instance

    monkeypatch.setattr("olympus.cli.session_cmd.build_session_provider", _builder)


def _goal_spec(
    *,
    ambiguity: bool = False,
    two_ambiguities: bool = False,
    include_subtract: bool = False,
    version: int = 1,
) -> dict[str, object]:
    requirements: list[dict[str, object]] = [
        {
            "id": "R1",
            "description": (
                "Expose divide(a, b) returning a / b and rejecting a zero divisor"
            ),
            "actor": "developer",
            "ears_criteria": [
                {
                    "id": "R1-C1",
                    "pattern": "ubiquitous",
                    "outcome": "return a / b for any non-zero divisor",
                },
                {
                    "id": "R1-C2",
                    "pattern": "unwanted_behavior",
                    "condition": "b equals zero",
                    "outcome": "raise ValueError with message 'zero'",
                },
            ],
        }
    ]
    if include_subtract:
        requirements.append(
            {
                "id": "R2",
                "description": "Expose subtract(a, b) returning a - b",
                "actor": "developer",
                "ears_criteria": [
                    {
                        "id": "R2-C1",
                        "pattern": "ubiquitous",
                        "outcome": "return a - b for integer inputs",
                    }
                ],
            }
        )
    spec: dict[str, object] = {
        "spec_id": "spec-calc",
        "version": version,
        "goal": "Build a small Python calculator library for developers",
        "deliverable_kind": "library",
        "observable_surfaces": ["calculator module"],
        "actors": ["developer"],
        "test_seams": ["calculator.divide(a, b)"],
        "requirements": requirements,
        "out_of_scope": ["graphical interface"],
        "constraints": [],
    }
    if ambiguity:
        spec["open_ambiguities"] = [
            {
                "id": "AMB-1",
                "question": "What behavior should divide(1, 0) have?",
                "impact": "Error contract",
                "options": ["raise ValueError", "return None"],
                "blocking": True,
            }
        ]
        if two_ambiguities:
            spec["open_ambiguities"].append(
                {
                    "id": "AMB-2",
                    "question": "Which numeric types must be accepted?",
                    "impact": "Input contract",
                    "options": ["integers only", "int and float"],
                    "blocking": True,
                }
            )
    return spec


def _design(*, requirements: tuple[str, ...] = ("R1",)) -> dict[str, object]:
    return {
        "architecture_summary": "A single calculator module with an explicit error contract",
        "spec_id": "spec-calc",
        "spec_hash": "sha256:pending",
        "shared_invariants": [
            {
                "id": "INV-ERR",
                "category": "error_envelope",
                "rule": "Raise ValueError('zero') when the divisor is zero",
                "enforcement_hint": "guard clause before division",
                "applies_to": ["M1"],
            }
        ],
        "shared_contract_paths": [],
        "modules": [
            {
                "id": "M1",
                "name": "Calculator",
                "responsibility": "Provide arithmetic helpers with explicit errors",
                "requirement_refs": list(requirements),
                "target_paths": ["calculator.py"],
            }
        ],
        "data_models": [],
        "interfaces": [],
        "execution_strategy": "serial_handoff",
        "strategy_rationale": (
            "One module and one task keep the change atomic and easy to verify."
        ),
    }


def _plan(
    *,
    requirements: tuple[str, ...] = ("R1",),
    objective: str = "Implement divide(a, b) in calculator.py with a zero guard",
    command: str = _DIVIDE_CHECK,
    version: int = 1,
) -> dict[str, object]:
    return {
        "plan_id": "plan-calc",
        "spec_id": "spec-calc",
        "spec_hash": "sha256:pending",
        "version": version,
        "tasks": [
            {
                "task_id": "T1",
                "title": "Implement calculator helpers",
                "objective": objective,
                "requirement_refs": list(requirements),
                "design_refs": ["M1"],
                "write_scope": ["calculator.py"],
                "acceptance_checks": [
                    {
                        "id": "C1",
                        "kind": "command",
                        "description": "Verify the calculator contract",
                        "command": command,
                    }
                ],
            }
        ],
    }


def _invalid_overlapping_plan() -> dict[str, object]:
    return {
        "plan_id": "plan-bad",
        "spec_id": "spec-calc",
        "spec_hash": "sha256:pending",
        "version": 1,
        "tasks": [
            {
                "task_id": "T1",
                "title": "Write calculator",
                "objective": "Implement the calculator module",
                "requirement_refs": ["R1"],
                "design_refs": ["M1"],
                "write_scope": ["calculator.py"],
                "acceptance_checks": [
                    {
                        "id": "C1",
                        "kind": "command",
                        "description": "Check",
                        "command": _DIVIDE_CHECK,
                    }
                ],
            },
            {
                "task_id": "T2",
                "title": "Write calculator again",
                "objective": "Duplicate the same module concurrently",
                "requirement_refs": ["R1"],
                "design_refs": ["M1"],
                "write_scope": ["calculator.py"],
                "acceptance_checks": [
                    {
                        "id": "C2",
                        "kind": "command",
                        "description": "Check",
                        "command": _DIVIDE_CHECK,
                    }
                ],
            },
        ],
    }


def _invoke(args: list[str]) -> tuple[int, dict[str, object]]:
    result = RUNNER.invoke(app, ["session", *args])
    assert result.stdout.strip(), f"no JSON envelope on stdout (stderr={result.stderr})"
    return result.exit_code, json.loads(result.stdout)


def _base(repo: Path, state: Path) -> list[str]:
    return [
        "--workspace",
        str(repo),
        "--state-dir",
        str(state),
        "--provider",
        "mock",
    ]


# --------------------------------------------------------------------------- tests


def test_start_clarify_reply_then_execute_completes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    state = tmp_path / "state"
    script = SessionScript()
    script.align.append(json.dumps(_goal_spec(ambiguity=True, two_ambiguities=True)))
    script.design.append(json.dumps(_design()))
    script.plan.append(json.dumps(_plan()))
    script.router.append(
        json.dumps(
            {
                "action": "clarify",
                "answers": {
                    "AMB-1": "raise ValueError('zero')",
                    "AMB-2": "int and float",
                },
            }
        )
    )
    script.workers.append(("calculator.py", _CALC_CODE))
    _patch_provider(monkeypatch, script)

    code, start = _invoke(["start", "Build a calculator library", *_base(repo, state)])
    assert code == 0
    assert start["status"] == "needs_input"
    assert start["pause_kind"] == "clarify"
    assert start["unresolved_blockers"] == ["ambiguity:AMB-1", "ambiguity:AMB-2"]
    assert start["step_mode"] == "execute"
    assert start["workflow"] == "full_sdlc_tdd"
    assert start["changed_files"] == []
    assert start["error"] is None
    assert "AMB-1" in str(start["message"])
    assert "AMB-2" in str(start["message"])
    assert _git_status(repo) == ""
    assert not (repo / ".olympus").exists()

    code, reply = _invoke(
        [
            "reply",
            str(start["run_id"]),
            "Make divide(1, 0) raise ValueError('zero'), and accept int and float inputs.",
            *_base(repo, state),
        ]
    )
    assert code == 0, json.dumps(reply, indent=2)
    assert reply["status"] == "completed"
    assert reply["pause_kind"] is None
    assert reply["changed_files"] == ["calculator.py"]
    assert (repo / "calculator.py").read_text(encoding="utf-8") == _CALC_CODE
    assert not (repo / ".olympus").exists()
    assert (state / "olympus.db").is_file()

    store = RunStore(state / "olympus.db")
    snap = store.get_run(str(start["run_id"]))
    assert snap["status"].value == "completed"
    assert snap["design"] is not None and snap["plan"] is not None
    assert store.is_approved(
        run_id=str(start["run_id"]),
        target=ApprovalTarget.DESIGN_SPEC,
        target_id=snap["design"].design_id,
        target_hash=snap["design"].content_hash,
    )
    assert store.is_approved(
        run_id=str(start["run_id"]),
        target=ApprovalTarget.TASK_PLAN,
        target_id=snap["plan"].plan_id,
        target_hash=snap["plan"].content_hash,
    )


def test_plan_only_handoff_mutates_nothing_then_execute_applies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    state = tmp_path / "state"
    script = SessionScript()
    script.align.append(json.dumps(_goal_spec()))
    script.design.append(json.dumps(_design()))
    script.plan.append(json.dumps(_plan()))
    script.router.append(json.dumps({"action": "approve", "message": "go ahead"}))
    script.workers.append(("calculator.py", _CALC_CODE))
    _patch_provider(monkeypatch, script)

    code, started = _invoke(
        [
            "start",
            "Build a calculator library",
            "--step-mode",
            "plan_only",
            *_base(repo, state),
        ]
    )
    assert code == 0
    assert started["status"] == "needs_input"
    assert started["pause_kind"] == "plan_handoff"
    assert started["step_mode"] == "plan_only"
    assert started["changed_files"] == []
    assert _git_status(repo) == ""
    assert not (repo / ".olympus").exists()
    assert (state / "olympus.db").is_file()

    code, executed = _invoke(
        [
            "reply",
            str(started["run_id"]),
            "Plan looks good, implement it.",
            "--step-mode",
            "execute",
            *_base(repo, state),
        ]
    )
    assert code == 0
    assert executed["status"] == "completed"
    assert executed["changed_files"] == ["calculator.py"]
    assert (repo / "calculator.py").is_file()
    assert not (repo / ".olympus").exists()


def test_default_gate_policy_pauses_for_every_approval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    state = tmp_path / "state"
    script = SessionScript()
    script.align.append(json.dumps(_goal_spec()))
    script.design.append(json.dumps(_design()))
    script.plan.append(json.dumps(_plan()))
    script.router.extend(json.dumps({"action": "approve"}) for _ in range(3))
    script.workers.append(("calculator.py", _CALC_CODE))
    _patch_provider(monkeypatch, script)

    code, first = _invoke(
        ["start", "Build a calculator library", "--gate-policy", "default", *_base(repo, state)]
    )
    assert code == 0
    assert first["status"] == "needs_input"
    assert first["pause_kind"] == "approval"
    run_id = str(first["run_id"])

    code, second = _invoke(["reply", run_id, "Approve the spec.", *_base(repo, state)])
    assert code == 0
    assert second["pause_kind"] == "approval", json.dumps(second, indent=2)
    assert "DesignSpec" in str(second["message"])

    code, third = _invoke(["reply", run_id, "Approve the design.", *_base(repo, state)])
    assert code == 0
    assert third["pause_kind"] == "approval"
    assert "TaskPlan" in str(third["message"])

    code, final = _invoke(["reply", run_id, "Approve the plan.", *_base(repo, state)])
    assert code == 0
    assert final["status"] == "completed"
    assert (repo / "calculator.py").is_file()


def test_free_form_revision_restarts_alignment_with_feedback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    state = tmp_path / "state"
    script = SessionScript()
    script.align.extend(
        [
            json.dumps(_goal_spec(ambiguity=True)),
            json.dumps(_goal_spec(version=2)),
        ]
    )
    script.design.append(json.dumps(_design()))
    script.plan.append(json.dumps(_plan()))
    script.router.append(
        json.dumps(
            {
                "action": "revise",
                "feedback": "Do not ask about zero division; always raise ValueError instead.",
            }
        )
    )
    script.workers.append(("calculator.py", _CALC_CODE))
    _patch_provider(monkeypatch, script)

    code, started = _invoke(["start", "Build a calculator library", *_base(repo, state)])
    assert code == 0
    assert started["pause_kind"] == "clarify"
    run_id = str(started["run_id"])

    code, revised = _invoke(
        [
            "reply",
            run_id,
            "Instead of asking, always raise ValueError for zero division.",
            *_base(repo, state),
        ]
    )
    assert code == 0, json.dumps(revised, indent=2)
    assert revised["status"] == "completed"
    assert (repo / "calculator.py").is_file()

    store = RunStore(state / "olympus.db")
    snap = store.get_run(run_id)
    assert snap["spec"].version == 2
    event_types = [event.event_type for event in store.list_events(run_id)]
    assert "run.revision_requested" in event_types
    assert "session.continued" in event_types


def test_blueprint_workflow_completes_at_approved_plan_without_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    state = tmp_path / "state"
    script = SessionScript()
    script.align.append(json.dumps(_goal_spec()))
    script.design.append(json.dumps(_design()))
    script.plan.append(json.dumps(_plan()))
    _patch_provider(monkeypatch, script)

    code, envelope = _invoke(
        [
            "start",
            "Build a calculator library",
            "--workflow",
            "product_blueprint",
            *_base(repo, state),
        ]
    )
    assert code == 0
    assert envelope["status"] == "completed"
    assert envelope["workflow"] == "product_blueprint"
    assert envelope["changed_files"] == []
    assert _git_status(repo) == ""


def test_malformed_reply_preserves_state_and_asks_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    state = tmp_path / "state"
    script = SessionScript()
    script.align.append(json.dumps(_goal_spec(ambiguity=True)))
    script.router.append("this is definitely not json {{{")
    _patch_provider(monkeypatch, script)

    code, started = _invoke(["start", "Build a calculator library", *_base(repo, state)])
    assert code == 0
    assert started["pause_kind"] == "clarify"
    run_id = str(started["run_id"])
    before = RunStore(state / "olympus.db").get_run(run_id)
    before_spec = before["spec"]

    code, after = _invoke(
        ["reply", run_id, "hmm, I am not sure what you mean", *_base(repo, state)]
    )
    assert code == 0
    assert after["status"] == "needs_input"
    assert after["pause_kind"] == "clarify"
    assert after["unresolved_blockers"] == ["ambiguity:AMB-1"]
    assert "AMB-1" in str(after["message"])
    current = RunStore(state / "olympus.db").get_run(run_id)
    assert current["spec"].version == before_spec.version
    assert current["spec"].content_hash == before_spec.content_hash
    assert current["status"].value == "awaiting_human"


def test_deterministic_validation_failure_pauses_blocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    state = tmp_path / "state"
    script = SessionScript()
    script.align.append(json.dumps(_goal_spec()))
    script.design.append(json.dumps(_design()))
    script.plan.extend(json.dumps(_invalid_overlapping_plan()) for _ in range(3))
    _patch_provider(monkeypatch, script)

    code, envelope = _invoke(["start", "Build a calculator library", *_base(repo, state)])
    assert code == 0
    assert envelope["status"] == "needs_input"
    assert envelope["pause_kind"] == "blocked"
    assert "overlapping write_scope" in str(envelope["message"])
    assert envelope["changed_files"] == []
    assert _git_status(repo) == ""


def test_post_completion_reply_reopens_and_applies_follow_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    state = tmp_path / "state"
    script = SessionScript()
    script.align.extend(
        [
            json.dumps(_goal_spec()),
            json.dumps(_goal_spec(include_subtract=True, version=2)),
        ]
    )
    script.design.extend(
        [json.dumps(_design()), json.dumps(_design(requirements=("R1", "R2")))]
    )
    script.plan.extend(
        [
            json.dumps(_plan()),
            json.dumps(
                _plan(
                    requirements=("R1", "R2"),
                    objective="Implement divide and subtract in calculator.py",
                    command=_SUBTRACT_CHECK,
                    version=2,
                )
            ),
        ]
    )
    script.router.append(
        json.dumps(
            {
                "action": "reopen",
                "feedback": "Also add a subtract(a, b) helper.",
            }
        )
    )
    script.workers.extend([("calculator.py", _CALC_CODE), ("calculator.py", _CALC_CODE_V2)])
    _patch_provider(monkeypatch, script)

    code, started = _invoke(["start", "Build a calculator library", *_base(repo, state)])
    assert code == 0
    assert started["status"] == "completed"
    run_id = str(started["run_id"])

    code, reopened = _invoke(
        [
            "reply",
            run_id,
            "Please also add a subtract(a, b) helper.",
            *_base(repo, state),
        ]
    )
    assert code == 0
    assert reopened["status"] == "completed"
    assert reopened["changed_files"] == ["calculator.py"]
    content = (repo / "calculator.py").read_text(encoding="utf-8")
    assert "def subtract" in content
    assert "def divide" in content

    store = RunStore(state / "olympus.db")
    snap = store.get_run(run_id)
    assert snap["spec"].version == 2
    assert {req.id for req in snap["spec"].requirements} == {"R1", "R2"}
    event_types = {event.event_type for event in store.list_events(run_id)}
    assert "run.reopened" in event_types
    assert "session.continued" in event_types


def test_state_dir_inside_workspace_is_rejected_without_creating_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    script = SessionScript()
    _patch_provider(monkeypatch, script)

    code, envelope = _invoke(
        [
            "start",
            "Build a calculator library",
            "--workspace",
            str(repo),
            "--state-dir",
            str(repo / ".olympus"),
            "--provider",
            "mock",
        ]
    )
    assert code == 1
    assert envelope["status"] == "failed"
    assert "inside the target workspace" in str(envelope["error"])
    assert not (repo / ".olympus").exists()
    assert _git_status(repo) == ""


def test_default_state_dir_and_message_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    monkeypatch.setattr(
        "olympus.session.runner.tempfile.gettempdir", lambda: str(tmp_path / "tmproot")
    )
    script = SessionScript()
    script.align.append(json.dumps(_goal_spec()))
    script.design.append(json.dumps(_design()))
    script.plan.append(json.dumps(_plan()))
    _patch_provider(monkeypatch, script)

    code, envelope = _invoke(
        [
            "start",
            "--message",
            "Build a calculator library",
            "--step-mode",
            "plan_only",
            "--workspace",
            str(repo),
            "--provider",
            "mock",
        ]
    )
    assert code == 0
    assert envelope["status"] == "needs_input"
    assert envelope["pause_kind"] == "plan_handoff"
    session_root = tmp_path / "tmproot" / "olympus-session"
    assert session_root.is_dir()
    assert any(session_root.iterdir())
    assert not (repo / ".olympus").exists()
    assert _git_status(repo) == ""


def test_resolve_state_dir_defaults_outside_repository(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    resolved = resolve_state_dir(repo, None)
    assert not resolved.is_relative_to(repo)
    assert resolved.is_dir()


# --------------------------------------------------------------------------- review A1-A5


def test_reply_rejects_a_different_workspace_without_writing_anywhere(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    other = tmp_path / "other"
    state = tmp_path / "state"
    _init_git_repo(repo)
    _init_git_repo(other)
    script = SessionScript()
    script.align.append(json.dumps(_goal_spec()))
    script.design.append(json.dumps(_design()))
    script.plan.append(json.dumps(_plan()))
    script.router.append(json.dumps({"action": "approve"}))
    script.workers.append(("calculator.py", _CALC_CODE))
    _patch_provider(monkeypatch, script)

    code, started = _invoke(
        [
            "start",
            "Build a calculator library",
            "--gate-policy",
            "interactive_minimal",
            "--step-mode",
            "plan_only",
            *_base(repo, state),
        ]
    )
    assert code == 0
    assert started["pause_kind"] == "plan_handoff"
    run_id = str(started["run_id"])

    code, rebound = _invoke(
        [
            "reply",
            run_id,
            "approve",
            "--step-mode",
            "execute",
            "--workspace",
            str(other),
            "--state-dir",
            str(state),
            "--provider",
            "mock",
        ]
    )
    assert code == 1, json.dumps(rebound, indent=2)
    assert rebound["status"] == "failed"
    assert "does not match the run's recorded workspace" in str(rebound["error"])
    assert _git_status(repo) == ""
    assert _git_status(other) == ""
    assert not (repo / "calculator.py").exists()
    assert not (other / "calculator.py").exists()

    store = RunStore(state / "olympus.db")
    assert Path(str(store.get_run(run_id)["workspace_root"])) == repo
    assert store.get_run(run_id)["status"].value == "awaiting_human"


def test_reply_without_workspace_resumes_the_recorded_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    state = tmp_path / "state"
    _init_git_repo(repo)
    script = SessionScript()
    script.align.append(json.dumps(_goal_spec()))
    script.design.append(json.dumps(_design()))
    script.plan.append(json.dumps(_plan()))
    script.router.append(json.dumps({"action": "approve"}))
    script.workers.append(("calculator.py", _CALC_CODE))
    _patch_provider(monkeypatch, script)

    code, started = _invoke(
        [
            "start",
            "Build a calculator library",
            "--step-mode",
            "plan_only",
            *_base(repo, state),
        ]
    )
    assert code == 0
    assert started["pause_kind"] == "plan_handoff"

    # No --workspace: the recorded workspace must win over the invocation cwd.
    code, executed = _invoke(
        [
            "reply",
            str(started["run_id"]),
            "Plan looks good, implement it.",
            "--step-mode",
            "execute",
            "--state-dir",
            str(state),
            "--provider",
            "mock",
        ]
    )
    assert code == 0, json.dumps(executed, indent=2)
    assert executed["status"] == "completed"
    assert (repo / "calculator.py").read_text(encoding="utf-8") == _CALC_CODE
    store = RunStore(state / "olympus.db")
    assert Path(str(store.get_run(str(started["run_id"]))["workspace_root"])) == repo


def test_state_dir_inside_enclosing_git_repository_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    nested = repo / "nested"
    nested.mkdir()
    script = SessionScript()
    _patch_provider(monkeypatch, script)

    code, envelope = _invoke(
        [
            "start",
            "Build a calculator library",
            "--workspace",
            str(nested),
            "--state-dir",
            str(repo / "session-state"),
            "--provider",
            "mock",
        ]
    )
    assert code == 1
    assert envelope["status"] == "failed"
    assert "inside" in str(envelope["error"])
    assert "git repository" in str(envelope["error"])
    assert not (repo / "session-state").exists()
    assert _git_status(repo) == ""


@pytest.mark.parametrize(
    "message",
    [
        "do not proceed",
        "Please don't proceed.",
        "I do not approve this",
        "no, not this version",
        "reject",
        "cancel",
        "不要继续",
        "别继续了",
        "拒绝这个方案",
        "取消",
    ],
)
def test_negated_approval_never_routes_to_approve(message: str) -> None:
    provider = MockLLMProvider(responses=[LLMResponse(content="invalid JSON")])
    intent = parse_reply(
        provider=provider,
        message=message,
        snapshot={"status": RunStatus.AWAITING_HUMAN},
        pause_kind=PauseKind.APPROVAL,
        gate="spec",
    )
    assert intent.action != "approve"
    assert intent.answers == {}
    assert intent.resolutions == {}


@pytest.mark.parametrize("message", ["looks good, proceed", "特别好的方案，批准"])
def test_unambiguous_approval_fallback_still_approves(message: str) -> None:
    provider = MockLLMProvider(responses=[LLMResponse(content="invalid JSON")])
    intent = parse_reply(
        provider=provider,
        message=message,
        snapshot={"status": RunStatus.AWAITING_HUMAN},
        pause_kind=PauseKind.APPROVAL,
        gate="spec",
    )
    assert intent.action == "approve"


def test_negated_reply_never_approves_the_pending_spec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    state = tmp_path / "state"
    script = SessionScript()
    script.align.append(json.dumps(_goal_spec()))
    script.router.append("invalid JSON")
    _patch_provider(monkeypatch, script)

    code, started = _invoke(
        ["start", "Build a calculator library", "--gate-policy", "default", *_base(repo, state)]
    )
    assert code == 0
    assert started["pause_kind"] == "approval"
    run_id = str(started["run_id"])
    store = RunStore(state / "olympus.db")
    before = store.get_run(run_id)
    assert not store.is_approved(
        run_id=run_id,
        target=ApprovalTarget.GOAL_SPEC,
        target_id=before["spec"].spec_id,
        target_hash=before["spec"].content_hash,
    )

    code, refused = _invoke(["reply", run_id, "Please do not proceed.", *_base(repo, state)])
    assert code == 0, json.dumps(refused, indent=2)
    assert refused["status"] == "needs_input"
    assert refused["pause_kind"] == "approval"
    after = store.get_run(run_id)
    assert after["spec"].version == before["spec"].version
    assert after["spec"].content_hash == before["spec"].content_hash
    assert after["status"].value == "awaiting_human"
    assert not store.is_approved(
        run_id=run_id,
        target=ApprovalTarget.GOAL_SPEC,
        target_id=after["spec"].spec_id,
        target_hash=after["spec"].content_hash,
    )

    script.router.append(json.dumps({"action": "approve"}))
    _invoke(["reply", run_id, "Approve the spec.", *_base(repo, state)])
    assert store.is_approved(
        run_id=run_id,
        target=ApprovalTarget.GOAL_SPEC,
        target_id=after["spec"].spec_id,
        target_hash=after["spec"].content_hash,
    )


def test_completed_run_ignores_unparseable_follow_up_reply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    state = tmp_path / "state"
    script = SessionScript()
    script.align.append(json.dumps(_goal_spec()))
    script.design.append(json.dumps(_design()))
    script.plan.append(json.dumps(_plan()))
    script.workers.append(("calculator.py", _CALC_CODE))
    _patch_provider(monkeypatch, script)

    code, started = _invoke(
        [
            "start",
            "Build a calculator library",
            "--gate-policy",
            "interactive_minimal",
            *_base(repo, state),
        ]
    )
    assert code == 0
    assert started["status"] == "completed"
    run_id = str(started["run_id"])
    store = RunStore(state / "olympus.db")
    before_events = [event.event_type for event in store.list_events(run_id)]

    script.router.append("invalid JSON")
    code, reply = _invoke(["reply", run_id, "???", *_base(repo, state)])
    assert code == 0, json.dumps(reply, indent=2)
    assert reply["status"] == "needs_input"
    assert "already completed" in str(reply["message"])

    after_events = [event.event_type for event in store.list_events(run_id)]
    assert "run.reopened" not in after_events
    assert "session.continued" not in after_events[len(before_events) :]
    assert store.get_run(run_id)["status"].value == "completed"
    assert script.align == []


def test_unavailable_provider_on_reply_emits_a_failure_envelope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    state = tmp_path / "state"
    with pytest.MonkeyPatch.context() as scoped:
        script = SessionScript()
        script.align.append(json.dumps(_goal_spec()))
        _patch_provider(scoped, script)
        code, started = _invoke(["start", "Build a calculator library", *_base(repo, state)])
    assert code == 0

    code, reply = _invoke(
        [
            "reply",
            str(started["run_id"]),
            "approve",
            "--workspace",
            str(repo),
            "--state-dir",
            str(state),
            "--provider",
            "definitely-unsupported",
        ]
    )
    assert code == 1
    assert reply["status"] == "failed"
    assert "Unknown provider" in str(reply["error"])


def test_unavailable_provider_on_start_emits_a_failure_envelope(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    state = tmp_path / "state"

    code, envelope = _invoke(
        [
            "start",
            "Build a calculator library",
            "--workspace",
            str(repo),
            "--state-dir",
            str(state),
            "--provider",
            "definitely-unsupported",
        ]
    )
    assert code == 1
    assert envelope["status"] == "failed"
    assert "Unknown provider" in str(envelope["error"])


def _custom_workflow(tmp_path: Path) -> Path:
    custom = tmp_path / "review_workflow.yaml"
    custom.write_text(
        (WORKFLOW_DIR / "full_sdlc_tdd.yaml")
        .read_text(encoding="utf-8")
        .replace("name: full_sdlc_tdd", "name: review_workflow"),
        encoding="utf-8",
    )
    return custom


def test_custom_yaml_workflow_resumes_on_reply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    state = tmp_path / "state"
    _init_git_repo(repo)
    custom = _custom_workflow(tmp_path)
    script = SessionScript()
    script.align.append(json.dumps(_goal_spec()))
    script.design.append(json.dumps(_design()))
    script.plan.append(json.dumps(_plan()))
    script.router.extend(json.dumps({"action": "approve"}) for _ in range(3))
    script.workers.append(("calculator.py", _CALC_CODE))
    _patch_provider(monkeypatch, script)

    common = [
        "--workspace",
        str(repo),
        "--state-dir",
        str(state),
        "--provider",
        "mock",
        "--gate-policy",
        "default",
    ]
    code, started = _invoke(
        ["start", "Build a calculator library", "--workflow", str(custom), *common]
    )
    assert code == 0, json.dumps(started, indent=2)
    assert started["pause_kind"] == "approval"
    assert started["workflow"] == "review_workflow"
    run_id = str(started["run_id"])

    payloads = [
        event.payload
        for event in RunStore(state / "olympus.db").list_events(run_id)
        if event.event_type == "session.started"
    ]
    assert Path(str(payloads[0]["workflow_path"])) == custom.resolve()

    # Explicit --workflow <path> on reply reloads the same custom YAML.
    code, second = _invoke(
        ["reply", run_id, "Approve the spec.", "--workflow", str(custom), *common]
    )
    assert code == 0, json.dumps(second, indent=2)
    assert second["pause_kind"] == "approval"
    assert second["workflow"] == "review_workflow"

    # Omitted --workflow on reply falls back to the persisted workflow path.
    code, third = _invoke(["reply", run_id, "Approve the design.", *common])
    assert code == 0, json.dumps(third, indent=2)
    assert third["pause_kind"] == "approval"
    assert third["workflow"] == "review_workflow"

    code, final = _invoke(["reply", run_id, "Approve the plan.", *common])
    assert code == 0, json.dumps(final, indent=2)
    assert final["status"] == "completed"
    assert (repo / "calculator.py").is_file()


def test_reply_rejects_a_workflow_that_does_not_match_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    state = tmp_path / "state"
    _init_git_repo(repo)
    custom = _custom_workflow(tmp_path)
    script = SessionScript()
    script.align.append(json.dumps(_goal_spec()))
    _patch_provider(monkeypatch, script)

    code, started = _invoke(
        [
            "start",
            "Build a calculator library",
            "--workflow",
            str(custom),
            "--gate-policy",
            "default",
            *_base(repo, state),
        ]
    )
    assert code == 0
    assert started["workflow"] == "review_workflow"

    code, mismatch = _invoke(
        [
            "reply",
            str(started["run_id"]),
            "Approve the spec.",
            "--workflow",
            "full_sdlc_tdd",
            *_base(repo, state),
        ]
    )
    assert code == 1, json.dumps(mismatch, indent=2)
    assert mismatch["status"] == "failed"
    assert "does not match the run's workflow" in str(mismatch["error"])



def test_reply_rejects_empty_db_inside_repo_without_writing_bytes(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    nested = repo / "nested"
    nested.mkdir()
    state = repo / "session-state"
    state.mkdir()
    db = state / "olympus.db"
    db.write_bytes(b"")

    code, env = _invoke(["reply", "run-missing", "Approve.", *_base(nested, state)])
    assert code == 1
    assert env["status"] == "failed"
    assert db.stat().st_size == 0


def test_router_guards_valid_json_negations_gibberish_and_provider_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from olympus.providers.types import LLMResponse, MockLLMProvider, ProviderError
    from olympus.session.contracts import PauseKind
    from olympus.session.router import parse_reply

    snap = {"status": RunStatus.AWAITING_HUMAN, "spec": None, "plan": None}
    approve_llm = MockLLMProvider(
        responder=lambda _m, _t: LLMResponse(content='{"action": "approve"}')
    )
    bad_json_llm = MockLLMProvider(responder=lambda _m, _t: LLMResponse(content="not json"))
    reopen_llm = MockLLMProvider(
        responder=lambda _m, _t: LLMResponse(
            content='{"action": "reopen", "feedback": "invented task"}'
        )
    )

    assert (
        parse_reply(
            provider=bad_json_llm,
            message="I am not ready to proceed.",
            snapshot=snap,
            pause_kind=PauseKind.APPROVAL,
        ).action
        != "approve"
    )
    assert (
        parse_reply(
            provider=approve_llm,
            message="Please do not proceed.",
            snapshot=snap,
            pause_kind=PauseKind.APPROVAL,
        ).action
        != "approve"
    )
    assert (
        parse_reply(
            provider=reopen_llm,
            message="???",
            snapshot=snap,
            pause_kind=None,
            completed=True,
        ).action
        == "unknown"
    )

    def _fail(_m: object, _t: object) -> LLMResponse:
        raise ProviderError("boom")

    with pytest.raises(ProviderError, match="boom"):
        parse_reply(
            provider=MockLLMProvider(responder=_fail),
            message="yes",
            snapshot=snap,
            pause_kind=PauseKind.APPROVAL,
        )


def test_reply_rejects_same_name_tampered_yaml_workflow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    state = tmp_path / "state"
    first = _custom_workflow(tmp_path)
    tampered = tmp_path / "wf_b.yaml"
    tampered.write_text(
        first.read_text(encoding="utf-8").replace(
            "approve_design: true", "approve_design: false"
        )
    )
    script = SessionScript()
    script.align.append(json.dumps(_goal_spec()))
    _patch_provider(monkeypatch, script)

    code, started = _invoke(
        [
            "start",
            "Add add(a, b)",
            "--workflow",
            str(first),
            "--gate-policy",
            "default",
            *_base(repo, state),
        ]
    )
    assert code == 0
    code, rejected = _invoke(
        [
            "reply",
            str(started["run_id"]),
            "Approve the spec.",
            "--workflow",
            str(tampered),
            *_base(repo, state),
        ]
    )
    assert code == 1
    assert rejected["status"] == "failed"
    assert "does not match the workflow that started this run" in str(rejected["error"])


def test_negated_approval_and_no_change_replies_never_mutate_or_reopen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    state = tmp_path / "state"
    script = SessionScript()
    script.align.append(json.dumps(_goal_spec()))
    script.router.append("invalid JSON")
    _patch_provider(monkeypatch, script)

    code, started = _invoke(
        ["start", "Build a calculator library", "--gate-policy", "default", *_base(repo, state)]
    )
    assert code == 0
    run_id = str(started["run_id"])
    store = RunStore(state / "olympus.db")
    snap = store.get_run(run_id)

    code, refused = _invoke(
        ["reply", run_id, "I am not going to approve this.", *_base(repo, state)]
    )
    assert code == 0
    assert refused["pause_kind"] == "approval"
    assert not store.is_approved(
        run_id=run_id,
        target=ApprovalTarget.GOAL_SPEC,
        target_id=snap["spec"].spec_id,
        target_hash=snap["spec"].content_hash,
    )

    repo2 = tmp_path / "repo2"
    _init_git_repo(repo2)
    state2 = tmp_path / "state2"
    script2 = SessionScript()
    script2.align.extend([json.dumps(_goal_spec()), json.dumps(_goal_spec(version=2))])
    script2.design.extend([json.dumps(_design())] * 2)
    script2.plan.extend([json.dumps(_plan())] * 2)
    script2.workers.extend([("calculator.py", _CALC_CODE)] * 2)
    script2.router.extend(["invalid JSON", json.dumps({"action": "reopen", "feedback": "edit"})])
    _patch_provider(monkeypatch, script2)

    code, done = _invoke(["start", "Build a calculator library", *_base(repo2, state2)])
    assert code == 0
    assert done["status"] == "completed"
    done_id = str(done["run_id"])

    for _ in range(2):
        code, stayed = _invoke(
            ["reply", done_id, "Do not change anything.", *_base(repo2, state2)]
        )
        assert code == 0
        assert stayed["status"] == "needs_input"

    store2 = RunStore(state2 / "olympus.db")
    assert store2.get_run(done_id)["status"] == RunStatus.COMPLETED
    assert not any(ev.event_type == "run.reopened" for ev in store2.list_events(done_id))


def test_explicit_ambiguity_answer_resolves_blocker_when_router_json_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    _init_git_repo(repo)
    state = tmp_path / "state"
    script = SessionScript()
    script.align.append(json.dumps(_goal_spec(ambiguity=True)))
    script.design.append(json.dumps(_design()))
    script.plan.append(json.dumps(_plan()))
    script.router.append("invalid JSON")
    _patch_provider(monkeypatch, script)

    code, started = _invoke(["start", "Build a calculator library", *_base(repo, state)])
    assert code == 0
    assert started["pause_kind"] == "clarify"
    assert started["unresolved_blockers"]

    code, clarified = _invoke(
        [
            "reply",
            str(started["run_id"]),
            "AMB-1: Use your default assumptions and proceed.",
            *_base(repo, state),
        ]
    )
    assert code == 0, json.dumps(clarified, indent=2)
    assert not clarified["unresolved_blockers"]


def test_delivery_checks_and_symlinked_sandboxes_never_pollute_target_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from olympus.harness import delivery

    repo = tmp_path / "repo"
    state = tmp_path / "state"
    _init_git_repo(repo)
    plan = _plan()
    plan["delivery_checks"] = [
        {"requirement_id": "R1", "criterion_index": 1, "command": _DIVIDE_CHECK},
    ]
    script = SessionScript()
    script.align.append(json.dumps(_goal_spec()))
    script.design.append(json.dumps(_design()))
    script.plan.append(json.dumps(plan))
    script.workers.append(("calculator.py", _CALC_CODE))
    _patch_provider(monkeypatch, script)

    observed: list[tuple[Path, Path]] = []
    real_create = delivery.create_sandbox

    def _observe(*args: object, **kwargs: object) -> object:
        handle = real_create(*args, **kwargs)  # type: ignore[arg-type] -- forwarding mock wrapper
        observed.append((handle.workspace_path, handle.tmp_dir))
        return handle

    monkeypatch.setattr(delivery, "create_sandbox", _observe)
    code, result = _invoke(["start", "Build a calculator library", *_base(repo, state)])
    assert code == 0, json.dumps(result, indent=2)
    assert result["status"] == "completed"
    assert observed
    for worktree, tmp_dir in observed:
        assert not worktree.resolve().is_relative_to(repo.resolve())
        assert not tmp_dir.resolve().is_relative_to(repo.resolve())
    assert not (repo / ".olympus").exists()

    repo2 = tmp_path / "repo2"
    state2 = tmp_path / "state2"
    _init_git_repo(repo2)
    state2.mkdir()
    (state2 / "sandboxes").symlink_to(repo2 / ".olympus" / "sandboxes", target_is_directory=True)
    code2, rejected = _invoke(["start", "Build a calculator library", *_base(repo2, state2)])
    assert code2 == 1
    assert rejected["status"] == "failed"
    assert "inside" in str(rejected["error"])
    assert not (repo2 / ".olympus").exists()
