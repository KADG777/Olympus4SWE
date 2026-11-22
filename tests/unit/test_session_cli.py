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
from olympus.core.contracts import ApprovalTarget
from olympus.core.store import RunStore
from olympus.providers.base import ChatMessage, LLMResponse, MockLLMProvider, ToolCall, ToolSpec
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
