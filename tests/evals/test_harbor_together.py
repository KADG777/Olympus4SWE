"""Offline contract tests for the SWE-Together Harbor adapter and Docker judge.

No Harbor installation, no Docker daemon, no SWE-Together checkout, and no
network access are required: the adapter runs against a fake Harbor
``BaseEnvironment`` with a stub reactive ``UserAgent``, and the judge runs
against an injected fake sandbox / mocked ``docker`` CLI.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import tarfile
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tests.evals.benchmarks.harbor_together import (
    SessionEnvelopeView,
    UserEnabledOlympusAgent,
    build_install_script,
    is_stop_decision,
    parse_session_envelope,
    split_diff_output,
    stage_source_archive,
    strip_junk,
    validate_state_dir,
)
from tests.evals.benchmarks.local_docker_judge import (
    CmdResult,
    DockerCli,
    DockerJudgeSandbox,
    JudgeJob,
    apply_candidates,
    build_apply_script,
    build_first_message,
    build_judge_command,
    derive_judge_score,
    load_judge_inputs,
    parse_plan,
    run_judge_docker,
    run_stage2,
    run_user_behavior,
    task_docker_image,
)

# --------------------------------------------------------------------------- fakes


@dataclass
class _ExecResult:
    stdout: str = ""
    stderr: str = ""
    return_code: int = 0


@dataclass
class _ExecCall:
    command: str
    cwd: str | None
    env: dict[str, str] | None
    timeout_sec: int | None


class _FakeEnvironment:
    def __init__(self, handler: Any) -> None:
        self.handler = handler
        self.exec_calls: list[_ExecCall] = []
        self.upload_calls: list[tuple[Path, str]] = []

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
    ) -> _ExecResult:
        self.exec_calls.append(_ExecCall(command, cwd, env, timeout_sec))
        return self.handler(command)

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        self.upload_calls.append((Path(source_path), target_path))

    def commands(self) -> list[str]:
        return [call.command for call in self.exec_calls]


class _SessionHandler:
    """Scripted container behavior: envelopes, diffs, setup probes."""

    def __init__(
        self,
        envelopes: list[dict[str, Any]],
        *,
        diffs: list[str] | None = None,
        repo: str = "/workspace",
        pi_installed: bool = False,
        install_rc: int = 0,
    ) -> None:
        self.envelopes = deque(envelopes)
        self.diffs = deque(diffs or [])
        self.repo = repo
        self.pi_installed = pi_installed
        self.install_rc = install_rc

    def __call__(self, command: str) -> _ExecResult:
        if "HARBOR_DIFF_BEGIN_CUM" in command:
            return _ExecResult(self.diffs.popleft() if self.diffs else "")
        if "session start" in command or "session reply" in command:
            payload = self.envelopes.popleft()
            return _ExecResult(json.dumps(payload))
        if "npm install" in command:
            self.pi_installed = True
            return _ExecResult("/usr/local/bin/pi\n")
        if "command -v pi" in command:
            return _ExecResult("/usr/local/bin/pi\n" if self.pi_installed else "", return_code=1)
        if "command -v olympus" in command:
            return _ExecResult("/usr/local/bin/olympus\n")
        if "pip install" in command:
            return _ExecResult("", "", self.install_rc)
        if "harbor-base" in command:
            return _ExecResult("")
        if "OLYMPUS_REPO=" in command:
            return _ExecResult(f"OLYMPUS_REPO={self.repo}\n")
        return _ExecResult("")


@dataclass
class _Decision:
    action: str = "question"
    content: str = ""
    raw_response: str = ""
    stop: bool = False

    @property
    def has_message(self) -> bool:
        return self.action != "no-op" and bool(self.content.strip())

    def format_for_injection(self) -> str:
        return self.content


class _StubUser:
    def __init__(self, decisions: list[_Decision]) -> None:
        self.decisions = deque(decisions)
        self.calls: list[dict[str, Any]] = []
        self.advanced = 0

    async def process(self, **kwargs: Any) -> _Decision:
        self.calls.append(kwargs)
        return self.decisions.popleft()

    def advance_original_index(self, n: int = 1) -> None:
        self.advanced += n

    def get_stats(self) -> dict[str, Any]:
        return {"version": "stub", "total_calls": len(self.calls)}


def _envelope(
    status: str,
    message: str,
    *,
    run_id: str = "run-1",
    pause_kind: str | None = None,
    changed_files: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "status": status,
        "pause_kind": pause_kind,
        "step_mode": "execute",
        "workflow": "full_sdlc_tdd",
        "message": message,
        "unresolved_blockers": [],
        "changed_files": changed_files or [],
        "error": None,
    }


def _framed_diff(cumulative: str, incremental: str) -> str:
    return (
        "===HARBOR_DIFF_BEGIN_CUM===\n"
        f"{cumulative}\n"
        "===HARBOR_DIFF_END_CUM===\n"
        "===HARBOR_DIFF_BEGIN_INC===\n"
        f"{incremental}\n"
        "===HARBOR_DIFF_END_INC===\n"
    )


def _make_agent(
    tmp_path: Path,
    handler: _SessionHandler,
    user: _StubUser,
    **kwargs: Any,
) -> tuple[UserEnabledOlympusAgent, _FakeEnvironment]:
    environment = _FakeEnvironment(handler)
    agent = UserEnabledOlympusAgent(
        logs_dir=tmp_path / "logs",
        model_name="anthropic/claude-sonnet-4-6",
        olympus_install="none",
        user_agent=user,
        **kwargs,
    )
    return agent, environment


# --------------------------------------------------------------------------- adapter tests


def test_turn0_start_mid_session_clarification_clean_patch_and_stop(tmp_path: Path) -> None:
    instruction = "Add a strict mode flag to the widget parser"
    cumulative = (
        "=== /workspace (cumulative vs harbor-base) ===\n"
        "diff --git a/app.py b/app.py\n"
        "--- a/app.py\n"
        "+++ b/app.py\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
        "diff --git a/.venv/lib/junk.py b/.venv/lib/junk.py\n"
        "--- a/.venv/lib/junk.py\n"
        "+++ b/.venv/lib/junk.py\n"
        "@@ -0,0 +1 @@\n"
        "+noise\n"
    )
    incremental = (
        "=== /workspace (incremental vs harbor-base) ===\n"
        "diff --git a/app.py b/app.py\n"
        "--- a/app.py\n"
        "+++ b/app.py\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
    )
    handler = _SessionHandler(
        [
            _envelope(
                "needs_input", "Should strict mode reject empty input?", pause_kind="clarify"
            ),
            _envelope("completed", "Implemented strict mode."),
        ],
        diffs=[_framed_diff(cumulative, incremental)],
    )
    user = _StubUser(
        [
            _Decision(action="question", content="Yes, reject empty input."),
            _Decision(action="stop", content="STOP"),
        ]
    )
    agent, environment = _make_agent(tmp_path, handler, user)

    asyncio.run(agent.setup(environment))
    context = SimpleNamespace(metadata=None)
    asyncio.run(agent.run(instruction, environment, context))

    commands = environment.commands()
    start_commands = [command for command in commands if "session start" in command]
    reply_commands = [command for command in commands if "session reply" in command]
    assert len(start_commands) == 1
    assert len(reply_commands) == 1

    start = start_commands[0]
    assert instruction in start
    assert "--state-dir /tmp/olympus-session" in start
    assert "--workspace /workspace" in start
    assert "--workflow full_sdlc_tdd" in start
    assert "--gate-policy interactive_minimal" in start
    assert "--provider anthropic" in start
    assert "--model claude-sonnet-4-6" in start

    reply = reply_commands[0]
    assert "run-1" in reply
    assert "Yes, reject empty input." in reply

    # Fidelity: only prose + the per-turn diff reach the simulated user.
    first_consult = user.calls[0]
    assert first_consult["recent_trajectory"] == ""
    assert first_consult["latest_observation"] == "Should strict mode reject empty input?"
    assert first_consult["latest_analysis"] is None
    assert first_consult["is_completion_attempt"] is False
    assert "+new" in first_consult["code_changes_diff"]
    assert ".venv" not in first_consult["code_changes_diff"]

    final_patch = (tmp_path / "logs" / "final.patch").read_text()
    assert "diff --git a/app.py b/app.py" in final_patch
    assert ".venv" not in final_patch
    assert (tmp_path / "logs" / "patches" / "turn-0.patch").exists()
    assert (tmp_path / "logs" / "trajectory.json").exists()
    assert (tmp_path / "logs" / "conversation.jsonl").exists()

    trajectory = json.loads((tmp_path / "logs" / "trajectory.json").read_text())
    assert trajectory["schema_version"] == "ATIF-v1.6"
    assert [step["step_id"] for step in trajectory["steps"]] == list(
        range(1, len(trajectory["steps"]) + 1)
    )
    assert trajectory["steps"][0]["source"] == "user"
    assert trajectory["steps"][0]["message"] == instruction
    assert trajectory["steps"][1]["source"] == "agent"
    assert trajectory["steps"][1]["message"] == "Should strict mode reject empty input?"

    conversation = [
        json.loads(line)
        for line in (tmp_path / "logs" / "conversation.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert [row["role"] for row in conversation if "role" in row] == [
        "user",
        "agent",
        "user",
        "agent",
    ]
    assert context.metadata["olympus_status"] == "completed"
    assert context.metadata["olympus_run_id"] == "run-1"
    assert user.advanced == 1


def test_post_completion_user_correction_reopens_session(tmp_path: Path) -> None:
    handler = _SessionHandler(
        [
            _envelope("completed", "Implemented strict mode."),
            _envelope("completed", "Added the regression tests."),
        ]
    )
    user = _StubUser(
        [
            _Decision(action="new_requirement", content="also add regression tests"),
            _Decision(action="stop", content="STOP"),
        ]
    )
    agent, environment = _make_agent(tmp_path, handler, user)

    asyncio.run(agent.setup(environment))
    asyncio.run(agent.run("Implement strict mode", environment, SimpleNamespace(metadata=None)))

    reply_commands = [cmd for cmd in environment.commands() if "session reply" in cmd]
    assert len(reply_commands) == 1
    assert "also add regression tests" in reply_commands[0]
    assert len(user.calls) == 2
    assert all(call["is_completion_attempt"] is True for call in user.calls)
    assert user.calls[0]["latest_observation"] == "Implemented strict mode."


def test_failed_start_terminates_without_consulting_user(tmp_path: Path) -> None:
    handler = _SessionHandler([_envelope("failed", "workflow crashed")])
    user = _StubUser([])
    agent, environment = _make_agent(tmp_path, handler, user)

    asyncio.run(agent.setup(environment))
    asyncio.run(agent.run("Do the thing", environment, SimpleNamespace(metadata=None)))

    assert user.calls == []
    assert len([cmd for cmd in environment.commands() if "session reply" in cmd]) == 0


def test_completion_callback_off_stops_without_consulting_user(tmp_path: Path) -> None:
    handler = _SessionHandler([_envelope("completed", "Done.")])
    user = _StubUser([])
    agent, environment = _make_agent(
        tmp_path, handler, user, call_user_on_completion=False
    )

    asyncio.run(agent.setup(environment))
    asyncio.run(agent.run("Do the thing", environment, SimpleNamespace(metadata=None)))

    assert user.calls == []
    assert len([cmd for cmd in environment.commands() if "session " in cmd]) == 1


def test_silent_user_on_blocked_pause_terminates_cleanly(tmp_path: Path) -> None:
    handler = _SessionHandler(
        [_envelope("needs_input", "Which API shape?", pause_kind="clarify")]
    )
    user = _StubUser([_Decision(action="no-op", content="")])
    agent, environment = _make_agent(tmp_path, handler, user)

    asyncio.run(agent.setup(environment))
    asyncio.run(agent.run("Add the API", environment, SimpleNamespace(metadata=None)))

    assert len([cmd for cmd in environment.commands() if "session reply" in cmd]) == 0


def test_post_completion_noop_budget_sends_continue_then_stops(tmp_path: Path) -> None:
    handler = _SessionHandler(
        [
            _envelope("completed", "Done."),
            _envelope("completed", "Still done."),
        ]
    )
    user = _StubUser([_Decision(action="no-op", content=""), _Decision(action="no-op", content="")])
    agent, environment = _make_agent(
        tmp_path, handler, user, max_consecutive_noops=2
    )

    asyncio.run(agent.setup(environment))
    asyncio.run(agent.run("Do the thing", environment, SimpleNamespace(metadata=None)))

    reply_commands = [cmd for cmd in environment.commands() if "session reply" in cmd]
    assert len(reply_commands) == 1
    assert "continue" in reply_commands[0]


def test_state_dir_inside_workspace_is_rejected_before_session_start(tmp_path: Path) -> None:
    handler = _SessionHandler([_envelope("completed", "Done.")])
    user = _StubUser([])
    agent, environment = _make_agent(
        tmp_path,
        handler,
        user,
        workspace="/workspace",
        state_dir="/workspace/.olympus-state",
    )

    with pytest.raises(ValueError, match="inside the task workspace"):
        asyncio.run(agent.run("Do the thing", environment, SimpleNamespace(metadata=None)))
    assert environment.exec_calls == []


def test_setup_uploads_source_archive_and_installs_olympus(tmp_path: Path) -> None:
    source = tmp_path / "olympus-src"
    (source / "src" / "olympus").mkdir(parents=True)
    (source / "pyproject.toml").write_text("[project]\nname='x'\n")
    (source / "README.md").write_text("readme")
    (source / "src" / "olympus" / "__init__.py").write_text("")
    handler = _SessionHandler([], pi_installed=True)
    user = _StubUser([])
    agent = UserEnabledOlympusAgent(
        logs_dir=tmp_path / "logs",
        model_name="anthropic/claude-sonnet-4-6",
        olympus_install="source",
        olympus_source=str(source),
        user_agent=user,
    )
    environment = _FakeEnvironment(handler)

    asyncio.run(agent.setup(environment))

    assert environment.upload_calls == [
        (tmp_path / "logs" / "olympus-src.tar.gz", "/tmp/olympus/olympus-src.tar.gz")
    ]
    with tarfile.open(tmp_path / "logs" / "olympus-src.tar.gz", "r:gz") as archive:
        names = set(archive.getnames())
    assert {"pyproject.toml", "README.md", "src/olympus/__init__.py"} <= names
    assert any(
        "install --quiet /tmp/olympus/olympus-src.tar.gz" in command
        for command in environment.commands()
    )
    assert any("harbor-base" in command for command in environment.commands())
    assert agent._discovered_repo == "/workspace"
    assert (tmp_path / "logs" / "setup-install.txt").exists()


def test_setup_installs_pi_for_pi_worker_driver(tmp_path: Path) -> None:
    handler = _SessionHandler([])
    agent, environment = _make_agent(
        tmp_path, handler, _StubUser([]), worker_driver="pi"
    )

    asyncio.run(agent.setup(environment))

    assert any("npm install -g --ignore-scripts" in cmd for cmd in environment.commands())
    assert (tmp_path / "logs" / "setup-pi.txt").exists()


def test_parse_session_envelope_ignores_noise_and_prefers_last_envelope() -> None:
    text = (
        "INFO workflow starting\n"
        '{"status": "needs_input", "run_id": "run-1", "message": "first"}\n'
        "INFO more logging\n"
        '{"status": "completed", "run_id": "run-1", "message": "use {braces} ok",'
        ' "pause_kind": null}\n'
        "tail\n"
    )
    envelope = parse_session_envelope(text)
    assert envelope is not None
    assert envelope.status == "completed"
    assert envelope.message == "use {braces} ok"
    assert parse_session_envelope("no json here") is None
    assert parse_session_envelope('{"status": "unknown"}') is None


def test_static_helpers(tmp_path: Path) -> None:
    assert is_stop_decision(_Decision(action="stop", content="anything"))
    assert is_stop_decision(_Decision(action="no-op", content="STOP"))
    assert not is_stop_decision(_Decision(action="question", content="why?"))

    cumulative, incremental = split_diff_output(
        _framed_diff("diff --git a/a.py b/a.py\n", "diff --git a/b.py b/b.py\n")
    )
    assert cumulative == "diff --git a/a.py b/a.py"
    assert incremental == "diff --git a/b.py b/b.py"

    junk = (
        "diff --git a/src/a.py b/src/a.py\n+ok\n"
        "diff --git a/node_modules/x.js b/node_modules/x.js\n+bad\n"
    )
    cleaned = strip_junk(junk)
    assert "src/a.py" in cleaned
    assert "node_modules" not in cleaned

    with pytest.raises(ValueError):
        validate_state_dir("/workspace", "relative/state")
    with pytest.raises(ValueError):
        validate_state_dir("/workspace", "/workspace/state")
    assert validate_state_dir("/workspace/repo", "/tmp/olympus-session") == (
        "/tmp/olympus-session"
    )

    source = tmp_path / "src-root"
    (source / "src" / "olympus").mkdir(parents=True)
    (source / "pyproject.toml").write_text("[project]\nname='x'\n")
    (source / "src" / "olympus" / "__init__.py").write_text("")
    archive = stage_source_archive(source, tmp_path / "out.tar.gz")
    assert archive.exists()
    assert "install --quiet /tmp/olympus/olympus-src.tar.gz" in build_install_script(
        install_target="/tmp/olympus/olympus-src.tar.gz"
    )


# --------------------------------------------------------------------------- docker fakes


class _FakeDockerRunner:
    def __init__(self, responder: Any) -> None:
        self.responder = responder
        self.calls: list[list[str]] = []
        self.side_effects: list[Any] = []

    def __call__(
        self,
        argv: list[str],
        *,
        input: Any = None,
        capture_output: bool = True,
        text: bool = True,
        timeout: float | None = None,
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append(list(argv))
        outcome = self.responder(argv)
        if isinstance(outcome, Exception):
            raise outcome
        if callable(outcome):
            outcome = outcome()
        return_code, stdout, stderr = outcome
        return subprocess.CompletedProcess(argv, return_code, stdout=stdout, stderr=stderr)


def _ok_docker(argv: list[str]) -> tuple[int, str, str]:
    if len(argv) > 1 and argv[1] == "run":
        return 0, "container-id\n", ""
    if len(argv) > 1 and argv[1] == "exec":
        command = argv[-1]
        if command.startswith("cat "):
            return 0, '{"hello": 1}', ""
        return 0, "ok\n", ""
    return 0, "", ""


def test_docker_sandbox_lifecycle_write_run_read_kill() -> None:
    source_exists: list[bool] = []
    source_content: list[bytes] = []
    runner = _FakeDockerRunner(_ok_docker)

    original = runner.responder

    def responder(argv: list[str]) -> Any:
        if len(argv) > 1 and argv[1] == "cp":
            source = Path(argv[2])
            source_exists.append(source.exists())
            source_content.append(source.read_bytes() if source.exists() else b"")
        return original(argv)

    runner.responder = responder
    cli = DockerCli(runner=runner)
    sandbox = asyncio.run(
        DockerJudgeSandbox.create(image="task-image:local", name="judge-test", cli=cli)
    )

    assert sandbox.sandbox_id == "judge-test"
    run_call = next(call for call in runner.calls if call[1] == "run")
    assert run_call[:4] == ["docker", "run", "-d", "--name"]
    assert "--entrypoint" in run_call
    assert "task-image:local" in run_call
    assert run_call[-2] == "-c"

    asyncio.run(sandbox.write("/tmp/judge_inputs/README.md", "read me"))
    assert source_exists == [True]
    assert source_content == [b"read me"]
    assert any(
        call[1] == "cp" and call[3] == "judge-test:/tmp/judge_inputs/README.md"
        for call in runner.calls
    )

    result = asyncio.run(
        sandbox.run("echo hi", timeout=10, cwd="/tmp", user="root")
    )
    assert result == CmdResult(stdout="ok\n", stderr="", exit_code=0)
    exec_call = next(
        call for call in reversed(runner.calls) if call[1] == "exec" and call[-1] == "echo hi"
    )
    assert exec_call[2:6] == ["-u", "root", "-w", "/tmp"]

    text = asyncio.run(sandbox.read("/tmp/verdict.json"))
    assert json.loads(text) == {"hello": 1}

    asyncio.run(sandbox.kill())
    assert runner.calls[-1] == ["docker", "rm", "-f", "judge-test"]


def test_docker_sandbox_read_failure_and_timeout() -> None:
    def responder(argv: list[str]) -> Any:
        if len(argv) > 1 and argv[1] == "exec":
            if argv[-1].startswith("cat "):
                return 1, "", "missing file"
            return subprocess.TimeoutExpired(argv, 5)
        return 0, "", ""

    cli = DockerCli(runner=_FakeDockerRunner(responder))
    sandbox = DockerJudgeSandbox(name="judge-test", image="img:local", cli=cli)

    with pytest.raises(RuntimeError, match="failed to read"):
        asyncio.run(sandbox.read("/tmp/missing"))

    timed_out = asyncio.run(sandbox.run("sleep 10", timeout=5))
    assert timed_out.exit_code == 124
    assert "timed out" in timed_out.stderr


def test_docker_sandbox_write_failure_raises() -> None:
    def responder(argv: list[str]) -> Any:
        if len(argv) > 1 and argv[1] == "cp":
            return 1, "", "copy failed"
        return 0, "", ""

    cli = DockerCli(runner=_FakeDockerRunner(responder))
    sandbox = DockerJudgeSandbox(name="judge-test", image="img:local", cli=cli)
    with pytest.raises(RuntimeError, match="failed to write"):
        asyncio.run(sandbox.write("/tmp/file", "data"))


# --------------------------------------------------------------------------- judge flow fakes


class _FakeJudgeSandbox:
    def __init__(
        self,
        *,
        verdict: dict[str, Any] | None = None,
        apply_code: int = 0,
        judge_error: Exception | None = None,
        read_error: Exception | None = None,
    ) -> None:
        self.sandbox_id = "fake-judge"
        self.verdict = verdict or {
            "goal_results": [{"id": "g1", "met": True}],
            "judge_score": 0.5,
            "verdict": "partial",
        }
        self.apply_code = apply_code
        self.judge_error = judge_error
        self.read_error = read_error
        self.writes: dict[str, str | bytes] = {}
        self.runs: list[tuple[str, int, str | None, str | None]] = []
        self.killed = False

    async def write(self, path: str, content: str | bytes) -> None:
        self.writes[path] = content

    async def run(
        self,
        cmd: str,
        *,
        timeout: int,
        cwd: str | None = None,
        user: str | None = None,
    ) -> CmdResult:
        self.runs.append((cmd, timeout, cwd, user))
        if "git" in cmd and "apply" in cmd:
            return CmdResult("applying to /workspace\napplied candidate 0", "", self.apply_code)
        if self.judge_error is not None:
            raise self.judge_error
        return CmdResult("judge stdout", "", 0)

    async def read(self, path: str) -> str:
        if self.read_error is not None:
            raise self.read_error
        return json.dumps(self.verdict)

    async def kill(self) -> None:
        self.killed = True


def _make_task_and_trial(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "swe-together"
    prompts = root / "eval" / "correctness" / "prompts"
    prompts.mkdir(parents=True)
    (prompts / "judge_phase1_system.md").write_text("PHASE1 PROMPT")
    (prompts / "judge_phase2_system.md").write_text("PHASE2 PROMPT")
    behavior = root / "eval" / "user_behavior"
    behavior.mkdir(parents=True)
    (behavior / "run_batch.py").write_text("# stub user-behavior runner\n")

    task = tmp_path / "tasks" / "task-a"
    (task / "tests").mkdir(parents=True)
    (task / "task.toml").write_text('[environment]\ndocker_image = "task-image:local"\n')
    (task / "README.md").write_text("task readme")
    (task / "user_simulation_prompt.md").write_text("user prompt")
    (task / "tests" / "test.sh").write_text("#!/bin/sh\ntrue\n")
    (task / "canonical_goals.json").write_text(
        json.dumps({"completeness_goals": [{"id": "g1", "weight": 0.6}]})
    )
    trial = tmp_path / "trials" / "task-a__t1"
    (trial / "agent").mkdir(parents=True)
    (trial / "agent" / "final.patch").write_text(
        "diff --git a/app.py b/app.py\n--- a/app.py\n+++ b/app.py\n"
        "@@ -1,4 +1,4 @@\n line one\n line two\n-old value\n+new value\n line four\n"
    )
    return root, task, trial


def test_run_judge_docker_applies_patch_runs_judge_and_cleans_up(tmp_path: Path) -> None:
    root, task, trial = _make_task_and_trial(tmp_path)
    inputs = load_judge_inputs(task_dir=task, trial_dir=trial, swe_together_root=root)
    sandbox = _FakeJudgeSandbox()
    holder: dict[str, Any] = {}

    async def factory(*, image: str, name: str, envs: Any) -> _FakeJudgeSandbox:
        holder["image"] = image
        holder["name"] = name
        holder["envs"] = envs
        return sandbox

    result = asyncio.run(
        run_judge_docker(
            task_name="task-a",
            trial_id="task-a__t1",
            inputs=inputs,
            image="task-image:local",
            judge_command="echo fake judge",
            sandbox_factory=factory,
        )
    )

    assert holder["image"] == "task-image:local"
    assert holder["envs"]["IS_SANDBOX"] == "1"
    assert sandbox.killed is True
    assert "/tmp/agent.patch" in sandbox.writes
    assert "/tmp/judge_inputs/README.md" in sandbox.writes
    assert "/tmp/judge_inputs/canonical_goals.json" in sandbox.writes
    assert "/tmp/judge_inputs/tests/test.sh" in sandbox.writes
    assert any("git" in cmd and "apply" in cmd for cmd, *_ in sandbox.runs)
    assert any(cmd == "echo fake judge" for cmd, *_ in sandbox.runs)
    assert result.verdict["judge_score"] == 0.5
    assert result.verdict["patch_applied_candidate"] == 0
    assert result.sandbox_id == "fake-judge"


def test_run_judge_docker_patch_failure_returns_structured_error_and_cleans_up(
    tmp_path: Path,
) -> None:
    root, task, trial = _make_task_and_trial(tmp_path)
    inputs = load_judge_inputs(task_dir=task, trial_dir=trial, swe_together_root=root)
    sandbox = _FakeJudgeSandbox(apply_code=1)

    async def factory(*, image: str, name: str, envs: Any) -> _FakeJudgeSandbox:
        return sandbox

    result = asyncio.run(
        run_judge_docker(
            task_name="task-a",
            trial_id="task-a__t1",
            inputs=inputs,
            image="task-image:local",
            judge_command="echo fake judge",
            sandbox_factory=factory,
        )
    )

    assert result.verdict["error"] == "patch_apply_failed"
    assert sandbox.killed is True
    assert not any(cmd == "echo fake judge" for cmd, *_ in sandbox.runs)


def test_run_judge_docker_cleans_up_when_judge_raises(tmp_path: Path) -> None:
    root, task, trial = _make_task_and_trial(tmp_path)
    inputs = load_judge_inputs(task_dir=task, trial_dir=trial, swe_together_root=root)
    sandbox = _FakeJudgeSandbox(judge_error=RuntimeError("judge crashed"))

    async def factory(*, image: str, name: str, envs: Any) -> _FakeJudgeSandbox:
        return sandbox

    with pytest.raises(RuntimeError, match="judge crashed"):
        asyncio.run(
            run_judge_docker(
                task_name="task-a",
                trial_id="task-a__t1",
                inputs=inputs,
                image="task-image:local",
                judge_command="echo fake judge",
                sandbox_factory=factory,
            )
        )
    assert sandbox.killed is True


def test_run_stage2_scores_correctly_and_runs_user_behavior(tmp_path: Path) -> None:
    root, task, trial = _make_task_and_trial(tmp_path)
    sandbox = _FakeJudgeSandbox()

    async def factory(*, image: str, name: str, envs: Any) -> _FakeJudgeSandbox:
        return sandbox

    behavior_calls: list[list[str]] = []

    def user_behavior_runner(
        argv: list[str],
        *,
        cwd: str,
        capture_output: bool,
        text: bool,
        timeout: float,
    ) -> SimpleNamespace:
        behavior_calls.append(argv)
        plan = Path(argv[argv.index("--plan") + 1])
        plan.with_suffix(".summary.json").write_text(json.dumps({"status_counts": {"ok": 1}}))
        return SimpleNamespace(returncode=0, stdout="done", stderr="")

    summary = asyncio.run(
        run_stage2(
            jobs=[JudgeJob(trial_dir=trial, task_dir=task)],
            swe_together_root=root,
            sandbox_factory=factory,
            user_behavior_runner=user_behavior_runner,
        )
    )

    assert summary["status"] == "ok"
    assert summary["correctness"][0]["status"] == "ok"
    assert summary["user_behavior"]["summary"] == {"status_counts": {"ok": 1}}
    assert behavior_calls and behavior_calls[0][1:3] == ["-m", "eval.user_behavior.run_batch"]

    verdict = json.loads((trial / "judge_verdict.json").read_text())
    assert verdict["judge_score"] == 0.6
    assert verdict["verdict"] == "partial"
    assert verdict["judge_reported_score"] == 0.5
    assert verdict["judge_backend"] == "docker"
    assert verdict["task"] == "task-a"
    assert sandbox.killed is True


def test_user_behavior_delegation_writes_plan_and_parses_summary(tmp_path: Path) -> None:
    root = tmp_path / "swe"
    module = root / "eval" / "user_behavior"
    module.mkdir(parents=True)
    (module / "run_batch.py").write_text("# stub\n")
    captured: dict[str, Any] = {}

    def runner(
        argv: list[str],
        *,
        cwd: str,
        capture_output: bool,
        text: bool,
        timeout: float,
    ) -> SimpleNamespace:
        captured["argv"] = argv
        captured["cwd"] = cwd
        plan = Path(argv[argv.index("--plan") + 1])
        captured["plan"] = json.loads(plan.read_text())
        plan.with_suffix(".summary.json").write_text(json.dumps({"results": [{"status": "ok"}]}))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    result = run_user_behavior(
        swe_together_root=root,
        jobs=[{"trial_dir": "/tmp/trial", "task_dir": "/tmp/task"}],
        workers=3,
        model="anthropic/claude-sonnet-4-6",
        runner=runner,
    )

    assert result["status"] == "ok"
    assert result["summary"] == {"results": [{"status": "ok"}]}
    assert captured["plan"][0]["trial_dir"] == "/tmp/trial"
    assert "--workers" in captured["argv"]
    assert captured["cwd"] == str(root.resolve())


# --------------------------------------------------------------------------- judge helper tests


def test_task_image_resolution_and_plan_parsing(tmp_path: Path) -> None:
    task = tmp_path / "tasks" / "task-b"
    task.mkdir(parents=True)
    (task / "task.toml").write_text('[environment]\ndocker_image = "ghcr.io/x:y"\n')
    assert task_docker_image(task) == "ghcr.io/x:y"
    assert task_docker_image(task, image="override:latest") == "override:latest"

    jobs = parse_plan(
        [
            {"trial_dir": "/tmp/t1", "task_dir": "/tmp/a"},
            {"trial_dir": "/tmp/t2", "task_dir": "/tmp/b", "out_name": "v2.json"},
        ]
    )
    assert [job.out_name for job in jobs] == ["judge_verdict.json", "v2.json"]


def test_judge_scripts_and_scoring_helpers() -> None:
    script = build_apply_script(candidates=2, repo_hint="/workspace/agent-swarm")
    assert "/tmp/agent.patch.$i" in script
    assert "seq 0 1" in script
    assert 'git -c safe.directory="*" apply --check' in script
    assert "NO_GIT_REPO_FOUND" in script
    assert "applying to" in script

    skip_script = build_apply_script(candidates=1, skip_apply=True)
    assert "chmod -R a+rwX" in skip_script
    assert "git -c safe.directory" not in skip_script

    candidates = apply_candidates(
        "=== /workspace (cumulative vs harbor-base) ===\n"
        "diff --git a/app.py b/app.py\n"
        "--- a/app.py\n"
        "+++ b/app.py\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
    )
    assert len(candidates) == 1
    assert "=== /workspace" not in candidates[0]
    assert candidates[0].startswith("diff --git a/app.py")

    phase2_message = build_first_message(inputs_dir="/tmp/judge_inputs", phase=2)
    assert "FROZEN" in phase2_message
    assert "/tmp/judge_inputs/verdict.json" in phase2_message
    assert build_first_message(inputs_dir="/tmp/judge_inputs", phase=1)
    command = build_judge_command(phase=2, model="claude-opus-4-6")
    assert "claude --print" in command
    assert "--model claude-opus-4-6" in command

    rubric = {"completeness_goals": [{"id": "g1", "weight": 0.6}, {"id": "g2", "weight": 0.4}]}
    score, bucket = derive_judge_score(
        rubric,
        [{"id": "g1", "met": True}, {"id": "g2", "met": False}],
    )
    assert score == 0.6
    assert bucket == "partial"
    assert derive_judge_score(rubric, [{"id": "g1", "met": True}, {"id": "g2", "met": True}]) == (
        1.0,
        "equivalent",
    )


def test_session_envelope_view_failure_defaults() -> None:
    failed = SessionEnvelopeView.failure("boom")
    assert failed.status == "failed"
    assert failed.error == "boom"
    assert failed.run_id == ""