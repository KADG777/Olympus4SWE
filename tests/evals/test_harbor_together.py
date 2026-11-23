"""Offline contract tests for the SWE-Together Harbor adapter and Docker judge.

No Harbor installation, no Docker daemon, no SWE-Together checkout, and no
network access are required: the adapter runs against a fake Harbor
``BaseEnvironment`` with a stub reactive ``UserAgent``, and the judge runs
against an injected fake sandbox / mocked ``docker`` CLI.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import tarfile
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tests.evals.benchmarks.harbor_together import (
    OLYMPUS_AGENT_IMPORT_PATH,
    ExecOutcome,
    SessionEnvelopeView,
    UserEnabledOlympusAgent,
    _trial_token_usage,
    build_install_script,
    build_launcher_parser,
    build_olympus_trial_config,
    is_stop_decision,
    parse_session_envelope,
    redact_base_url,
    redact_text,
    split_diff_output,
    split_pi_model_name,
    stage_source_archive,
    strip_junk,
    validate_state_dir,
)
from tests.evals.benchmarks.harbor_together import (
    main as launch_main,
)
from tests.evals.benchmarks.local_docker_judge import (
    CmdResult,
    DockerCli,
    DockerJudgeSandbox,
    JudgeJob,
    apply_candidates,
    build_apply_script,
    derive_judge_score,
    install_local_docker_judge_backend,
    judge_jobs,
    load_judge_inputs,
    load_upstream_judge_module,
    parse_plan,
    run_judge_docker,
    run_stage2,
    run_user_behavior,
    task_docker_image,
)
from tests.evals.benchmarks.token_usage import BUDGET_SETTLED_DB_MARKER

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
        budget_rows: list[list[Any]] | None = None,
    ) -> None:
        self.envelopes = deque(envelopes)
        self.diffs = deque(diffs or [])
        self.repo = repo
        self.pi_installed = pi_installed
        self.install_rc = install_rc
        self.budget_rows = budget_rows

    def __call__(self, command: str) -> _ExecResult:
        if BUDGET_SETTLED_DB_MARKER in command:
            return _ExecResult(json.dumps(self.budget_rows or []))
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
    unresolved_blockers: list[str] | None = None,
    changed_files: list[str] | None = None,
    token_usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    envelope = {
        "run_id": run_id,
        "status": status,
        "pause_kind": pause_kind,
        "step_mode": "execute",
        "workflow": "full_sdlc_tdd",
        "message": message,
        "unresolved_blockers": unresolved_blockers or [],
        "changed_files": changed_files or [],
        "error": None,
    }
    if token_usage is not None:
        envelope["token_usage"] = token_usage
    return envelope


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
    assert "--stage-driver auto" in start
    assert "--worker-driver auto" in start
    assert "--reviewer-driver auto" in start
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
        [_envelope("needs_input", "Task blocked on missing secret.", pause_kind="blocked")]
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


def test_setup_installs_pi_by_default_for_auto_drivers(tmp_path: Path) -> None:
    handler = _SessionHandler([])
    agent, environment = _make_agent(tmp_path, handler, _StubUser([]))

    assert agent._stage_driver == "auto"
    assert agent._worker_driver == "auto"
    assert agent._reviewer_driver == "auto"
    assert agent._needs_pi is True

    asyncio.run(agent.setup(environment))

    assert any("npm install -g --ignore-scripts" in cmd for cmd in environment.commands())
    assert (tmp_path / "logs" / "setup-pi.txt").exists()


def test_native_driver_overrides_and_install_pi_false_skip_pi(tmp_path: Path) -> None:
    handler = _SessionHandler([])
    native_agent, native_environment = _make_agent(
        tmp_path / "native",
        handler,
        _StubUser([]),
        stage_driver="native",
        worker_driver="native",
        reviewer_driver="native",
    )
    asyncio.run(native_agent.setup(native_environment))
    assert native_agent._needs_pi is False
    assert not any("npm install" in cmd for cmd in native_environment.commands())

    handler = _SessionHandler([])
    disabled_agent, disabled_environment = _make_agent(
        tmp_path / "disabled", handler, _StubUser([]), install_pi=False
    )
    asyncio.run(disabled_agent.setup(disabled_environment))
    assert disabled_agent._needs_pi is False
    assert not any("npm install" in cmd for cmd in disabled_environment.commands())


def test_session_turn_forwards_gateway_env_without_leaking_into_artifacts(
    tmp_path: Path,
) -> None:
    secret = "sk-gateway-secret-987654"
    gateway = "https://gateway.example/v1"
    handler = _SessionHandler([_envelope("completed", "Done.")])
    agent, environment = _make_agent(
        tmp_path,
        handler,
        _StubUser([]),
        call_user_on_completion=False,
        extra_env={"OPENAI_API_KEY": secret, "OPENAI_BASE_URL": gateway},
    )

    asyncio.run(agent.setup(environment))
    asyncio.run(agent.run("Do the thing", environment, SimpleNamespace(metadata=None)))

    session_calls = [
        call for call in environment.exec_calls if "session start" in call.command
    ]
    assert len(session_calls) == 1
    assert session_calls[0].env["OPENAI_API_KEY"] == secret
    assert session_calls[0].env["OPENAI_BASE_URL"] == gateway
    assert secret not in session_calls[0].command
    leaked = [
        str(path.relative_to(tmp_path / "logs"))
        for path in (tmp_path / "logs").rglob("*")
        if path.is_file() and secret in path.read_text(encoding="utf-8", errors="ignore")
    ]
    assert not leaked, leaked


def test_trial_config_does_not_embed_active_gateway_secrets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, task, _trial = _make_task_and_trial(tmp_path)
    (task / "instruction.md").write_text("Fix bug")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-host-openai-secret-111")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-host-anthropic-secret-222")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://gateway.example/v1")

    config = build_olympus_trial_config(
        task, model="openai/gpt-4o", swe_together_root=root
    )

    serialized = (
        config.model_dump_json()
        if hasattr(config, "model_dump_json")
        else json.dumps(config.agent.kwargs)
    )
    assert "sk-host-openai-secret-111" not in serialized
    assert "sk-host-anthropic-secret-222" not in serialized
    assert config.agent.env == {}


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


def test_capabilities_declare_base_url_and_atif() -> None:
    assert UserEnabledOlympusAgent.capabilities.base_url is True
    assert UserEnabledOlympusAgent.capabilities.atif is True
    assert UserEnabledOlympusAgent.SUPPORTS_ATIF is True


def test_turn_artifacts_redact_base_url_credentials_and_env_secrets(
    tmp_path: Path,
) -> None:
    secret = "sk-live-secret-123456"
    base_url = "https://user:hunter2@llm.example/v1?api_key=query-secret"
    agent = UserEnabledOlympusAgent(
        logs_dir=tmp_path / "logs",
        model_name="openai/gpt-4o",
        extra_env={"OPENAI_API_KEY": secret},
        base_url=base_url,
    )
    command = f"olympus session start --message fix --base-url {base_url}"
    agent._write_turn_artifacts(
        0,
        command,
        ExecOutcome(
            stdout=f"key={secret} url={base_url}",
            stderr=f"leaked {secret}",
            return_code=0,
        ),
    )

    written_command = (tmp_path / "logs" / "command-0" / "command.txt").read_text()
    written_stdout = (tmp_path / "logs" / "command-0" / "stdout.txt").read_text()
    written_stderr = (tmp_path / "logs" / "command-0" / "stderr.txt").read_text()
    for text in (written_command, written_stdout, written_stderr):
        assert secret not in text
        assert "hunter2" not in text
        assert "query-secret" not in text
    assert "--base-url" in written_command

    assert redact_base_url(base_url) == (
        "https://***@llm.example/v1?api_key=***REDACTED***"
    )
    assert redact_text(f"token={secret}", [secret]) == "token=***REDACTED***"


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


def test_docker_sandbox_uses_private_env_file_and_never_leaks_secrets() -> None:
    secret = "sk-ant-secret-123456"
    captured: dict[str, Any] = {}

    def responder(argv: list[str]) -> Any:
        if len(argv) > 1 and argv[1] == "run":
            env_file = Path(argv[argv.index("--env-file") + 1])
            captured["run_argv"] = list(argv)
            captured["env_path"] = env_file
            captured["env_exists"] = env_file.exists()
            captured["env_mode"] = env_file.stat().st_mode & 0o777
            captured["env_content"] = env_file.read_text()
            return 0, "container-id\n", ""
        return 0, "", ""

    cli = DockerCli(runner=_FakeDockerRunner(responder))
    asyncio.run(
        DockerJudgeSandbox.create(
            image="task-image:local",
            name="judge-test",
            envs={"ANTHROPIC_API_KEY": secret, "IS_SANDBOX": "1"},
            cli=cli,
        )
    )

    run_argv = captured["run_argv"]
    assert "--env-file" in run_argv
    assert "-e" not in run_argv
    assert secret not in " ".join(run_argv)
    assert captured["env_exists"] is True
    assert captured["env_mode"] == 0o600
    assert f"ANTHROPIC_API_KEY={secret}" in captured["env_content"]
    assert captured["env_path"].exists() is False


def test_docker_sandbox_kill_surfaces_non_zero_exit(caplog: pytest.LogCaptureFixture) -> None:
    def responder(argv: list[str]) -> Any:
        if len(argv) > 1 and argv[1] == "rm":
            return 1, "", "Error: No such container"
        return 0, "", ""

    cli = DockerCli(runner=_FakeDockerRunner(responder))
    sandbox = DockerJudgeSandbox(name="judge-test", image="img:local", cli=cli)
    with caplog.at_level("WARNING"), pytest.raises(
        RuntimeError, match="failed to remove container judge-test"
    ):
        asyncio.run(sandbox.kill())
    assert any("exited 1" in record.message for record in caplog.records)


def test_docker_sandbox_kill_surfaces_cli_exception() -> None:
    def responder(argv: list[str]) -> Any:
        if len(argv) > 1 and argv[1] == "rm":
            return subprocess.TimeoutExpired(argv, 60)
        return 0, "", ""

    cli = DockerCli(runner=_FakeDockerRunner(responder))
    sandbox = DockerJudgeSandbox(name="judge-test", image="img:local", cli=cli)
    with pytest.raises(RuntimeError, match="failed to remove container judge-test"):
        asyncio.run(sandbox.kill())


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
    (root / "eval" / "__init__.py").write_text("")
    behavior = root / "eval" / "user_behavior"
    behavior.mkdir(parents=True)
    (behavior / "__init__.py").write_text("")
    (behavior / "run_batch.py").write_text("# stub user-behavior runner\n")
    (behavior / "tag_messages.py").write_text("# stub message tagger\n")
    (behavior / "user_metrics.py").write_text(
        "def metrics_from_rows(rows):\n"
        "    rows = rows or []\n"
        "    tags = [row.get('tags', []) for row in rows]\n"
        "    corrections = sum(1 for item in tags if 'correction' in item)\n"
        "    nudges = sum(1 for item in tags if 'nudge' in item)\n"
        "    return {\n"
        "        'user_correction': None if not rows else corrections + 0.2 * nudges,\n"
        "        'n_tagged_msgs': len(rows),\n"
        "        'n_correction': None if not rows else corrections,\n"
        "        'n_nudge': None if not rows else nudges,\n"
        "    }\n"
    )

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
        if "eval.user_behavior.tag_messages" in argv:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        plan = Path(argv[argv.index("--plan") + 1])
        for entry in json.loads(plan.read_text()):
            verdict = Path(entry["trial_dir"]) / "intent_coverage_verdict.json"
            verdict.write_text(json.dumps({"coverage_rate": 1.0}))
        plan.with_suffix(".summary.json").write_text(json.dumps({"status_counts": {"ok": 1}}))
        return SimpleNamespace(returncode=0, stdout="done", stderr="")

    summary = asyncio.run(
        run_stage2(
            jobs=[JudgeJob(trial_dir=trial, task_dir=task)],
            swe_together_root=root,
            sandbox_factory=factory,
            judge_command="echo fake judge",
            user_behavior_runner=user_behavior_runner,
        )
    )

    assert summary["status"] == "ok"
    assert summary["correctness"][0]["status"] == "ok"
    assert summary["user_behavior"]["summary"] == {"status_counts": {"ok": 1}}
    assert [call[1:3] for call in behavior_calls] == [
        ["-m", "eval.user_behavior.run_batch"],
        ["-m", "eval.user_behavior.tag_messages"],
    ]
    assert summary["user_correction"] == {str(trial.resolve()): None}

    verdict = json.loads((trial / "judge_verdict.json").read_text())
    assert verdict["judge_score"] == 0.6
    assert verdict["verdict"] == "partial"
    assert verdict["judge_reported_score"] == 0.5
    assert verdict["judge_backend"] == "docker"
    assert verdict["task"] == "task-a"
    assert sandbox.killed is True


def test_user_behavior_runs_coverage_and_tagging_with_user_correction(tmp_path: Path) -> None:
    root, task, trial = _make_task_and_trial(tmp_path)
    episode = trial / "agent" / "episode-1"
    episode.mkdir(parents=True)
    (episode / "user_decision.json").write_text(
        json.dumps(
            {
                "turn": 1,
                "action": "send_message",
                "has_message": True,
                "content": "The calculation has the sign backwards.",
            }
        )
    )
    calls: list[list[str]] = []
    staged_roots: list[Path] = []

    def runner(
        argv: list[str],
        *,
        cwd: str,
        capture_output: bool,
        text: bool,
        timeout: float,
    ) -> SimpleNamespace:
        calls.append(argv)
        if "eval.user_behavior.run_batch" in argv:
            plan_path = Path(argv[argv.index("--plan") + 1])
            for entry in json.loads(plan_path.read_text()):
                verdict = Path(entry["trial_dir"]) / "intent_coverage_verdict.json"
                verdict.write_text(
                    json.dumps(
                        {
                            "trial_dir": entry["trial_dir"],
                            "task_dir": entry["task_dir"],
                            "coverage_rate": 1.0,
                        }
                    )
                )
            plan_path.with_suffix(".summary.json").write_text(
                json.dumps({"status_counts": {"ok": 1}})
            )
            return SimpleNamespace(returncode=0, stdout="coverage", stderr="")
        if "eval.user_behavior.tag_messages" in argv:
            staged = Path(argv[argv.index("--trials-root") + 1])
            staged_roots.append(staged)
            for entry in staged.iterdir():
                verdict = entry / "intent_coverage_verdict.json"
                payload = json.loads(verdict.read_text())
                payload["trial_msg_tags"] = [
                    {"trial_idx": 1, "tags": ["correction", "request"], "frustration": 0}
                ]
                payload["user_correction"] = 1.0
                verdict.write_text(json.dumps(payload))
            return SimpleNamespace(returncode=0, stdout="tagged", stderr="")
        raise AssertionError(f"unexpected argv: {argv}")

    result = run_user_behavior(
        swe_together_root=root,
        jobs=[{"trial_dir": trial, "task_dir": task}],
        workers=3,
        model="anthropic/claude-sonnet-4-6",
        runner=runner,
    )

    assert result["status"] == "ok"
    assert result["summary"] == {"status_counts": {"ok": 1}}
    assert [call[1:3] for call in calls] == [
        ["-m", "eval.user_behavior.run_batch"],
        ["-m", "eval.user_behavior.tag_messages"],
    ]
    assert "--workers" in calls[0] and "--workers" in calls[1]
    assert all(call[0] == sys.executable for call in calls)
    assert staged_roots and staged_roots[0].name.startswith("olympus-user-tagging-")
    trial_key = str(trial.resolve())
    assert result["user_correction"][trial_key] == 1.0
    assert result["trials"][0]["expected_tags"] is True
    assert result["trials"][0]["trial_msg_tags"] is True
    assert result["trials"][0]["user_correction_persisted"] == 1.0
    assert result["trials"][0]["user_correction_recomputed"] == 1.0
    verdict = json.loads((trial / "intent_coverage_verdict.json").read_text())
    assert verdict["trial_msg_tags"] == [
        {"trial_idx": 1, "tags": ["correction", "request"], "frustration": 0}
    ]
    assert verdict["user_correction"] == 1.0


def test_user_behavior_errors_when_tagging_leaves_user_correction_null(
    tmp_path: Path,
) -> None:
    root, task, trial = _make_task_and_trial(tmp_path)
    episode = trial / "agent" / "episode-1"
    episode.mkdir(parents=True)
    (episode / "user_decision.json").write_text(
        json.dumps({"turn": 1, "action": "send_message", "has_message": True, "content": "fix it"})
    )

    def runner(
        argv: list[str],
        *,
        cwd: str,
        capture_output: bool,
        text: bool,
        timeout: float,
    ) -> SimpleNamespace:
        if "eval.user_behavior.run_batch" in argv:
            plan_path = Path(argv[argv.index("--plan") + 1])
            for entry in json.loads(plan_path.read_text()):
                verdict = Path(entry["trial_dir"]) / "intent_coverage_verdict.json"
                verdict.write_text(json.dumps({"coverage_rate": 1.0}))
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        return SimpleNamespace(returncode=0, stdout="tagging skipped", stderr="")

    result = run_user_behavior(
        swe_together_root=root,
        jobs=[{"trial_dir": trial, "task_dir": task}],
        runner=runner,
    )

    assert result["status"] == "error"
    assert result["user_correction"][str(trial.resolve())] is None
    assert any("user_correction is null" in error for error in result["errors"])


def test_user_behavior_requires_the_upstream_message_tagger(tmp_path: Path) -> None:
    root = tmp_path / "swe"
    behavior = root / "eval" / "user_behavior"
    behavior.mkdir(parents=True)
    (behavior / "run_batch.py").write_text("# stub\n")
    with pytest.raises(FileNotFoundError, match="message tagger"):
        run_user_behavior(
            swe_together_root=root,
            jobs=[{"trial_dir": tmp_path / "trial", "task_dir": tmp_path / "task"}],
        )


_UPSTREAM_JUDGE_SOURCE = '''\
import json
from types import SimpleNamespace

CALLS = []


async def open_judge_sandbox(**kwargs):
    raise AssertionError("install_local_docker_judge_backend must replace this")


async def run_judge(task_name, trial_id, inputs, oauth_token, *, timeout_sec=0,
                    max_turns=0, api_key=None):
    CALLS.append({
        "task_name": task_name,
        "trial_id": trial_id,
        "phase": inputs.phase,
        "oauth_token": oauth_token,
        "api_key": api_key,
    })
    sandbox = await open_judge_sandbox(
        task_name=task_name, e2b_template="unused",
        envs={"ANTHROPIC_API_KEY": "sk-test"}, timeout_sec=timeout_sec, buffer_sec=5,
    )
    await sandbox.write("/tmp/judge_inputs/README.md", inputs.readme)
    result = await sandbox.run("judge --print", timeout=5)
    raw = await sandbox.read("/tmp/judge_inputs/verdict.json")
    await sandbox.kill()
    return SimpleNamespace(
        verdict=json.loads(raw),
        stdout=result.stdout,
        stderr=result.stderr,
        exit_code=result.exit_code,
        sandbox_id=sandbox.sandbox_id,
        judge_model="fake-judge",
        judge_backend="native",
    )
'''


def test_judge_jobs_delegates_to_upstream_run_judge_with_docker_sandbox(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, task, trial = _make_task_and_trial(tmp_path)
    (root / "eval" / "correctness" / "sandbox.py").write_text(_UPSTREAM_JUDGE_SOURCE)
    (root / "eval" / "correctness" / "__init__.py").write_text("")
    secret = "sk-ant-upstream-secret"
    docker_calls: list[list[str]] = []

    def responder(argv: list[str]) -> Any:
        docker_calls.append(list(argv))
        if len(argv) > 1 and argv[1] == "exec" and "verdict.json" in argv[-1]:
            verdict = {
                "goal_results": [{"id": "g1", "met": True}],
                "judge_score": 0.5,
                "verdict": "partial",
            }
            return 0, json.dumps(verdict), ""
        if len(argv) > 1 and argv[1] == "run":
            return 0, "container-id\n", ""
        return 0, "ok\n", ""

    monkeypatch.delenv("E2B_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", secret)

    results = asyncio.run(
        judge_jobs(
            jobs=[JudgeJob(trial_dir=trial, task_dir=task)],
            swe_together_root=root,
            docker_cli=DockerCli(runner=_FakeDockerRunner(responder)),
        )
    )

    assert results[0]["status"] == "ok"
    module = load_upstream_judge_module(root)
    assert [
        {
            "task_name": "task-a",
            "trial_id": "task-a__t1",
            "phase": 2,
            "oauth_token": "",
            "api_key": secret,
        }
    ] == module.CALLS
    assert module.open_judge_sandbox.__name__ == "open_judge_sandbox"
    verdict = json.loads((trial / "judge_verdict.json").read_text())
    assert verdict["judge_score"] == 0.6
    assert verdict["judge_reported_score"] == 0.5
    assert verdict["judge_backend"] == "native"
    assert verdict["judge_model"] == "fake-judge"
    run_argv = next(call for call in docker_calls if call[1] == "run")
    assert "--env-file" in run_argv
    assert all(secret not in " ".join(call) for call in docker_calls)


def test_install_backend_returns_none_without_upstream_sources(tmp_path: Path) -> None:
    root, _task, _trial = _make_task_and_trial(tmp_path)
    assert load_upstream_judge_module(root) is None
    assert (
        install_local_docker_judge_backend(
            swe_together_root=root,
            cli=DockerCli(runner=_FakeDockerRunner(_ok_docker)),
        )
        is None
    )


def test_build_olympus_trial_config_matches_swe_together_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SWT_SANDBOX", raising=False)
    root, task, trial = _make_task_and_trial(tmp_path)
    (task / "instruction.md").write_text("Fix the append bug")
    (task / "user_simulation_prompt.md").write_text("SIM PROMPT")
    (task / "original_session.json").write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "please fix the bug"},
                    {"role": "assistant", "content": "on it"},
                    {"role": "user", "content": "<command-name>skip"},
                    {"role": "user", "content": "also add tests"},
                ]
            }
        )
    )

    config = build_olympus_trial_config(
        task,
        model="openai/gpt-4o",
        user_model="anthropic/claude-opus-4-6",
        swe_together_root=root,
        user_context_chars=1234,
        agent_kwargs={"workflow": "full_sdlc_tdd", "max_turns": 3},
    )

    assert config.agent.import_path == OLYMPUS_AGENT_IMPORT_PATH
    assert config.agent.import_path.endswith(":UserEnabledOlympusAgent")
    assert config.agent.model_name == "openai/gpt-4o"
    assert config.agent.base_url is None
    assert config.task.path == task.resolve()
    assert config.trials_dir == (root / "trials").resolve()
    environment_type = getattr(config.environment.type, "value", config.environment.type)
    assert environment_type == "docker"
    assert config.agent.kwargs["user_model_name"] == "anthropic/claude-opus-4-6"
    assert config.agent.kwargs["original_user_messages"] == [
        "please fix the bug",
        "also add tests",
    ]
    assert "SIM PROMPT" in config.agent.kwargs["session_analysis"]
    assert "Message Guidance" in config.agent.kwargs["session_analysis"]
    assert config.agent.kwargs["max_messages"] is None
    assert config.agent.kwargs["user_context_chars"] == 1234
    assert config.agent.kwargs["call_user_on_completion"] is True
    assert config.agent.kwargs["workflow"] == "full_sdlc_tdd"
    assert config.agent.kwargs["max_turns"] == 3


def test_trial_launcher_dry_run_targets_olympus_agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("SWT_SANDBOX", raising=False)
    root, task, _trial = _make_task_and_trial(tmp_path)
    (task / "instruction.md").write_text("Fix the append bug")

    code = launch_main(
        [
            "--swe-together-root",
            str(root),
            "--task-dir",
            str(task),
            "--model",
            "openai/gpt-4o",
            "--dry-run",
            "--json",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == [
        {
            "task_dir": str(task.resolve()),
            "import_path": OLYMPUS_AGENT_IMPORT_PATH,
            "model_name": "openai/gpt-4o",
            "trials_dir": str((root / "trials").resolve()),
            "environment": "docker",
        }
    ]


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



def test_together_redacts_secrets_in_conversation_envelope_and_trial_config(
    tmp_path: Path,
) -> None:
    secret = "synthetic-review-key-only"
    root, task, _trial = _make_task_and_trial(tmp_path)
    (task / "instruction.md").write_text("Fix bug")
    log_dir = tmp_path / "logs"
    agent = UserEnabledOlympusAgent(
        logs_dir=log_dir,
        model_name="openai/gpt-4o",
        extra_env={"OPENAI_API_KEY": secret},
    )
    raw = {"status": "needs_input", "run_id": "probe", "message": f"Echoed {secret}"}
    result = SimpleNamespace(stdout=json.dumps(raw), stderr="", return_code=0, exit_code=0)

    class _FakeEnv:
        async def exec(self, **kwargs: Any) -> Any:
            if "session start" in kwargs["command"]:
                return result
            return SimpleNamespace(stdout="", stderr="", return_code=0, exit_code=0)

    agent._repo = "/workspace"
    asyncio.run(agent._execute_turn(_FakeEnv(), action="start", message="probe", turn=0))
    leaked = [
        str(path.relative_to(log_dir))
        for path in log_dir.rglob("*")
        if path.is_file() and secret in path.read_text(encoding="utf-8")
    ]
    assert not leaked, leaked

    cfg = build_olympus_trial_config(
        task, model="openai/gpt-4o", user_api_key=secret, swe_together_root=root
    )
    assert secret not in json.dumps(cfg.agent.kwargs)
    rebuilt = UserEnabledOlympusAgent(logs_dir=tmp_path / "rebuilt", **cfg.agent.kwargs)
    assert rebuilt._user_api_key == secret


def test_run_user_behavior_stages_custom_out_name_for_tag_messages(tmp_path: Path) -> None:
    root, task, trial = _make_task_and_trial(tmp_path)
    behavior_dir = root / "eval" / "user_behavior"
    behavior_dir.mkdir(parents=True, exist_ok=True)
    (behavior_dir / "run_batch.py").write_text("# stub\n", encoding="utf-8")
    (behavior_dir / "tag_messages.py").write_text("# stub\n", encoding="utf-8")
    episode = trial / "agent" / "episode-1"
    episode.mkdir(parents=True, exist_ok=True)
    (episode / "user_decision.json").write_text(
        json.dumps({"turn": 1, "has_message": True, "action": "redirect", "content": "Fix it."})
    )
    custom = "intent_coverage_verdict_custom.json"

    def _runner(argv: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        if argv[2].endswith("run_batch"):
            (trial / custom).write_text(json.dumps({"trial_dir": str(trial)}))
        else:
            staged_root = Path(argv[argv.index("--trials-root") + 1])
            verdicts = list(staged_root.glob("*/intent_coverage_verdict.json"))
            assert len(verdicts) == 1
            payload = json.loads(verdicts[0].read_text(encoding="utf-8"))
            payload["trial_msg_tags"] = [
                {"trial_idx": 1, "tags": ["correction"], "frustration": 0}
            ]
            payload["user_correction"] = 1.0
            verdicts[0].write_text(json.dumps(payload), encoding="utf-8")
        return subprocess.CompletedProcess(list(argv), 0, "", "")

    res = run_user_behavior(
        swe_together_root=root,
        jobs=[{"trial_dir": trial, "task_dir": task, "out_name": custom}],
        runner=_runner,
        tag_model="openai/mock",
    )
    assert res["status"] == "ok"
    assert res["trials"][0]["user_correction"] == 1.0


def test_build_install_script_provisions_isolated_py313_venv() -> None:
    script = build_install_script(install_target="/tmp/olympus/olympus-src.tar.gz")
    assert "sys.version_info >= (3, 13)" in script
    assert "--python 3.13" in script
    assert "VENV_DIR=/opt/olympus/venv" in script
    assert "VENV_DIR=/tmp/olympus-venv" in script


def test_build_repo_discovery_command_executes_cleanly_in_bash(tmp_path: Path) -> None:
    from tests.evals.benchmarks.harbor_together import build_repo_discovery_command

    nested = tmp_path / "repo"
    (nested / ".git").mkdir(parents=True)
    script = build_repo_discovery_command(repo_roots=str(tmp_path))
    assert "${{#g}}" not in script
    proc = subprocess.run(
        ["bash", "-c", script],
        capture_output=True,
        text=True,
        check=True,
    )
    assert f"OLYMPUS_REPO={nested}" in proc.stdout


def test_local_docker_judge_forwards_anthropic_base_url_for_custom_gateway(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from tests.evals.benchmarks.local_docker_judge import (
        DockerJudgeBackend,
        JudgeInputs,
        _run_upstream_judge,
    )

    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://gateway.example/apps/anthropic")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "tok-gateway-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "tok-gateway-test")

    captured: dict[str, Any] = {}

    class _Resolved:
        def __init__(self, *, seat: str, backend: str, model: str) -> None:
            self.seat = seat
            self.backend = backend
            self.model = model

    def _orig_judge_model(*_a: Any, **_kw: Any) -> Any:
        raise SystemExit("should be overridden for custom gateway")

    llm_cfg = SimpleNamespace(ResolvedModel=_Resolved, judge_model=_orig_judge_model)

    async def _fake_run_judge(*_a: Any, **_kw: Any) -> Any:
        resolved = llm_cfg.judge_model()
        captured["resolved_model"] = resolved.model
        return SimpleNamespace(
            verdict={"verdict": "equivalent", "judge_score": 1.0},
            stdout="ok",
            stderr="",
            exit_code=0,
            sandbox_id="sb-1",
            judge_model=resolved.model,
            judge_backend="native",
        )

    fake_mod = SimpleNamespace(
        open_judge_sandbox=lambda **_kw: None,
        run_judge=_fake_run_judge,
        llm_config=llm_cfg,
    )
    backend = DockerJudgeBackend(fake_mod, image_resolver=lambda _t: "img:1")
    assert backend._extra_envs["ANTHROPIC_BASE_URL"] == "https://gateway.example/apps/anthropic"
    res = asyncio.run(
        _run_upstream_judge(
            backend,
            task_name="demo",
            trial_id="t1",
            inputs=JudgeInputs(phase=2),
            timeout_sec=60,
            max_turns=5,
            model="gateway-model-v1",
        )
    )
    assert captured["resolved_model"] == "anthropic/gateway-model-v1"
    assert res.verdict["verdict"] == "equivalent"
    assert llm_cfg.judge_model is _orig_judge_model


def test_user_api_key_ref_shared_across_dual_module_loads() -> None:
    import importlib.util
    import sys

    from tests.evals.benchmarks import harbor_together as qualified

    name = "_harbor_together_main_copy"
    spec = importlib.util.spec_from_file_location(name, qualified.__file__)
    assert spec is not None and spec.loader is not None
    duplicate = importlib.util.module_from_spec(spec)
    sys.modules[name] = duplicate
    try:
        spec.loader.exec_module(duplicate)
        ref = duplicate._register_user_api_key("sk-dual-load-secret-999")
        assert qualified._resolve_user_api_key(user_api_key=None, user_api_key_ref=ref) == (
            "sk-dual-load-secret-999"
        )
    finally:
        sys.modules.pop(name, None)


def test_silent_user_on_clarify_proceeds_with_defaults_then_accepts_correction(
    tmp_path: Path,
) -> None:
    handler = _SessionHandler(
        [
            _envelope(
                "needs_input",
                "Any special Windows flags?",
                pause_kind="clarify",
                unresolved_blockers=["ambiguity:AMB-1", "ambiguity:AMB-2"],
            ),
            _envelope("completed", "Implemented Windows path fix."),
            _envelope("completed", "Also fixed _build_project_name."),
        ],
        diffs=[
            "=== /workspace/repo (cumulative vs harbor-base) ===\n",
            "=== /workspace/repo (cumulative vs harbor-base) ===\n"
            "diff --git a/a.py b/a.py\n+win\n",
            "=== /workspace/repo (cumulative vs harbor-base) ===\n"
            "diff --git a/a.py b/a.py\n+win\n+proj\n",
        ],
    )
    user = _StubUser(
        [
            _Decision(action="no-op", content=""),
            _Decision(action="redirect", content="Also handle _build_project_name."),
            _Decision(action="stop", content="", stop=True),
        ]
    )
    agent, environment = _make_agent(tmp_path, handler, user)
    asyncio.run(agent.setup(environment))
    asyncio.run(agent.run("Support Windows", environment, SimpleNamespace(metadata=None)))
    replies = [cmd for cmd in environment.commands() if "session reply" in cmd]
    assert len(replies) == 2
    assert "AMB-1: Use your default assumptions and proceed." in replies[0]
    assert "AMB-2: Use your default assumptions and proceed." in replies[0]
    assert "_build_project_name" in replies[1]
    records = [
        json.loads(line)
        for line in (agent.logs_dir / "conversation.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    user_turns = [rec for rec in records if rec.get("role") == "user" and rec.get("turn", 0) > 0]
    assert [rec.get("kind") for rec in user_turns] == ["adapter_default", "simulated_user"]


def test_together_hides_user_api_base_token_in_trial_config_and_redacts_setup_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from unittest.mock import AsyncMock

    root, task, _trial = _make_task_and_trial(tmp_path)
    (task / "instruction.md").write_text("Fix bug")
    for idx, token in enumerate(("synthetic-gateway-token-987654", "Q7zK")):
        gateway = f"https://gateway.example/v1?token={token}"
        cfg = build_olympus_trial_config(
            task,
            model="openai/gpt-4o",
            user_api_base=gateway,
            swe_together_root=root,
        )
        serialized = (
            cfg.model_dump_json()
            if hasattr(cfg, "model_dump_json")
            else json.dumps(cfg.agent.kwargs)
        )
        assert token not in serialized
        logs_dir = tmp_path / f"rebuilt-{idx}"
        rebuilt = UserEnabledOlympusAgent(logs_dir=logs_dir, **cfg.agent.kwargs)
        assert rebuilt._user_api_base == gateway

        monkeypatch.setattr(
            rebuilt,
            "_install_olympus",
            AsyncMock(return_value=ExecOutcome("", f"install failed for {token}", 1)),
        )
        with pytest.raises(RuntimeError) as excinfo:
            asyncio.run(rebuilt.setup(None))
        assert token not in str(excinfo.value)
        assert token not in (logs_dir / "setup-install.txt").read_text(encoding="utf-8")


def test_together_install_script_falls_back_to_curl_when_pip_fails(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "curl-called"
    python_bin = bin_dir / "python3"
    python_bin.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    curl_bin = bin_dir / "curl"
    curl_bin.write_text('#!/bin/sh\nprintf called > "$CALLS"\nexit 1\n', encoding="utf-8")
    python_bin.chmod(0o755)
    curl_bin.chmod(0o755)
    (bin_dir / "mkdir").symlink_to("/bin/mkdir")
    (bin_dir / "sh").symlink_to("/bin/sh")
    script = (
        build_install_script(install_target="olympus4swe")
        .replace("/opt/olympus", f"{tmp_path}/opt/olympus")
        .replace("/tmp/olympus", f"{tmp_path}/tmp/olympus")
    )
    env = {**os.environ, "PATH": str(bin_dir), "HOME": str(tmp_path), "CALLS": str(calls)}
    subprocess.run(["/bin/bash", "-c", script], capture_output=True, env=env, check=False)
    assert calls.exists()


def test_overlapping_upstream_judge_calls_isolate_and_restore_gateway_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.evals.benchmarks.local_docker_judge import (
        DockerJudgeBackend,
        JudgeInputs,
        _run_upstream_judge,
    )

    def baseline() -> Any:
        return SimpleNamespace(model="baseline")

    llm_cfg = SimpleNamespace(
        ResolvedModel=lambda **kw: SimpleNamespace(**kw),
        judge_model=baseline,
    )

    async def _run_overlap() -> dict[str, str]:
        first_started, second_started = asyncio.Event(), asyncio.Event()
        finish_first, finish_second = asyncio.Event(), asyncio.Event()
        seen: dict[str, str] = {}

        async def _fake_run_judge(_task: str, trial_id: str, *_a: Any, **_kw: Any) -> Any:
            seen[trial_id] = llm_cfg.judge_model().model
            (first_started if trial_id == "A" else second_started).set()
            await (finish_first if trial_id == "A" else finish_second).wait()
            return SimpleNamespace(
                verdict={},
                stdout="",
                stderr="",
                exit_code=0,
                sandbox_id=trial_id,
                judge_model=seen[trial_id],
                judge_backend="native",
            )

        mod = SimpleNamespace(
            llm_config=llm_cfg,
            run_judge=_fake_run_judge,
            open_judge_sandbox=lambda **_kw: None,
        )

        async def _call(trial_id: str) -> Any:
            backend = DockerJudgeBackend(mod, image_resolver=lambda _t: "img:1")
            return await _run_upstream_judge(
                backend,
                task_name="demo",
                trial_id=trial_id,
                inputs=JudgeInputs(),
                timeout_sec=10,
                max_turns=3,
                model=f"gateway-{trial_id}",
            )

        first = asyncio.create_task(_call("A"))
        await first_started.wait()
        second = asyncio.create_task(_call("B"))
        await second_started.wait()
        finish_first.set()
        await first
        finish_second.set()
        await second
        return seen

    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://gateway.example/v1")
    observed = asyncio.run(_run_overlap())
    assert observed == {"A": "anthropic/gateway-A", "B": "anthropic/gateway-B"}
    assert llm_cfg.judge_model is baseline


def test_together_install_pi_uploads_and_extracts_local_bundle(tmp_path: Path) -> None:
    bundle = tmp_path / "pi-bundle.tar.gz"
    bundle.write_bytes(b"fake-tarball")
    uploaded: list[tuple[Path, str]] = []
    commands: list[str] = []

    class _FakeEnv:
        async def upload_file(self, source_path: Path, target_path: str) -> None:
            uploaded.append((Path(source_path), target_path))

        async def exec(self, command: str, **_kwargs: Any) -> Any:
            commands.append(command)
            if "tar -xzf /tmp/olympus-pi-bundle.tar.gz" in command:
                return SimpleNamespace(
                    stdout="/opt/olympus-pi/bin/pi\n", stderr="", return_code=0
                )
            return SimpleNamespace(stdout="", stderr="", return_code=0)

    agent = UserEnabledOlympusAgent(
        logs_dir=tmp_path / "logs",
        model_name="openai/gpt-4o",
        pi_bundle=str(bundle),
        olympus_install="none",
    )
    asyncio.run(agent._install_pi(_FakeEnv()))
    assert uploaded == [(bundle.resolve(), "/tmp/olympus-pi-bundle.tar.gz")]
    assert agent._extra_env["OLYMPUS_PI_BIN"] == "/opt/olympus-pi/bin/pi"
    assert any("tar -xzf /tmp/olympus-pi-bundle.tar.gz" in cmd for cmd in commands)


def test_together_agent_forwards_host_provider_env_in_memory_and_redacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "sk-runtime-openai-secret-999"
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    agent = UserEnabledOlympusAgent(
        logs_dir=tmp_path / "logs",
        model_name="openai/gpt-4o",
        base_url="https://gateway.example/v1",
        olympus_install="none",
        install_pi=False,
    )
    assert agent._extra_env["OPENAI_API_KEY"] == secret
    assert agent._extra_env["OPENAI_BASE_URL"] == "https://gateway.example/v1"
    assert secret in agent._redaction_secrets


def test_together_trial_config_keeps_agent_env_secrets_out_of_persisted_config(
    tmp_path: Path,
) -> None:
    secret = "sk-agent-env-sentinel-12345"
    root, task, _trial = _make_task_and_trial(tmp_path)
    (task / "instruction.md").write_text("Fix bug", encoding="utf-8")
    cfg = build_olympus_trial_config(
        task,
        model="openai/gpt-4o",
        agent_env={"OPENAI_API_KEY": secret, "OLYMPUS_SAFE_FLAG": "1"},
        swe_together_root=root,
    )
    assert "OPENAI_API_KEY" not in cfg.agent.env
    assert cfg.agent.env == {"OLYMPUS_SAFE_FLAG": "1"}
    assert secret not in json.dumps(cfg.agent.kwargs)
    assert secret not in json.dumps(cfg.agent.env)
    rebuilt = UserEnabledOlympusAgent(logs_dir=tmp_path / "logs", **cfg.agent.kwargs)
    assert rebuilt._extra_env["OPENAI_API_KEY"] == secret
    assert secret in rebuilt._redaction_secrets


# --------------------------------------------------------------------------- Pi passthrough


def test_session_commands_forward_pi_controls(tmp_path: Path) -> None:
    handler = _SessionHandler(
        [
            _envelope("needs_input", "Question?", pause_kind="clarify"),
            _envelope("completed", "Done."),
        ]
    )
    user = _StubUser(
        [
            _Decision(action="question", content="Yes."),
            _Decision(action="stop", content="STOP"),
        ]
    )
    agent, environment = _make_agent(
        tmp_path,
        handler,
        user,
        pi_provider="qwen-token-plan-cn",
        thinking_level="high",
        context_window=262144,
    )

    asyncio.run(agent.setup(environment))
    asyncio.run(agent.run("Do the thing", environment, SimpleNamespace(metadata=None)))

    start = next(cmd for cmd in environment.commands() if "session start" in cmd)
    reply = next(cmd for cmd in environment.commands() if "session reply" in cmd)
    for command in (start, reply):
        assert "--pi-provider qwen-token-plan-cn" in command
        assert "--thinking high" in command
        assert "--context-window 262144" in command


def test_pi_provider_owns_model_prefix_without_breaking_model_id() -> None:
    assert split_pi_model_name(
        "qwen-token-plan-cn/qwen3-max", pi_provider="qwen-token-plan-cn"
    ) == (None, "qwen-token-plan-cn", "qwen3-max")
    # Known API providers still split into the native --provider flag.
    assert split_pi_model_name("anthropic/claude-sonnet-4-6") == (
        "anthropic",
        None,
        "claude-sonnet-4-6",
    )
    # Unknown prefixes with no Pi provider stay untouched (org/model ids survive).
    assert split_pi_model_name("meta-llama/Llama-3-8B") == (
        None,
        None,
        "meta-llama/Llama-3-8B",
    )


def test_pi_provider_kwarg_strips_only_matching_model_prefix(tmp_path: Path) -> None:
    agent = UserEnabledOlympusAgent(
        logs_dir=tmp_path / "logs",
        model_name="qwen-token-plan-cn/qwen3-max",
        pi_provider="qwen-token-plan-cn",
        olympus_install="none",
    )
    assert agent._provider == "auto"
    assert agent._pi_provider == "qwen-token-plan-cn"
    assert agent._model == "qwen3-max"
    command = agent._session_command("start", message="hi")
    assert "--pi-provider qwen-token-plan-cn" in command
    assert "--model qwen3-max" in command
    assert "--model qwen-token-plan-cn/qwen3-max" not in command

    untouched = UserEnabledOlympusAgent(
        logs_dir=tmp_path / "logs",
        model_name="meta-llama/Llama-3-8B",
        pi_provider="openrouter",
        olympus_install="none",
    )
    assert untouched._model == "meta-llama/Llama-3-8B"


def test_context_window_rejects_non_positive_values(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="context_window"):
        UserEnabledOlympusAgent(
            logs_dir=tmp_path / "logs",
            model_name="openai/gpt-4o",
            context_window="not-a-number",
            olympus_install="none",
        )
    with pytest.raises(ValueError, match="context_window"):
        UserEnabledOlympusAgent(
            logs_dir=tmp_path / "logs",
            model_name="openai/gpt-4o",
            context_window=0,
            olympus_install="none",
        )


# --------------------------------------------------------------------------- token usage


_TOKEN_USAGE_PAYLOAD = {
    "llm_calls": 3,
    "input_tokens": 1000,
    "output_tokens": 200,
    "reasoning_tokens": 50,
    "cache_read_tokens": 500,
    "cache_write_tokens": 10,
    "by_actor": {
        "align:proposer": {
            "calls": 3,
            "input_tokens": 1000,
            "output_tokens": 200,
            "reasoning_tokens": 50,
            "cache_read_tokens": 500,
            "cache_write_tokens": 10,
        }
    },
}


def test_token_usage_persisted_to_artifact_context_and_trajectory(tmp_path: Path) -> None:
    handler = _SessionHandler(
        [_envelope("completed", "Done.", token_usage=_TOKEN_USAGE_PAYLOAD)]
    )
    agent, environment = _make_agent(
        tmp_path, handler, _StubUser([]), call_user_on_completion=False
    )
    context = SimpleNamespace(metadata=None)

    asyncio.run(agent.setup(environment))
    asyncio.run(agent.run("Do the thing", environment, context))

    expected = {
        "llm_calls": 3,
        "input_tokens": 1000,
        "output_tokens": 200,
        "reasoning_tokens": 50,
        "cache_read_tokens": 500,
        "cache_write_tokens": 10,
        "total_tokens": 1000 + 200 + 500 + 10,
        "cache_hit_rate": round(500 / (1000 + 500), 4),
        "by_actor": {
            "align:proposer": {
                "calls": 3,
                "input_tokens": 1000,
                "output_tokens": 200,
                "reasoning_tokens": 50,
                "cache_read_tokens": 500,
                "cache_write_tokens": 10,
                "total_tokens": 1000 + 200 + 500 + 10,
            }
        },
    }
    artifact = json.loads((tmp_path / "logs" / "token_usage.json").read_text())
    assert artifact == expected
    assert context.metadata["token_usage"] == expected
    assert context.n_input_tokens == 1510
    assert context.n_output_tokens == 200
    assert context.n_cache_tokens == 500
    trajectory = json.loads((tmp_path / "logs" / "trajectory.json").read_text())
    assert trajectory["token_usage"] == expected
    assert trajectory["final_metrics"]["extra"]["token_usage"] == expected


def test_token_usage_falls_back_to_budget_settled_events(tmp_path: Path) -> None:
    handler = _SessionHandler(
        [_envelope("completed", "Done.")],
        budget_rows=[
            ["align:proposer", {"used": 1, "input_tokens": 100, "cache_read_tokens": 40}],
            ["build:worker", {"used": 2, "input_tokens": 50, "output_tokens": 10}],
        ],
    )
    agent, environment = _make_agent(
        tmp_path, handler, _StubUser([]), call_user_on_completion=False
    )
    context = SimpleNamespace(metadata=None)

    asyncio.run(agent.setup(environment))
    asyncio.run(agent.run("Do the thing", environment, context))

    artifact = json.loads((tmp_path / "logs" / "token_usage.json").read_text())
    assert artifact["llm_calls"] == 2
    assert artifact["input_tokens"] == 150
    assert artifact["output_tokens"] == 10
    assert artifact["cache_read_tokens"] == 40
    assert artifact["total_tokens"] == 150 + 10 + 40
    assert artifact["by_actor"]["align:proposer"]["calls"] == 1
    assert context.metadata["token_usage"] == artifact
    assert context.n_input_tokens == 190
    assert context.n_cache_tokens == 40


def test_token_usage_artifact_is_zeroed_when_nothing_reported(tmp_path: Path) -> None:
    handler = _SessionHandler([_envelope("completed", "Done.")])
    agent, environment = _make_agent(
        tmp_path, handler, _StubUser([]), call_user_on_completion=False
    )
    context = SimpleNamespace(metadata=None)

    asyncio.run(agent.setup(environment))
    asyncio.run(agent.run("Do the thing", environment, context))

    artifact = json.loads((tmp_path / "logs" / "token_usage.json").read_text())
    assert artifact["llm_calls"] == 0
    assert artifact["total_tokens"] == 0
    assert artifact["by_actor"] == {}
    assert "token_usage" not in context.metadata
    assert not hasattr(context, "n_input_tokens")


def test_trial_token_usage_reads_agent_result_context() -> None:
    result = SimpleNamespace(
        agent_result=SimpleNamespace(
            metadata={"token_usage": _TOKEN_USAGE_PAYLOAD},
            n_input_tokens=None,
            n_output_tokens=None,
            n_cache_tokens=None,
        )
    )
    usage = _trial_token_usage(result)
    assert usage is not None
    assert usage["total_tokens"] == 1710

    fields_only = SimpleNamespace(
        agent_result=SimpleNamespace(
            metadata=None,
            n_input_tokens=1500,
            n_output_tokens=200,
            n_cache_tokens=500,
        )
    )
    rebuilt = _trial_token_usage(fields_only)
    assert rebuilt is not None
    assert rebuilt["input_tokens"] == 1000
    assert rebuilt["cache_read_tokens"] == 500
    assert rebuilt["total_tokens"] == 1700
    assert _trial_token_usage(SimpleNamespace(agent_result=None)) is None


# --------------------------------------------------------------------------- setup timeout


def test_trial_config_sets_setup_timeout_and_forwards_pi_kwargs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SWT_SANDBOX", raising=False)
    root, task, _trial = _make_task_and_trial(tmp_path)
    (task / "instruction.md").write_text("Fix the append bug")
    config = build_olympus_trial_config(
        task,
        model="openai/gpt-4o",
        swe_together_root=root,
        setup_timeout=1234.5,
        agent_kwargs={
            "pi_provider": "qwen-token-plan-cn",
            "thinking_level": "high",
            "context_window": 262144,
        },
    )
    assert config.agent.override_setup_timeout_sec == 1234.5
    assert config.agent.kwargs["pi_provider"] == "qwen-token-plan-cn"
    assert config.agent.kwargs["thinking_level"] == "high"
    assert config.agent.kwargs["context_window"] == 262144

    default_config = build_olympus_trial_config(
        task, model="openai/gpt-4o", swe_together_root=root
    )
    assert default_config.agent.override_setup_timeout_sec is None
    assert UserEnabledOlympusAgent.DEFAULT_SETUP_TIMEOUT_SEC == 900.0


def test_launcher_forwards_pi_controls_and_setup_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SWT_SANDBOX", raising=False)
    root, task, _trial = _make_task_and_trial(tmp_path)
    (task / "instruction.md").write_text("Fix the append bug")

    import tests.evals.benchmarks.harbor_together as harbor_module

    captured: dict[str, Any] = {}
    real_builder = harbor_module.build_olympus_trial_config

    def _capture(task_dir: Path, **kwargs: Any) -> Any:
        captured.update(kwargs)
        return real_builder(task_dir, **kwargs)

    monkeypatch.setattr(harbor_module, "build_olympus_trial_config", _capture)
    code = launch_main(
        [
            "--swe-together-root",
            str(root),
            "--task-dir",
            str(task),
            "--model",
            "openai/gpt-4o",
            "--pi-provider",
            "qwen-token-plan-cn",
            "--thinking",
            "high",
            "--context-window",
            "262144",
            "--agent-setup-timeout",
            "1500",
            "--dry-run",
            "--json",
        ]
    )

    assert code == 0
    assert captured["setup_timeout"] == 1500.0
    assert captured["agent_kwargs"] == {
        "pi_provider": "qwen-token-plan-cn",
        "thinking_level": "high",
        "context_window": 262144,
    }


def test_launcher_parser_accepts_setup_timeout_alias() -> None:
    args = build_launcher_parser().parse_args(
        [
            "--swe-together-root",
            "/tmp/root",
            "--model",
            "openai/gpt-4o",
            "--override-setup-timeout-sec",
            "1200",
        ]
    )
    assert args.override_setup_timeout_sec == 1200.0


def test_token_usage_recovers_from_db_when_later_turn_drops_envelope(
    tmp_path: Path,
) -> None:
    turn0 = _envelope(
        "needs_input",
        "Proceed?",
        pause_kind="clarify",
        token_usage={
            "llm_calls": 1,
            "input_tokens": 100,
            "output_tokens": 20,
            "cache_read_tokens": 50,
            "cache_write_tokens": 0,
        },
    )
    # Turn 1 produces no valid JSON envelope (e.g. timed out after settling extra tokens in DB).
    handler = _SessionHandler(
        [turn0, "unparseable timeout output"],
        budget_rows=[
            [
                "align:proposer",
                {"used": 1, "input_tokens": 100, "output_tokens": 20, "cache_read_tokens": 50},
            ],
            [
                "build:worker",
                {"used": 2, "input_tokens": 200, "output_tokens": 40, "cache_read_tokens": 100},
            ],
        ],
    )
    agent, environment = _make_agent(
        tmp_path, handler, _StubUser(["Go ahead"]), call_user_on_completion=False
    )
    context = SimpleNamespace(metadata=None)

    asyncio.run(agent.setup(environment))
    asyncio.run(agent.run("Do the thing", environment, context))

    artifact = json.loads((tmp_path / "logs" / "token_usage.json").read_text())
    assert artifact["llm_calls"] == 2
    assert artifact["input_tokens"] == 300
    assert artifact["output_tokens"] == 60
    assert artifact["cache_read_tokens"] == 150
    assert context.n_input_tokens == 450


def test_token_usage_recovers_from_db_when_later_turn_raises_or_returns_empty_by_actor(
    tmp_path: Path,
) -> None:
    turn0 = _envelope(
        "needs_input",
        "Proceed?",
        pause_kind="clarify",
        token_usage={
            "llm_calls": 1,
            "input_tokens": 100,
            "output_tokens": 20,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
        },
    )
    empty_actor_turn1 = _envelope(
        "completed",
        "Done.",
        token_usage={"by_actor": {"align:proposer": {}}},
    )
    budget_rows = [
        ["align:proposer", {"used": 1, "input_tokens": 100, "output_tokens": 20}],
        ["build:worker", {"used": 2, "input_tokens": 200, "output_tokens": 40}],
    ]

    # Case A: later turn returns empty-count by_actor payload -> must recover 300 from DB.
    handler_a = _SessionHandler([turn0, empty_actor_turn1], budget_rows=budget_rows)
    agent_a, env_a = _make_agent(
        tmp_path / "case_a", handler_a, _StubUser(["Go ahead"]), call_user_on_completion=False
    )
    ctx_a = SimpleNamespace(metadata=None)
    asyncio.run(agent_a.setup(env_a))
    asyncio.run(agent_a.run("Do the thing", env_a, ctx_a))
    art_a = json.loads((tmp_path / "case_a" / "logs" / "token_usage.json").read_text())
    assert art_a["input_tokens"] == 300

    # Case B: later turn raises ConnectionError -> finally must recover 300 from DB.
    handler_b = _SessionHandler([turn0], budget_rows=budget_rows)
    orig_call = handler_b.__call__
    turn_calls = 0

    def _raising_handler(command: str) -> Any:
        nonlocal turn_calls
        if "olympus session" in command:
            turn_calls += 1
            if turn_calls == 2:
                raise ConnectionError("container dropped")
        return orig_call(command)

    agent_b, env_b = _make_agent(
        tmp_path / "case_b",
        _raising_handler,
        _StubUser(["Go ahead"]),
        call_user_on_completion=False,
    )
    ctx_b = SimpleNamespace(metadata=None)
    asyncio.run(agent_b.setup(env_b))
    with pytest.raises(ConnectionError):
        asyncio.run(agent_b.run("Do the thing", env_b, ctx_b))
    art_b = json.loads((tmp_path / "case_b" / "logs" / "token_usage.json").read_text())
    assert art_b["input_tokens"] == 300
