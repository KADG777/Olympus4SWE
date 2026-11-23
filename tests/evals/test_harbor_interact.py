"""Offline contract tests for the ``SWE-Interact`` Harbor adapter (Ticket T3).

The suite exercises three seams without Docker:

* the stdlib FastMCP streamable-HTTP client against a real local HTTP server
  (JSON and SSE responses, session headers, tool and transport errors),
* the container-side bridge against a fake ``olympus`` executable and — for one
  end-to-end case — the real ``olympus session`` CLI with a scripted
  ``MockLLMProvider`` in a real temporary git repository,
* the Harbor agent adapter against a fake ``BaseEnvironment``.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tests.evals.benchmarks import harbor_interact
from tests.evals.benchmarks.fastmcp_client import (
    McpError,
    McpHttpClient,
    McpToolError,
    parse_sse_messages,
)
from tests.evals.benchmarks.harbor_interact import (
    ExecResult,
    OlympusInteractAgent,
    classify_step,
    extract_handoff_script,
)
from tests.evals.benchmarks.interact_bridge import (
    EXIT_CONFIG_ERROR,
    EXIT_SESSION_FAILED,
    EXIT_SIDECAR_ERROR,
    EXIT_TURN_CAP,
    redact_text,
)

BENCHMARKS_DIR = Path(harbor_interact.__file__).resolve().parent
BRIDGE_SCRIPT = BENCHMARKS_DIR / "interact_bridge.py"
PROJECT_ROOT = BENCHMARKS_DIR.parents[2]
PROJECT_SRC = PROJECT_ROOT / "src"

PLAN_INSTRUCTION = (
    "You're working in the repository at `/app`.\n"
    "\n"
    "A maintainer needs a change implemented.\n"
    "\n"
    "- Use the `ask_user` tool (MCP server at `http://user-server:8000/mcp`) "
    "to talk to the maintainer.\n"
    "- This step is only for understanding the request and agreeing on the plan. "
    "Do not make implementation edits in this step.\n"
    "- If they approve your plan, save it in `/tmp/plan.md`, then stop.\n"
)

EXECUTE_INSTRUCTION = (
    "You're working in the repository at `/app`.\n"
    "\n"
    "**Read `/tmp/plan.md` first** - it is the plan you agreed with the maintainer "
    "in the previous step. Carry it out now in `/app`.\n"
)

HANDOFF_INSTRUCTION = """You're working in the repository at `/app`.

This is a handoff step. Do not make intentional implementation changes.

Save the local commit history for handoff:

```bash
set -euo pipefail
cd /app
mkdir -p /logs/agent
git status --short > /logs/agent/final-status.txt
```
"""

UPSTREAM_IMPLEMENT_INSTRUCTION = "\n".join(
    [
        "You're working in the repository at `/app`.",
        "",
        "**Read `/tmp/plan.md` first** - it is the plan you agreed with the maintainer "
        "in the previous step. Carry it out now in `/app`.",
        "",
        "- Make the minimal production-code and documentation changes needed to "
        "implement the agreed behavior.",
        "- Do not modify test files, test harness files, or anything under `/tests`.",
        "- Keep existing behavior intact except for the requested change.",
        "- Before editing, configure a local git identity if this repo does not "
        "already have one:",
        "",
        "```bash",
        'git config user.name "Dev"',
        'git config user.email "dev@example.com"',
        "```",
        "",
        "- Save the starting commit before your first implementation commit:",
        "",
        "```bash",
        "git rev-parse HEAD > /tmp/base_commit.txt",
        "```",
        "",
        "- After each coherent implementation pass, commit the current repo changes "
        "before asking the maintainer for review. A coherent pass is the initial "
        "implementation, or a later revision made in response to maintainer feedback. "
        "Use short commit messages like `initial implementation` or "
        "`address feedback: parser behavior`.",
        "- The maintainer can inspect the repository state directly after your commit, "
        "so a concise review request is fine. Still make sure each coherent "
        "implementation pass is committed before calling `ask_user`.",
        "- Do not commit `/logs`, temporary investigation files, or unrelated "
        "generated files.",
        "- When you believe it is done, run focused validation that is practical in "
        "this environment, review the latest commit and working tree, then report "
        "the key changes to the maintainer via `ask_user`. If they ask for changes, "
        "make another coherent pass and commit it. If they say ship it, then finish.",
    ]
)

UPSTREAM_HANDOFF_INSTRUCTION = """You're working in the repository at `/app`.

This is a handoff step. Do not make intentional implementation changes.

Save the local commit history for handoff:

```bash
set -euo pipefail
cd /app
mkdir -p /logs/agent
base="$(cat /tmp/base_commit.txt)"

{
  echo "Base commit: $base"
  echo
  echo "Commits:"
  git log --oneline --reverse "$base"..HEAD
  echo
  echo "Patches:"
  git log --patch --reverse --find-renames "$base"..HEAD
} > /logs/agent/change-log.txt

git status --short > /logs/agent/final-status.txt
git diff "$base"..HEAD > /logs/agent/final.diff
```
"""

UPSTREAM_WRITE_TESTS_INSTRUCTION = (
    "You're working in the repository at `/app`.\n"
    "\n"
    "Now add tests for the implementation changes you made. "
    "Keep the test changes in a separate commit, then hand off.\n"
)


# --------------------------------------------------------------------------- HTTP fixtures


class FakeMcpServer:
    """A minimal MCP streamable-HTTP server backed by canned ``ask_user`` replies."""

    def __init__(
        self,
        replies: list[str] | None = None,
        *,
        mode: str = "json",
        tool_error: str | None = None,
        http_error: int | None = None,
        default_reply: str | None = None,
        expire_after_tool_calls: int | None = None,
        expiry_mode: str = "http",
    ) -> None:
        self.replies = list(replies or [])
        self.mode = mode
        self.tool_error = tool_error
        self.http_error = http_error
        self.default_reply = default_reply
        self.expire_after_tool_calls = expire_after_tool_calls
        self.expiry_mode = expiry_mode
        self.session_generation = 0
        self.active_session: str | None = None
        self.tool_call_count = 0
        self.expired = False
        self.questions: list[tuple[str, str]] = []
        self.requests: list[dict[str, Any]] = []
        self.init_notified = False
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def url(self) -> str:
        host, port = self.server.server_address[:2]
        return f"http://{host}:{port}/mcp"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:
                return

            def do_POST(self) -> None:  # noqa: N802 - http.server API
                length = int(self.headers.get("content-length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                headers = {key.lower(): value for key, value in self.headers.items()}
                outer.requests.append({"headers": headers, "body": body})
                method = body.get("method")
                if method == "initialize":
                    protocol = (body.get("params") or {}).get("protocolVersion")
                    outer.session_generation += 1
                    outer.active_session = f"fake-session-{outer.session_generation}"
                    outer.expired = outer.expire_after_tool_calls == 0
                    self._send(
                        200,
                        {
                            "jsonrpc": "2.0",
                            "id": body.get("id"),
                            "result": {
                                "protocolVersion": protocol,
                                "capabilities": {"tools": {}},
                                "serverInfo": {"name": "fake-user", "version": "0"},
                            },
                        },
                        session=outer.active_session,
                    )
                    return
                if method == "notifications/initialized":
                    outer.init_notified = True
                    self.send_response(202)
                    self.send_header("content-length", "0")
                    self.end_headers()
                    return
                if method == "tools/call":
                    session = headers.get("mcp-session-id")
                    if session is None:
                        self._send(400, {"error": "missing mcp-session-id"})
                        return
                    if session != outer.active_session or outer.expired:
                        payload = {
                            "jsonrpc": "2.0",
                            "id": body.get("id"),
                            "error": {"code": -32600, "message": "Session not found"},
                        }
                        self._send(
                            200 if outer.expiry_mode == "jsonrpc" else 404, payload
                        )
                        return
                    if outer.http_error:
                        self._send(outer.http_error, {"error": "simulated transport failure"})
                        return
                    params = body.get("params") or {}
                    name = str(params.get("name") or "")
                    question = str((params.get("arguments") or {}).get("question") or "")
                    outer.questions.append((name, question))
                    if outer.tool_error:
                        result = {
                            "content": [{"type": "text", "text": outer.tool_error}],
                            "isError": True,
                        }
                    else:
                        reply = (
                            outer.replies.pop(0)
                            if outer.replies
                            else (outer.default_reply or "No further replies.")
                        )
                        result = {
                            "content": [{"type": "text", "text": reply}],
                            "isError": False,
                        }
                    self._send(200, {"jsonrpc": "2.0", "id": body.get("id"), "result": result})
                    outer.tool_call_count += 1
                    if (
                        outer.expire_after_tool_calls is not None
                        and outer.tool_call_count >= outer.expire_after_tool_calls
                    ):
                        outer.expired = True
                    return
                self._send(200, {"jsonrpc": "2.0", "id": body.get("id"), "result": {}})

            def _send(
                self,
                status: int,
                payload: dict[str, Any],
                session: str | None = None,
            ) -> None:
                content_type = "application/json"
                if outer.mode == "sse":
                    head = {"jsonrpc": payload.get("jsonrpc"), "id": payload.get("id")}
                    rest = {key: value for key, value in payload.items() if key not in head}
                    body = json.dumps(head)[:-1] + ",\n" + json.dumps(rest)[1:]
                    data_lines = "\n".join(f"data: {line}" for line in body.splitlines())
                    raw = (
                        ": keepalive\n\n"
                        'event: message\ndata: {"jsonrpc":"2.0","method":"notifications/progress",'
                        '"params":{}}\n\n'
                        f"event: message\n{data_lines}\n\n"
                    ).encode()
                    content_type = "text/event-stream"
                else:
                    raw = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("content-type", content_type)
                if session:
                    self.send_header("mcp-session-id", session)
                self.send_header("content-length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        return Handler


# --------------------------------------------------------------------------- helpers


def _init_git_repo(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "README.md").write_text("# Calculator\n", encoding="utf-8")
    for command in (
        ["git", "init"],
        ["git", "config", "user.email", "dev@example.com"],
        ["git", "config", "user.name", "Dev"],
        ["git", "add", "README.md"],
        ["git", "commit", "-m", "init"],
    ):
        subprocess.run(command, cwd=root, check=True, capture_output=True)
    return root


def _git_status(root: Path) -> str:
    return subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def _run_bridge(
    args: list[str], *, env: dict[str, str] | None = None, timeout: float = 180
) -> tuple[subprocess.CompletedProcess[str], dict[str, Any]]:
    process = subprocess.run(
        [sys.executable, str(BRIDGE_SCRIPT), *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        env={**os.environ, **(env or {})},
    )
    summary: dict[str, Any] = {}
    for line in reversed(process.stdout.strip().splitlines()):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            summary = payload
            break
    return process, summary


def _bridge_common(
    phase: str,
    olympus: Path,
    repo: Path,
    state: Path,
    server: FakeMcpServer,
    trajectory: Path,
    plan_path: Path,
    *,
    max_turns: int = 4,
    base_commit_file: Path | None = None,
) -> list[str]:
    args = [
        "--phase",
        phase,
        "--olympus-bin",
        str(olympus),
        "--workspace",
        str(repo),
        "--state-dir",
        str(state),
        "--max-turns",
        str(max_turns),
        "--trajectory",
        str(trajectory),
        "--plan-out",
        str(plan_path),
        "--mcp-url",
        server.url,
    ]
    if base_commit_file is not None:
        args += ["--base-commit-file", str(base_commit_file)]
    return args


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout


def _git_subjects(repo: Path) -> list[str]:
    return [line for line in _git(repo, "log", "--format=%s", "--reverse").splitlines() if line]


def _pinned_git_env() -> dict[str, str]:
    return {"GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}


def _upstream_handoff_script(repo: Path, logs: Path, base_file: Path) -> str:
    instruction = (
        UPSTREAM_HANDOFF_INSTRUCTION.replace("/app", str(repo))
        .replace("/logs/agent", str(logs))
        .replace("/tmp/base_commit.txt", str(base_file))
    )
    return extract_handoff_script(instruction, logs_dir=logs, workspace=str(repo))


def _run_upstream_handoff(
    repo: Path, logs: Path, base_file: Path
) -> subprocess.CompletedProcess[str]:
    script = _upstream_handoff_script(repo, logs, base_file)
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True)


def _jsonl_events(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


FAKE_OLYMPUS = r'''#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

PLAN_MESSAGE = "Plan plan-fake is approved with 1 task(s): T1: Implement calculator."


def opt(name, default=None):
    argv = sys.argv[1:]
    for index, item in enumerate(argv):
        if item == name and index + 1 < len(argv):
            return argv[index + 1]
        if item.startswith(name + "="):
            return item.split("=", 1)[1]
    return default


state_dir_arg = opt("--state-dir")
db_arg = opt("--db")
if state_dir_arg:
    state_dir = Path(state_dir_arg)
elif db_arg:
    state_dir = Path(db_arg).parent
else:
    state_dir = Path("/tmp/olympus-session")
state_dir.mkdir(parents=True, exist_ok=True)
workspace = Path(opt("--workspace", "/app"))
step_mode = opt("--step-mode", "execute")
message = opt("--message", "")
workflow = opt("--workflow", "full_sdlc_tdd")
state_path = state_dir / "fake-state.json"
CALC = (
    "def divide(a, b):\n"
    "    if b == 0:\n"
    "        raise ValueError('zero')\n"
    "    return a / b\n"
)
CALC_REVISED = (
    "def divide(a, b):\n"
    "    if b == 0:\n"
    "        raise ValueError('zero divisor')\n"
    "    return a / b\n"
)
TEST_FILE = (
    "from calculator import divide\n"
    "\n"
    "\n"
    "def test_divide_by_zero():\n"
    "    try:\n"
    "        divide(1, 0)\n"
    "    except ValueError as exc:\n"
    "        assert str(exc) == 'zero'\n"
    "    else:\n"
    "        raise AssertionError('expected ValueError')\n"
)

with (state_dir / "fake-invocations.jsonl").open("a", encoding="utf-8") as handle:
    handle.write(json.dumps({"argv": sys.argv[1:], "cwd": os.getcwd()}) + "\n")


def load():
    try:
        return json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def save(state):
    state_path.write_text(json.dumps(state), encoding="utf-8")


def emit(status, *, pause_kind=None, message="", changed_files=None, error=None):
    payload = {
        "run_id": load().get("run_id", ""),
        "status": status,
        "pause_kind": pause_kind,
        "step_mode": step_mode,
        "workflow": workflow,
        "message": message,
        "unresolved_blockers": [],
        "changed_files": changed_files or [],
        "error": error,
    }
    usage = os.environ.get("FAKE_OLYMPUS_TOKEN_USAGE")
    if usage:
        try:
            payload["token_usage"] = json.loads(usage)
        except ValueError:
            pass
    print(json.dumps(payload))
    raise SystemExit(0)


argv = sys.argv[1:]
command = argv[0] if argv else ""

if command == "events":
    state = load()
    kind = state.get("phase")
    if kind in ("clarify", "plan_handoff"):
        print(json.dumps([{"event_type": "session.paused", "payload": {"pause_kind": kind}}]))
    else:
        print(json.dumps([{"event_type": "session.completed", "payload": {}}]))
    raise SystemExit(0)

if command != "session":
    print(json.dumps({"status": "failed", "error": "unknown command " + command}))
    raise SystemExit(1)

subcommand = argv[1]
state = load()
infra_path = state_dir / "fake-infra.json"


def take_infra_failure():
    try:
        payload = json.loads(infra_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        payload = {"remaining": int(os.environ.get("FAKE_OLYMPUS_INFRA_RETRIES", "0"))}
    if int(payload.get("remaining", 0)) <= 0:
        return False
    payload["remaining"] = int(payload["remaining"]) - 1
    try:
        infra_path.write_text(json.dumps(payload), encoding="utf-8")
    except OSError:
        return False
    return True


if subcommand in ("start", "reply") and take_infra_failure():
    emit(
        "needs_input",
        pause_kind="blocked",
        message=(
            "Task execution blocked: T1: provider_error: The read operation timed out"
        ),
    )
if subcommand == "start":
    state = {"run_id": "run-fake-1", "phase": "clarify", "goal": message}
    save(state)
    state = load()
    emit(
        "needs_input",
        pause_kind="clarify",
        message="AMB-1: What behavior should divide(1, 0) have?",
    )
if subcommand == "reply":
    run_id = argv[2]
    if run_id != state.get("run_id"):
        emit("failed", error="unknown run " + run_id)
    if os.environ.get("FAKE_OLYMPUS_ALWAYS_CLARIFY") == "1":
        emit(
            "needs_input",
            pause_kind="clarify",
            message="AMB-1: Still unclear, please answer again.",
        )
    if step_mode == "plan_only":
        if state.get("phase") == "clarify":
            state["phase"] = "plan_handoff"
            save(state)
        emit("needs_input", pause_kind="plan_handoff", message=PLAN_MESSAGE)
    if step_mode == "execute":
        instruction = message.casefold()
        if state.get("phase") == "plan_handoff":
            (workspace / "calculator.py").write_text(CALC, encoding="utf-8")
            state["phase"] = "completed"
            save(state)
            emit(
                "completed",
                message="Session completed with 1 changed file(s).",
                changed_files=["calculator.py"],
            )
        if "add tests" in instruction and not state.get("tests_written"):
            (workspace / "tests").mkdir(exist_ok=True)
            (workspace / "tests" / "test_calculator.py").write_text(
                TEST_FILE, encoding="utf-8"
            )
            state["tests_written"] = True
            save(state)
            emit(
                "completed",
                message="Session completed with 1 changed file(s).",
                changed_files=["tests/test_calculator.py"],
            )
        if not state.get("revised") and ("change" in instruction or "fix" in instruction):
            (workspace / "calculator.py").write_text(CALC_REVISED, encoding="utf-8")
            state["revised"] = True
            save(state)
            emit(
                "completed",
                message="Session completed with 1 changed file(s).",
                changed_files=["calculator.py"],
            )
        emit("completed", message="Session completed with no workspace changes.")
    emit("failed", error="unexpected step mode " + step_mode)

emit("failed", error="unknown session command " + subcommand)
'''


def _write_fake_olympus(tmp_path: Path) -> Path:
    script = tmp_path / "fake-olympus"
    script.write_text(
        FAKE_OLYMPUS.replace("#!/usr/bin/env python3", f"#!{sys.executable}", 1),
        encoding="utf-8",
    )
    script.chmod(0o755)
    return script


_REAL_OLYMPUS_TEMPLATE = r'''#!/usr/bin/env python3
import json
import re
import sys

sys.path.insert(0, "__SRC__")

from olympus.cli import session_cmd
from olympus.providers.base import LLMResponse, MockLLMProvider, ToolCall

SPEC = __SPEC__
DESIGN = __DESIGN__
PLAN = __PLAN__
CALC = "def divide(a, b):\n    if b == 0:\n        raise ValueError('zero')\n    return a / b\n"


def respond(messages, tools):
    del tools
    system = messages[0].content if messages else ""
    user = messages[1].content if len(messages) > 1 else ""
    if "Session Reply Router" in system:
        ids = re.findall(r"^- (AMB-\d+):", user, re.MULTILINE)
        if ids:
            answers = {amb_id: "raise ValueError('zero')" for amb_id in ids}
            return LLMResponse(
                content=json.dumps({"action": "clarify", "answers": answers})
            )
        return LLMResponse(content=json.dumps({"action": "approve", "message": "approved"}))
    if "Requirements Alignment Architect" in system:
        return LLMResponse(content=json.dumps(SPEC))
    if "White-Box Solution Architect" in system:
        return LLMResponse(content=json.dumps(DESIGN))
    if "Task Planner" in system:
        return LLMResponse(content=json.dumps(PLAN))
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
            return LLMResponse(content="Implementation finished.")
        match = re.search(r"Sandbox workspace root: (.+)", user)
        if match is None:
            raise AssertionError("missing sandbox workspace root")
        sandbox = match.group(1).strip()
        return LLMResponse(
            content="Writing the calculator implementation.",
            tool_calls=(
                ToolCall(
                    id="write-1",
                    name="write_file",
                    arguments={"path": sandbox + "/calculator.py", "content": CALC},
                ),
            ),
        )
    raise AssertionError("unexpected system prompt: " + system[:120])


session_cmd.build_session_provider = lambda **kwargs: MockLLMProvider(responder=respond)

from olympus.cli.app import app

app()
'''

_GOAL_SPEC = {
    "spec_id": "spec-calc",
    "version": 1,
    "goal": "Build a small Python calculator library for developers",
    "deliverable_kind": "library",
    "observable_surfaces": ["calculator module"],
    "actors": ["developer"],
    "test_seams": ["calculator.divide(a, b)"],
    "requirements": [
        {
            "id": "R1",
            "description": "Expose divide(a, b) returning a / b and rejecting a zero divisor",
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
    ],
    "out_of_scope": ["graphical interface"],
    "constraints": [],
    "open_ambiguities": [
        {
            "id": "AMB-1",
            "question": "What behavior should divide(1, 0) have?",
            "impact": "Error contract",
            "options": ["raise ValueError", "return None"],
            "blocking": True,
        }
    ],
}

_DESIGN_SPEC = {
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
            "requirement_refs": ["R1"],
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

_TASK_PLAN = {
    "plan_id": "plan-calc",
    "spec_id": "spec-calc",
    "spec_hash": "sha256:pending",
    "version": 1,
    "tasks": [
        {
            "task_id": "T1",
            "title": "Implement calculator helpers",
            "objective": "Implement divide(a, b) in calculator.py with a zero guard",
            "requirement_refs": ["R1"],
            "design_refs": ["M1"],
            "write_scope": ["calculator.py"],
            "acceptance_checks": [
                {
                    "id": "C1",
                    "kind": "command",
                    "description": "Verify the calculator contract",
                    "command": (
                        'python3 -c "import calculator; '
                        "assert calculator.divide(6, 2) == 3\""
                    ),
                }
            ],
        }
    ],
}


def _write_real_olympus(tmp_path: Path) -> Path:
    script = tmp_path / "olympus-real"
    content = (
        _REAL_OLYMPUS_TEMPLATE.replace("__SRC__", str(PROJECT_SRC))
        .replace("__SPEC__", repr(_GOAL_SPEC))
        .replace("__DESIGN__", repr(_DESIGN_SPEC))
        .replace("__PLAN__", repr(_TASK_PLAN))
    )
    script.write_text(content, encoding="utf-8")
    script.chmod(0o755)
    return script


class FakeEnvironment:
    """Records container commands and returns scripted ``ExecResult`` objects."""

    def __init__(self, handler: Any = None) -> None:
        self.commands: list[dict[str, Any]] = []
        self.uploads: list[tuple[Path, str, str]] = []
        self.uploaded_dirs: list[tuple[Path, str]] = []
        self.handler = handler or (
            lambda command, cwd, env, timeout_sec: ExecResult(return_code=0)
        )

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: Any = None,
    ) -> Any:
        self.commands.append(
            {"command": command, "cwd": cwd, "env": env, "timeout_sec": timeout_sec}
        )
        return self.handler(command, cwd, env, timeout_sec)

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        content = Path(source_path).read_text(encoding="utf-8")
        self.uploads.append((Path(source_path), str(target_path), content))

    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        self.uploaded_dirs.append((Path(source_dir), str(target_dir)))


def _agent(tmp_path: Path, **kwargs: Any) -> OlympusInteractAgent:
    kwargs.setdefault("model_name", "openai/gpt-test")
    return OlympusInteractAgent(logs_dir=tmp_path / "logs", **kwargs)


def _context() -> SimpleNamespace:
    return SimpleNamespace(metadata=None)


def _ok_bridge(phase: str, run_id: str = "run-1", turns: int = 0) -> Any:
    payload = json.dumps(
        {"status": "ok", "phase": phase, "run_id": run_id, "turns": turns}
    )
    return lambda command, cwd, env, timeout_sec: ExecResult(
        return_code=0, stdout=payload + "\n"
    )


# --------------------------------------------------------------------------- MCP client


def test_mcp_client_json_handshake_and_ask_user() -> None:
    server = FakeMcpServer(replies=["answer one"])
    try:
        with McpHttpClient(server.url) as client:
            reply = client.ask_user("What should I do?")
        assert reply == "answer one"
        assert server.questions == [("ask_user", "What should I do?")]
        assert server.init_notified is True
        tool_requests = [
            request
            for request in server.requests
            if request["body"].get("method") == "tools/call"
        ]
        assert len(tool_requests) == 1
        headers = tool_requests[0]["headers"]
        assert headers.get("mcp-session-id") == "fake-session-1"
        assert headers.get("mcp-protocol-version")
        assert "text/event-stream" in headers.get("accept", "")
    finally:
        server.close()


def test_mcp_client_parses_sse_with_multiline_data_and_session_header() -> None:
    server = FakeMcpServer(replies=["answer over sse"], mode="sse")
    try:
        with McpHttpClient(server.url) as client:
            reply = client.ask_user("Stream it")
        assert reply == "answer over sse"
        assert server.questions == [("ask_user", "Stream it")]
        tool_requests = [
            request
            for request in server.requests
            if request["body"].get("method") == "tools/call"
        ]
        assert tool_requests[0]["headers"].get("mcp-session-id") == "fake-session-1"
    finally:
        server.close()


def test_parse_sse_messages_joins_multiline_data() -> None:
    messages = parse_sse_messages(
        'event: message\ndata: {"a": 1,\ndata: "b": 2}\n\n'
        ": comment\n\n"
        "data: [DONE]\n\n"
    )
    assert messages == [{"a": 1, "b": 2}]


def test_mcp_client_maps_tool_errors() -> None:
    server = FakeMcpServer(tool_error="simulated backend failure")
    try:
        with McpHttpClient(server.url) as client, pytest.raises(
            McpToolError, match="simulated backend failure"
        ):
            client.ask_user("Will this fail?")
    finally:
        server.close()


def test_mcp_client_maps_transport_errors() -> None:
    server = FakeMcpServer(http_error=500)
    try:
        with McpHttpClient(server.url) as client, pytest.raises(
            McpError, match="MCP HTTP 500"
        ):
            client.ask_user("Will this fail?")
    finally:
        server.close()


def test_mcp_client_reconnects_transparently_after_expired_http_session() -> None:
    server = FakeMcpServer(replies=["first answer", "second answer"], expire_after_tool_calls=1)
    try:
        with McpHttpClient(server.url) as client:
            assert client.ask_user("First question?") == "first answer"
            assert client.ask_user("Second question?") == "second answer"
        assert [question for _, question in server.questions] == [
            "First question?",
            "Second question?",
        ]
        assert server.session_generation == 2
        tool_calls = [
            request
            for request in server.requests
            if request["body"].get("method") == "tools/call"
        ]
        assert [
            call["headers"].get("mcp-session-id") for call in tool_calls
        ] == ["fake-session-1", "fake-session-1", "fake-session-2"]
        assert [
            request["body"].get("method") for request in server.requests
        ].count("initialize") == 2
    finally:
        server.close()


def test_mcp_client_reconnects_on_jsonrpc_session_not_found() -> None:
    server = FakeMcpServer(
        replies=["before expiry", "after expiry"],
        expire_after_tool_calls=1,
        expiry_mode="jsonrpc",
    )
    try:
        with McpHttpClient(server.url) as client:
            assert client.ask_user("Before?") == "before expiry"
            assert client.ask_user("After?") == "after expiry"
        assert server.session_generation == 2
        assert [question for _, question in server.questions] == ["Before?", "After?"]
    finally:
        server.close()


def test_mcp_client_does_not_reconnect_for_tool_errors_or_other_http_failures() -> None:
    session_tool_error = FakeMcpServer(tool_error="session not found")
    try:
        with McpHttpClient(session_tool_error.url) as client, pytest.raises(McpToolError):
            client.ask_user("Is this an expiry?")
        assert session_tool_error.session_generation == 1
    finally:
        session_tool_error.close()

    server_error = FakeMcpServer(http_error=500)
    try:
        with McpHttpClient(server_error.url) as client, pytest.raises(
            McpError, match="MCP HTTP 500"
        ):
            client.ask_user("Will this fail?")
        assert server_error.session_generation == 1
    finally:
        server_error.close()


def test_mcp_client_surfaces_second_failure_after_reconnect() -> None:
    server = FakeMcpServer(expire_after_tool_calls=0)
    try:
        with McpHttpClient(server.url) as client, pytest.raises(McpError, match="404"):
            client.ask_user("Always expired?")
        assert server.session_generation == 2
    finally:
        server.close()


# --------------------------------------------------------------------------- step roles


@pytest.mark.parametrize(
    ("instruction", "index", "expected"),
    [
        (PLAN_INSTRUCTION, 1, "plan"),
        (EXECUTE_INSTRUCTION, 2, "execute"),
        (HANDOFF_INSTRUCTION, 3, "handoff"),
        ("Now add tests for the implementation changes you made.", 4, "execute"),
        ("This is the final handoff step.", 5, "handoff"),
        ("", 1, "plan"),
        ("", 2, "execute"),
    ],
)
def test_classify_step(instruction: str, index: int, expected: str) -> None:
    assert classify_step(instruction, index) == expected


def test_classify_step_honours_explicit_roles() -> None:
    assert classify_step(PLAN_INSTRUCTION, 2, "execute,handoff") == "handoff"
    with pytest.raises(ValueError, match="step_roles"):
        classify_step(PLAN_INSTRUCTION, 3, "plan,execute")
    with pytest.raises(ValueError, match="unknown step role"):
        classify_step(PLAN_INSTRUCTION, 1, "bogus")


def test_extract_handoff_script_prefers_fenced_shell_blocks() -> None:
    script = extract_handoff_script(HANDOFF_INSTRUCTION)
    assert "git status --short > /logs/agent/final-status.txt" in script
    assert script.startswith("set -euo pipefail")


def test_extract_handoff_script_falls_back_to_status_capture() -> None:
    script = extract_handoff_script(
        "This is a handoff step.", logs_dir="/logs/agent", workspace="/app"
    )
    assert "mkdir -p /logs/agent" in script
    assert "handoff-status.txt" in script


# --------------------------------------------------------------------------- bridge


def test_bridge_plan_then_execute_preserves_state_and_scope(tmp_path: Path) -> None:
    repo = _init_git_repo(tmp_path / "repo")
    state = tmp_path / "state"
    plan_path = tmp_path / "plan.md"
    trajectory = tmp_path / "trajectory.jsonl"
    fake = _write_fake_olympus(tmp_path)
    server = FakeMcpServer(
        replies=[
            "AMB-1: raise ValueError('zero')",
            "Looks good, ship it.",
            "Looks good, ship it.",
        ],
        mode="sse",
    )
    try:
        plan_args = _bridge_common(
            "plan", fake, repo, state, server, trajectory, plan_path
        ) + [
            "--goal",
            "Build a calculator library",
            "--workflow",
            "repo_fast_patch",
            "--worker-driver",
            "native",
            "--monitor-model",
            "monitor-1",
        ]
        process, summary = _run_bridge(plan_args)
        assert process.returncode == 0, process.stdout + process.stderr
        assert summary["status"] == "ok"
        assert summary["run_id"] == "run-fake-1"
        assert summary["turns"] == 2
        assert plan_path.is_file()
        plan_text = plan_path.read_text(encoding="utf-8")
        assert plan_text.startswith("# Agreed Implementation Plan")
        assert "Approved Task Plan" in plan_text
        assert "# Maintainer Dialogue" in plan_text
        assert (state / "run-id.txt").read_text(encoding="utf-8").strip() == "run-fake-1"
        assert _git_status(repo) == ""
        assert not (repo / ".olympus").exists()

        reply_file = tmp_path / "execute-instruction.md"
        reply_file.write_text(EXECUTE_INSTRUCTION, encoding="utf-8")
        base_file = tmp_path / "base_commit.txt"
        execute_args = _bridge_common(
            "execute",
            fake,
            repo,
            state,
            server,
            trajectory,
            plan_path,
            base_commit_file=base_file,
        ) + ["--reply-file", str(reply_file)]
        process, summary = _run_bridge(execute_args)
        assert process.returncode == 0, process.stdout + process.stderr
        assert summary["status"] == "ok"
        assert (repo / "calculator.py").is_file()
        assert _git_status(repo) == ""
        assert "calculator.py" in _git(repo, "ls-files").splitlines()
        assert _git_subjects(repo)[-1] == "initial implementation"
        assert not (repo / ".olympus").exists()

        invocations = [
            json.loads(line)
            for line in (state / "fake-invocations.jsonl").read_text().splitlines()
        ]
        session_modes = [
            invocation["argv"][invocation["argv"].index("--step-mode") + 1]
            for invocation in invocations
            if invocation["argv"][0] == "session"
        ]
        assert session_modes == ["plan_only", "plan_only", "plan_only", "execute"]
        assert any(invocation["argv"][0] == "events" for invocation in invocations)
        workflow = invocations[0]["argv"][invocations[0]["argv"].index("--workflow") + 1]
        assert workflow == "repo_fast_patch"
        assert "--monitor-model" in invocations[0]["argv"]

        event_names = [record["event"] for record in _jsonl_events(trajectory)]
        assert "ask_user" in event_names
        assert "plan.saved" in event_names
        assert "session.envelope" in event_names
    finally:
        server.close()


def test_bridge_plan_asks_maintainer_for_initial_goal(tmp_path: Path) -> None:
    repo = _init_git_repo(tmp_path / "repo")
    state = tmp_path / "state"
    plan_path = tmp_path / "plan.md"
    trajectory = tmp_path / "trajectory.jsonl"
    fake = _write_fake_olympus(tmp_path)
    maintainer_goal = "Add a deterministic cookie store to the HTTP client"
    server = FakeMcpServer(
        replies=[maintainer_goal, "AMB-1: raise ValueError('zero')", "Looks good."]
    )
    try:
        args = _bridge_common(
            "plan", fake, repo, state, server, trajectory, plan_path
        )
        process, summary = _run_bridge(args)
        assert process.returncode == 0, process.stdout + process.stderr
        assert summary["status"] == "ok"
        assert summary["turns"] == 3
        assert server.questions[0][0] == "ask_user"
        assert "What change would you like me to make" in server.questions[0][1]
        fake_state = json.loads((state / "fake-state.json").read_text())
        assert fake_state["goal"] == maintainer_goal
        goal_events = [
            record for record in _jsonl_events(trajectory) if record["event"] == "plan.goal"
        ]
        assert goal_events and goal_events[0]["goal"] == maintainer_goal
    finally:
        server.close()


def test_bridge_enforces_turn_cap(tmp_path: Path) -> None:
    repo = _init_git_repo(tmp_path / "repo")
    state = tmp_path / "state"
    trajectory = tmp_path / "trajectory.jsonl"
    fake = _write_fake_olympus(tmp_path)
    server = FakeMcpServer(default_reply="I still do not understand.")
    try:
        args = _bridge_common(
            "plan",
            fake,
            repo,
            state,
            server,
            trajectory,
            tmp_path / "plan.md",
            max_turns=2,
        ) + ["--goal", "Build a calculator library"]
        process, summary = _run_bridge(
            args, env={"FAKE_OLYMPUS_ALWAYS_CLARIFY": "1"}
        )
        assert process.returncode == EXIT_TURN_CAP
        assert summary["status"] == "error"
        assert "interaction cap" in summary["error"]
        assert len(server.questions) == 2
        events = _jsonl_events(trajectory)
        assert any(record["event"] == "turn_cap_exceeded" for record in events)
    finally:
        server.close()


def test_bridge_reports_sidecar_errors(tmp_path: Path) -> None:
    repo = _init_git_repo(tmp_path / "repo")
    state = tmp_path / "state"
    trajectory = tmp_path / "trajectory.jsonl"
    fake = _write_fake_olympus(tmp_path)
    server = FakeMcpServer(tool_error="simulated backend failure")
    try:
        args = _bridge_common(
            "plan", fake, repo, state, server, trajectory, tmp_path / "plan.md"
        ) + ["--goal", "Build a calculator library"]
        process, summary = _run_bridge(args)
        assert process.returncode == EXIT_SIDECAR_ERROR
        assert summary["status"] == "error"
        assert "ask_user sidecar failed" in summary["error"]
        events = _jsonl_events(trajectory)
        assert any(record["event"] == "ask_user.error" for record in events)
    finally:
        server.close()


def test_bridge_execute_requires_persisted_run_id(tmp_path: Path) -> None:
    repo = _init_git_repo(tmp_path / "repo")
    fake = _write_fake_olympus(tmp_path)
    server = FakeMcpServer()
    try:
        reply_file = tmp_path / "reply.md"
        reply_file.write_text(EXECUTE_INSTRUCTION, encoding="utf-8")
        args = _bridge_common(
            "execute",
            fake,
            repo,
            tmp_path / "state",
            server,
            tmp_path / "trajectory.jsonl",
            tmp_path / "plan.md",
        ) + ["--reply-file", str(reply_file)]
        process, summary = _run_bridge(args)
        assert process.returncode == EXIT_CONFIG_ERROR
        assert "persisted run id" in summary["error"]
    finally:
        server.close()


def test_bridge_redacts_secret_values(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    secret = "supersecretvalue123"
    monkeypatch.setenv("DEPLOY_API_TOKEN", secret)
    assert secret not in redact_text(f"token is {secret}")
    repo = _init_git_repo(tmp_path / "repo")
    state = tmp_path / "state"
    plan_path = tmp_path / "plan.md"
    trajectory = tmp_path / "trajectory.jsonl"
    fake = _write_fake_olympus(tmp_path)
    server = FakeMcpServer(replies=[f"use {secret}", f"approved, keep {secret}"])
    try:
        args = _bridge_common(
            "plan", fake, repo, state, server, trajectory, plan_path
        ) + ["--goal", f"Build a calculator with {secret}"]
        process, summary = _run_bridge(args)
        assert process.returncode == 0, process.stdout + process.stderr
        assert summary["status"] == "ok"
        assert secret not in trajectory.read_text(encoding="utf-8")
        assert secret not in plan_path.read_text(encoding="utf-8")
        assert "***" in trajectory.read_text(encoding="utf-8")
    finally:
        server.close()


def test_real_session_cli_plan_then_execute(tmp_path: Path) -> None:
    repo = _init_git_repo(tmp_path / "repo")
    state = tmp_path / "state"
    plan_path = tmp_path / "plan.md"
    trajectory = tmp_path / "trajectory.jsonl"
    olympus = _write_real_olympus(tmp_path)
    server = FakeMcpServer(
        replies=[
            "AMB-1: raise ValueError('zero') and accept int and float inputs.",
            "Looks good, ship it.",
            "Looks good, ship it.",
        ]
    )
    try:
        plan_args = _bridge_common(
            "plan", olympus, repo, state, server, trajectory, plan_path
        ) + ["--goal", "Build a small calculator library", "--provider", "mock"]
        process, summary = _run_bridge(plan_args)
        assert process.returncode == 0, process.stdout + process.stderr
        assert summary["status"] == "ok", summary
        assert plan_path.is_file()
        plan_text = plan_path.read_text(encoding="utf-8")
        assert "## Approved Task Plan" in plan_text
        assert _git_status(repo) == ""
        assert not (repo / ".olympus").exists()
        assert (state / "olympus.db").is_file()

        reply_file = tmp_path / "execute-instruction.md"
        reply_file.write_text(EXECUTE_INSTRUCTION, encoding="utf-8")
        execute_args = _bridge_common(
            "execute",
            olympus,
            repo,
            state,
            server,
            trajectory,
            plan_path,
            base_commit_file=tmp_path / "base_commit.txt",
        ) + ["--reply-file", str(reply_file), "--provider", "mock"]
        process, summary = _run_bridge(execute_args)
        assert process.returncode == 0, process.stdout + process.stderr
        assert summary["status"] == "ok", summary
        calculator = repo / "calculator.py"
        assert calculator.is_file()
        assert "def divide" in calculator.read_text(encoding="utf-8")
        assert _git_status(repo) == ""
        assert "calculator.py" in _git(repo, "ls-files").splitlines()
        assert _git_subjects(repo)[-1] == "initial implementation"
        assert not (repo / ".olympus").exists()
    finally:
        server.close()


def test_bridge_execute_bootstraps_git_commits_reviews_and_hands_off(
    tmp_path: Path,
) -> None:
    repo = _init_git_repo(tmp_path / "repo")
    for key in ("user.name", "user.email"):
        subprocess.run(["git", "config", "--unset", key], cwd=repo, check=True)
    state = tmp_path / "state"
    plan_path = tmp_path / "plan.md"
    trajectory = tmp_path / "trajectory.jsonl"
    base_file = tmp_path / "base_commit.txt"
    logs = tmp_path / "logs"
    olympus = _write_real_olympus(tmp_path)
    server = FakeMcpServer(
        replies=[
            "AMB-1: raise ValueError('zero').",
            "Looks good, ship it.",
            "Looks good, ship it.",
        ]
    )
    try:
        plan_args = _bridge_common(
            "plan", olympus, repo, state, server, trajectory, plan_path
        ) + ["--goal", "Build a small calculator library", "--provider", "mock"]
        process, summary = _run_bridge(plan_args)
        assert process.returncode == 0, process.stdout + process.stderr
        start_commit = _git(repo, "rev-parse", "HEAD").strip()

        instruction_path = tmp_path / "implement.md"
        instruction_path.write_text(UPSTREAM_IMPLEMENT_INSTRUCTION, encoding="utf-8")
        execute_args = _bridge_common(
            "execute",
            olympus,
            repo,
            state,
            server,
            trajectory,
            plan_path,
            base_commit_file=base_file,
        ) + ["--reply-file", str(instruction_path), "--provider", "mock"]
        process, summary = _run_bridge(execute_args, env=_pinned_git_env())
        assert process.returncode == 0, process.stdout + process.stderr
        assert summary["status"] == "ok", summary
        assert summary["turns"] == 1

        assert base_file.read_text(encoding="utf-8").strip() == start_commit
        assert _git(repo, "rev-parse", "HEAD").strip() != start_commit
        assert _git_subjects(repo) == ["init", "initial implementation"]
        assert _git(repo, "config", "--local", "user.name").strip() == "Dev"
        assert _git(repo, "config", "--local", "user.email").strip() == "dev@example.com"
        assert _git_status(repo) == ""

        session_events = [
            record
            for record in _jsonl_events(trajectory)
            if record["event"] == "session.invoke" and record["step_mode"] == "execute"
        ]
        assert session_events
        assert session_events[0]["message"] == UPSTREAM_IMPLEMENT_INSTRUCTION.strip()

        reviews = [
            record
            for record in _jsonl_events(trajectory)
            if record["event"] == "ask_user" and record["phase"] == "execute_review"
        ]
        assert len(reviews) == 1
        assert "review" in reviews[0]["question"].casefold()
        assert "latest commit" in reviews[0]["question"].casefold()

        result = _run_upstream_handoff(repo, logs, base_file)
        assert result.returncode == 0, result.stderr
        change_log = (logs / "change-log.txt").read_text(encoding="utf-8")
        assert f"Base commit: {start_commit}" in change_log
        assert "initial implementation" in change_log
        final_diff = (logs / "final.diff").read_text(encoding="utf-8")
        assert "calculator.py" in final_diff
        assert "def divide" in final_diff
        assert (logs / "final-status.txt").read_text(encoding="utf-8").strip() == ""
    finally:
        server.close()


def test_bridge_execute_revision_round_commits_and_hands_off(tmp_path: Path) -> None:
    repo = _init_git_repo(tmp_path / "repo")
    state = tmp_path / "state"
    plan_path = tmp_path / "plan.md"
    trajectory = tmp_path / "trajectory.jsonl"
    base_file = tmp_path / "base_commit.txt"
    logs = tmp_path / "logs"
    fake = _write_fake_olympus(tmp_path)
    server = FakeMcpServer(
        replies=[
            "AMB-1: raise ValueError('zero').",
            "Looks good, ship it.",
            "Please change the zero-division message to 'zero divisor'.",
            "Looks good, ship it.",
        ]
    )
    try:
        plan_args = _bridge_common(
            "plan", fake, repo, state, server, trajectory, plan_path
        ) + ["--goal", "Build a calculator library"]
        process, summary = _run_bridge(plan_args)
        assert process.returncode == 0, process.stdout + process.stderr
        start_commit = _git(repo, "rev-parse", "HEAD").strip()

        reply_file = tmp_path / "implement.md"
        reply_file.write_text(UPSTREAM_IMPLEMENT_INSTRUCTION, encoding="utf-8")
        execute_args = _bridge_common(
            "execute",
            fake,
            repo,
            state,
            server,
            trajectory,
            plan_path,
            base_commit_file=base_file,
        ) + ["--reply-file", str(reply_file)]
        process, summary = _run_bridge(execute_args)
        assert process.returncode == 0, process.stdout + process.stderr
        assert summary["status"] == "ok", summary
        assert summary["turns"] == 2

        assert base_file.read_text(encoding="utf-8").strip() == start_commit
        assert _git_subjects(repo) == [
            "init",
            "initial implementation",
            "address feedback",
        ]
        calculator = (repo / "calculator.py").read_text(encoding="utf-8")
        assert "zero divisor" in calculator
        assert _git_status(repo) == ""

        result = _run_upstream_handoff(repo, logs, base_file)
        assert result.returncode == 0, result.stderr
        change_log = (logs / "change-log.txt").read_text(encoding="utf-8")
        assert "initial implementation" in change_log
        assert "address feedback" in change_log
        final_diff = (logs / "final.diff").read_text(encoding="utf-8")
        assert "zero divisor" in final_diff
        assert (logs / "final-status.txt").read_text(encoding="utf-8").strip() == ""
    finally:
        server.close()


def test_bridge_write_tests_step_commits_separately(tmp_path: Path) -> None:
    repo = _init_git_repo(tmp_path / "repo")
    state = tmp_path / "state"
    plan_path = tmp_path / "plan.md"
    trajectory = tmp_path / "trajectory.jsonl"
    base_file = tmp_path / "base_commit.txt"
    fake = _write_fake_olympus(tmp_path)
    server = FakeMcpServer(
        replies=[
            "AMB-1: raise ValueError('zero').",
            "Looks good, ship it.",
            "Looks good, ship it.",
        ]
    )
    try:
        plan_args = _bridge_common(
            "plan", fake, repo, state, server, trajectory, plan_path
        ) + ["--goal", "Build a calculator library"]
        process, _ = _run_bridge(plan_args)
        assert process.returncode == 0, process.stdout + process.stderr

        implement_file = tmp_path / "implement.md"
        implement_file.write_text(UPSTREAM_IMPLEMENT_INSTRUCTION, encoding="utf-8")
        implement_args = _bridge_common(
            "execute",
            fake,
            repo,
            state,
            server,
            trajectory,
            plan_path,
            base_commit_file=base_file,
        ) + ["--reply-file", str(implement_file)]
        process, summary = _run_bridge(implement_args)
        assert process.returncode == 0, process.stdout + process.stderr
        assert summary["status"] == "ok", summary
        assert _git_subjects(repo) == ["init", "initial implementation"]

        tests_file = tmp_path / "write-tests.md"
        tests_file.write_text(UPSTREAM_WRITE_TESTS_INSTRUCTION, encoding="utf-8")
        review_turns = len(server.questions)
        tests_args = _bridge_common(
            "execute",
            fake,
            repo,
            state,
            server,
            trajectory,
            plan_path,
            base_commit_file=base_file,
        ) + ["--reply-file", str(tests_file)]
        process, summary = _run_bridge(tests_args)
        assert process.returncode == 0, process.stdout + process.stderr
        assert summary["status"] == "ok", summary
        assert summary["turns"] == 0
        assert len(server.questions) == review_turns

        assert _git_subjects(repo) == [
            "init",
            "initial implementation",
            "add tests",
        ]
        assert "tests/test_calculator.py" in _git(repo, "ls-files").splitlines()
        assert _git_status(repo) == ""
        assert base_file.read_text(encoding="utf-8").strip() == _git(
            repo, "rev-list", "--max-parents=0", "HEAD"
        ).strip()
    finally:
        server.close()


def test_bridge_run_id_write_failure_exits_session_failed(tmp_path: Path) -> None:
    repo = _init_git_repo(tmp_path / "repo")
    state = tmp_path / "state"
    fake = _write_fake_olympus(tmp_path)
    blocker = tmp_path / "blocked"
    blocker.write_text("not a directory\n", encoding="utf-8")
    server = FakeMcpServer()
    try:
        args = _bridge_common(
            "plan",
            fake,
            repo,
            state,
            server,
            tmp_path / "trajectory.jsonl",
            tmp_path / "plan.md",
        ) + [
            "--goal",
            "Build a calculator library",
            "--run-id-file",
            str(blocker / "run-id.txt"),
        ]
        process, summary = _run_bridge(args)
        assert process.returncode == EXIT_SESSION_FAILED
        assert summary["status"] == "error"
        assert "could not persist run id" in summary["error"]
        assert "run-id.txt" in summary["error"]
    finally:
        server.close()


def test_bridge_retries_infrastructure_block_without_asking_maintainer(
    tmp_path: Path,
) -> None:
    repo = _init_git_repo(tmp_path / "repo")
    state = tmp_path / "state"
    plan_path = tmp_path / "plan.md"
    trajectory = tmp_path / "trajectory.jsonl"
    fake = _write_fake_olympus(tmp_path)
    server = FakeMcpServer(
        replies=["AMB-1: raise ValueError('zero')", "Looks good, ship it."]
    )
    try:
        args = _bridge_common(
            "plan", fake, repo, state, server, trajectory, plan_path
        ) + [
            "--goal",
            "Build a calculator library",
            "--infra-retries",
            "2",
            "--infra-retry-backoff",
            "0",
        ]
        process, summary = _run_bridge(args, env={"FAKE_OLYMPUS_INFRA_RETRIES": "1"})
        assert process.returncode == 0, process.stdout + process.stderr
        assert summary["status"] == "ok", summary
        assert summary["turns"] == 2
        questions = [question for _, question in server.questions]
        assert len(questions) == 2
        assert all(
            "provider_error" not in question and "timed out" not in question
            for question in questions
        )
        retries = [
            record
            for record in _jsonl_events(trajectory)
            if record["event"] == "session.infra_retry"
        ]
        assert len(retries) == 1
        assert retries[0]["attempt"] == 1
        assert retries[0]["step_mode"] == "plan_only"
        assert plan_path.is_file()
    finally:
        server.close()


def test_bridge_execute_retries_infrastructure_block_without_asking_maintainer(
    tmp_path: Path,
) -> None:
    repo = _init_git_repo(tmp_path / "repo")
    state = tmp_path / "state"
    plan_path = tmp_path / "plan.md"
    trajectory = tmp_path / "trajectory.jsonl"
    base_file = tmp_path / "base_commit.txt"
    fake = _write_fake_olympus(tmp_path)
    server = FakeMcpServer(
        replies=[
            "AMB-1: raise ValueError('zero').",
            "Looks good, ship it.",
            "Looks good, ship it.",
        ]
    )
    try:
        plan_args = _bridge_common(
            "plan", fake, repo, state, server, trajectory, plan_path
        ) + ["--goal", "Build a calculator library"]
        process, _ = _run_bridge(plan_args)
        assert process.returncode == 0, process.stdout + process.stderr

        reply_file = tmp_path / "implement.md"
        reply_file.write_text(UPSTREAM_IMPLEMENT_INSTRUCTION, encoding="utf-8")
        execute_args = _bridge_common(
            "execute",
            fake,
            repo,
            state,
            server,
            trajectory,
            plan_path,
            base_commit_file=base_file,
        ) + [
            "--reply-file",
            str(reply_file),
            "--infra-retries",
            "2",
            "--infra-retry-backoff",
            "0",
        ]
        process, summary = _run_bridge(
            execute_args, env={"FAKE_OLYMPUS_INFRA_RETRIES": "1"}
        )
        assert process.returncode == 0, process.stdout + process.stderr
        assert summary["status"] == "ok", summary
        assert summary["turns"] == 1
        assert (repo / "calculator.py").is_file()
        questions = [question for _, question in server.questions]
        assert all(
            "provider_error" not in question and "timed out" not in question
            for question in questions
        )
        retries = [
            record
            for record in _jsonl_events(trajectory)
            if record["event"] == "session.infra_retry"
        ]
        assert len(retries) == 1
        assert retries[0]["step_mode"] == "execute"
    finally:
        server.close()


def test_bridge_fails_cleanly_when_infrastructure_retries_exhaust(tmp_path: Path) -> None:
    repo = _init_git_repo(tmp_path / "repo")
    state = tmp_path / "state"
    trajectory = tmp_path / "trajectory.jsonl"
    fake = _write_fake_olympus(tmp_path)
    server = FakeMcpServer()
    try:
        args = _bridge_common(
            "plan",
            fake,
            repo,
            state,
            server,
            trajectory,
            tmp_path / "plan.md",
        ) + [
            "--goal",
            "Build a calculator library",
            "--infra-retries",
            "2",
            "--infra-retry-backoff",
            "0",
        ]
        process, summary = _run_bridge(args, env={"FAKE_OLYMPUS_INFRA_RETRIES": "10"})
        assert process.returncode == EXIT_SESSION_FAILED
        assert summary["status"] == "error"
        assert "infrastructure" in summary["error"].casefold()
        assert server.questions == []
        records = _jsonl_events(trajectory)
        retries = [
            record for record in records if record["event"] == "session.infra_retry"
        ]
        assert len(retries) == 2
        assert [record["attempt"] for record in retries] == [1, 2]
        assert any(record["event"] == "bridge.failed" for record in records)
    finally:
        server.close()


def test_bridge_forwards_stage_driver_to_session_cli(tmp_path: Path) -> None:
    repo = _init_git_repo(tmp_path / "repo")
    state = tmp_path / "state"
    trajectory = tmp_path / "trajectory.jsonl"
    fake = _write_fake_olympus(tmp_path)
    server = FakeMcpServer(
        replies=["AMB-1: raise ValueError('zero').", "Looks good, ship it."]
    )
    try:
        args = _bridge_common(
            "plan", fake, repo, state, server, trajectory, tmp_path / "plan.md"
        ) + [
            "--goal",
            "Build a calculator library",
            "--stage-driver",
            "pi",
            "--worker-driver",
            "native",
        ]
        process, summary = _run_bridge(args)
        assert process.returncode == 0, process.stdout + process.stderr
        assert summary["status"] == "ok", summary
        invocations = [
            json.loads(line)
            for line in (state / "fake-invocations.jsonl").read_text().splitlines()
        ]
        session_argv = next(
            invocation["argv"]
            for invocation in invocations
            if invocation["argv"][0] == "session"
        )
        assert session_argv[session_argv.index("--stage-driver") + 1] == "pi"
        assert session_argv[session_argv.index("--worker-driver") + 1] == "native"
    finally:
        server.close()


# --------------------------------------------------------------------------- adapter setup


def test_setup_bundles_olympus_bridge_and_pi(tmp_path: Path) -> None:
    source = tmp_path / "source"
    package = source / "src" / "olympus"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (source / "pyproject.toml").write_text(
        "[project]\nname = 'olympus4swe'\nversion = '0.1.0'\n", encoding="utf-8"
    )
    (source / "README.md").write_text("# Olympus\n", encoding="utf-8")
    environment = FakeEnvironment()
    agent = _agent(
        tmp_path,
        olympus_source=str(source),
        olympus_editable="true",
        worker_driver="pi",
        install_pi="true",
    )
    asyncio.run(agent.setup(environment))

    commands = "\n".join(record["command"] for record in environment.commands)
    assert "mkdir -p" in commands
    assert "/tmp/olympus-bridge" in commands
    assert "/tmp/olympus-session" in commands
    assert "pip install" in commands
    assert "-e /tmp/olympus-src" in commands
    assert "/tmp/olympus-src" in commands
    assert "@earendil-works/pi-coding-agent" in commands
    assert "npm install" in commands
    targets = [target for _, target, _ in environment.uploads]
    assert "/tmp/olympus-bridge/interact_bridge.py" in targets
    assert "/tmp/olympus-bridge/fastmcp_client.py" in targets
    assert "/tmp/olympus-bridge/token_usage.py" in targets
    assert "/tmp/olympus-src/pyproject.toml" in targets
    assert (package, "/tmp/olympus-src/src/olympus") in environment.uploaded_dirs
    assert agent._installed_pi_bin == "/tmp/pi-cli/bin/pi"


def test_setup_defaults_to_project_source_without_pi_when_disabled(tmp_path: Path) -> None:
    environment = FakeEnvironment()
    agent = _agent(
        tmp_path,
        stage_driver="native",
        worker_driver="native",
        reviewer_driver="native",
        install_pi="false",
    )
    asyncio.run(agent.setup(environment))
    assert (
        PROJECT_SRC / "olympus",
        "/tmp/olympus-src/src/olympus",
    ) in environment.uploaded_dirs
    commands = "\n".join(record["command"] for record in environment.commands)
    assert "npm install" not in commands
    assert "pip install" in commands


def test_setup_installs_pi_by_default_for_auto_drivers(tmp_path: Path) -> None:
    environment = FakeEnvironment()
    agent = _agent(tmp_path)
    options = agent.interact_options
    assert options.stage_driver == "auto"
    assert options.worker_driver == "auto"
    assert options.reviewer_driver == "auto"
    asyncio.run(agent.setup(environment))
    commands = "\n".join(record["command"] for record in environment.commands)
    assert "npm install" in commands
    assert "@earendil-works/pi-coding-agent" in commands
    assert agent._installed_pi_bin == "/tmp/pi-cli/bin/pi"


def test_setup_respects_explicit_native_driver_overrides(tmp_path: Path) -> None:
    environment = FakeEnvironment()
    agent = _agent(
        tmp_path,
        stage_driver="native",
        worker_driver="native",
        reviewer_driver="native",
    )
    asyncio.run(agent.setup(environment))
    commands = "\n".join(record["command"] for record in environment.commands)
    assert "npm install" not in commands


# --------------------------------------------------------------------------- adapter run


def test_run_passes_kwargs_through_to_session_cli(tmp_path: Path) -> None:
    environment = FakeEnvironment(handler=_ok_bridge("plan", run_id="run-9", turns=2))
    agent = _agent(
        tmp_path,
        workflow="repo_fast_patch",
        gate_policy="default",
        worker_driver="pi@model-x",
        reviewer_driver="native",
        monitor_model="monitor-1",
        max_turns_per_step="3",
        provider="openai",
        model="gpt-5",
        base_url="http://127.0.0.1:9999/v1",
    )
    asyncio.run(agent.run(PLAN_INSTRUCTION, environment, _context()))
    command = environment.commands[-1]["command"]
    for fragment in (
        "--phase plan",
        "--workflow repo_fast_patch",
        "--gate-policy default",
        "--worker-driver pi@model-x",
        "--reviewer-driver native",
        "--monitor-model monitor-1",
        "--provider openai",
        "--model gpt-5",
        "--base-url http://127.0.0.1:9999/v1",
        "--max-turns 3",
        "--state-dir /tmp/olympus-session",
        "--initial-question",
        "--plan-out /tmp/plan.md",
    ):
        assert fragment in command, command
    assert agent.run_id == "run-9"

    environment.handler = _ok_bridge("execute", run_id="run-9")
    asyncio.run(agent.run(EXECUTE_INSTRUCTION, environment, _context()))
    command = environment.commands[-1]["command"]
    assert "--phase execute" in command
    assert "--reply-file /tmp/olympus-bridge/step-02-instruction.md" in command
    assert any(
        target == "/tmp/olympus-bridge/step-02-instruction.md"
        for _, target, _ in environment.uploads
    )
    uploaded = [
        content
        for _, target, content in environment.uploads
        if target == "/tmp/olympus-bridge/step-02-instruction.md"
    ]
    assert uploaded and uploaded[-1] == EXECUTE_INSTRUCTION


def test_run_splits_harbor_model_name_into_provider_and_model(tmp_path: Path) -> None:
    environment = FakeEnvironment(handler=_ok_bridge("plan"))
    agent = _agent(tmp_path, model_name="anthropic/claude-test")
    asyncio.run(agent.run(PLAN_INSTRUCTION, environment, _context()))
    command = environment.commands[-1]["command"]
    assert "--provider anthropic" in command
    assert "--model claude-test" in command


def test_run_forwards_auto_and_explicit_stage_driver(tmp_path: Path) -> None:
    environment = FakeEnvironment(handler=_ok_bridge("plan"))
    agent = _agent(tmp_path)
    asyncio.run(agent.run(PLAN_INSTRUCTION, environment, _context()))
    command = environment.commands[-1]["command"]
    assert "--stage-driver auto" in command
    assert "--worker-driver auto" in command
    assert "--reviewer-driver auto" in command

    override_env = FakeEnvironment(handler=_ok_bridge("plan"))
    override_agent = _agent(
        tmp_path,
        stage_driver="pi@model-stage",
        worker_driver="native",
        reviewer_driver="native",
    )
    asyncio.run(override_agent.run(PLAN_INSTRUCTION, override_env, _context()))
    override_command = override_env.commands[-1]["command"]
    assert "--stage-driver pi@model-stage" in override_command
    assert "--worker-driver native" in override_command
    assert "--reviewer-driver native" in override_command


def test_run_bridge_step_forwards_gateway_env_without_leaking(
    tmp_path: Path,
) -> None:
    secret = "sk-gateway-secret-123456"
    gateway = "https://gateway.example/v1"
    environment = FakeEnvironment(handler=_ok_bridge("plan", run_id="run-9"))
    agent = _agent(
        tmp_path,
        extra_env={
            "OPENAI_API_KEY": secret,
            "OPENAI_BASE_URL": gateway,
            "ANTHROPIC_API_KEY": "sk-ant-gateway-123456",
            "UNRELATED": "ignored",
        },
    )
    asyncio.run(agent.run(PLAN_INSTRUCTION, environment, _context()))
    bridge_calls = [
        call for call in environment.commands if "interact_bridge.py" in call["command"]
    ]
    assert bridge_calls
    env = bridge_calls[-1]["env"]
    assert env["OPENAI_API_KEY"] == secret
    assert env["OPENAI_BASE_URL"] == gateway
    assert env["ANTHROPIC_API_KEY"] == "sk-ant-gateway-123456"
    assert "UNRELATED" not in env
    assert secret not in bridge_calls[-1]["command"]
    assert agent._redact(f"key={secret} url={gateway}") == f"key=*** url={gateway}"


def test_run_handoff_executes_instruction_script(tmp_path: Path) -> None:
    environment = FakeEnvironment()
    agent = _agent(tmp_path)
    asyncio.run(agent.run(HANDOFF_INSTRUCTION, environment, _context()))
    assert len(environment.commands) == 1
    command = environment.commands[0]["command"]
    assert command.startswith("bash ")
    assert "/tmp/olympus-bridge/handoff-step-01.sh" in command
    _, target, content = environment.uploads[-1]
    assert target == "/tmp/olympus-bridge/handoff-step-01.sh"
    assert "final-status.txt" in content


def test_run_handoff_failure_raises(tmp_path: Path) -> None:
    environment = FakeEnvironment(
        handler=lambda command, cwd, env, timeout_sec: ExecResult(
            return_code=7, stderr="git exploded"
        )
    )
    agent = _agent(tmp_path)
    with pytest.raises(RuntimeError, match="handoff step 1 failed"):
        asyncio.run(agent.run(HANDOFF_INSTRUCTION, environment, _context()))


def test_run_bridge_failure_is_reported_and_redacted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "topsecretvalue999"
    monkeypatch.setenv("PROVIDER_SECRET_KEY", secret)
    payload = json.dumps(
        {"status": "error", "error": f"auth failed for {secret}", "exit_code": 3}
    )
    environment = FakeEnvironment(
        handler=lambda command, cwd, env, timeout_sec: ExecResult(
            return_code=3, stdout=payload + "\n"
        )
    )
    agent = _agent(tmp_path)
    with pytest.raises(RuntimeError) as excinfo:
        asyncio.run(agent.run(EXECUTE_INSTRUCTION, environment, _context()))
    message = str(excinfo.value)
    assert secret not in message
    assert "***" in message


def test_populate_context_post_run_reads_trajectories(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "olympus-interact-step-01.jsonl").write_text(
        json.dumps({"event": "ask_user", "turn": 1})
        + "\n"
        + json.dumps(
            {
                "event": "session.envelope",
                "envelope": {"status": "needs_input", "run_id": "run-42"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    agent = _agent(tmp_path)
    context = _context()
    agent.populate_context_post_run(context)
    summary = context.metadata["olympus_interact"]
    assert summary["steps"][0]["turns"] == 1
    assert summary["steps"][0]["status"] == "needs_input"


# --------------------------------------------------------------------------- Pi + usage passthrough


_PI_USAGE_PAYLOAD = {
    "llm_calls": 2,
    "input_tokens": 800,
    "output_tokens": 100,
    "reasoning_tokens": 40,
    "cache_read_tokens": 300,
    "cache_write_tokens": 0,
}


def test_interact_options_parse_pi_controls_and_strip_pi_prefix(tmp_path: Path) -> None:
    agent = _agent(
        tmp_path,
        model_name="qwen-token-plan-cn/qwen3-max",
        pi_provider="qwen-token-plan-cn",
        thinking_level=" high ",
        context_window="262144",
    )
    options = agent.interact_options
    assert options.pi_provider == "qwen-token-plan-cn"
    assert options.thinking_level == "high"
    assert options.context_window == 262144
    assert options.model == "qwen3-max"

    default = _agent(tmp_path, model_name="openai/gpt-test").interact_options
    assert default.pi_provider is None
    assert default.thinking_level is None
    assert default.context_window is None


def test_bridge_argv_forwards_pi_controls(tmp_path: Path) -> None:
    agent = _agent(
        tmp_path,
        pi_provider="qwen-token-plan-cn",
        thinking_level="max",
        context_window=262144,
    )
    argv = agent._bridge_argv(
        "plan", "/tmp/instruction.md", "/logs/agent/olympus-interact-step-01.jsonl"
    )
    assert argv[argv.index("--pi-provider") + 1] == "qwen-token-plan-cn"
    assert argv[argv.index("--thinking") + 1] == "max"
    assert argv[argv.index("--context-window") + 1] == "262144"


def test_bridge_forwards_pi_controls_and_writes_token_usage(tmp_path: Path) -> None:
    repo = _init_git_repo(tmp_path / "repo")
    state = tmp_path / "state"
    logs = tmp_path / "agent-logs"
    logs.mkdir()
    trajectory = logs / "olympus-interact-step-01.jsonl"
    fake = _write_fake_olympus(tmp_path)
    server = FakeMcpServer(
        replies=["AMB-1: raise ValueError('zero').", "Looks good, ship it."]
    )
    try:
        args = _bridge_common(
            "plan", fake, repo, state, server, trajectory, tmp_path / "plan.md"
        ) + [
            "--goal",
            "Build a calculator library",
            "--pi-provider",
            "qwen-token-plan-cn",
            "--thinking",
            "high",
            "--context-window",
            "262144",
        ]
        process, summary = _run_bridge(
            args, env={"FAKE_OLYMPUS_TOKEN_USAGE": json.dumps(_PI_USAGE_PAYLOAD)}
        )
        assert process.returncode == 0, process.stdout + process.stderr
        assert summary["status"] == "ok", summary

        expected = {
            "llm_calls": 2,
            "input_tokens": 800,
            "output_tokens": 100,
            "reasoning_tokens": 40,
            "cache_read_tokens": 300,
            "cache_write_tokens": 0,
            "total_tokens": 800 + 100 + 300,
            "cache_hit_rate": round(300 / (800 + 300), 4),
            "by_actor": {},
        }
        assert summary["token_usage"] == expected
        assert json.loads((logs / "token_usage.json").read_text()) == expected
        assert json.loads((logs / "token_usage-step-01.json").read_text()) == expected

        events = _jsonl_events(trajectory)
        assert any(event.get("event") == "session.token_usage" for event in events)
        envelope_events = [
            event for event in events if event.get("event") == "session.envelope"
        ]
        assert envelope_events[-1]["envelope"]["token_usage"] == _PI_USAGE_PAYLOAD

        invocations = [
            json.loads(line)
            for line in (state / "fake-invocations.jsonl").read_text().splitlines()
        ]
        session_argv = next(
            invocation["argv"]
            for invocation in invocations
            if invocation["argv"][0] == "session"
        )
        assert session_argv[session_argv.index("--pi-provider") + 1] == (
            "qwen-token-plan-cn"
        )
        assert session_argv[session_argv.index("--thinking") + 1] == "high"
        assert session_argv[session_argv.index("--context-window") + 1] == "262144"
    finally:
        server.close()


def test_populate_context_reads_token_usage_from_step_log(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "olympus-interact-step-01.jsonl").write_text(
        json.dumps({"event": "ask_user", "turn": 1})
        + "\n"
        + json.dumps(
            {
                "event": "session.envelope",
                "envelope": {
                    "status": "completed",
                    "run_id": "run-7",
                    "token_usage": _PI_USAGE_PAYLOAD,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    agent = _agent(tmp_path)
    context = _context()
    agent.populate_context_post_run(context)
    usage = context.metadata["token_usage"]
    assert usage["total_tokens"] == 1200
    assert context.n_input_tokens == 1100
    assert context.n_output_tokens == 100
    assert context.n_cache_tokens == 300


def test_populate_context_reads_token_usage_artifact_without_step_logs(
    tmp_path: Path,
) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "token_usage.json").write_text(json.dumps(_PI_USAGE_PAYLOAD), encoding="utf-8")
    agent = _agent(tmp_path)
    context = _context()
    agent.populate_context_post_run(context)
    assert context.metadata["token_usage"]["total_tokens"] == 1200
    assert context.n_cache_tokens == 300


def test_run_bridge_step_captures_summary_token_usage(tmp_path: Path) -> None:
    payload = json.dumps(
        {
            "status": "ok",
            "phase": "plan",
            "run_id": "run-9",
            "turns": 0,
            "token_usage": _PI_USAGE_PAYLOAD,
        }
    )
    environment = FakeEnvironment(
        handler=lambda command, cwd, env, timeout_sec: ExecResult(
            return_code=0, stdout=payload + "\n"
        )
    )
    agent = _agent(tmp_path)
    asyncio.run(agent.run(PLAN_INSTRUCTION, environment, _context()))
    assert agent._token_usage is not None
    assert agent._token_usage["total_tokens"] == 1200

    context = _context()
    agent.populate_context_post_run(context)
    assert context.metadata["token_usage"]["total_tokens"] == 1200
    assert context.n_input_tokens == 1100
    assert context.n_output_tokens == 100
    assert context.n_cache_tokens == 300


# --------------------------------------------------------------------------- invariants


def test_core_package_stays_free_of_harbor_and_fastmcp_imports() -> None:
    offenders: list[str] = []
    pattern = re.compile(r"^\s*(?:from|import)\s+(harbor|fastmcp)\b", re.MULTILINE)
    for path in (PROJECT_SRC / "olympus").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if pattern.search(text):
            offenders.append(str(path.relative_to(PROJECT_SRC)))
    assert offenders == []



def test_is_ship_reply_rejects_negated_ship_phrases() -> None:
    from tests.evals.benchmarks.interact_bridge import InteractBridge

    assert not InteractBridge._is_ship_reply("I don't think you should ship it.")
    assert not InteractBridge._is_ship_reply("We are not ready to ship this yet.")
    assert InteractBridge._is_ship_reply("Looks good, ship it!")


def test_interact_agent_declares_mcp_servers_and_uses_isolated_py313_venv(
    tmp_path: Path,
) -> None:
    assert OlympusInteractAgent.capabilities.base_url is True
    assert OlympusInteractAgent.capabilities.mcp_servers is True
    script = OlympusInteractAgent._pip_install_command("/tmp/olympus-src")
    assert "sys.version_info >= (3, 13)" in script
    assert "--python 3.13" in script
    assert "VENV_DIR=/opt/olympus/venv" in script
    assert "VENV_DIR=/tmp/olympus-venv" in script

    def _resolve_bin(command: str, *_args: Any) -> ExecResult:
        if "pip install" in command:
            return ExecResult(stdout="/tmp/olympus-venv/bin/olympus\n", stderr="", return_code=0)
        return ExecResult(stdout="", stderr="", return_code=0)

    env = FakeEnvironment(handler=_resolve_bin)
    agent = _agent(tmp_path)
    asyncio.run(agent.setup(env))
    assert agent._installed_olympus_bin == "/tmp/olympus-venv/bin/olympus"
    argv = agent._bridge_argv("plan", "/tmp/instr.md", "/logs/agent/step.jsonl")
    assert argv[argv.index("--olympus-bin") + 1] == "/tmp/olympus-venv/bin/olympus"


def test_interact_install_script_falls_back_to_curl_when_pip_fails(tmp_path: Path) -> None:
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
        OlympusInteractAgent._pip_install_command("olympus4swe")
        .replace("/opt/olympus", f"{tmp_path}/opt/olympus")
        .replace("/tmp/olympus", f"{tmp_path}/tmp/olympus")
    )
    env = {**os.environ, "PATH": str(bin_dir), "HOME": str(tmp_path), "CALLS": str(calls)}
    subprocess.run(["/bin/bash", "-c", script], capture_output=True, env=env, check=False)
    assert calls.exists()


def test_interact_install_pi_uploads_and_extracts_local_bundle(tmp_path: Path) -> None:
    bundle = tmp_path / "pi-bundle.tar.gz"
    bundle.write_bytes(b"fake-tarball")

    def _handler(command: str, *_args: Any) -> ExecResult:
        if "tar -xzf /tmp/olympus-pi-bundle.tar.gz" in command:
            return ExecResult(stdout="/opt/olympus/pi/bin/pi\n", stderr="", return_code=0)
        return ExecResult(stdout="", stderr="", return_code=0)

    env = FakeEnvironment(handler=_handler)
    agent = _agent(tmp_path, pi_bundle=str(bundle))
    asyncio.run(agent.setup(env))
    assert any(dst == "/tmp/olympus-pi-bundle.tar.gz" for _src, dst, _text in env.uploads)
    assert agent._installed_pi_bin == "/opt/olympus/pi/bin/pi"


def test_interact_gateway_env_falls_back_to_host_provider_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "sk-interact-host-secret-777"
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    agent = _agent(
        tmp_path,
        model_name="openai/gpt-4o",
        base_url="https://gateway.example/v1",
    )
    gw_env = agent._gateway_env()
    assert gw_env is not None
    assert gw_env["OPENAI_API_KEY"] == secret
    assert gw_env["OPENAI_BASE_URL"] == "https://gateway.example/v1"


def test_fastmcp_does_not_replay_non_expiry_rpc_error_mentioning_32600() -> None:
    from tests.evals.benchmarks.fastmcp_client import (
        McpHttpClient,
        McpRpcError,
        is_session_expiry_error,
    )

    err = McpRpcError(
        -32042,
        "Unhandled upstream exception",
        "MCP 'tools/list' failed: {'code': -32042, 'data': {'previous': -32600}}",
    )
    assert is_session_expiry_error(err) is False

    calls: list[str] = []

    class _FakeClient(McpHttpClient):
        def _post(self, message: dict[str, Any]) -> dict[str, Any] | None:
            method = str(message.get("method") or "")
            calls.append(method)
            if method == "initialize":
                self.session_id = "sess-1"
                return {"jsonrpc": "2.0", "id": message["id"], "result": {}}
            if method == "notifications/initialized":
                return None
            return {
                "jsonrpc": "2.0",
                "id": message["id"],
                "error": {
                    "code": -32042,
                    "message": "Unhandled upstream exception",
                    "data": {"previous": -32600},
                },
            }

    client = _FakeClient(url="http://localhost:8000/mcp")
    with pytest.raises(McpRpcError):
        client.list_tools()
    assert calls == ["initialize", "notifications/initialized", "tools/list"]


def test_interact_bridge_retries_pi_error_failed_timeout_and_subprocess_timeout(
    tmp_path: Path,
) -> None:
    from unittest.mock import Mock, patch

    from tests.evals.benchmarks.interact_bridge import (
        InteractBridge,
        build_parser,
        config_from_args,
    )

    repo = tmp_path / "repo"
    repo.mkdir()
    args = build_parser().parse_args(
        [
            "--phase",
            "plan",
            "--workspace",
            str(repo),
            "--state-dir",
            str(tmp_path / "state"),
            "--plan-out",
            str(tmp_path / "plan.md"),
            "--goal",
            "Build feature",
            "--infra-retry-backoff",
            "0",
        ]
    )
    cfg = config_from_args(args)
    client = Mock()
    cases = [
        subprocess.CompletedProcess(
            [],
            0,
            json.dumps(
                {
                    "status": "needs_input",
                    "pause_kind": "blocked",
                    "message": "pi driver stopped before returning TaskPlan: pi_error",
                    "run_id": "run-1",
                }
            ),
            "",
        ),
        subprocess.CompletedProcess(
            [],
            1,
            json.dumps(
                {
                    "status": "failed",
                    "error": "provider_error: read timed out",
                    "run_id": "run-1",
                }
            ),
            "",
        ),
        subprocess.TimeoutExpired([], 1),
    ]
    for first in cases:
        client.reset_mock()
        bridge = InteractBridge(cfg, client=client, environ={})
        with patch(
            "tests.evals.benchmarks.interact_bridge.subprocess.run",
            side_effect=[
                first,
                subprocess.CompletedProcess([], 0, '{"status":"completed"}', ""),
            ],
        ) as run_mock:
            bridge.run()
        assert run_mock.call_count == 2
        assert client.ask_user.call_count == 0


def test_interact_populate_context_reports_step_deltas_and_sums_distinct_runs(
    tmp_path: Path,
) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    agent = OlympusInteractAgent(logs_dir=logs)

    # Step 1 (plan, run-1): cumulative 100 in / 20 out / 40 cache_r
    (logs / "olympus-interact-step-01.jsonl").write_text(
        json.dumps(
            {
                "event": "session.envelope",
                "envelope": {
                    "run_id": "run-1",
                    "status": "completed",
                    "token_usage": {
                        "llm_calls": 1,
                        "input_tokens": 100,
                        "output_tokens": 20,
                        "cache_read_tokens": 40,
                        "cache_write_tokens": 0,
                    },
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    agent._last_step_role = "plan"
    ctx1 = SimpleNamespace(metadata=None)
    agent.populate_context_post_run(ctx1)
    assert (ctx1.n_input_tokens, ctx1.n_output_tokens, ctx1.n_cache_tokens) == (140, 20, 40)

    # Step 2 (execute, run-1): cumulative 160 in / 35 out / 100 cache_r -> delta 120 in, 15 out
    (logs / "olympus-interact-step-02.jsonl").write_text(
        json.dumps(
            {
                "event": "session.envelope",
                "envelope": {
                    "run_id": "run-1",
                    "status": "completed",
                    "token_usage": {
                        "llm_calls": 2,
                        "input_tokens": 160,
                        "output_tokens": 35,
                        "cache_read_tokens": 100,
                        "cache_write_tokens": 0,
                    },
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    agent._last_step_role = "execute"
    ctx2 = SimpleNamespace(metadata=None)
    agent.populate_context_post_run(ctx2)
    assert (ctx2.n_input_tokens, ctx2.n_output_tokens, ctx2.n_cache_tokens) == (120, 15, 60)

    # Step 3 (handoff): no LLM calls -> Harbor context token fields must be explicitly 0
    agent._last_step_role = "handoff"
    ctx3 = SimpleNamespace(metadata=None)
    agent.populate_context_post_run(ctx3)
    assert (ctx3.n_input_tokens, ctx3.n_output_tokens, ctx3.n_cache_tokens) == (0, 0, 0)

    # Step 4 (second distinct run-2): 50 in / 10 out -> delta 50/10, trial total sums run-1 + run-2
    (logs / "olympus-interact-step-04.jsonl").write_text(
        json.dumps(
            {
                "event": "session.envelope",
                "envelope": {
                    "run_id": "run-2",
                    "status": "completed",
                    "token_usage": {
                        "llm_calls": 1,
                        "input_tokens": 50,
                        "output_tokens": 10,
                        "cache_read_tokens": 0,
                        "cache_write_tokens": 0,
                    },
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    agent._last_step_role = "plan"
    ctx4 = SimpleNamespace(metadata=None)
    agent.populate_context_post_run(ctx4)
    assert (ctx4.n_input_tokens, ctx4.n_output_tokens, ctx4.n_cache_tokens) == (50, 10, 0)
    # Harbor summing step contexts equals true total across both runs (140+120+50 = 310)
    assert ctx1.n_input_tokens + ctx2.n_input_tokens + ctx4.n_input_tokens == 310
    assert ctx4.metadata["token_usage"]["input_tokens"] == 210
    assert ctx4.metadata["token_usage"]["cache_read_tokens"] == 100
    assert ctx4.metadata["token_usage"]["output_tokens"] == 45
    assert ctx4.metadata["token_usage"]["total_tokens"] == 355
