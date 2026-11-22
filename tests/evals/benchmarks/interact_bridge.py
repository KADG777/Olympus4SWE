"""In-container bridge between ``olympus session`` and SWE-Interact's user server.

The script is stdlib-only so it can be uploaded into a task container (and also
run directly on the host during offline tests). It drives the
``olympus session start/reply`` CLI, forwards every ``needs_input`` pause to the
task's FastMCP ``ask_user`` tool over streamable HTTP, feeds the maintainer's
reply back into the session, and appends a redacted JSONL trajectory.

Exits codes are part of the adapter contract:
``0`` success, ``1`` session failed, ``2`` user-server/MCP failure, ``3`` the
per-step interaction cap was reached, ``4`` invalid bridge configuration.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:  # Imported as ``tests.evals.benchmarks.interact_bridge`` (host tests).
    from .fastmcp_client import McpError, McpHttpClient
except ImportError:  # Executed as a standalone script inside the task container.
    from fastmcp_client import McpError, McpHttpClient

EXIT_OK = 0
EXIT_SESSION_FAILED = 1
EXIT_SIDECAR_ERROR = 2
EXIT_TURN_CAP = 3
EXIT_CONFIG_ERROR = 4

DEFAULT_INITIAL_QUESTION = (
    "Hi! I'm starting work in this repository. What change would you like me to make? "
    "Please describe the requirements and any constraints you care about."
)
PLAN_REVIEW_PREFIX = "Please review my implementation plan:"
_SECRET_NAME = re.compile(
    r"(API_KEY|APIKEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|PRIVATE_KEY)", re.IGNORECASE
)
_MAX_LOGGED_CHARS = 20_000


def redact_text(text: str, environ: dict[str, str] | None = None) -> str:
    """Replace values of secret-looking environment variables inside ``text``."""
    source = os.environ if environ is None else environ
    secrets = sorted(
        {
            value
            for name, value in source.items()
            if value and len(value) >= 8 and _SECRET_NAME.search(name)
        },
        key=len,
        reverse=True,
    )
    redacted = text
    for value in secrets:
        redacted = redacted.replace(value, "***")
    return redacted


def _redact_value(value: Any, environ: dict[str, str]) -> Any:
    if isinstance(value, str):
        return redact_text(value, environ)
    if isinstance(value, dict):
        return {key: _redact_value(item, environ) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact_value(item, environ) for item in value]
    return value


def append_jsonl(
    path: Path | None, record: dict[str, Any], environ: dict[str, str]
) -> None:
    """Append one redacted JSONL record; logging failures must not break a run."""
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"timestamp": time.time(), **_redact_value(record, environ)}
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")
    except OSError:
        pass


class BridgeError(RuntimeError):
    """A bridge-level failure carrying the process exit code to report."""

    def __init__(self, message: str, exit_code: int = EXIT_SESSION_FAILED) -> None:
        super().__init__(message)
        self.exit_code = exit_code


@dataclass(frozen=True)
class BridgeConfig:
    phase: str
    olympus_bin: str
    workspace: str
    state_dir: Path
    run_id_file: Path
    mcp_url: str
    ask_user_tool: str
    max_turns: int
    trajectory: Path | None
    workflow: str
    gate_policy: str
    provider: str | None
    model: str | None
    base_url: str | None
    worker_driver: str | None
    reviewer_driver: str | None
    monitor_model: str | None
    goal: str
    goal_file: str | None
    initial_question: str | None
    reply: str
    reply_file: str | None
    plan_out: Path | None
    session_timeout: float
    pi_bin: str | None


class InteractBridge:
    """Drive one SWE-Interact step through the headless session CLI."""

    def __init__(
        self,
        config: BridgeConfig,
        client: McpHttpClient,
        environ: dict[str, str] | None = None,
    ) -> None:
        self.config = config
        self.client = client
        self.environ = dict(os.environ if environ is None else environ)
        self.turns = 0
        self.dialogue: list[dict[str, str]] = []
        self.run_id = ""

    # ------------------------------------------------------------------ entry point

    def run(self) -> None:
        if self.config.phase == "plan":
            self._run_plan()
        elif self.config.phase == "execute":
            self._run_execute()
        else:
            raise BridgeError(
                f"unsupported phase {self.config.phase!r}", EXIT_CONFIG_ERROR
            )

    # ------------------------------------------------------------------ plan phase

    def _run_plan(self) -> None:
        goal = self._initial_goal()
        self.log("plan.goal", goal=goal)
        envelope = self.invoke("start", goal, step_mode="plan_only")
        while True:
            status = envelope.get("status")
            self._record_envelope(envelope)
            if status == "completed":
                self._write_plan(goal, str(envelope.get("message") or ""))
                return
            if status != "needs_input":
                detail = envelope.get("error") or envelope.get("message")
                raise BridgeError(
                    f"session failed in plan phase: {detail}",
                    EXIT_SESSION_FAILED,
                )
            pause_kind = str(envelope.get("pause_kind") or "")
            plan_message = str(envelope.get("message") or "")
            answer = self.ask_user("plan", envelope)
            next_envelope = self.invoke("reply", answer, step_mode="plan_only")
            if pause_kind == "plan_handoff" and self._is_plan_approval(
                next_envelope, plan_message
            ):
                self._write_plan(goal, plan_message)
                return
            envelope = next_envelope

    @staticmethod
    def _is_plan_approval(envelope: dict[str, Any], plan_message: str) -> bool:
        """An approval reply re-emits the same handed-off plan in plan-only mode."""
        if envelope.get("status") == "completed":
            return True
        return (
            envelope.get("pause_kind") == "plan_handoff"
            and str(envelope.get("message") or "") == plan_message
        )

    def _initial_goal(self) -> str:
        goal = self.config.goal.strip()
        if not goal and self.config.goal_file:
            goal = Path(self.config.goal_file).read_text(encoding="utf-8").strip()
        if goal:
            return goal
        question = (self.config.initial_question or "").strip()
        if not question:
            raise BridgeError(
                "plan phase requires --goal/--goal-file or --initial-question",
                EXIT_CONFIG_ERROR,
            )
        envelope = {"pause_kind": "initial_goal", "message": question}
        reply = self.ask_user("plan", envelope).strip()
        if not reply:
            raise BridgeError("the maintainer returned an empty initial goal")
        return reply

    def _write_plan(self, goal: str, plan_message: str) -> None:
        path = self.config.plan_out
        if path is None:
            return
        lines = [
            "# Agreed Implementation Plan",
            "",
            "## Goal",
            goal.strip(),
            "",
            "## Maintainer Dialogue",
        ]
        for turn in self.dialogue:
            lines += [
                "",
                "### Question",
                turn["question"],
                "",
                "### Reply",
                turn["reply"],
            ]
        if plan_message:
            lines += ["", "## Approved Task Plan", plan_message]
        content = redact_text("\n".join(lines).rstrip() + "\n", self.environ)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        self.log("plan.saved", path=str(path), chars=len(content))

    # ------------------------------------------------------------------ execute phase

    def _run_execute(self) -> None:
        run_id = self._load_run_id()
        if not run_id:
            raise BridgeError(
                f"execute phase requires a persisted run id at {self.config.run_id_file}",
                EXIT_CONFIG_ERROR,
            )
        self.run_id = run_id
        reply = self._reply_text()
        pause_kind = self.last_pause_kind()
        first_message = "approve" if pause_kind == "plan_handoff" else reply
        if pause_kind == "plan_handoff":
            self.log("execute.approve_plan", run_id=run_id)
        envelope = self.invoke("reply", first_message, step_mode="execute")
        while True:
            status = envelope.get("status")
            self._record_envelope(envelope)
            if status == "completed":
                return
            if status != "needs_input":
                raise BridgeError(
                    "session failed in execute phase: "
                    f"{envelope.get('error') or envelope.get('message')}",
                    EXIT_SESSION_FAILED,
                )
            answer = self.ask_user("execute", envelope)
            envelope = self.invoke("reply", answer, step_mode="execute")

    def _reply_text(self) -> str:
        text = self.config.reply.strip()
        if not text and self.config.reply_file:
            text = Path(self.config.reply_file).read_text(encoding="utf-8").strip()
        if not text:
            raise BridgeError(
                "execute phase requires --reply or --reply-file", EXIT_CONFIG_ERROR
            )
        return text

    def last_pause_kind(self) -> str | None:
        """Read the live run's most recent pause kind, falling back to saved state."""
        argv = [
            self.config.olympus_bin,
            "events",
            self.run_id,
            "--db",
            str(self.config.state_dir / "olympus.db"),
        ]
        try:
            proc = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=60,
                env=self.environ,
                check=False,
            )
            events = json.loads(proc.stdout or "[]")
        except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
            return self._saved_pause_kind()
        if not isinstance(events, list):
            return self._saved_pause_kind()
        pause_kind: str | None = None
        for event in events:
            if not isinstance(event, dict):
                continue
            event_type = event.get("event_type")
            if event_type == "session.paused":
                payload = event.get("payload") or {}
                raw = payload.get("pause_kind") if isinstance(payload, dict) else None
                pause_kind = str(raw) if raw else None
            elif event_type in ("session.completed", "session.failed"):
                pause_kind = None
        return pause_kind

    # ------------------------------------------------------------------ session IO

    def invoke(self, subcommand: str, message: str, *, step_mode: str) -> dict[str, Any]:
        argv = self.session_argv(subcommand, message, step_mode)
        env = dict(self.environ)
        if self.config.pi_bin:
            env.setdefault("OLYMPUS_PI_BIN", self.config.pi_bin)
        try:
            proc = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=self.config.session_timeout,
                env=env,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            self.log("session.timeout", subcommand=subcommand, step_mode=step_mode)
            raise BridgeError(
                f"olympus session {subcommand} timed out after {self.config.session_timeout}s"
            ) from exc
        except OSError as exc:
            raise BridgeError(
                f"could not launch olympus ({self.config.olympus_bin!r}): {exc}",
                EXIT_CONFIG_ERROR,
            ) from exc
        self.log(
            "session.invoke",
            subcommand=subcommand,
            step_mode=step_mode,
            return_code=proc.returncode,
            message=message,
            stderr=(proc.stderr or "")[-_MAX_LOGGED_CHARS:],
        )
        envelope = self._parse_envelope(proc)
        run_id = str(envelope.get("run_id") or "")
        if run_id:
            self.run_id = run_id
            self._save_run_id()
        self.log("session.envelope", envelope=envelope)
        return envelope

    def session_argv(self, subcommand: str, message: str, step_mode: str) -> list[str]:
        argv = [self.config.olympus_bin, "session", subcommand]
        if subcommand == "reply":
            argv.append(self.run_id)
        argv.append(f"--message={message}")
        argv += [
            "--workspace",
            self.config.workspace,
            "--state-dir",
            str(self.config.state_dir),
            "--workflow",
            self.config.workflow,
            "--gate-policy",
            self.config.gate_policy,
            "--step-mode",
            step_mode,
        ]
        for flag, value in (
            ("--provider", self.config.provider),
            ("--model", self.config.model),
            ("--base-url", self.config.base_url),
            ("--worker-driver", self.config.worker_driver),
            ("--reviewer-driver", self.config.reviewer_driver),
            ("--monitor-model", self.config.monitor_model),
        ):
            if value:
                argv += [flag, value]
        return argv

    @staticmethod
    def _parse_envelope(proc: subprocess.CompletedProcess[str]) -> dict[str, Any]:
        stdout = (proc.stdout or "").strip()
        if not stdout:
            raise BridgeError(
                f"olympus session produced no output (rc={proc.returncode}): "
                f"{(proc.stderr or '')[-2000:]}"
            )
        payload: Any = None
        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError:
            start = stdout.find("{")
            if start >= 0:
                try:
                    payload = json.loads(stdout[start:])
                except json.JSONDecodeError:
                    payload = None
        if not isinstance(payload, dict) or "status" not in payload:
            raise BridgeError(
                f"olympus session emitted an unparseable envelope (rc={proc.returncode}): "
                f"{stdout[-2000:]}"
            )
        return payload

    # ------------------------------------------------------------------ user server

    def ask_user(self, phase: str, envelope: dict[str, Any]) -> str:
        self.turns += 1
        if self.turns > self.config.max_turns:
            self.log(
                "turn_cap_exceeded",
                phase=phase,
                turns=self.turns,
                max_turns=self.config.max_turns,
            )
            raise BridgeError(
                f"interaction cap reached ({self.config.max_turns} ask_user turns) "
                f"in {phase} phase",
                EXIT_TURN_CAP,
            )
        question = self._question_for(envelope)
        try:
            reply = self.client.ask_user(question, tool=self.config.ask_user_tool)
        except McpError as exc:
            self.log("ask_user.error", phase=phase, turn=self.turns, error=str(exc))
            raise BridgeError(f"ask_user sidecar failed: {exc}", EXIT_SIDECAR_ERROR) from exc
        self.dialogue.append({"question": question, "reply": reply})
        self.log(
            "ask_user",
            phase=phase,
            turn=self.turns,
            pause_kind=envelope.get("pause_kind"),
            question=question,
            reply=reply,
        )
        return reply

    @staticmethod
    def _question_for(envelope: dict[str, Any]) -> str:
        message = str(envelope.get("message") or "").strip()
        if envelope.get("pause_kind") == "plan_handoff":
            return f"{PLAN_REVIEW_PREFIX}\n\n{message}" if message else PLAN_REVIEW_PREFIX
        return message or "The session needs your input to continue."

    # ------------------------------------------------------------------ persistence

    def _run_id_path(self) -> Path:
        return self.config.run_id_file

    def _save_run_id(self) -> None:
        if not self.run_id:
            return
        try:
            path = self._run_id_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(self.run_id + "\n", encoding="utf-8")
        except OSError:
            pass

    def _load_run_id(self) -> str:
        try:
            return self._run_id_path().read_text(encoding="utf-8").strip()
        except OSError:
            return ""

    def _state_path(self) -> Path:
        return self.config.state_dir / "bridge-state.json"

    def _record_envelope(self, envelope: dict[str, Any]) -> None:
        try:
            self.config.state_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "run_id": self.run_id,
                "phase": self.config.phase,
                "status": envelope.get("status"),
                "pause_kind": envelope.get("pause_kind"),
            }
            self._state_path().write_text(
                json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8"
            )
        except OSError:
            pass

    def _saved_pause_kind(self) -> str | None:
        try:
            payload = json.loads(self._state_path().read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        raw = payload.get("pause_kind") if isinstance(payload, dict) else None
        return str(raw) if raw else None

    # ------------------------------------------------------------------ logging

    def log(self, event: str, **fields: Any) -> None:
        append_jsonl(self.config.trajectory, {"event": event, **fields}, self.environ)


# --------------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bridge olympus session pauses to a FastMCP ask_user sidecar."
    )
    parser.add_argument("--phase", required=True, choices=("plan", "execute"))
    parser.add_argument("--olympus-bin", default="olympus")
    parser.add_argument("--workspace", default="/app")
    parser.add_argument("--state-dir", default="/tmp/olympus-session")
    parser.add_argument("--run-id-file", default=None)
    parser.add_argument("--mcp-url", default="http://user-server:8000/mcp")
    parser.add_argument("--ask-user-tool", default="ask_user")
    parser.add_argument("--mcp-timeout", type=float, default=120.0)
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--trajectory", default=None)
    parser.add_argument("--workflow", default="full_sdlc_tdd")
    parser.add_argument("--gate-policy", default="interactive_minimal")
    parser.add_argument("--provider", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--worker-driver", default=None)
    parser.add_argument("--reviewer-driver", default=None)
    parser.add_argument("--monitor-model", default=None)
    parser.add_argument("--goal", default="")
    parser.add_argument("--goal-file", default=None)
    parser.add_argument("--initial-question", default=DEFAULT_INITIAL_QUESTION)
    parser.add_argument("--reply", default="")
    parser.add_argument("--reply-file", default=None)
    parser.add_argument("--plan-out", default="/tmp/plan.md")
    parser.add_argument("--session-timeout", type=float, default=3600.0)
    parser.add_argument("--pi-bin", default=None)
    return parser


def config_from_args(args: argparse.Namespace) -> BridgeConfig:
    state_dir = Path(args.state_dir).expanduser()
    run_id_file = (
        Path(args.run_id_file).expanduser()
        if args.run_id_file
        else state_dir / "run-id.txt"
    )
    return BridgeConfig(
        phase=args.phase,
        olympus_bin=args.olympus_bin,
        workspace=args.workspace,
        state_dir=state_dir,
        run_id_file=run_id_file,
        mcp_url=args.mcp_url,
        ask_user_tool=args.ask_user_tool,
        max_turns=args.max_turns,
        trajectory=Path(args.trajectory).expanduser() if args.trajectory else None,
        workflow=args.workflow,
        gate_policy=args.gate_policy,
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
        worker_driver=args.worker_driver,
        reviewer_driver=args.reviewer_driver,
        monitor_model=args.monitor_model,
        goal=args.goal,
        goal_file=args.goal_file,
        initial_question=args.initial_question,
        reply=args.reply,
        reply_file=args.reply_file,
        plan_out=Path(args.plan_out).expanduser() if args.plan_out else None,
        session_timeout=args.session_timeout,
        pi_bin=args.pi_bin,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = config_from_args(args)
    client = McpHttpClient(url=config.mcp_url, timeout=args.mcp_timeout)
    bridge = InteractBridge(config, client)
    exit_code = EXIT_OK
    error: str | None = None
    try:
        client.initialize()
        bridge.run()
    except BridgeError as exc:
        exit_code = exc.exit_code
        error = str(exc)
    except McpError as exc:
        exit_code = EXIT_SIDECAR_ERROR
        error = f"MCP failure: {exc}"
    except Exception as exc:  # noqa: BLE001 - always emit a structured summary
        exit_code = EXIT_SESSION_FAILED
        error = f"{type(exc).__name__}: {exc}"
    finally:
        client.close()
        if error:
            bridge.log("bridge.failed", exit_code=exit_code, error=error)
        summary = {
            "status": "ok" if exit_code == EXIT_OK else "error",
            "phase": config.phase,
            "run_id": bridge.run_id,
            "turns": bridge.turns,
            "exit_code": exit_code,
            "error": redact_text(error, bridge.environ) if error else None,
        }
        print(json.dumps(summary))
        if error:
            print(redact_text(error, bridge.environ), file=sys.stderr)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "EXIT_CONFIG_ERROR",
    "EXIT_OK",
    "EXIT_SESSION_FAILED",
    "EXIT_SIDECAR_ERROR",
    "EXIT_TURN_CAP",
    "BridgeConfig",
    "BridgeError",
    "InteractBridge",
    "append_jsonl",
    "build_parser",
    "config_from_args",
    "main",
    "redact_text",
]
