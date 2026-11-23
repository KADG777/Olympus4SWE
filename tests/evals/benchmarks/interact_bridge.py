"""In-container bridge between ``olympus session`` and SWE-Interact's user server.

The script is stdlib-only so it can be uploaded into a task container (and also
run directly on the host during offline tests). It drives the
``olympus session start/reply`` CLI, forwards every ``needs_input`` pause to the
task's FastMCP ``ask_user`` tool over streamable HTTP, feeds the maintainer's
reply back into the session, and appends a redacted JSONL trajectory.

In the execute phase it also honours the upstream maintainer contract: configure
a local git identity, record the starting commit, commit every coherent pass,
and ask the maintainer to review the committed state before finishing or
addressing revision feedback.

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

DEFAULT_INFRA_RETRIES = 2
DEFAULT_INFRA_RETRY_BACKOFF_SEC = 0.5

DEFAULT_INITIAL_QUESTION = (
    "Hi! I'm starting work in this repository. What change would you like me to make? "
    "Please describe the requirements and any constraints you care about."
)
PLAN_REVIEW_PREFIX = "Please review my implementation plan:"
_SECRET_NAME = re.compile(
    r"(API_KEY|APIKEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|PRIVATE_KEY)", re.IGNORECASE
)
_MAX_LOGGED_CHARS = 20_000
_DEFAULT_GIT_IDENTITY = (("user.name", "Dev"), ("user.email", "dev@example.com"))
_TEST_STEP_MARKERS = (
    "add tests",
    "write tests",
    "test changes in a separate commit",
)
_SHIP_MARKERS = (
    "ship it",
    "ship this",
    "looks good",
    "lgtm",
    "approved",
    "approve",
    "go ahead",
    "proceed",
    "可以",
    "同意",
    "批准",
    "通过",
    "没问题",
    "继续",
)
_REVISION_MARKERS = (
    "change",
    "revise",
    "rework",
    "fix",
    "update",
    "instead",
    "reject",
    "not right",
    "wrong",
    "missing",
    "redo",
    "rename",
    "remove",
    "move ",
    "ensure",
    "make sure",
    "修改",
    "调整",
    "重做",
    "不对",
    "改一下",
)
_NEGATED_SHIP_RE = re.compile(
    r"\b(?:don'?t|do\s+not|doesn'?t|didn'?t|not|never|cannot|can'?t|won'?t|will\s+not|"
    r"wouldn'?t|shouldn'?t|should\s+not|hold\s+off)\b",
    re.IGNORECASE,
)
_NEGATED_SHIP_MARKERS = (
    "can't approve",
    "cannot approve",
    "can not approve",
    "won't approve",
    "don't approve",
    "do not approve",
    "not approved",
    "can't ship",
    "cannot ship",
    "don't ship",
    "do not ship",
    "not ship",
    "do not proceed",
    "don't proceed",
    "拒绝",
)
_INFRA_ERROR_RE = re.compile(
    r"provider_error"
    r"|timed\s*out"
    r"|timeout"
    r"|connection\s*(?:error|refused|reset|aborted)"
    r"|rate\s*limit"
    r"|http\s+5\d{2}"
    r"|http\s+5xx"
    r"|\b50[234]\b",
    re.IGNORECASE,
)


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


def is_infrastructure_block(envelope: dict[str, Any]) -> bool:
    """Return True when a session pause is a transient provider/infra failure.

    Such pauses must never be forwarded to the maintainer through ``ask_user``:
    a human cannot answer a provider timeout, so the bridge retries locally and
    fails cleanly if the infrastructure stays down.
    """
    if envelope.get("status") != "needs_input":
        return False
    pause_kind = str(envelope.get("pause_kind") or "").strip().lower()
    if pause_kind not in ("", "blocked"):
        return False
    detail = " ".join(
        str(envelope.get(field) or "") for field in ("message", "error")
    )
    return bool(_INFRA_ERROR_RE.search(detail))


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
    stage_driver: str | None
    monitor_model: str | None
    goal: str
    goal_file: str | None
    initial_question: str | None
    reply: str
    reply_file: str | None
    plan_out: Path | None
    session_timeout: float
    infra_retries: int
    infra_retry_backoff: float
    pi_bin: str | None
    base_commit_file: Path


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

    def run(self) -> None:
        if self.config.phase == "plan":
            self._run_plan()
        elif self.config.phase == "execute":
            self._run_execute()
        else:
            raise BridgeError(
                f"unsupported phase {self.config.phase!r}", EXIT_CONFIG_ERROR
            )

    def _run_plan(self) -> None:
        goal = self._initial_goal()
        self.log("plan.goal", goal=goal)
        envelope = self.invoke_with_infra_retry("start", goal, step_mode="plan_only")
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
            next_envelope = self.invoke_with_infra_retry(
                "reply", answer, step_mode="plan_only"
            )
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

    def _run_execute(self) -> None:
        run_id = self._load_run_id()
        if not run_id:
            raise BridgeError(
                f"execute phase requires a persisted run id at {self.config.run_id_file}",
                EXIT_CONFIG_ERROR,
            )
        self.run_id = run_id
        instruction = self._reply_text()
        self._prepare_git_workspace()
        if self.last_pause_kind() == "plan_handoff":
            self.log("execute.continue_after_plan", run_id=run_id)
        envelope = self.invoke_with_infra_retry("reply", instruction, step_mode="execute")
        test_step = self._is_test_step(instruction)
        commit_message = "add tests" if test_step else "initial implementation"
        while True:
            status = envelope.get("status")
            self._record_envelope(envelope)
            if status == "completed":
                commit = self._commit_pass(commit_message)
                if test_step:
                    return
                feedback = self.ask_user(
                    "execute_review", {"message": self._review_message(commit)}
                )
                if self._is_ship_reply(feedback):
                    self.log("execute.review_approved", run_id=run_id, turns=self.turns)
                    return
                commit_message = "address feedback"
                envelope = self.invoke_with_infra_retry(
                    "reply", feedback, step_mode="execute"
                )
                continue
            if status != "needs_input":
                raise BridgeError(
                    "session failed in execute phase: "
                    f"{envelope.get('error') or envelope.get('message')}",
                    EXIT_SESSION_FAILED,
                )
            answer = self.ask_user("execute", envelope)
            envelope = self.invoke_with_infra_retry(
                "reply", answer, step_mode="execute"
            )

    @staticmethod
    def _is_test_step(instruction: str) -> bool:
        lowered = instruction.casefold()
        return any(marker in lowered for marker in _TEST_STEP_MARKERS)

    @staticmethod
    def _is_ship_reply(reply: str) -> bool:
        lowered = reply.casefold()
        if not lowered.strip():
            return False
        if any(marker in lowered for marker in _NEGATED_SHIP_MARKERS):
            return False
        if _NEGATED_SHIP_RE.search(lowered) is not None:
            return False
        if any(marker in lowered for marker in _REVISION_MARKERS):
            return False
        return any(marker in lowered for marker in _SHIP_MARKERS)

    def _prepare_git_workspace(self) -> None:
        for key, value in _DEFAULT_GIT_IDENTITY:
            current = self._git_probe("config", "--get", key, exit_code=EXIT_CONFIG_ERROR)
            if current.returncode == 0 and current.stdout.strip():
                continue
            self._git("config", key, value, exit_code=EXIT_CONFIG_ERROR)
            self.log("execute.git_identity", key=key, value=value)
        path = self.config.base_commit_file
        try:
            existing = path.read_text(encoding="utf-8").strip()
        except OSError:
            existing = ""
        if existing:
            self.log("execute.base_commit", path=str(path), commit=existing, status="kept")
            return
        head = self._git("rev-parse", "HEAD", exit_code=EXIT_CONFIG_ERROR).strip()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(head + "\n", encoding="utf-8")
        except OSError as exc:
            raise BridgeError(
                f"could not save base commit to {path}: {exc}", EXIT_SESSION_FAILED
            ) from exc
        self.log("execute.base_commit", path=str(path), commit=head, status="created")

    def _commit_pass(self, message: str) -> dict[str, Any] | None:
        self._git("add", "-A")
        staged = self._git_probe("diff", "--cached", "--quiet")
        if staged.returncode == 0:
            self.log("execute.commit", status="no_changes")
            return None
        if staged.returncode != 1:
            detail = (staged.stderr or staged.stdout or "").strip()
            raise BridgeError(
                f"git diff --cached failed ({staged.returncode}): {detail}",
                EXIT_SESSION_FAILED,
            )
        self._git("commit", "-m", message)
        commit = self._git("rev-parse", "--short", "HEAD").strip()
        subject = self._git("log", "-1", "--pretty=%s").strip()
        summary = self._git("show", "--stat", "--oneline", "--no-renames", "HEAD").strip()
        self.log("execute.commit", status="committed", commit=commit, subject=subject)
        return {"commit": commit, "subject": subject, "summary": summary}

    def _review_message(self, commit: dict[str, Any] | None) -> str:
        lines = ["I finished this implementation pass."]
        if commit is None:
            lines.append("This pass produced no new working-tree changes.")
        else:
            lines.append(f"Latest commit: {commit['commit']} {commit['subject']}")
        try:
            base = self.config.base_commit_file.read_text(encoding="utf-8").strip()
        except OSError:
            base = ""
        log = self._git_probe(
            "log", "--oneline", "--max-count=20", f"{base}..HEAD" if base else "HEAD"
        )
        if log.returncode == 0 and log.stdout.strip():
            lines += ["", "Commits since the base commit:", log.stdout.strip()]
        if commit is not None and commit["summary"]:
            lines += ["", "Key changes:", commit["summary"][:4000]]
        lines += [
            "",
            "Please review the committed repository state. If it looks good, say "
            "'ship it'; otherwise describe the changes you want.",
        ]
        return "\n".join(lines)

    def _git_probe(
        self, *args: str, exit_code: int = EXIT_SESSION_FAILED
    ) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                ["git", *args],
                cwd=self.config.workspace,
                capture_output=True,
                text=True,
                timeout=120,
                env=self.environ,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise BridgeError(
                f"could not run git {' '.join(args)}: {exc}", exit_code
            ) from exc

    def _git(self, *args: str, exit_code: int = EXIT_SESSION_FAILED) -> str:
        proc = self._git_probe(*args, exit_code=exit_code)
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip()
            raise BridgeError(
                f"git {' '.join(args)} failed ({proc.returncode}): {detail}",
                exit_code,
            )
        return proc.stdout

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

    def invoke_with_infra_retry(
        self, subcommand: str, message: str, *, step_mode: str
    ) -> dict[str, Any]:
        """Invoke the session, retrying transient infrastructure blocks locally."""
        envelope = self.invoke(subcommand, message, step_mode=step_mode)
        attempt = 0
        while (
            is_infrastructure_block(envelope)
            and attempt < self.config.infra_retries
        ):
            attempt += 1
            self.log(
                "session.infra_retry",
                subcommand=subcommand,
                step_mode=step_mode,
                attempt=attempt,
                max_retries=self.config.infra_retries,
                detail=str(envelope.get("error") or envelope.get("message") or ""),
            )
            if self.config.infra_retry_backoff > 0:
                time.sleep(self.config.infra_retry_backoff)
            envelope = self.invoke(subcommand, message, step_mode=step_mode)
        if is_infrastructure_block(envelope):
            detail = str(
                envelope.get("error")
                or envelope.get("message")
                or "unknown infrastructure failure"
            )
            raise BridgeError(
                f"session {subcommand} stopped on an infrastructure error after "
                f"{attempt + 1} attempt(s): {detail}",
                EXIT_SESSION_FAILED,
            )
        return envelope

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
            ("--stage-driver", self.config.stage_driver),
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

    def _run_id_path(self) -> Path:
        return self.config.run_id_file

    def _save_run_id(self) -> None:
        if not self.run_id:
            return
        path = self._run_id_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(self.run_id + "\n", encoding="utf-8")
        except OSError as exc:
            raise BridgeError(
                f"could not persist run id to {path}: {exc}", EXIT_SESSION_FAILED
            ) from exc

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

    def log(self, event: str, **fields: Any) -> None:
        append_jsonl(self.config.trajectory, {"event": event, **fields}, self.environ)


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
    parser.add_argument("--stage-driver", default=None)
    parser.add_argument("--worker-driver", default=None)
    parser.add_argument("--reviewer-driver", default=None)
    parser.add_argument("--monitor-model", default=None)
    parser.add_argument("--infra-retries", type=int, default=DEFAULT_INFRA_RETRIES)
    parser.add_argument(
        "--infra-retry-backoff",
        type=float,
        default=DEFAULT_INFRA_RETRY_BACKOFF_SEC,
        help="Seconds slept between local retries of transient session blocks.",
    )
    parser.add_argument("--goal", default="")
    parser.add_argument("--goal-file", default=None)
    parser.add_argument("--initial-question", default=DEFAULT_INITIAL_QUESTION)
    parser.add_argument("--reply", default="")
    parser.add_argument("--reply-file", default=None)
    parser.add_argument("--plan-out", default="/tmp/plan.md")
    parser.add_argument("--base-commit-file", default="/tmp/base_commit.txt")
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
        stage_driver=args.stage_driver,
        monitor_model=args.monitor_model,
        goal=args.goal,
        goal_file=args.goal_file,
        initial_question=args.initial_question,
        reply=args.reply,
        reply_file=args.reply_file,
        plan_out=Path(args.plan_out).expanduser() if args.plan_out else None,
        session_timeout=args.session_timeout,
        infra_retries=max(0, args.infra_retries),
        infra_retry_backoff=max(0.0, args.infra_retry_backoff),
        pi_bin=args.pi_bin,
        base_commit_file=Path(args.base_commit_file).expanduser(),
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
    "DEFAULT_INFRA_RETRIES",
    "DEFAULT_INFRA_RETRY_BACKOFF_SEC",
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
    "is_infrastructure_block",
    "main",
    "redact_text",
]
