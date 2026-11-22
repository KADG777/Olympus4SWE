"""Local-Docker Stage 2 backend for SWE-Together's correctness judge.

The upstream judge (``eval/correctness``) requires an E2B or enroot sandbox.
This module provides a thin ``DockerJudgeSandbox`` implementing the same
``sandbox_id`` / ``write`` / ``run`` / ``read`` / ``kill`` protocol on top of
the local Docker CLI, a faithful port of the Phase 1/2 judge flow
(``run_judge_docker``), and a CLI that scores one or more SWE-Together trials
without ever touching ``E2B_API_KEY``:

* correctness — agent patch applied in a fresh container from the task image,
  then the agentic judge runs and writes ``verdict.json``;
* user behavior — the host-side ``User Correction`` tagging pipeline is
  delegated to the upstream ``eval.user_behavior.run_batch`` module.

Everything Harbor/SWE-Together-specific is confined to this file; core
``olympus`` code stays benchmark-free.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import posixpath
import re
import shlex
import subprocess
import sys
import tempfile
import time
import tomllib
import uuid
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

JUDGE_TIMEOUT_SEC = 600
JUDGE_MAX_TURNS = 50
PHASE1_PROMPT = "judge_phase1_system.md"
PHASE2_PROMPT = "judge_phase2_system.md"
DEFAULT_PHASE2_PROMPT_PATH = (
    "eval/correctness/prompts/judge_phase2_system.md"
)
DEFAULT_PHASE1_PROMPT_PATH = (
    "eval/correctness/prompts/judge_phase1_system.md"
)


@dataclass(frozen=True)
class CmdResult:
    """Mirror of ``eval.correctness.judge_sandbox.CmdResult``."""

    stdout: str
    stderr: str
    exit_code: int


class DockerCli:
    """Tiny injectable wrapper around the ``docker`` executable."""

    def __init__(
        self,
        binary: str = "docker",
        runner: Callable[..., Any] | None = None,
    ) -> None:
        self.binary = binary
        self._runner = runner or subprocess.run

    def run(
        self,
        args: Sequence[str],
        *,
        timeout: float | None = None,
    ) -> subprocess.CompletedProcess[str]:
        argv = [self.binary, *(str(item) for item in args)]
        return self._runner(
            argv,
            input=None,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    async def arun(
        self, args: Sequence[str], *, timeout: float | None = None
    ) -> subprocess.CompletedProcess[str]:
        return await asyncio.to_thread(self.run, args, timeout=timeout)


class DockerJudgeSandbox:
    """Judge sandbox backed by one long-lived local Docker container."""

    def __init__(
        self,
        *,
        name: str,
        image: str,
        cli: DockerCli | None = None,
    ) -> None:
        self.sandbox_id = name
        self.image = image
        self._cli = cli or DockerCli()

    @classmethod
    async def create(
        cls,
        *,
        image: str,
        name: str | None = None,
        envs: Mapping[str, str] | None = None,
        cli: DockerCli | None = None,
        ready_timeout_sec: float = 30.0,
    ) -> DockerJudgeSandbox:
        """Start a fresh container from ``image`` and wait until exec works."""
        docker = cli or DockerCli()
        container = name or f"olympus-judge-{uuid.uuid4().hex[:12]}"
        args = ["run", "-d", "--name", container]
        for key, value in (envs or {}).items():
            args.extend(["-e", f"{key}={value}"])
        args.extend(
            ["--entrypoint", "/bin/sh", image, "-c", "sleep infinity"]
        )
        proc = await docker.arun(args, timeout=180)
        if proc.returncode != 0:
            raise RuntimeError(
                f"docker run failed for {image!r}: "
                f"{(proc.stderr or proc.stdout or '').strip()[-1200:]}"
            )
        sandbox = cls(name=container, image=image, cli=docker)
        deadline = time.monotonic() + ready_timeout_sec
        while True:
            probe = await docker.arun(
                ["exec", container, "/bin/sh", "-c", "true"], timeout=15
            )
            if probe.returncode == 0:
                return sandbox
            if time.monotonic() >= deadline:
                await sandbox.kill()
                raise RuntimeError(
                    f"container {container!r} did not become ready within "
                    f"{ready_timeout_sec:.0f}s"
                )
            await asyncio.sleep(0.5)

    async def write(self, path: str, content: str | bytes) -> None:
        """Copy ``content`` into the container at ``path`` (``docker cp``)."""
        target_dir = posixpath.dirname(path) or "/"
        mkdir = await self._cli.arun(
            ["exec", self.sandbox_id, "/bin/sh", "-c", f"mkdir -p {shlex.quote(target_dir)}"],
            timeout=30,
        )
        if mkdir.returncode != 0:
            raise RuntimeError(
                f"failed to create {target_dir} in {self.sandbox_id}: "
                f"{(mkdir.stderr or mkdir.stdout).strip()[-500:]}"
            )
        payload = content.encode("utf-8") if isinstance(content, str) else content
        descriptor, temp_path = tempfile.mkstemp(prefix="olympus-judge-")
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
            proc = await self._cli.arun(
                ["cp", temp_path, f"{self.sandbox_id}:{path}"], timeout=120
            )
        finally:
            with suppress(OSError):
                os.unlink(temp_path)
        if proc.returncode != 0:
            raise RuntimeError(
                f"failed to write {path} in {self.sandbox_id}: "
                f"{(proc.stderr or proc.stdout).strip()[-500:]}"
            )

    async def run(
        self,
        cmd: str,
        *,
        timeout: int,
        cwd: str | None = None,
        user: str | None = None,
    ) -> CmdResult:
        """Execute ``cmd`` via ``docker exec`` and never raise on non-zero exit."""
        args = ["exec"]
        if user:
            args.extend(["-u", user])
        if cwd:
            args.extend(["-w", cwd])
        args.extend([self.sandbox_id, "/bin/sh", "-c", cmd])
        try:
            proc = await self._cli.arun(args, timeout=timeout)
        except subprocess.TimeoutExpired:
            return CmdResult(
                stdout="",
                stderr=f"docker exec timed out after {timeout}s",
                exit_code=124,
            )
        return CmdResult(
            stdout=proc.stdout or "",
            stderr=proc.stderr or "",
            exit_code=proc.returncode,
        )

    async def read(self, path: str) -> str:
        result = await self.run(f"cat {shlex.quote(path)}", timeout=30)
        if result.exit_code != 0:
            raise RuntimeError(
                f"failed to read {path} from {self.sandbox_id}: "
                f"{result.stderr.strip()[-500:]}"
            )
        return result.stdout

    async def kill(self) -> None:
        with suppress(Exception):
            await self._cli.arun(["rm", "-f", self.sandbox_id], timeout=60)


# --------------------------------------------------------------------------- judge inputs


@dataclass
class JudgeInputs:
    """Mirror of ``eval.correctness.sandbox.JudgeInputs`` (subset)."""

    readme: str = ""
    user_sim_prompt: str = ""
    oracle_patch: str = ""
    agent_patch: str = ""
    test_sh: str = ""
    system_prompt: str = ""
    tests_files: dict[str, bytes] = field(default_factory=dict)
    phase: int = 2
    canonical_goals_json: str = ""
    user_dialogue: str = ""


@dataclass
class JudgeRunResult:
    """Mirror of ``eval.correctness.sandbox.JudgeRunResult``."""

    verdict: dict[str, Any]
    stdout: str
    stderr: str
    exit_code: int
    sandbox_id: str
    judge_model: str = ""
    judge_backend: str = "docker"


def _read_optional(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def load_tests_files(task_dir: Path) -> dict[str, bytes]:
    tests_dir = task_dir / "tests"
    files: dict[str, bytes] = {}
    if not tests_dir.exists():
        return files
    for entry in tests_dir.iterdir():
        if entry.is_file():
            with suppress(OSError):
                files[entry.name] = entry.read_bytes()
    return files


def load_reference_patch(task_dir: Path) -> str:
    """Canonical oracle patch for Phase 1, from either upstream artifact."""
    reference = task_dir / "reference_patch.json"
    if reference.exists():
        try:
            patch = json.loads(reference.read_text(encoding="utf-8")).get("patch")
            if isinstance(patch, str) and patch.strip():
                return patch
        except (json.JSONDecodeError, OSError):
            pass
    session_file = task_dir / "oracle_session.jsonl"
    if session_file.exists():
        try:
            with session_file.open(encoding="utf-8") as handle:
                header = json.loads(handle.readline() or "{}")
            patch = header.get("_grading_patch")
            if isinstance(patch, str) and patch.strip():
                return patch
        except (json.JSONDecodeError, OSError):
            pass
    return ""


def load_user_dialogue(task_dir: Path, limit: int = 20_000) -> str:
    """Best-effort user-intent transcript for Phase 1 tasks without a patch."""
    parts: list[str] = []
    intents = task_dir / "oracle_intents.json"
    if intents.exists():
        with suppress(OSError, json.JSONDecodeError):
            parts.append(json.dumps(json.loads(intents.read_text(encoding="utf-8")), indent=2))
    session_file = task_dir / "oracle_session.jsonl"
    if session_file.exists():
        try:
            for index, line in enumerate(session_file.read_text(encoding="utf-8").splitlines()):
                if not line.strip():
                    continue
                record = json.loads(line)
                if index == 0 and record.get("_is_header"):
                    continue
                role = record.get("role") or record.get("type") or "user"
                if role != "user" and role != "human":
                    continue
                content = record.get("content") or record.get("message") or ""
                if isinstance(content, str) and content.strip():
                    parts.append(f"[user] {content.strip()}")
        except (json.JSONDecodeError, OSError):
            pass
    return "\n\n".join(parts)[:limit]


def phase_prompt_path(swe_together_root: Path | str, phase: int) -> Path:
    relative = DEFAULT_PHASE1_PROMPT_PATH if phase == 1 else DEFAULT_PHASE2_PROMPT_PATH
    return Path(swe_together_root).expanduser().resolve() / relative


def load_judge_inputs(
    *,
    task_dir: Path | str,
    trial_dir: Path | str,
    phase: int = 2,
    system_prompt_path: Path | str | None = None,
    swe_together_root: Path | str | None = None,
) -> JudgeInputs:
    """Assemble the judge inputs from a SWE-Together task/trial pair."""
    task = Path(task_dir).expanduser().resolve()
    trial = Path(trial_dir).expanduser().resolve()
    if phase not in (1, 2):
        raise ValueError(f"unsupported judge phase {phase!r}; expected 1 or 2")
    prompt_path = (
        Path(system_prompt_path).expanduser().resolve()
        if system_prompt_path
        else phase_prompt_path(swe_together_root or _infer_swe_together_root(task), phase)
    )
    if not prompt_path.exists():
        raise FileNotFoundError(
            f"judge system prompt not found: {prompt_path} "
            "(pass --swe-together-root or --system-prompt)"
        )
    agent_patch = _read_optional(trial / "agent" / "final.patch")
    if phase == 2 and not agent_patch.strip():
        raise FileNotFoundError(f"trial patch not found: {trial / 'agent' / 'final.patch'}")
    return JudgeInputs(
        readme=_read_optional(task / "README.md"),
        user_sim_prompt=_read_optional(task / "user_simulation_prompt.md"),
        oracle_patch=load_reference_patch(task) if phase == 1 else "",
        agent_patch=agent_patch,
        test_sh=_read_optional(task / "tests" / "test.sh"),
        system_prompt=prompt_path.read_text(encoding="utf-8"),
        tests_files=load_tests_files(task),
        phase=phase,
        canonical_goals_json=_read_optional(task / "canonical_goals.json"),
        user_dialogue=load_user_dialogue(task) if phase == 1 else "",
    )


def _infer_swe_together_root(task_dir: Path) -> Path:
    """Walk up from ``tasks/<task>`` to the benchmark checkout root."""
    for parent in [task_dir, *task_dir.parents]:
        if (parent / "eval" / "correctness" / "prompts").is_dir():
            return parent
    return task_dir.parent


# --------------------------------------------------------------------------- patch normalisation


_BANNER_RE = re.compile(r"^=== .* ===$")
_BANNER_PATH_RE = re.compile(r"^=== (\S+) \(", re.M)
_HUNK_RE = re.compile(r"^@@ -\d+(?:,(\d+))? \+\d+(?:,(\d+))? @@")
_SUBMODULE_RE = re.compile(r"^index [0-9a-f]+\.\.[0-9a-f]+ 160000$", re.M)
_BINARY_RE = re.compile(r"^Binary files .* differ$", re.M)


def banner_repos(diff: str) -> list[str]:
    return [match.group(1) for match in _BANNER_PATH_RE.finditer(diff)]


def main_repo_path(diff: str) -> str | None:
    repos = banner_repos(diff)
    if not repos:
        return None
    for repo in repos:
        if repo != "/tmp" and not repo.startswith("/tmp/"):
            return repo
    return repos[0]


def _main_repo_section(diff: str) -> str:
    lines = diff.split("\n")
    starts = [index for index, line in enumerate(lines) if _BANNER_RE.match(line)]
    if len(starts) <= 1:
        return diff
    target = main_repo_path(diff)
    for position, start in enumerate(starts):
        match = _BANNER_PATH_RE.match(lines[start])
        if match and match.group(1) == target:
            end = starts[position + 1] if position + 1 < len(starts) else len(lines)
            return "\n".join(lines[start:end])
    return diff


def _strip_banners_and_unapplyable(diff: str) -> str:
    section = _main_repo_section(diff)
    base = "\n".join(line for line in section.split("\n") if not _BANNER_RE.match(line))
    blocks = re.split(r"^(?=diff --git )", base, flags=re.M)
    kept = [
        block
        for block in blocks
        if not _SUBMODULE_RE.search(block) and not _BINARY_RE.search(block)
    ]
    return "".join(kept)


def _rstrip_empty(diff: str) -> list[str]:
    lines = diff.split("\n")
    while lines and lines[-1] == "":
        lines.pop()
    return lines


def _short_last_hunk(lines: list[str]) -> tuple[int | None, tuple[int, int]]:
    hunks = [index for index, line in enumerate(lines) if line.startswith("@@")]
    if not hunks:
        return None, (0, 0)
    header = hunks[-1]
    match = _HUNK_RE.match(lines[header])
    if not match:
        return None, (0, 0)
    old_n = int(match.group(1) or 1)
    new_n = int(match.group(2) or 1)
    body = [line for line in lines[header + 1 :] if not line.startswith("\\ ")]
    old_count = sum(1 for line in body if line[:1] in (" ", "-"))
    new_count = sum(1 for line in body if line[:1] in (" ", "+"))
    missing = old_n - old_count
    if missing >= 1 and new_n - new_count == missing:
        return header, (old_count, new_count)
    return None, (0, 0)


def _rewrite_hunk_counts(header: str, old: int, new: int) -> str:
    match = re.match(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@(.*)$", header)
    if match is None:
        return header
    return f"@@ -{match.group(1)},{old} +{match.group(2)},{new} @@{match.group(3)}"


def _restore_trailing_context(diff: str) -> str:
    lines = _rstrip_empty(diff)
    header, counts = _short_last_hunk(lines)
    if header is None:
        return "\n".join(lines)
    lines[header] = _rewrite_hunk_counts(lines[header], *counts)
    return "\n".join(lines)


def apply_candidates(diff: str) -> list[str]:
    """Normalised patch candidates, mirroring ``src/patch_normalize.py``."""
    if not diff.strip():
        return []
    base = _strip_banners_and_unapplyable(diff)
    if not base.strip():
        return []
    candidates = [_restore_trailing_context(base) + "\n"]
    lines = _rstrip_empty(base)
    header, counts = _short_last_hunk(lines)
    if header is not None:
        match = _HUNK_RE.match(lines[header])
        missing = int(match.group(1) or 1) - counts[0] if match else 0
        if missing:
            candidates.append("\n".join([*lines, *([" "] * missing)]) + "\n")
    return candidates


# --------------------------------------------------------------------------- judge flow


def build_apply_script(*, candidates: int, repo_hint: str = "", skip_apply: bool = False) -> str:
    """Apply the best patch candidate as root, discovering the task repo first."""
    hint = shlex.quote(repo_hint or "")
    if skip_apply:
        apply_branch = 'chmod -R a+rwX "$REPO" 2>/dev/null || true'
    else:
        apply_branch = (
            'APPLIED=""; '
            'for flags in "" "-C2"; do '
            f"  for i in $(seq 0 {max(0, candidates - 1)}); do "
            '    if git -c safe.directory="*" apply --check --whitespace=nowarn '
            '$flags "/tmp/agent.patch.$i" 2>/dev/null; then '
            '      git -c safe.directory="*" apply --whitespace=nowarn '
            '$flags "/tmp/agent.patch.$i" && cp "/tmp/agent.patch.$i" '
            '/tmp/agent.patch && APPLIED="$i${flags:+ $flags}"; break 2; '
            "    fi; "
            "  done; "
            "done; "
            'if [ -z "$APPLIED" ]; then git -c safe.directory="*" apply --check '
            "--whitespace=nowarn /tmp/agent.patch.0; exit 1; fi; "
            'echo "applied candidate $APPLIED"; '
            '{ chmod -R a+rwX "$REPO" 2>/dev/null || true; }'
        )
    return "".join(
        [
            "set -e; ",
            f"HINT={hint}; ",
            'ROOTS="/workspace /opt /home /app /repo /tmp /entire-cli '
            '/entireio-cli /no-magic"; ',
            'if [ -n "${HARBOR_REPO_PATHS:-}" ]; then '
            '  ROOTS="$ROOTS $(echo "$HARBOR_REPO_PATHS" | tr ":" " ")"; '
            "fi; ",
            'EXISTING=""; ',
            'for r in $ROOTS; do [ -e "$r" ] && EXISTING="$EXISTING $r"; done; ',
            'if [ -z "$EXISTING" ]; then echo "NO_REPO_ROOTS_EXIST" >&2; exit 1; fi; ',
            'REPO=""; ',
            'if [ -n "$HINT" ] && [ -e "$HINT/.git" ]; then REPO="$HINT"; fi; ',
            'if [ -z "$REPO" ]; then ',
            "  REPO=$(find $EXISTING -maxdepth 3 -name .git \\( -type d -o -type f \\) "
            '2>/dev/null | while IFS= read -r g; do printf "%d %s\\n" "${#g}" "$g"; '
            'done | sort -n | head -1 | cut -d" " -f2- | xargs -I{} dirname {}); ',
            "fi; ",
            'if [ -z "$REPO" ]; then echo "NO_GIT_REPO_FOUND" >&2; exit 1; fi; ',
            'cd "$REPO" && echo "applying to $(pwd)" && ',
            apply_branch,
        ]
    )


def build_first_message(
    *,
    inputs_dir: str,
    phase: int,
    oracle_present: bool = True,
    repo_hint: str = "/workspace",
) -> str:
    if phase == 1:
        if oracle_present:
            return (
                f"Begin by reading {inputs_dir}/README.md and "
                f"{inputs_dir}/user_simulation_prompt.md to understand the task. "
                f"Then read {inputs_dir}/oracle.patch (the reference solution, "
                f"already applied to {repo_hint}) and explore the workspace. "
                f"Decompose the task into completeness goals and write the FROZEN "
                f"rubric to {inputs_dir}/canonical_goals.json."
            )
        return (
            f"Begin by reading {inputs_dir}/README.md and "
            f"{inputs_dir}/user_simulation_prompt.md to understand the task. "
            f"This task has NO canonical oracle patch. Read "
            f"{inputs_dir}/user_dialogue.md for the user's stated intent, derive "
            f"goals from it and from {inputs_dir}/test.sh, and write the FROZEN "
            f"rubric to {inputs_dir}/canonical_goals.json."
        )
    return (
        f"Begin by reading {inputs_dir}/canonical_goals.json — this is the FROZEN "
        f"rubric. DO NOT re-derive goals; for each goal in the rubric, mark "
        f"met:true/false with concrete evidence. Then read {inputs_dir}/README.md "
        f"and {inputs_dir}/user_simulation_prompt.md for context, inspect the "
        f"agent's patch at {inputs_dir}/agent.patch (already applied to "
        f"{repo_hint}), explore the workspace, and optionally run tests. Write "
        f"your verdict to {inputs_dir}/verdict.json."
    )


def build_judge_command(
    *,
    inputs_dir: str = "/tmp/judge_inputs",
    phase: int = 2,
    timeout_sec: int = JUDGE_TIMEOUT_SEC,
    max_turns: int = JUDGE_MAX_TURNS,
    model: str | None = None,
    oracle_present: bool = True,
    repo_hint: str = "/workspace",
) -> str:
    """Default headless ``claude --print`` invocation inside the sandbox."""
    first_message = build_first_message(
        inputs_dir=inputs_dir,
        phase=phase,
        oracle_present=oracle_present,
        repo_hint=repo_hint,
    )
    model_arg = f"--model {shlex.quote(model)} " if model else ""
    return (
        'export PATH="$HOME/.local/bin:/root/.local/bin:$PATH"; '
        f"timeout {int(timeout_sec)} claude --print --max-turns {int(max_turns)} "
        f"{model_arg}--dangerously-skip-permissions --setting-sources user "
        f'--append-system-prompt "$(cat {inputs_dir}/judge_system.md)" '
        f"{shlex.quote(first_message)}"
    )


def _judge_envs(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    source = os.environ if environ is None else environ
    keys = (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_BASE_URL",
        "CLAUDE_CODE_OAUTH_TOKEN",
    )
    envs = {key: source[key] for key in keys if source.get(key)}
    envs["IS_SANDBOX"] = "1"
    return envs


async def _ensure_judge_cli(sandbox: Any) -> CmdResult:
    prefix = 'export PATH="$HOME/.local/bin:/root/.local/bin:$PATH"; '
    check = await sandbox.run(prefix + "command -v claude || true", timeout=10)
    if check.stdout.strip():
        return check
    install = await sandbox.run(
        "curl -fsSL https://claude.ai/install.sh | bash -s -- 2.1.108", timeout=300
    )
    return install


async def run_judge_docker(
    *,
    task_name: str,
    trial_id: str,
    inputs: JudgeInputs,
    image: str,
    timeout_sec: int = JUDGE_TIMEOUT_SEC,
    max_turns: int = JUDGE_MAX_TURNS,
    model: str | None = None,
    judge_command: str | None = None,
    sandbox_factory: Callable[..., Any] | None = None,
    envs: Mapping[str, str] | None = None,
) -> JudgeRunResult:
    """Run one judge phase in a fresh local-Docker container, then clean up."""
    inputs_dir = "/tmp/judge_inputs"
    tests_dir = f"{inputs_dir}/tests"
    output_filename = "canonical_goals.json" if inputs.phase == 1 else "verdict.json"
    sandbox: Any | None = None
    factory = sandbox_factory or _default_sandbox_factory
    container = f"olympus-judge-{uuid.uuid4().hex[:12]}"
    try:
        sandbox = await factory(
            image=image,
            name=container,
            envs=_judge_envs(envs),
        )
        patch_to_apply = inputs.oracle_patch if inputs.phase == 1 else inputs.agent_patch
        skip_apply = inputs.phase == 1 and not patch_to_apply.strip()
        candidates = apply_candidates(patch_to_apply) or [patch_to_apply or ""]
        for index, candidate in enumerate(candidates):
            await sandbox.write(f"/tmp/agent.patch.{index}", candidate)
        await sandbox.write("/tmp/agent.patch", candidates[0])
        repo_hint = main_repo_path(patch_to_apply) or ""
        apply = await sandbox.run(
            build_apply_script(
                candidates=len(candidates),
                repo_hint=repo_hint,
                skip_apply=skip_apply,
            ),
            timeout=180,
            user="root",
        )
        if apply.exit_code != 0:
            return JudgeRunResult(
                verdict={
                    "error": "patch_apply_failed",
                    "stdout": apply.stdout[-2000:],
                    "stderr": apply.stderr[-2000:],
                },
                stdout=apply.stdout,
                stderr=apply.stderr,
                exit_code=apply.exit_code,
                sandbox_id=sandbox.sandbox_id,
                judge_backend="docker",
            )
        for line in apply.stdout.splitlines():
            if line.startswith("applying to "):
                repo_hint = line.removeprefix("applying to ").strip()
                break

        await sandbox.run(f"mkdir -p {inputs_dir} {tests_dir}", timeout=10, user="root")
        await sandbox.write(f"{inputs_dir}/README.md", inputs.readme)
        await sandbox.write(
            f"{inputs_dir}/user_simulation_prompt.md", inputs.user_sim_prompt
        )
        await sandbox.write(f"{inputs_dir}/oracle.patch", inputs.oracle_patch)
        await sandbox.write(f"{inputs_dir}/agent.patch", inputs.agent_patch)
        await sandbox.write(f"{inputs_dir}/test.sh", inputs.test_sh)
        await sandbox.write(f"{inputs_dir}/judge_system.md", inputs.system_prompt)
        if inputs.user_dialogue:
            await sandbox.write(f"{inputs_dir}/user_dialogue.md", inputs.user_dialogue)
        if inputs.phase == 2 and inputs.canonical_goals_json:
            await sandbox.write(
                f"{inputs_dir}/canonical_goals.json", inputs.canonical_goals_json
            )
        for filename, content in inputs.tests_files.items():
            await sandbox.write(f"{tests_dir}/{filename}", content)
        await sandbox.run(
            f"chmod +x {tests_dir}/test.sh 2>/dev/null || true",
            timeout=10,
            user="root",
        )

        command = judge_command
        if command is None:
            await _ensure_judge_cli(sandbox)
            command = build_judge_command(
                inputs_dir=inputs_dir,
                phase=inputs.phase,
                timeout_sec=timeout_sec,
                max_turns=max_turns,
                model=model,
                oracle_present=bool(inputs.oracle_patch.strip()),
                repo_hint=repo_hint or "/workspace",
            )
        result = await sandbox.run(command, timeout=timeout_sec + 60, cwd="/tmp")
        try:
            verdict = json.loads(await sandbox.read(f"{inputs_dir}/{output_filename}"))
            if not isinstance(verdict, dict):
                verdict = {"error": "non_dict_verdict", "value": verdict}
        except Exception as exc:
            verdict = {
                "error": "verdict_read_failed",
                "exception": str(exc),
                "expected_filename": output_filename,
                "judge_exit_code": result.exit_code,
                "judge_stdout_tail": result.stdout[-2000:],
                "judge_stderr_tail": result.stderr[-2000:],
            }
        applied = re.search(r"applied candidate (\d+)((?: -[-A-Za-z0-9]+)*)", apply.stdout)
        verdict["patch_applied_candidate"] = int(applied.group(1)) if applied else None
        verdict["patch_applied_flags"] = applied.group(2).strip() if applied else None
        verdict["patch_applied_repo"] = repo_hint or "/workspace"
        return JudgeRunResult(
            verdict=verdict,
            stdout=result.stdout,
            stderr=result.stderr,
            exit_code=result.exit_code,
            sandbox_id=sandbox.sandbox_id,
            judge_backend="docker",
        )
    finally:
        if sandbox is not None:
            try:
                await sandbox.kill()
            except Exception as exc:  # pragma: no cover - cleanup is best-effort
                log.warning("sandbox kill failed for %s: %s", container, exc)


async def _default_sandbox_factory(
    *, image: str, name: str, envs: Mapping[str, str] | None = None
) -> DockerJudgeSandbox:
    return await DockerJudgeSandbox.create(image=image, name=name, envs=envs)


# --------------------------------------------------------------------------- task images


def task_docker_image(
    task_dir: Path | str,
    *,
    image: str | None = None,
    dockerfile: Path | str | None = None,
    tag: str | None = None,
    build: bool = True,
    cli: DockerCli | None = None,
) -> str:
    """Resolve the task image (``task.toml``) or build its Dockerfile locally."""
    if image:
        return image
    task = Path(task_dir).expanduser().resolve()
    task_toml = task / "task.toml"
    if task_toml.exists():
        with suppress(OSError, tomllib.TOMLDecodeError):
            config = tomllib.loads(task_toml.read_text(encoding="utf-8"))
            configured = (config.get("environment") or {}).get("docker_image")
            if isinstance(configured, str) and configured.strip():
                return configured.strip()
    dockerfile_path = (
        Path(dockerfile).expanduser().resolve()
        if dockerfile
        else task / "environment" / "Dockerfile"
    )
    if not build or not dockerfile_path.exists():
        raise FileNotFoundError(
            f"no docker image for {task}: task.toml has no [environment].docker_image "
            f"and {dockerfile_path} does not exist"
        )
    image_tag = tag or f"olympus-judge-{_slug(task.name)}:local"
    docker = cli or DockerCli()
    proc = docker.run(
        [
            "build",
            "-t",
            image_tag,
            "-f",
            str(dockerfile_path),
            str(dockerfile_path.parent),
        ],
        timeout=3600,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"docker build failed for {task.name}: "
            f"{(proc.stderr or proc.stdout or '').strip()[-1500:]}"
        )
    return image_tag


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9_.-]+", "-", value.lower()).strip("-") or "task"


# --------------------------------------------------------------------------- user behavior


def run_user_behavior(
    *,
    swe_together_root: Path | str,
    jobs: Sequence[Mapping[str, Any]],
    workers: int = 5,
    model: str | None = None,
    force: bool = False,
    python: str | None = None,
    timeout: float = 3600,
    runner: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Delegate ``User Correction`` tagging to upstream ``eval.user_behavior``."""
    root = Path(swe_together_root).expanduser().resolve()
    module = root / "eval" / "user_behavior" / "run_batch.py"
    if not module.exists():
        raise FileNotFoundError(f"SWE-Together user-behavior runner not found: {module}")
    workspace = Path(tempfile.mkdtemp(prefix="olympus-user-behavior-"))
    plan_path = workspace / "plan.json"
    plan = [
        {
            "trial_dir": str(job["trial_dir"]),
            "task_dir": str(job["task_dir"]),
            "out_name": job.get("out_name", "intent_coverage_verdict.json"),
            **({"force": True} if force else {}),
        }
        for job in jobs
    ]
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    argv = [
        python or sys.executable,
        "-m",
        "eval.user_behavior.run_batch",
        "--plan",
        str(plan_path),
        "--workers",
        str(workers),
    ]
    if model:
        argv.extend(["--model", model])
    if force:
        argv.append("--force")
    invoke = runner or subprocess.run
    proc = invoke(argv, cwd=str(root), capture_output=True, text=True, timeout=timeout)
    summary_path = plan_path.with_suffix(".summary.json")
    summary: dict[str, Any] = {}
    with suppress(OSError, json.JSONDecodeError):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return {
        "status": "ok" if proc.returncode == 0 else "error",
        "return_code": proc.returncode,
        "plan": str(plan_path),
        "stdout_tail": (proc.stdout or "")[-2000:],
        "stderr_tail": (proc.stderr or "")[-2000:],
        "summary": summary,
    }


# --------------------------------------------------------------------------- batch scoring


def derive_judge_score(
    rubric: Mapping[str, Any], goal_results: Sequence[Mapping[str, Any]]
) -> tuple[float, str]:
    """Mechanically re-derive the score from frozen weights × ``met`` flags."""
    weights = {
        str(goal["id"]): float(goal["weight"])
        for goal in rubric.get("completeness_goals", [])
        if "id" in goal and "weight" in goal
    }
    met = {str(result.get("id")): bool(result.get("met")) for result in goal_results}
    score = round(sum(weight for goal_id, weight in weights.items() if met.get(goal_id)), 2)
    if score >= 0.85:
        return score, "equivalent"
    if score >= 0.30:
        return score, "partial"
    return score, "incorrect"


@dataclass
class JudgeJob:
    trial_dir: Path
    task_dir: Path
    out_name: str = "judge_verdict.json"


def parse_plan(plan: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> list[JudgeJob]:
    entries = plan.get("jobs", []) if isinstance(plan, Mapping) else plan
    jobs: list[JudgeJob] = []
    for entry in entries:
        jobs.append(
            JudgeJob(
                trial_dir=Path(str(entry["trial_dir"])).expanduser(),
                task_dir=Path(str(entry["task_dir"])).expanduser(),
                out_name=str(entry.get("out_name") or "judge_verdict.json"),
            )
        )
    return jobs


async def judge_jobs(
    *,
    jobs: Sequence[JudgeJob],
    swe_together_root: Path | str,
    image: str | None = None,
    phase: int = 2,
    force: bool = False,
    timeout_sec: int = JUDGE_TIMEOUT_SEC,
    max_turns: int = JUDGE_MAX_TURNS,
    model: str | None = None,
    judge_command: str | None = None,
    sandbox_factory: Callable[..., Any] | None = None,
    auto_rubric: bool = False,
) -> list[dict[str, Any]]:
    """Score every job's patch in its own fresh Docker container."""
    results: list[dict[str, Any]] = []
    for job in jobs:
        trial = job.trial_dir.resolve()
        task = job.task_dir.resolve()
        output = trial / job.out_name
        result: dict[str, Any] = {
            "trial_dir": str(trial),
            "task_dir": str(task),
            "out_name": job.out_name,
        }
        if output.exists() and not force:
            result["status"] = "skipped_existing"
            results.append(result)
            continue
        rubric_path = task / "canonical_goals.json"
        try:
            if phase == 2:
                patch_text = _read_optional(trial / "agent" / "final.patch")
                if len(patch_text.strip()) < 100 or not apply_candidates(patch_text):
                    result["status"] = "skipped_no_patch"
                    result["patch_bytes"] = len(patch_text)
                    results.append(result)
                    continue
            if phase == 2 and not rubric_path.exists():
                if not auto_rubric:
                    result["status"] = "skipped_no_rubric"
                    result["reason"] = f"missing {rubric_path}"
                    results.append(result)
                    continue
                phase1_inputs = load_judge_inputs(
                    task_dir=task,
                    trial_dir=trial,
                    phase=1,
                    swe_together_root=swe_together_root,
                )
                if not (
                    phase1_inputs.oracle_patch.strip()
                    or phase1_inputs.user_dialogue.strip()
                ):
                    result["status"] = "skipped_no_rubric"
                    result["reason"] = "phase 1 has neither an oracle patch nor dialogue"
                    results.append(result)
                    continue
                phase1 = await run_judge_docker(
                    task_name=task.name,
                    trial_id=trial.name,
                    inputs=phase1_inputs,
                    image=task_docker_image(task, image=image),
                    timeout_sec=timeout_sec,
                    max_turns=max_turns,
                    model=model,
                    judge_command=judge_command,
                    sandbox_factory=sandbox_factory,
                )
                if "error" in phase1.verdict:
                    result["status"] = "phase1_error"
                    result["error"] = phase1.verdict.get("error")
                    results.append(result)
                    continue
                rubric_path.write_text(
                    json.dumps(phase1.verdict, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
            inputs = load_judge_inputs(
                task_dir=task,
                trial_dir=trial,
                phase=phase,
                swe_together_root=swe_together_root,
            )
            started = time.monotonic()
            judged = await run_judge_docker(
                task_name=task.name,
                trial_id=trial.name,
                inputs=inputs,
                image=task_docker_image(task, image=image),
                timeout_sec=timeout_sec,
                max_turns=max_turns,
                model=model,
                judge_command=judge_command,
                sandbox_factory=sandbox_factory,
            )
            elapsed = round(time.monotonic() - started, 1)
            verdict = dict(judged.verdict)
            rubric: dict[str, Any] = {}
            with suppress(OSError, json.JSONDecodeError):
                rubric = json.loads(rubric_path.read_text(encoding="utf-8"))
            goal_results = verdict.get("goal_results")
            if isinstance(goal_results, list) and rubric:
                verdict["judge_reported_score"] = verdict.get("judge_score")
                verdict["judge_reported_verdict"] = verdict.get("verdict")
                score, bucket = derive_judge_score(rubric, goal_results)
                verdict["judge_score"] = score
                verdict["verdict"] = bucket
            reward_path = trial / "verifier" / "reward.txt"
            if reward_path.exists():
                with suppress(ValueError):
                    verdict["test_reward_raw"] = float(
                        reward_path.read_text(encoding="utf-8").strip()
                    )
            if (
                verdict.get("judge_score") is not None
                and verdict.get("test_reward_raw") is not None
            ):
                delta = float(verdict["judge_score"]) - float(verdict["test_reward_raw"])
                verdict["score_delta"] = round(delta, 4)
                verdict["direction"] = (
                    "unchanged" if abs(delta) < 1e-6 else "upgrade" if delta > 0 else "downgrade"
                )
            verdict.setdefault("task", task.name)
            verdict.setdefault("trial_id", trial.name)
            verdict["judge_elapsed_sec"] = elapsed
            verdict["sandbox_id"] = judged.sandbox_id
            verdict["judge_exit_code"] = judged.exit_code
            verdict["judge_backend"] = "docker"
            verdict["judge_phase"] = phase
            verdict["rubric_n_goals"] = len(rubric.get("completeness_goals", []))
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps(verdict, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            result["status"] = "ok" if "error" not in verdict else "verdict_error"
            result["judge_score"] = verdict.get("judge_score")
            result["verdict"] = verdict.get("verdict")
            result["elapsed_sec"] = elapsed
        except Exception as exc:  # noqa: BLE001 - one bad trial must not stop the batch
            result["status"] = "error"
            result["error"] = f"{type(exc).__name__}: {exc}"[:2000]
            log.warning("judge failed for %s: %s", trial.name, result["error"])
        results.append(result)
    return results


async def run_stage2(
    *,
    jobs: Sequence[JudgeJob],
    swe_together_root: Path | str,
    image: str | None = None,
    phase: int = 2,
    force: bool = False,
    timeout_sec: int = JUDGE_TIMEOUT_SEC,
    max_turns: int = JUDGE_MAX_TURNS,
    model: str | None = None,
    judge_command: str | None = None,
    sandbox_factory: Callable[..., Any] | None = None,
    auto_rubric: bool = False,
    user_behavior: bool = True,
    user_behavior_workers: int = 5,
    user_behavior_model: str | None = None,
    user_behavior_force: bool = False,
    user_behavior_runner: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Run Stage 2 correctness and (optionally) User Correction locally."""
    correctness = await judge_jobs(
        jobs=jobs,
        swe_together_root=swe_together_root,
        image=image,
        phase=phase,
        force=force,
        timeout_sec=timeout_sec,
        max_turns=max_turns,
        model=model,
        judge_command=judge_command,
        sandbox_factory=sandbox_factory,
        auto_rubric=auto_rubric,
    )
    behavior: dict[str, Any] | None = None
    if user_behavior and jobs:
        behavior = run_user_behavior(
            swe_together_root=swe_together_root,
            jobs=[
                {
                    "trial_dir": str(job.trial_dir.resolve()),
                    "task_dir": str(job.task_dir.resolve()),
                    "out_name": "intent_coverage_verdict.json",
                }
                for job in jobs
            ],
            workers=user_behavior_workers,
            model=user_behavior_model,
            force=user_behavior_force,
            runner=user_behavior_runner,
        )
    return {
        "correctness": correctness,
        "user_behavior": behavior,
        "status": (
            "ok"
            if all(item.get("status") != "error" for item in correctness)
            and (behavior is None or behavior.get("status") == "ok")
            else "error"
        ),
    }


# --------------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Judge SWE-Together trials locally on Docker (no E2B_API_KEY): "
            "eval/correctness via DockerJudgeSandbox plus eval/user_behavior "
            "User Correction tagging."
        )
    )
    parser.add_argument(
        "--swe-together-root",
        required=True,
        type=Path,
        help="Path to the SWE-Together checkout (provides prompts and eval modules).",
    )
    parser.add_argument("--plan", type=Path, help="JSON plan: [{trial_dir, task_dir, out_name}].")
    parser.add_argument("--trial-dir", type=Path, help="Single trial dir (with --task-dir).")
    parser.add_argument("--task-dir", type=Path, help="Single task dir (with --trial-dir).")
    parser.add_argument("--image", default=None, help="Override the task Docker image.")
    parser.add_argument("--phase", type=int, default=2, choices=[1, 2], help="Judge phase.")
    parser.add_argument(
        "--out-name", default="judge_verdict.json", help="Verdict file inside each trial dir."
    )
    parser.add_argument("--force", action="store_true", help="Re-run existing verdicts.")
    parser.add_argument(
        "--auto-rubric",
        action="store_true",
        help="Generate missing canonical_goals.json via Phase 1 in Docker.",
    )
    parser.add_argument(
        "--timeout-sec", type=int, default=JUDGE_TIMEOUT_SEC, help="Judge wall-clock budget."
    )
    parser.add_argument(
        "--max-turns", type=int, default=JUDGE_MAX_TURNS, help="Judge agent turn budget."
    )
    parser.add_argument("--model", default=None, help="Judge Claude model passed to `claude`.")
    parser.add_argument(
        "--judge-command",
        default=None,
        help="Override the in-container judge command (advanced/testing).",
    )
    parser.add_argument(
        "--skip-user-behavior",
        action="store_true",
        help="Skip the eval/user_behavior User Correction pass.",
    )
    parser.add_argument(
        "--user-behavior-workers", type=int, default=5, help="Concurrent tagging calls."
    )
    parser.add_argument(
        "--user-behavior-model", default=None, help="Model for the User Correction tagger."
    )
    parser.add_argument(
        "--user-behavior-force", action="store_true", help="Re-tag existing verdicts."
    )
    parser.add_argument("--json", action="store_true", help="Emit the summary as JSON.")
    parser.add_argument("--dry-run", action="store_true", help="Resolve jobs and exit.")
    return parser


def _load_jobs(args: argparse.Namespace) -> list[JudgeJob]:
    if args.plan:
        payload = json.loads(args.plan.read_text(encoding="utf-8"))
        jobs = parse_plan(payload)
    elif args.trial_dir and args.task_dir:
        jobs = [JudgeJob(trial_dir=args.trial_dir, task_dir=args.task_dir, out_name=args.out_name)]
    else:
        raise SystemExit("provide --plan or both --trial-dir and --task-dir")
    if not jobs:
        raise SystemExit("plan contains no jobs")
    for job in jobs:
        if job.out_name == "judge_verdict.json" and args.out_name:
            job.out_name = args.out_name
    return jobs


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    jobs = _load_jobs(args)
    root = args.swe_together_root.expanduser().resolve()
    if args.dry_run:
        def _image_hint(job: JudgeJob) -> str:
            try:
                return task_docker_image(job.task_dir, image=args.image, build=False)
            except Exception:
                return f"build:{job.task_dir / 'environment' / 'Dockerfile'}"

        print(
            json.dumps(
                {
                    "jobs": [
                        {
                            "trial_dir": str(job.trial_dir.resolve()),
                            "task_dir": str(job.task_dir.resolve()),
                            "out_name": job.out_name,
                            "image": _image_hint(job),
                        }
                        for job in jobs
                    ],
                    "user_behavior": not args.skip_user_behavior,
                },
                indent=2,
            )
        )
        return 0
    summary = asyncio.run(
        run_stage2(
            jobs=jobs,
            swe_together_root=root,
            image=args.image,
            phase=args.phase,
            force=args.force,
            timeout_sec=args.timeout_sec,
            max_turns=args.max_turns,
            model=args.model,
            judge_command=args.judge_command,
            auto_rubric=args.auto_rubric,
            user_behavior=not args.skip_user_behavior,
            user_behavior_workers=args.user_behavior_workers,
            user_behavior_model=args.user_behavior_model,
            user_behavior_force=args.user_behavior_force,
        )
    )
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        for item in summary["correctness"]:
            print(
                f"{item.get('status', '?'):>16}  {item.get('trial_dir', '')}"
                f"  score={item.get('judge_score')}  verdict={item.get('verdict')}"
                + (f"  error={item['error']}" if item.get("error") else "")
            )
        behavior = summary.get("user_behavior")
        if behavior is not None:
            print(f"user_behavior: {behavior.get('status')}")
    return 0 if summary["status"] == "ok" else 1


__all__ = [
    "CmdResult",
    "DockerCli",
    "DockerJudgeSandbox",
    "JudgeInputs",
    "JudgeJob",
    "JudgeRunResult",
    "apply_candidates",
    "build_apply_script",
    "build_first_message",
    "build_judge_command",
    "derive_judge_score",
    "judge_jobs",
    "load_judge_inputs",
    "main",
    "parse_plan",
    "run_judge_docker",
    "run_stage2",
    "run_user_behavior",
    "task_docker_image",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())