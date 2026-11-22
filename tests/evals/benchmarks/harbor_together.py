"""Harbor ``BaseAgent`` adapter driving Olympus through SWE-Together's user loop.

Everything Harbor- or SWE-Together-specific lives here (or in
``local_docker_judge.py``); :mod:`olympus` core never imports this module. The
adapter mirrors upstream ``user_enabled_mini_swe_agent`` semantics:

* Turn 0 runs ``olympus session start`` inside the task container with the
  verbatim task instruction and an externalized ``--state-dir``.
* Turn N feeds the host-side reactive ``UserAgent`` reply (including a
  post-completion ``User Correction``) into ``olympus session reply``.
* The simulated user sees *only* the assistant's natural-language message and
  the per-turn ``git diff`` — never raw tool arguments or results.
* Clean cumulative/ incremental patches are recorded under ``logs_dir`` and
  ``final.patch`` always reflects the last captured state.
* The loop stops on a user-sim ``STOP`` decision, a silent user on a blocked
  pause, consecutive silent turns after completion, the turn budget, or the
  wall-clock budget.

Harbor is an optional dependency of the default Olympus dev venv, so the base
class and ``AgentContext`` gracefully fall back to local stand-ins; when Harbor
is importable the adapter subclasses the real ``harbor.agents.base.BaseAgent``.
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
import tarfile
import time
import tomllib
import uuid
from collections.abc import Callable, Iterable, Mapping
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_HarborBaseAgent: Any = None
_HarborAgentContext: Any = None
_HarborCapabilities: Any = None
_HarborModelConnectionSpec: Any = None
_HarborProviderAccess: Any = None
_HarborModelApiFormat: Any = None

try:  # pragma: no cover - optional benchmark dependency
    from harbor.agents.base import BaseAgent as _HarborBaseAgent
    from harbor.models.agent.context import AgentContext as _HarborAgentContext

    HARBOR_AVAILABLE = True
except Exception:  # pragma: no cover - default pytest venv without harbor
    HARBOR_AVAILABLE = False

try:  # pragma: no cover - old Harbor releases predate capability declarations
    from harbor.agents.capabilities import AgentCapabilities as _HarborCapabilities
except Exception:  # pragma: no cover - fallback capability record below
    _HarborCapabilities = None

try:  # pragma: no cover - connection specs ship with newer Harbor releases
    from harbor.agents import agent_auth_resolution as _harbor_auth
    from harbor.models import model_api_format as _harbor_api_format

    _HarborModelConnectionSpec = _harbor_auth.ModelConnectionSpec
    _HarborProviderAccess = _harbor_auth.ProviderAccess
    _HarborModelApiFormat = _harbor_api_format.ModelApiFormat
except Exception:  # pragma: no cover - old Harbor resolves connections natively
    _HarborModelConnectionSpec = None

log = logging.getLogger(__name__)

OLYMPUS_AGENT_IMPORT_PATH = (
    "tests.evals.benchmarks.harbor_together:UserEnabledOlympusAgent"
)
_REDACTED = "***REDACTED***"
_USER_API_KEY_REFS: dict[str, str] = {}


def _register_user_api_key(value: str) -> str:
    ref = f"ref:{uuid.uuid4().hex}"
    _USER_API_KEY_REFS[ref] = value
    return ref


def _resolve_user_api_key(
    *, user_api_key: str | None, user_api_key_ref: str | None
) -> str | None:
    if user_api_key:
        return user_api_key
    if user_api_key_ref:
        return _USER_API_KEY_REFS.get(user_api_key_ref)
    return None

_SESSION_STATUSES = frozenset({"needs_input", "completed", "failed"})
_STOP_WORDS = frozenset({"stop", "done", "finish", "finished", "terminate", "exit"})
_MODEL_PROVIDER_PREFIXES = frozenset(
    {
        "anthropic",
        "azure",
        "bedrock",
        "deepseek",
        "gemini",
        "google",
        "groq",
        "mistral",
        "openai",
        "openrouter",
        "together",
        "xai",
    }
)


class _FallbackAgentContext:
    """Minimal stand-in for ``harbor.models.agent.context.AgentContext``."""

    def __init__(self, metadata: dict[str, Any] | None = None) -> None:
        self.metadata = metadata


@dataclass(frozen=True)
class _FallbackCapabilities:
    """Capability record compatible with ``harbor.agents.capabilities``."""

    base_url: bool = True
    atif: bool = True


class _FallbackBaseAgent:
    """Minimal stand-in for ``harbor.agents.base.BaseAgent``."""

    SUPPORTS_ATIF: bool = False

    def __init__(self, logs_dir: Path, model_name: str | None = None, **_: Any) -> None:
        self.logs_dir = Path(logs_dir)
        self.model_name = model_name
        self.logger = log

    @staticmethod
    def name() -> str:
        raise NotImplementedError

    def version(self) -> str | None:
        return None

    async def setup(self, environment: Any) -> None:
        raise NotImplementedError

    async def run(self, instruction: str, environment: Any, context: Any) -> None:
        raise NotImplementedError


BaseAgent: Any = _HarborBaseAgent if HARBOR_AVAILABLE else _FallbackBaseAgent
AgentContext: Any = _HarborAgentContext if HARBOR_AVAILABLE else _FallbackAgentContext


DEFAULT_REPO_ROOTS = (
    "/workspace /opt /home /app /repo /tmp /entire-cli /entireio-cli /no-magic"
)
USER_SIM_DIFF_MAX_CHARS = 200_000

_JUNK_RE = re.compile(
    r"(^|/)(\.venv|venv|node_modules|__pycache__|\.pytest_cache|\.mypy_cache"
    r"|\.ruff_cache|\.tox|\.desloppify|\.coverage|\.cache|\.eggs|\.gradle"
    r"|\.next|site-packages"
    r"|\.go|pkg/mod|\.cargo|\.npm|\.pnpm-store|\.m2|\.ivy2|\.nuget|\.stack-work"
    r"|\.pub-cache"
    r")(/|$)"
)


def build_repo_discovery_command(*, repo_roots: str | None = None) -> str:
    """Shell command printing ``OLYMPUS_REPO=<path>`` for the task repository."""
    roots = repo_roots or DEFAULT_REPO_ROOTS
    return "".join(
        [
            "set +e\n",
            f'ROOTS="{roots}"\n',
            'if [ -n "${HARBOR_REPO_PATHS:-}" ]; then\n',
            '  ROOTS="$ROOTS $(echo "$HARBOR_REPO_PATHS" | tr ":" " ")"\n',
            "fi\n",
            "for r in /workspace $ROOTS; do\n",
            '  if [ -d "$r/.git" ] || [ -f "$r/.git" ]; then\n',
            '    echo "OLYMPUS_REPO=$r"\n',
            "    exit 0\n",
            "  fi\n",
            "done\n",
            'GITDIR=$(find $ROOTS -maxdepth 3 \\( -type d -o -type f \\) -name .git '
            '2>/dev/null | while IFS= read -r g; do printf "%d %s\\n" "${{#g}}" "$g"; '
            'done | sort -n | head -1 | cut -d" " -f2-)\n',
            'if [ -n "$GITDIR" ]; then echo "OLYMPUS_REPO=$(dirname "$GITDIR")"; fi\n',
            "exit 0\n",
        ]
    )


def _tag_per_repo_script() -> str:
    return (
        '  (cd "$d" || exit 0; \\\n'
        '     git -c safe.directory="$PWD" tag -f harbor-base HEAD 2>/dev/null; \\\n'
        '     _del=$(git -c safe.directory="$PWD" status --porcelain 2>/dev/null '
        '| grep -c "^.D"); \\\n'
        '     if [ "${_del:-0}" -lt 20 ]; then \\\n'
        '       git -c safe.directory="$PWD" add -A 2>/dev/null; \\\n'
        '       git -c safe.directory="$PWD" -c user.email=harbor@base '
        '-c user.name=harbor commit --allow-empty --no-verify '
        '-m "harbor-base" --quiet 2>/dev/null; \\\n'
        '       git -c safe.directory="$PWD" tag -f harbor-base HEAD 2>/dev/null; \\\n'
        '     fi) || true\n'
    )


def build_tag_command(*, repo_roots: str | None = None) -> str:
    """Snapshot every discovered repo as ``harbor-base`` before the agent runs."""
    roots = repo_roots or DEFAULT_REPO_ROOTS
    return "".join(
        [
            "set +e\n",
            f'ROOTS="{roots}"\n',
            'if [ -n "${HARBOR_REPO_PATHS:-}" ]; then\n',
            '  ROOTS="$ROOTS $(echo "$HARBOR_REPO_PATHS" | tr ":" " ")"\n',
            "fi\n",
            'EXISTING=""\n',
            'for r in $ROOTS; do [ -e "$r" ] && EXISTING="$EXISTING $r"; done\n',
            '[ -z "$EXISTING" ] && exit 0\n',
            'REPOS=$(find $EXISTING -maxdepth 3 \\( -type d -o -type f \\) -name .git '
            "2>/dev/null | sort -u)\n",
            "for gitdir in $REPOS; do\n",
            '  d=$(dirname "$gitdir")\n',
            _tag_per_repo_script(),
            "done\n",
        ]
    )


def _diff_per_repo_script() -> str:
    return (
        '  d=$(dirname "$gitdir")\n'
        '  cd "$d" || continue\n'
        '  HEAD_BEFORE=$(git -c safe.directory="$PWD" rev-parse --verify HEAD 2>/dev/null)\n'
        '  if ! git -c safe.directory="$PWD" rev-parse --verify harbor-base '
        ">/dev/null 2>&1; then\n"
        '    if git -c safe.directory="$PWD" rev-parse --verify harbor-turn-0 '
        ">/dev/null 2>&1; then\n"
        '      git -c safe.directory="$PWD" tag -f harbor-base harbor-turn-0 2>/dev/null\n'
        '    elif [ -n "$HEAD_BEFORE" ]; then\n'
        '      git -c safe.directory="$PWD" tag -f harbor-base "$HEAD_BEFORE" 2>/dev/null\n'
        "    fi\n"
        "  fi\n"
        '  if [ "$PREV_TURN" -ge 0 ] && git -c safe.directory="$PWD" rev-parse '
        '"harbor-turn-$PREV_TURN" >/dev/null 2>&1; then\n'
        '    PREV_REF="harbor-turn-$PREV_TURN"\n'
        '  elif git -c safe.directory="$PWD" rev-parse --verify harbor-base '
        ">/dev/null 2>&1; then\n"
        "    PREV_REF=harbor-base\n"
        '  elif [ -n "$HEAD_BEFORE" ]; then\n'
        '    PREV_REF="$HEAD_BEFORE"\n'
        "  else\n"
        "    PREV_REF=HEAD\n"
        "  fi\n"
        '  git -c safe.directory="$PWD" add -A 2>/dev/null\n'
        '  git -c safe.directory="$PWD" -c user.email=harbor@base '
        '-c user.name=harbor commit --allow-empty --no-verify '
        '-m "harbor-turn-$TURN" --quiet 2>/dev/null\n'
        '  git -c safe.directory="$PWD" tag -f "harbor-turn-$TURN" HEAD 2>/dev/null\n'
        '  if git -c safe.directory="$PWD" rev-parse --verify harbor-base '
        ">/dev/null 2>&1; then\n"
        '    printf "=== %s (cumulative vs harbor-base) ===\\n" "$d" '
        ">> /tmp/harbor_cum.diff\n"
        '    git -c safe.directory="$PWD" --no-pager diff harbor-base HEAD '
        "2>/dev/null >> /tmp/harbor_cum.diff\n"
        '    printf "\\n" >> /tmp/harbor_cum.diff\n'
        "  fi\n"
        '  printf "=== %s (incremental vs %s) ===\\n" "$d" "$PREV_REF" '
        ">> /tmp/harbor_inc.diff\n"
        '  git -c safe.directory="$PWD" --no-pager diff "$PREV_REF" HEAD '
        "2>/dev/null >> /tmp/harbor_inc.diff\n"
        '  printf "\\n" >> /tmp/harbor_inc.diff\n'
    )


def build_diff_command(turn: int, *, repo_roots: str | None = None) -> str:
    """Snapshot cumulative + incremental diffs for every discovered repo."""
    roots = repo_roots or DEFAULT_REPO_ROOTS
    return "".join(
        [
            "set +e\n",
            f"TURN={int(turn)}\n",
            f"PREV_TURN={int(turn) - 1}\n",
            f'ROOTS="{roots}"\n',
            'if [ -n "${HARBOR_REPO_PATHS:-}" ]; then\n',
            '  ROOTS="$ROOTS $(echo "$HARBOR_REPO_PATHS" | tr ":" " ")"\n',
            "fi\n",
            ": > /tmp/harbor_cum.diff\n",
            ": > /tmp/harbor_inc.diff\n",
            'EXISTING=""\n',
            'for r in $ROOTS; do [ -e "$r" ] && EXISTING="$EXISTING $r"; done\n',
            'if [ -n "$EXISTING" ]; then\n',
            "  REPOS=$(find $EXISTING -maxdepth 3 \\( -type d -o -type f \\) "
            "-name .git 2>/dev/null | sort -u)\n",
            "  for gitdir in $REPOS; do\n",
            _diff_per_repo_script(),
            "    cd - >/dev/null\n",
            "  done\n",
            "fi\n",
            'echo "===HARBOR_DIFF_BEGIN_CUM==="\n',
            "cat /tmp/harbor_cum.diff 2>/dev/null\n",
            'echo "===HARBOR_DIFF_END_CUM==="\n',
            'echo "===HARBOR_DIFF_BEGIN_INC==="\n',
            "cat /tmp/harbor_inc.diff 2>/dev/null\n",
            'echo "===HARBOR_DIFF_END_INC==="\n',
            "rm -f /tmp/harbor_cum.diff /tmp/harbor_inc.diff\n",
        ]
    )


def trim_diff(diff: str) -> str:
    """Drop leading/trailing empty lines without eating a trailing context line."""
    lines = diff.split("\n")
    while lines and lines[0] == "":
        lines.pop(0)
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)


def split_diff_output(raw: str) -> tuple[str, str]:
    """Split the framed diff output into ``(cumulative, incremental)``."""
    cum_lines: list[str] = []
    inc_lines: list[str] = []
    mode: str | None = None
    for line in raw.split("\n"):
        stripped = line.rstrip("\r")
        if stripped == "===HARBOR_DIFF_BEGIN_CUM===":
            mode = "cum"
            continue
        if stripped == "===HARBOR_DIFF_END_CUM===":
            mode = None
            continue
        if stripped == "===HARBOR_DIFF_BEGIN_INC===":
            mode = "inc"
            continue
        if stripped == "===HARBOR_DIFF_END_INC===":
            mode = None
            continue
        if mode == "cum":
            cum_lines.append(line)
        elif mode == "inc":
            inc_lines.append(line)
    return trim_diff("\n".join(cum_lines)), trim_diff("\n".join(inc_lines))


def strip_junk(diff: str) -> str:
    """Drop diff blocks for run-generated junk directories (``.venv``, …)."""
    if not diff:
        return diff
    kept: list[str] = []
    skip = False
    for line in diff.split("\n"):
        if line.startswith("=== ") and line.endswith(" ==="):
            skip = False
            kept.append(line)
            continue
        match = re.match(r"diff --git a/(.+?) b/", line)
        if match:
            skip = bool(_JUNK_RE.search(match.group(1)))
        if not skip:
            kept.append(line)
    return trim_diff("\n".join(kept))


def truncate_diff_for_user_sim(diff: str, limit: int = USER_SIM_DIFF_MAX_CHARS) -> str:
    """Bound the per-turn diff handed to the simulated user."""
    if len(diff) <= limit:
        return diff
    files = diff.count("\ndiff --git ") + int(diff.startswith("diff --git "))
    head = diff[:limit]
    cut = head.rfind("\n")
    if cut > 0:
        head = head[:cut]
    return (
        f"{head}\n[... diff truncated for the user simulator: {len(diff):,} chars "
        f"across {files} files; showing the first {len(head):,} chars ...]"
    )


async def tag_harbor_base(environment: Any, *, repo_roots: str | None = None) -> None:
    """Best-effort baseline tagging in every repo found in the container."""
    try:
        await environment.exec(
            command=build_tag_command(repo_roots=repo_roots),
            cwd="/",
            env={},
            timeout_sec=90,
        )
    except Exception as exc:  # pragma: no cover - defensive around sandbox APIs
        log.debug("harbor-base tagging failed (best-effort): %s", exc)


async def discover_repo_root(
    environment: Any, *, repo_roots: str | None = None
) -> str | None:
    """Return the repository root used as ``--workspace`` inside the container."""
    try:
        result = await environment.exec(
            command=build_repo_discovery_command(repo_roots=repo_roots),
            cwd="/",
            env={},
            timeout_sec=60,
        )
    except Exception as exc:  # pragma: no cover - defensive around sandbox APIs
        log.debug("repository discovery failed: %s", exc)
        return None
    matches = re.findall(r"^OLYMPUS_REPO=(.+)$", result.stdout or "", flags=re.MULTILINE)
    return matches[-1].strip() if matches else None


@dataclass(frozen=True)
class SessionEnvelopeView:
    """Parsed subset of ``olympus session``'s JSON envelope."""

    run_id: str
    status: str
    message: str
    pause_kind: str | None = None
    workflow: str = ""
    unresolved_blockers: tuple[str, ...] = ()
    changed_files: tuple[str, ...] = ()
    error: str | None = None
    raw: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def failure(cls, error: str) -> SessionEnvelopeView:
        return cls(
            run_id="",
            status="failed",
            message=error,
            error=error,
            raw={"status": "failed", "error": error},
        )


def parse_session_envelope(text: str) -> SessionEnvelopeView | None:
    """Return the last well-formed session envelope embedded in ``text``.

    ``olympus`` prints its envelope as pretty JSON, but workflow logging may add
    unrelated lines, so the parser scans every ``{`` candidate and keeps the
    last object carrying a known ``status`` value.
    """
    decoder = json.JSONDecoder()
    best: dict[str, Any] | None = None
    cursor = 0
    while True:
        start = text.find("{", cursor)
        if start < 0:
            break
        try:
            payload, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            cursor = start + 1
            continue
        if isinstance(payload, dict) and payload.get("status") in _SESSION_STATUSES:
            best = payload
        cursor = max(end, start + 1)
    if best is None:
        return None
    status = str(best["status"])
    return SessionEnvelopeView(
        run_id=str(best.get("run_id") or ""),
        status=status,
        message=str(best.get("message") or best.get("error") or ""),
        pause_kind=str(best["pause_kind"]) if best.get("pause_kind") else None,
        workflow=str(best.get("workflow") or ""),
        unresolved_blockers=tuple(str(item) for item in best.get("unresolved_blockers") or []),
        changed_files=tuple(str(item) for item in best.get("changed_files") or []),
        error=str(best["error"]) if best.get("error") is not None else None,
        raw=best,
    )


def validate_state_dir(workspace: str, state_dir: str) -> str:
    """Reject state directories nested inside the task repository workspace."""
    repo = posixpath.normpath(workspace.rstrip("/") or "/")
    state = posixpath.normpath(state_dir.rstrip("/") or "/")
    if not state.startswith("/"):
        raise ValueError(f"--state-dir {state_dir!r} must be an absolute container path")
    if state == repo or state.startswith(repo.rstrip("/") + "/"):
        raise ValueError(
            f"--state-dir {state_dir!r} is inside the task workspace {workspace!r}; "
            "session state must live outside the repository"
        )
    return state


def is_stop_decision(decision: Any) -> bool:
    """Return True when the simulated user asks the adapter to stop."""
    if bool(getattr(decision, "stop", False)):
        return True
    action = str(getattr(decision, "action", "") or "").strip().lower()
    if action in _STOP_WORDS:
        return True
    content = str(getattr(decision, "content", "") or "").strip().lower().strip(".! ")
    return content in _STOP_WORDS


def _decision_message(decision: Any) -> str:
    formatter = getattr(decision, "format_for_injection", None)
    if callable(formatter):
        return str(formatter()).strip()
    return str(getattr(decision, "content", "") or "").strip()


def _split_model_name(name: str | None) -> tuple[str | None, str | None]:
    if not name:
        return None, None
    cleaned = name.strip()
    if "/" in cleaned:
        prefix, rest = cleaned.split("/", 1)
        if prefix.lower() in _MODEL_PROVIDER_PREFIXES and rest:
            return prefix.lower(), rest
    return None, cleaned or None


def default_project_root() -> Path:
    """Repository root of this checkout (``pyproject.toml`` + ``src/olympus``)."""
    return Path(__file__).resolve().parents[3]


def _tar_filter(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
    parts = Path(info.name).parts
    if any(part in {"__pycache__", ".git"} for part in parts):
        return None
    if any(part.endswith((".pyc", ".pyo")) for part in parts):
        return None
    return info


def stage_source_archive(source_root: Path | str, archive_path: Path | str) -> Path:
    """Bundle the local Olympus source tree for upload into a task container."""
    root = Path(source_root).resolve()
    target = Path(archive_path)
    pyproject = root / "pyproject.toml"
    package = root / "src" / "olympus"
    if not pyproject.exists() or not package.is_dir():
        raise ValueError(
            f"{root} does not look like an Olympus4SWE checkout "
            "(expected pyproject.toml and src/olympus)"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(target, "w:gz") as archive:
        archive.add(pyproject, arcname="pyproject.toml", filter=_tar_filter)
        readme = root / "README.md"
        if readme.exists():
            archive.add(readme, arcname="README.md", filter=_tar_filter)
        archive.add(package, arcname="src/olympus", filter=_tar_filter)
    return target


def build_install_script(*, install_target: str, olympus_bin: str = "olympus") -> str:
    """Install Olympus (or a pip package/archive) and expose the CLI on PATH."""
    target = shlex.quote(install_target)
    binary = shlex.quote(olympus_bin)
    return "\n".join(
        [
            "set -eu",
            "mkdir -p /tmp/olympus",
            'if command -v python3 >/dev/null 2>&1; then PY=python3; '
            'elif command -v python >/dev/null 2>&1; then PY=python; '
            'else echo "python not found in task container" >&2; exit 1; fi',
            'if "$PY" -m venv /opt/olympus/venv >/dev/null 2>&1; then',
            "  PIP=/opt/olympus/venv/bin/pip",
            "  BIN_DIR=/opt/olympus/venv/bin",
            "else",
            '  PIP="$PY -m pip"',
            '  BIN_DIR=""',
            "fi",
            'if [ -n "$BIN_DIR" ]; then "$PIP" install --quiet --upgrade pip '
            ">/dev/null 2>&1 || true; fi",
            f'$PIP install --quiet {target} || '
            f"$PIP install --quiet --break-system-packages {target}",
            f'if [ -n "$BIN_DIR" ]; then ln -sf "$BIN_DIR"/{binary} '
            f"/usr/local/bin/{binary}; fi",
            'export PATH="/opt/olympus/venv/bin:$HOME/.local/bin:/usr/local/bin:$PATH"',
            f"command -v {binary}",
            f"{binary} --help >/dev/null 2>&1 || true",
        ]
    )


@dataclass(frozen=True)
class ExecOutcome:
    stdout: str
    stderr: str
    return_code: int
    timed_out: bool = False


def build_user_agent(
    *,
    model_name: str,
    api_base: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.5,
    original_user_messages: list[str] | None = None,
    session_analysis: str = "",
    max_messages: int | None = None,
) -> Any:
    """Construct SWE-Together's host-side ``UserAgent`` when the repo is present."""
    try:
        from user_agent.user_agent import UserAgent
    except ImportError as exc:  # pragma: no cover - depends on benchmark checkout
        raise RuntimeError(
            "SWE-Together's host-side UserAgent is not importable. Add the "
            "SWE-Together '<repo>/src' directory to PYTHONPATH (it provides "
            "`user_agent.user_agent`)."
        ) from exc
    try:
        from harbor.llms.lite_llm import LiteLLM
    except ImportError as exc:  # pragma: no cover - depends on Harbor install
        raise RuntimeError(
            "Harbor's LiteLLM wrapper is not importable; install laude-institute/harbor "
            "or pass `user_agent=` / `user_agent_factory=` explicitly."
        ) from exc
    llm = LiteLLM(
        model_name=model_name,
        api_base=api_base,
        api_key=api_key,
        temperature=temperature,
    )
    return UserAgent(
        llm=llm,
        original_user_messages=original_user_messages,
        session_analysis=session_analysis,
        max_messages=max_messages,
    )


_SENSITIVE_ENV_RE = re.compile(
    r"(?i)(api[_-]?key|token|secret|password|passwd|credential|authorization|auth)"
)
_URL_USERINFO_RE = re.compile(r"(?P<scheme>[a-z][a-z0-9+.-]*://)[^/@\s]+@", re.IGNORECASE)
_URL_SECRET_QUERY_RE = re.compile(
    r"(?i)([?&](?:api[_-]?key|key|token|access_token|auth|password|secret)=)[^&\s]+"
)


def redact_base_url(value: str) -> str:
    """Strip URL userinfo and sensitive query values from a persisted base URL."""
    redacted = _URL_USERINFO_RE.sub(r"\g<scheme>***@", value)
    return _URL_SECRET_QUERY_RE.sub(rf"\g<1>{_REDACTED}", redacted)


def collect_redaction_secrets(
    env: Mapping[str, str] | None = None,
    *,
    minimum_length: int = 8,
) -> frozenset[str]:
    """Values of sensitive environment variables that must not reach logs."""
    secrets: set[str] = set()
    sources: tuple[Mapping[str, str], ...] = (env or {}, os.environ)
    for source in sources:
        for key, value in source.items():
            text = str(value or "")
            if (
                text
                and len(text) >= minimum_length
                and _SENSITIVE_ENV_RE.search(str(key))
            ):
                secrets.add(text)
    return frozenset(secrets)


def redact_text(text: str, secrets: Iterable[str] = ()) -> str:
    """Redact URL credentials/tokens and known secret values from ``text``."""
    if not text:
        return text
    redacted = redact_base_url(text)
    for secret in sorted((item for item in secrets if item), key=len, reverse=True):
        redacted = redacted.replace(secret, _REDACTED)
    return redacted


def _redact_json_value(value: Any, secrets: Iterable[str]) -> Any:
    if isinstance(value, str):
        return redact_text(value, secrets)
    if isinstance(value, dict):
        return {k: _redact_json_value(v, secrets) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact_json_value(item, secrets) for item in value]
    return value


class UserEnabledOlympusAgent(BaseAgent):
    """Harbor agent wrapping ``olympus session start/reply`` with a user simulator."""

    capabilities = (
        _HarborCapabilities(base_url=True, atif=True)
        if _HarborCapabilities is not None
        else _FallbackCapabilities()
    )
    if _HarborModelConnectionSpec is not None:
        MODEL_CONNECTION = _HarborModelConnectionSpec(
            base_url_destination_names={"default": None},
            supported_api_formats=[
                _HarborModelApiFormat.OPENAI_CHAT_COMPLETIONS,
                _HarborModelApiFormat.OPENAI_RESPONSES,
                _HarborModelApiFormat.ANTHROPIC_MESSAGES,
            ],
            agent_provider_access=_HarborProviderAccess(
                api_key_candidate_names=["OPENAI_API_KEY", "ANTHROPIC_API_KEY"],
                base_url_candidate_names=["OPENAI_BASE_URL", "ANTHROPIC_BASE_URL"],
            ),
        )
    if _HarborCapabilities is None:
        SUPPORTS_ATIF: bool = True

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        *,
        # Olympus passthrough
        workflow: str = "full_sdlc_tdd",
        gate_policy: str = "interactive_minimal",
        worker_driver: str | None = None,
        reviewer_driver: str | None = None,
        monitor_model: str | None = None,
        provider: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        max_turns: int = 15,
        state_dir: str = "/tmp/olympus-session",
        workspace: str | None = None,
        repo_roots: str | None = None,
        # Container installation
        olympus_install: str = "source",
        olympus_wheel: str | None = None,
        olympus_source: str | None = None,
        olympus_bin: str = "olympus",
        pi_package: str = "@earendil-works/pi-coding-agent",
        install_pi: bool | None = None,
        extra_env: Mapping[str, str] | None = None,
        # Reactive user simulator
        user_model_name: str = "anthropic/claude-opus-4-6",
        user_api_base: str | None = None,
        user_api_key: str | None = None,
        user_api_key_ref: str | None = None,
        user_temperature: float = 0.5,
        user_context_chars: int = 3000,
        original_user_messages: list[str] | None = None,
        session_analysis: str = "",
        max_messages: int | None = None,
        call_user_on_completion: bool = True,
        user_agent: Any | None = None,
        user_agent_factory: Callable[[], Any] | None = None,
        # Budgets
        trial_budget_sec: float = 5400.0,
        per_turn_timeout_sec: float = 1800.0,
        max_consecutive_noops: int = 4,
        **kwargs: Any,
    ) -> None:
        super().__init__(logs_dir=logs_dir, model_name=model_name, **kwargs)
        self.logs_dir = Path(logs_dir)
        self.logger = (getattr(self, "logger", log) or log).getChild(__name__)

        derived_provider, derived_model = _split_model_name(model_name)
        self._workflow = str(workflow or "full_sdlc_tdd")
        self._gate_policy = str(gate_policy or "interactive_minimal")
        self._worker_driver = worker_driver.strip() if worker_driver else None
        self._reviewer_driver = reviewer_driver.strip() if reviewer_driver else None
        self._monitor_model = monitor_model.strip() if monitor_model else None
        self._provider = (provider or derived_provider or "auto").strip()
        self._model = (model or derived_model or "").strip() or None
        self._base_url = base_url.strip() if base_url else None
        self._max_turns = max(1, int(max_turns))
        self._state_dir = (state_dir or "/tmp/olympus-session").rstrip("/") or "/"
        self._workspace_override = workspace.rstrip("/") if workspace else None
        self._repo_roots = repo_roots

        install_kind = (olympus_install or "source").strip().lower()
        if install_kind not in {"source", "wheel", "pip", "none"}:
            raise ValueError(
                f"olympus_install={olympus_install!r}; expected source, wheel, pip, or none"
            )
        self._olympus_install = install_kind
        self._olympus_wheel = olympus_wheel
        self._olympus_source = olympus_source
        self._olympus_bin = olympus_bin.strip() or "olympus"
        self._pi_package = pi_package.strip()
        driver_name = (self._worker_driver or "").split("@", 1)[0].strip().lower()
        self._needs_pi = bool(install_pi) if install_pi is not None else driver_name == "pi"
        self._extra_env = {str(key): str(value) for key, value in (extra_env or {}).items()}
        self._user_agent = user_agent
        self._user_agent_factory = user_agent_factory
        self._user_model_name = user_model_name
        self._user_api_base = user_api_base
        self._user_api_key = _resolve_user_api_key(
            user_api_key=user_api_key, user_api_key_ref=user_api_key_ref
        )
        redaction_env = dict(self._extra_env)
        if self._user_api_key:
            redaction_env["USER_SIM_API_KEY"] = self._user_api_key
        self._redaction_secrets = collect_redaction_secrets(redaction_env)
        self._user_temperature = user_temperature
        self._user_context_chars = max(500, int(user_context_chars))
        self._original_user_messages = list(original_user_messages or [])
        self._session_analysis = session_analysis
        self._max_messages = max_messages
        self._check_on_completion = bool(call_user_on_completion)

        self._trial_budget_sec = float(trial_budget_sec)
        self._per_turn_timeout_sec = float(per_turn_timeout_sec)
        self._max_consecutive_noops = max(1, int(max_consecutive_noops))

        self._task_instruction = ""
        self._discovered_repo: str | None = None
        self._repo: str | None = None
        self._run_id = ""
        self._start_time = 0.0
        self._turn_start_time = 0.0
        self._last_turn_diff = ""
        self._last_cumulative_diff = ""
        self._conversation: list[dict[str, Any]] = []
        self._sim_user: Any | None = None

    @staticmethod
    def name() -> str:
        return "user-enabled-olympus"

    def version(self) -> str | None:
        try:
            from importlib.metadata import version

            return version("olympus4swe")
        except Exception:  # noqa: BLE001 - version reporting must never fail a trial
            return "unknown"


    async def setup(self, environment: Any) -> None:
        """Install Olympus (and optional Pi) into the task container, tag base."""
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        install = await self._install_olympus(environment)
        (self.logs_dir / "setup-install.txt").write_text(
            f"return_code={install.return_code}\n"
            f"timed_out={install.timed_out}\n"
            f"stdout:\n{redact_text(install.stdout, self._redaction_secrets)}\n\n"
            f"stderr:\n{redact_text(install.stderr, self._redaction_secrets)}\n",
            encoding="utf-8",
        )
        if install.return_code != 0:
            tail = (install.stderr or install.stdout or "").strip()[-2000:]
            raise RuntimeError(
                f"failed to install olympus in the task container "
                f"(exit {install.return_code}): {tail}"
            )
        resolved = _resolved_binary(install.stdout)
        if resolved:
            self._olympus_bin = resolved
        if self._needs_pi:
            await self._install_pi(environment)
        await tag_harbor_base(environment, repo_roots=self._repo_roots)
        self._discovered_repo = await discover_repo_root(
            environment, repo_roots=self._repo_roots
        )

    async def _install_olympus(self, environment: Any) -> ExecOutcome:
        kind = self._olympus_install
        if kind == "none":
            return await self._exec(
                environment,
                f"command -v {shlex.quote(self._olympus_bin)}",
                cwd="/",
                timeout_sec=60,
                env={},
            )
        if kind == "pip":
            script = build_install_script(
                install_target="olympus4swe", olympus_bin=self._olympus_bin
            )
            return await self._exec(environment, script, cwd="/", timeout_sec=900, env={})
        if kind == "wheel":
            if not self._olympus_wheel:
                raise ValueError("olympus_install='wheel' requires olympus_wheel=<path>")
            wheel = Path(self._olympus_wheel).expanduser().resolve()
            if not wheel.is_file():
                raise FileNotFoundError(f"Olympus wheel not found: {wheel}")
            await self._exec(environment, "mkdir -p /tmp/olympus", cwd="/", timeout_sec=30, env={})
            target = f"/tmp/olympus/{wheel.name}"
            await environment.upload_file(source_path=wheel, target_path=target)
        else:
            source_root = (
                Path(self._olympus_source).expanduser().resolve()
                if self._olympus_source
                else default_project_root()
            )
            archive = stage_source_archive(source_root, self.logs_dir / "olympus-src.tar.gz")
            await self._exec(environment, "mkdir -p /tmp/olympus", cwd="/", timeout_sec=30, env={})
            target = "/tmp/olympus/olympus-src.tar.gz"
            await environment.upload_file(source_path=archive, target_path=target)
        script = build_install_script(install_target=target, olympus_bin=self._olympus_bin)
        return await self._exec(environment, script, cwd="/", timeout_sec=900, env={})

    async def _install_pi(self, environment: Any) -> None:
        script = "\n".join(
            [
                "set -eu",
                "if command -v pi >/dev/null 2>&1; then command -v pi; exit 0; fi",
                "if ! command -v npm >/dev/null 2>&1; then "
                'echo "npm unavailable; Pi CLI not installed" >&2; exit 0; fi',
                f"npm install -g --ignore-scripts {shlex.quote(self._pi_package)}",
                "command -v pi",
            ]
        )
        result = await self._exec(environment, script, cwd="/", timeout_sec=900, env={})
        (self.logs_dir / "setup-pi.txt").write_text(
            f"return_code={result.return_code}\n"
            f"stdout:\n{redact_text(result.stdout, self._redaction_secrets)}\n\n"
            f"stderr:\n{redact_text(result.stderr, self._redaction_secrets)}\n",
            encoding="utf-8",
        )
        if result.return_code != 0:
            log.warning(
                "Pi CLI installation failed (worker_driver=%s): %s",
                self._worker_driver,
                (result.stderr or result.stdout).strip()[-1000:],
            )


    async def run(self, instruction: str, environment: Any, context: Any) -> None:
        """Drive the multi-turn Olympus session against the simulated user."""
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self._task_instruction = (instruction or "").strip()
        if not self._task_instruction:
            raise ValueError("UserEnabledOlympusAgent requires a non-empty instruction")
        repo = (self._workspace_override or self._discovered_repo or "/workspace").rstrip("/")
        repo = repo or "/"
        self._state_dir = validate_state_dir(repo, self._state_dir)
        self._repo = repo
        self._sim_user = self._resolve_user_agent()

        self._start_time = time.monotonic()
        self._turn_start_time = self._start_time
        self._record_message(
            role="user", content=self._task_instruction, turn=0, kind="initial_request"
        )

        envelope: SessionEnvelopeView | None = None
        final_status = "failed"
        turns = 0
        try:
            envelope = await self._execute_turn(
                environment, turn=0, action="start", message=self._task_instruction
            )
            final_status = envelope.status
            consecutive_noops = 0
            turns_max = 0 if envelope.status == "failed" else self._max_turns
            for turn in range(1, turns_max + 1):
                if self._budget_exhausted():
                    self._record_event(turn, "trial_budget_exhausted")
                    break
                completing = envelope.status == "completed"
                if completing and not self._check_on_completion:
                    break
                decision = await self._consult_user(
                    turn=turn, envelope=envelope, completing=completing
                )
                if is_stop_decision(decision):
                    self._record_event(turn, "sim_user_stop")
                    break
                if bool(getattr(decision, "has_message", False)):
                    message = _decision_message(decision)
                    if not message:
                        break
                    consecutive_noops = 0
                    with suppress(Exception):
                        self._sim_user.advance_original_index(1)
                else:
                    consecutive_noops += 1
                    if not completing:
                        # A blocked session cannot progress without user input.
                        self._record_event(turn, "sim_user_silent_on_pause")
                        break
                    if consecutive_noops >= self._max_consecutive_noops:
                        self._record_event(turn, "noop_budget_exhausted")
                        break
                    message = "continue"
                self._record_message(
                    role="user", content=message, turn=turn, kind="simulated_user"
                )
                self._turn_start_time = time.monotonic()
                envelope = await self._execute_turn(
                    environment, turn=turn, action="reply", message=message
                )
                final_status = envelope.status
                turns = turn
                if envelope.status == "failed":
                    break
        finally:
            with suppress(Exception):
                await self._capture_git_diff(environment, turn=999)
            if envelope is not None:
                self._run_id = envelope.run_id or self._run_id
            self._write_atif(final_status=final_status, turns=turns)
            self._populate_context(context, final_status=final_status, turns=turns)


    def _session_command(self, action: str, *, message: str) -> str:
        args = [self._olympus_bin, "session", action]
        if action == "reply":
            args.append(self._run_id)
        args.extend(["--message", message, "--workspace", self._repo or "/"])
        args.extend(["--state-dir", self._state_dir])
        for flag, value in (
            ("--workflow", self._workflow),
            ("--gate-policy", self._gate_policy),
            ("--provider", self._provider),
            ("--model", self._model),
            ("--base-url", self._base_url),
            ("--worker-driver", self._worker_driver),
            ("--reviewer-driver", self._reviewer_driver),
            ("--monitor-model", self._monitor_model),
        ):
            if value:
                args.extend([flag, value])
        return f"set -o pipefail; {shlex.join(args)}"

    async def _execute_turn(
        self,
        environment: Any,
        *,
        turn: int,
        action: str,
        message: str,
    ) -> SessionEnvelopeView:
        command = self._session_command(action, message=message)
        outcome = await self._exec(
            environment,
            command,
            cwd=self._repo or "/",
            timeout_sec=self._turn_timeout(),
        )
        self._write_turn_artifacts(turn, command, outcome)
        envelope = parse_session_envelope(outcome.stdout)
        if envelope is None:
            detail = (outcome.stderr or outcome.stdout or "").strip()
            if outcome.timed_out:
                detail = detail or "olympus session timed out"
            envelope = SessionEnvelopeView.failure(
                detail[:2000] or "olympus session produced no JSON envelope"
            )
        self._run_id = envelope.run_id or self._run_id
        await self._capture_git_diff(environment, turn=turn)
        self._record_message(
            role="agent",
            content=envelope.message,
            turn=turn,
            status=envelope.status,
            pause_kind=envelope.pause_kind,
        )
        (self.logs_dir / f"envelope-{turn}.json").write_text(
            json.dumps(
                _redact_json_value(dict(envelope.raw), self._redaction_secrets),
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        return envelope

    async def _exec(
        self,
        environment: Any,
        command: str,
        *,
        cwd: str,
        timeout_sec: float,
        env: Mapping[str, str] | None = None,
    ) -> ExecOutcome:
        capped = max(1, int(timeout_sec))
        try:
            result = await asyncio.wait_for(
                environment.exec(
                    command=command,
                    cwd=cwd,
                    env=dict(self._extra_env if env is None else env),
                    timeout_sec=capped,
                ),
                timeout=float(capped) + 30.0,
            )
        except TimeoutError:
            return ExecOutcome(
                stdout="",
                stderr=f"[olympus] exec capped at {capped}s — abandoning call",
                return_code=-1,
                timed_out=True,
            )
        return ExecOutcome(
            stdout=str(getattr(result, "stdout", "") or ""),
            stderr=str(getattr(result, "stderr", "") or ""),
            return_code=int(getattr(result, "return_code", 1)),
        )

    def _turn_timeout(self) -> float:
        remaining = self._trial_budget_sec - (time.monotonic() - self._start_time)
        return max(1.0, min(self._per_turn_timeout_sec, remaining))

    def _budget_exhausted(self) -> bool:
        return (time.monotonic() - self._start_time) >= self._trial_budget_sec


    def _resolve_user_agent(self) -> Any:
        if self._user_agent is not None:
            return self._user_agent
        if self._user_agent_factory is not None:
            return self._user_agent_factory()
        return build_user_agent(
            model_name=self._user_model_name,
            api_base=self._user_api_base,
            api_key=self._user_api_key,
            temperature=self._user_temperature,
            original_user_messages=self._original_user_messages,
            session_analysis=self._session_analysis,
            max_messages=self._max_messages,
        )

    async def _consult_user(
        self, *, turn: int, envelope: SessionEnvelopeView, completing: bool
    ) -> Any:
        now = time.monotonic()
        elapsed = now - self._start_time
        turn_duration = now - self._turn_start_time
        decision = await self._sim_user.process(
            task_description=self._task_instruction,
            # Fidelity: only natural-language prose is surfaced. Internal tool
            # arguments/results never existed for a human user to see.
            recent_trajectory="",
            latest_observation=envelope.message,
            latest_analysis=None,
            step_count=turn,
            is_completion_attempt=completing,
            total_steps_so_far=turn,
            elapsed_sec=elapsed,
            turn_duration_sec=turn_duration,
            code_changes_diff=self._last_turn_diff,
        )
        self._log_user_decision(turn, completing, decision)
        return decision

    def _log_user_decision(self, turn: int, completing: bool, decision: Any) -> None:
        directory = self.logs_dir / f"episode-{turn}"
        directory.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": _utc_now(),
            "turn": turn,
            "is_completion_attempt": completing,
            "action": str(getattr(decision, "action", "") or ""),
            "has_message": bool(getattr(decision, "has_message", False)),
            "content": redact_text(
                str(getattr(decision, "content", "") or ""), self._redaction_secrets
            ),
            "raw_response": redact_text(
                str(getattr(decision, "raw_response", "") or "")[:500],
                self._redaction_secrets,
            ),
        }
        stats_getter = getattr(self._sim_user, "get_stats", None)
        if callable(stats_getter):
            with suppress(Exception):
                record["stats"] = stats_getter()
        (directory / "user_decision.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )


    async def _capture_git_diff(self, environment: Any, *, turn: int) -> None:
        try:
            result = await environment.exec(
                command=build_diff_command(turn, repo_roots=self._repo_roots),
                cwd="/",
                env={},
                timeout_sec=90,
            )
        except Exception as exc:  # pragma: no cover - defensive around sandbox APIs
            log.debug("git diff capture failed at turn %d: %s", turn, exc)
            return
        raw = str(getattr(result, "stdout", "") or "")
        if not raw.strip():
            return
        cumulative, incremental = split_diff_output(raw)
        cumulative = strip_junk(cumulative)
        incremental = strip_junk(incremental)

        patches_dir = self.logs_dir / "patches"
        if cumulative:
            patches_dir.mkdir(parents=True, exist_ok=True)
            (patches_dir / f"turn-{turn}.patch").write_text(
                cumulative + "\n", encoding="utf-8"
            )
            (self.logs_dir / "final.patch").write_text(cumulative + "\n", encoding="utf-8")
            files = cumulative.count("diff --git a/")
            if files > 300:
                log.warning(
                    "turn %d cumulative diff spans %d files — likely diverged "
                    "harbor-base baseline; flagging",
                    turn,
                    files,
                )
                (self.logs_dir / "diff_polluted.flag").write_text(
                    f"{files}\n", encoding="utf-8"
                )
        if incremental:
            patches_dir.mkdir(parents=True, exist_ok=True)
            (patches_dir / f"turn-{turn}.incremental.patch").write_text(
                incremental + "\n", encoding="utf-8"
            )
        self._last_turn_diff = truncate_diff_for_user_sim(incremental)
        self._last_cumulative_diff = cumulative


    def _write_turn_artifacts(self, turn: int, command: str, outcome: ExecOutcome) -> None:
        directory = self.logs_dir / f"command-{turn}"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "command.txt").write_text(
            redact_text(command, self._redaction_secrets) + "\n", encoding="utf-8"
        )
        (directory / "return-code.txt").write_text(
            str(outcome.return_code), encoding="utf-8"
        )
        if outcome.timed_out:
            (directory / "timed-out.txt").write_text("1\n", encoding="utf-8")
        if outcome.stdout:
            (directory / "stdout.txt").write_text(
                redact_text(outcome.stdout, self._redaction_secrets), encoding="utf-8"
            )
        if outcome.stderr:
            (directory / "stderr.txt").write_text(
                redact_text(outcome.stderr, self._redaction_secrets), encoding="utf-8"
            )

    def _record_message(
        self,
        *,
        role: str,
        content: str,
        turn: int,
        kind: str | None = None,
        status: str | None = None,
        pause_kind: str | None = None,
    ) -> None:
        record: dict[str, Any] = {
            "timestamp": _utc_now(),
            "turn": turn,
            "role": role,
            "content": redact_text(content, self._redaction_secrets),
        }
        if kind:
            record["kind"] = kind
        if status:
            record["status"] = status
        if pause_kind:
            record["pause_kind"] = pause_kind
        self._conversation.append(record)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        with (self.logs_dir / "conversation.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _record_event(self, turn: int, event: str) -> None:
        record = {"timestamp": _utc_now(), "turn": turn, "event": event}
        with (self.logs_dir / "conversation.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _write_atif(self, *, final_status: str, turns: int) -> None:
        steps: list[dict[str, Any]] = []
        for record in self._conversation:
            step: dict[str, Any] = {
                "step_id": len(steps) + 1,
                "timestamp": record["timestamp"],
                "source": record["role"],
                "message": record["content"] or "(no message)",
            }
            if record["role"] == "agent":
                step["model_name"] = self._model or self.model_name
                step["extra"] = {
                    "turn": record["turn"],
                    "status": record.get("status"),
                    "pause_kind": record.get("pause_kind"),
                }
            steps.append(step)
        if not steps:
            steps.append(
                {
                    "step_id": 1,
                    "source": "system",
                    "message": "olympus session produced no conversation steps",
                }
            )
        trajectory = {
            "schema_version": "ATIF-v1.6",
            "session_id": self._run_id or f"olympus-{self.name()}",
            "agent": {
                "name": self.name(),
                "version": self.version() or "unknown",
                "model_name": self._model or self.model_name,
            },
            "steps": steps,
            "notes": (
                "Olympus headless-session adapter: simulated users see only the "
                "assistant's natural-language message and the per-turn git diff."
            ),
            "final_metrics": {
                "total_steps": len(steps),
                "extra": {"turns": turns, "final_status": final_status},
            },
        }
        (self.logs_dir / "trajectory.json").write_text(
            json.dumps(trajectory, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    def _populate_context(
        self, context: Any, *, final_status: str, turns: int
    ) -> None:
        metadata = {
            "olympus_run_id": self._run_id,
            "olympus_status": final_status,
            "olympus_turns": turns,
            "olympus_state_dir": self._state_dir,
            "olympus_workspace": self._repo,
            "olympus_final_patch": str(self.logs_dir / "final.patch"),
            "olympus_changed_files": [
                line[3:].strip()
                for line in (self._last_cumulative_diff or "").splitlines()
                if line.startswith("diff --git ")
            ],
        }
        try:
            context.metadata = metadata
        except Exception:  # pragma: no cover - defensive around foreign contexts
            log.debug("could not attach Olympus metadata to AgentContext")


_SYSTEM_USER_MESSAGE_PREFIXES = (
    "[Request interrupted",
    "<local-command-caveat>",
    "<command-name>",
    "<command-message>",
    "<command-args>",
    "<local-command-stdout>",
    "<task-",
    "Base directory",
)


@dataclass
class _FallbackTaskConfig:
    path: Path


@dataclass
class _FallbackEnvironmentConfig:
    type: str | None = None
    delete: bool = True
    force_build: bool = False


@dataclass
class _FallbackAgentConfig:
    import_path: str | None = None
    model_name: str | None = None
    base_url: str | None = None
    override_timeout_sec: float | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)


@dataclass
class _FallbackTrialConfig:
    task: _FallbackTaskConfig
    trials_dir: Path
    agent: _FallbackAgentConfig
    environment: _FallbackEnvironmentConfig
    trial_name: str = ""


def _harbor_config_classes() -> tuple[Any, Any, Any, Any, Any]:
    try:
        from harbor.models import environment_type as harbor_environment_type
        from harbor.models.trial import config as harbor_trial_config
    except ImportError:
        return (
            _FallbackAgentConfig,
            _FallbackEnvironmentConfig,
            _FallbackTaskConfig,
            _FallbackTrialConfig,
            None,
        )
    return (
        harbor_trial_config.AgentConfig,
        harbor_trial_config.EnvironmentConfig,
        harbor_trial_config.TaskConfig,
        harbor_trial_config.TrialConfig,
        harbor_environment_type.EnvironmentType,
    )


def _resolve_environment_type(
    environment: str | None, environment_enum: Any
) -> Any:
    value = (environment or os.environ.get("SWT_SANDBOX") or "docker").strip().lower()
    if environment_enum is None:
        return value
    try:
        return environment_enum(value)
    except ValueError as exc:
        allowed = ", ".join(item.value for item in environment_enum)
        raise ValueError(
            f"environment={value!r}; expected one of {allowed}"
        ) from exc


def _load_task_user_messages(task_dir: Path) -> list[str]:
    analysis_path = task_dir / "analysis.json"
    if analysis_path.is_file():
        analysis: dict[str, Any] = {}
        with suppress(OSError, json.JSONDecodeError):
            loaded = json.loads(analysis_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                analysis = loaded
        candidates = analysis.get("user_messages")
        if isinstance(candidates, list):
            return [
                message
                for message in candidates
                if isinstance(message, str)
                and not message.startswith("[Request interrupted")
            ]
    session_path = task_dir / "original_session.json"
    if not session_path.is_file():
        return []
    try:
        session = json.loads(session_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    messages = session.get("messages") if isinstance(session, dict) else None
    if not isinstance(messages, list):
        return []
    return [
        message.get("content", "")
        for message in messages
        if isinstance(message, dict)
        and message.get("role") == "user"
        and isinstance(message.get("content"), str)
        and not message["content"].startswith(_SYSTEM_USER_MESSAGE_PREFIXES)
    ]


def _load_task_session_analysis(task_dir: Path, message_count: int) -> str:
    prompt_path = task_dir / "user_simulation_prompt.md"
    analysis = (
        prompt_path.read_text(encoding="utf-8") if prompt_path.is_file() else ""
    )
    if message_count <= 0:
        return analysis
    low = max(1, (message_count + 1) // 2)
    high = max(low + 1, (message_count * 3 + 1) // 2)
    return analysis + (
        "\n\n## Message Guidance (auto-generated)\n"
        f"The real user sent {message_count} messages in the original session. "
        f"Aim for **{low}–{high} messages** total. This is a soft target — send "
        "fewer if the agent handles everything well, send more if it needs "
        "correction. Do NOT treat any cap in the session analysis above as a hard "
        "limit; use this range instead.\n"
    )


def _task_agent_kwargs(task_dir: Path) -> dict[str, Any]:
    task_toml = task_dir / "task.toml"
    if not task_toml.is_file():
        return {}
    try:
        config = tomllib.loads(task_toml.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return {}
    agent_section = config.get("agent") if isinstance(config, dict) else None
    kwargs = agent_section.get("kwargs") if isinstance(agent_section, dict) else None
    return dict(kwargs) if isinstance(kwargs, dict) else {}


def build_olympus_trial_config(
    task_dir: Path | str,
    *,
    model: str,
    user_model: str | None = None,
    user_api_key: str | None = None,
    user_api_base: str | None = None,
    agent_env: Mapping[str, str] | None = None,
    trials_dir: Path | str | None = None,
    swe_together_root: Path | str | None = None,
    environment: str | None = None,
    agent_timeout: float | None = None,
    user_context_chars: int = 3000,
    call_user_on_completion: bool = True,
    user_temperature: float | None = 0.5,
    force_build: bool = False,
    base_url: str | None = None,
    agent_kwargs: Mapping[str, Any] | None = None,
) -> Any:
    """Build a Harbor ``TrialConfig`` that runs Olympus with SWE-Together's user sim."""
    task = Path(task_dir).expanduser().resolve()
    if not (task / "task.toml").is_file():
        raise FileNotFoundError(f"not a SWE-Together task directory: {task}")
    root = (
        Path(swe_together_root).expanduser().resolve()
        if swe_together_root
        else task.parent.parent
    )
    original_messages = _load_task_user_messages(task)
    sim_kwargs: dict[str, Any] = {
        "user_model_name": user_model or model,
        "user_api_base": user_api_base,
        "original_user_messages": original_messages,
        "session_analysis": _load_task_session_analysis(task, len(original_messages)),
        "max_messages": None,
        "user_context_chars": user_context_chars,
        "call_user_on_completion": call_user_on_completion,
    }
    if user_api_key:
        sim_kwargs["user_api_key_ref"] = _register_user_api_key(user_api_key)
    if user_temperature != 0.5:
        sim_kwargs["user_temperature"] = user_temperature
    if base_url is not None:
        sim_kwargs["base_url"] = base_url
    sim_kwargs.update(_task_agent_kwargs(task))
    sim_kwargs.update(dict(agent_kwargs or {}))

    (
        AgentConfigClass,
        EnvironmentConfigClass,
        TaskConfigClass,
        TrialConfigClass,
        EnvironmentType,
    ) = _harbor_config_classes()
    env_type = _resolve_environment_type(environment, EnvironmentType)
    agent_config = AgentConfigClass(
        import_path=OLYMPUS_AGENT_IMPORT_PATH,
        model_name=model,
        base_url=base_url,
        override_timeout_sec=agent_timeout,
        kwargs=sim_kwargs,
        env=dict(agent_env or {}),
    )
    environment_config = EnvironmentConfigClass(
        type=env_type,
        delete=True,
        force_build=force_build,
    )
    return TrialConfigClass(
        task=TaskConfigClass(path=task),
        trials_dir=(
            Path(trials_dir).expanduser().resolve() if trials_dir else root / "trials"
        ),
        agent=agent_config,
        environment=environment_config,
    )


def discover_olympus_tasks(swe_together_root: Path | str) -> list[Path]:
    """Return runnable SWE-Together task dirs (task.toml + instruction + test.sh)."""
    tasks_dir = Path(swe_together_root).expanduser().resolve() / "tasks"
    if not tasks_dir.is_dir():
        return []
    return sorted(
        entry
        for entry in tasks_dir.iterdir()
        if entry.is_dir()
        and (entry / "task.toml").is_file()
        and (entry / "instruction.md").is_file()
        and (entry / "tests" / "test.sh").is_file()
    )


def _select_task_dirs(
    root: Path,
    tasks: str | None,
    explicit: Iterable[str] | None,
) -> list[Path]:
    if explicit:
        selected = [Path(item).expanduser().resolve() for item in explicit]
    elif tasks:
        selected = [
            (root / "tasks" / name.strip()).resolve()
            for name in tasks.split(",")
            if name.strip()
        ]
    else:
        selected = discover_olympus_tasks(root)
    missing = [path for path in selected if not (path / "task.toml").is_file()]
    if missing:
        raise SystemExit(
            "task directories not found: " + ", ".join(str(path) for path in missing)
        )
    return selected


async def launch_olympus_trials(
    trial_configs: Iterable[Any], *, workers: int = 1
) -> list[Any]:
    """Run Olympus trials through Harbor's ``LocalOrchestrator``."""
    try:
        from harbor.orchestrators.local import LocalOrchestrator
    except ImportError as exc:
        raise RuntimeError(
            "harbor.orchestrators.local is unavailable; install a Harbor build that "
            "ships LocalOrchestrator (SWE-Together's external/harbor) to launch trials"
        ) from exc
    orchestrator = LocalOrchestrator(
        trial_configs=list(trial_configs),
        n_concurrent_trials=max(1, int(workers)),
        metrics={},
        quiet=True,
    )
    return await orchestrator.run()


def _trial_reward(result: Any) -> float | None:
    verifier = getattr(result, "verifier_result", None)
    rewards = getattr(verifier, "rewards", None) if verifier is not None else None
    if isinstance(rewards, dict):
        rewards = rewards.get("reward")
    if rewards is None:
        return None
    try:
        return float(rewards)
    except (TypeError, ValueError):
        return None


def _trial_error(result: Any) -> str | None:
    info = getattr(result, "exception_info", None)
    return getattr(info, "exception_type", None) if info is not None else None


def build_launcher_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Launch SWE-Together tasks with Olympus as the Harbor agent "
            "(no dependency on run_eval.py's Claude Code default)."
        )
    )
    parser.add_argument(
        "--swe-together-root",
        required=True,
        type=Path,
        help="SWE-Together checkout providing tasks/ and the user simulator.",
    )
    parser.add_argument("--tasks", help="Comma-separated task names (default: all).")
    parser.add_argument(
        "--task-dir",
        action="append",
        default=None,
        help="Explicit task directory (repeatable; overrides --tasks).",
    )
    parser.add_argument("--model", required=True, help="Agent model, e.g. openai/gpt-4o.")
    parser.add_argument("--user-model", default=None, help="Simulated-user model.")
    parser.add_argument("--user-api-key", default=None, help="Simulated-user API key.")
    parser.add_argument("--user-api-base", default=None, help="Simulated-user API base.")
    parser.add_argument(
        "--base-url",
        default=None,
        help="Inference endpoint override passed to Olympus (AgentConfig.base_url).",
    )
    parser.add_argument(
        "--trials-dir",
        type=Path,
        default=None,
        help="Harbor trials root (default: <swe-together-root>/trials).",
    )
    parser.add_argument(
        "--environment",
        default=None,
        help="Harbor environment type (default: $SWT_SANDBOX, else docker).",
    )
    parser.add_argument("--agent-timeout", type=float, default=None)
    parser.add_argument("--user-context-chars", type=int, default=3000)
    parser.add_argument("--user-temperature", type=float, default=0.5)
    parser.add_argument(
        "--no-call-user-on-completion",
        action="store_true",
        help="Stop after the first completed session instead of asking the user sim.",
    )
    parser.add_argument("--force-build", action="store_true")
    parser.add_argument(
        "--agent-kwargs",
        default=None,
        help="JSON object of extra UserEnabledOlympusAgent kwargs.",
    )
    parser.add_argument(
        "--agent-env",
        default=None,
        help="JSON object of environment variables for Olympus inside the container.",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true", help="Print configs and exit.")
    parser.add_argument("--json", action="store_true", help="Emit JSON output.")
    return parser


def _trial_config_payload(task: Path, config: Any) -> dict[str, Any]:
    env_type = getattr(config.environment, "type", None)
    if hasattr(env_type, "value"):
        env_type = env_type.value
    return {
        "task_dir": str(task),
        "import_path": config.agent.import_path,
        "model_name": config.agent.model_name,
        "trials_dir": str(config.trials_dir),
        "environment": env_type,
    }


def main(argv: Iterable[str] | None = None) -> int:
    args = build_launcher_parser().parse_args(list(argv) if argv is not None else None)
    root = args.swe_together_root.expanduser().resolve()
    task_dirs = _select_task_dirs(root, args.tasks, args.task_dir)
    if not task_dirs:
        raise SystemExit(f"no runnable tasks found under {root / 'tasks'}")
    extra_kwargs = json.loads(args.agent_kwargs) if args.agent_kwargs else {}
    if not isinstance(extra_kwargs, dict):
        raise SystemExit("--agent-kwargs must be a JSON object")
    agent_env = json.loads(args.agent_env) if args.agent_env else {}
    if not isinstance(agent_env, dict):
        raise SystemExit("--agent-env must be a JSON object")
    configs = [
        build_olympus_trial_config(
            task,
            model=args.model,
            user_model=args.user_model,
            user_api_key=args.user_api_key,
            user_api_base=args.user_api_base,
            agent_env=agent_env,
            trials_dir=args.trials_dir,
            swe_together_root=root,
            environment=args.environment,
            agent_timeout=args.agent_timeout,
            user_context_chars=args.user_context_chars,
            call_user_on_completion=not args.no_call_user_on_completion,
            user_temperature=args.user_temperature,
            force_build=args.force_build,
            base_url=args.base_url,
            agent_kwargs=extra_kwargs,
        )
        for task in task_dirs
    ]
    if args.dry_run:
        payload = [
            _trial_config_payload(task, config)
            for task, config in zip(task_dirs, configs, strict=True)
        ]
        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            for item in payload:
                print(
                    f"{item['task_dir']}  ->  {item['import_path']}  "
                    f"model={item['model_name']}  env={item['environment']}"
                )
        return 0
    results = asyncio.run(launch_olympus_trials(configs, workers=args.workers))
    payload = [
        {
            "task": getattr(result, "task_name", None),
            "trial": getattr(result, "trial_name", None),
            "reward": _trial_reward(result),
            "error": _trial_error(result),
        }
        for result in results
    ]
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        for item in payload:
            print(
                f"{str(item['reward']):>6}  {item['task']}  {item['trial']}"
                + (f"  error={item['error']}" if item["error"] else "")
            )
    return 0 if all(item["error"] is None for item in payload) else 1


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _resolved_binary(stdout: str) -> str | None:
    for line in reversed([item.strip() for item in stdout.splitlines() if item.strip()]):
        if line.startswith("/"):
            return line
    return None


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DEFAULT_REPO_ROOTS",
    "HARBOR_AVAILABLE",
    "OLYMPUS_AGENT_IMPORT_PATH",
    "SessionEnvelopeView",
    "UserEnabledOlympusAgent",
    "build_diff_command",
    "build_install_script",
    "build_launcher_parser",
    "build_olympus_trial_config",
    "build_repo_discovery_command",
    "build_tag_command",
    "build_user_agent",
    "collect_redaction_secrets",
    "default_project_root",
    "discover_olympus_tasks",
    "discover_repo_root",
    "is_stop_decision",
    "launch_olympus_trials",
    "main",
    "parse_session_envelope",
    "redact_base_url",
    "redact_text",
    "split_diff_output",
    "stage_source_archive",
    "strip_junk",
    "tag_harbor_base",
    "truncate_diff_for_user_sim",
    "validate_state_dir",
]
