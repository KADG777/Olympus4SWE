"""Harbor agent adapter bridging SWE-Interact multi-step tasks to ``olympus session``.

Harbor imports this module from the host via
``--agent-import-path tests.evals.benchmarks.harbor_interact:OlympusInteractAgent``.
It must therefore also stay importable without Harbor installed so the core
pytest environment can exercise the adapter with a fake environment.

The adapter never touches core ``olympus`` modules: installation, the session
bridge, and the FastMCP client all live under ``tests/evals/benchmarks``.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shlex
from dataclasses import dataclass, fields, replace
from pathlib import Path, PurePosixPath
from typing import Any, ClassVar

try:  # Harbor is optional; only present in the benchmark operator environment.
    from harbor.agents.base import BaseAgent as _HarborAgentBase
    from harbor.agents.capabilities import AgentCapabilities as _HarborCapabilities
    from harbor.environments.base import ExecResult as _HarborExecResult
except Exception:  # pragma: no cover - the fallback branch runs in core pytest
    _HarborAgentBase = None
    _HarborCapabilities = None
    _HarborExecResult = None

try:
    from .interact_bridge import DEFAULT_INITIAL_QUESTION, redact_text
except ImportError:  # Imported as a top-level module by standalone tooling.
    from interact_bridge import DEFAULT_INITIAL_QUESTION, redact_text

VERSION = "0.1.0"
BRIDGE_DIR = "/tmp/olympus-bridge"
PI_PREFIX = "/tmp/pi-cli"
_PLAN_ONLY_ROLE = "plan"
_EXECUTE_ROLE = "execute"
_HANDOFF_ROLE = "handoff"
_VALID_ROLES = frozenset({_PLAN_ONLY_ROLE, _EXECUTE_ROLE, _HANDOFF_ROLE})
_HANDOFF_MARKERS = ("handoff step", "do not make intentional implementation changes")
_PLAN_MARKERS = ("agreeing on the plan", "do not make implementation edits")
_FENCED_BLOCK = re.compile(r"```(?:bash|sh|shell)?[ \t]*\n(.*?)```", re.DOTALL)
_BRIDGE_SCRIPTS = ("fastmcp_client.py", "interact_bridge.py")


@dataclass
class _FallbackExecResult:
    stdout: str | None = None
    stderr: str | None = None
    return_code: int = 0


@dataclass(frozen=True)
class _FallbackCapabilities:
    base_url: bool = True
    mcp_servers: bool = True


ExecResult = _HarborExecResult or _FallbackExecResult
_AgentBase = _HarborAgentBase or None


class _FallbackAgent:
    """Minimal stand-in matching the ``BaseAgent`` surface this adapter uses."""

    capabilities = _FallbackCapabilities()
    DEFAULT_SETUP_TIMEOUT_SEC: ClassVar[float] = 900.0
    environment_logs_dir: PurePosixPath = PurePosixPath("/logs/agent")

    def __init__(
        self,
        logs_dir: str | Path | None = None,
        model_name: str | None = None,
        logger: logging.Logger | None = None,
        mcp_servers: Any = None,
        skills_dir: str | None = None,
        extra_env: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        self.logs_dir = Path(logs_dir) if logs_dir else Path.cwd() / "logs"
        self.environment_logs_dir = PurePosixPath("/logs/agent")
        self.model_name = model_name
        self.logger = logger or logging.getLogger("olympus.interact")
        self.mcp_servers = mcp_servers or []
        self.skills_dir = skills_dir
        self._extra_env = dict(extra_env or {})
        self._model_connection_kwargs = dict(kwargs)

    @property
    def extra_env(self) -> dict[str, str]:
        return dict(self._extra_env)

    @staticmethod
    def name() -> str:
        return "olympus-interact"

    def version(self) -> str | None:
        return VERSION

    @classmethod
    def import_path(cls) -> str:
        return f"{cls.__module__}:{cls.__name__}"


def _as_int(value: Any, default: int) -> int:
    if value is None or value == "":
        return default
    return int(value)


def _as_float(value: Any, default: float) -> float:
    if value is None or value == "":
        return default
    return float(value)


def _as_optional_bool(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class InteractOptions:
    """Everything Harbor may pass through ``-ak key=value`` to the session CLI."""

    workflow: str = "full_sdlc_tdd"
    gate_policy: str = "interactive_minimal"
    provider: str | None = None
    model: str | None = None
    base_url: str | None = None
    worker_driver: str | None = None
    reviewer_driver: str | None = None
    monitor_model: str | None = None
    max_turns_per_step: int = 8
    workspace_path: str = "/app"
    state_dir: str = "/tmp/olympus-session"
    mcp_url: str = "http://user-server:8000/mcp"
    ask_user_tool: str = "ask_user"
    olympus_bin: str = "olympus"
    olympus_source: str | None = None
    olympus_wheel: str | None = None
    olympus_install_command: str | None = None
    olympus_editable: bool = False
    pi_package: str = "@earendil-works/pi-coding-agent"
    install_pi: bool | None = None
    pi_bin: str | None = None
    step_roles: str | None = None
    plan_path: str = "/tmp/plan.md"
    initial_question: str = DEFAULT_INITIAL_QUESTION
    session_timeout_sec: float = 3600.0
    mcp_timeout_sec: float = 120.0

    @classmethod
    def from_kwargs(
        cls, kwargs: dict[str, Any], *, model_name: str | None = None
    ) -> InteractOptions:
        known = {field.name for field in fields(cls)}
        raw = {key: kwargs.pop(key) for key in list(kwargs) if key in known}
        coerced = {
            "max_turns_per_step",
            "session_timeout_sec",
            "mcp_timeout_sec",
            "install_pi",
            "olympus_editable",
        }
        options = cls(**{key: value for key, value in raw.items() if key not in coerced})
        options = replace(
            options,
            max_turns_per_step=_as_int(raw.get("max_turns_per_step"), 8),
            session_timeout_sec=_as_float(raw.get("session_timeout_sec"), 3600.0),
            mcp_timeout_sec=_as_float(raw.get("mcp_timeout_sec"), 120.0),
            install_pi=_as_optional_bool(raw.get("install_pi")),
            olympus_editable=bool(_as_optional_bool(raw.get("olympus_editable"))),
        )
        provider = options.provider
        model = options.model or model_name
        if provider is None and model and "/" in model:
            head, tail = model.split("/", 1)
            if head in {"openai", "anthropic", "mock"}:
                provider, model = head, tail
        return replace(options, provider=provider, model=model)


def classify_step(
    instruction: str, step_index: int, step_roles: str | None = None
) -> str:
    """Resolve one Harbor step to ``plan``, ``execute``, or ``handoff``.

    Harbor passes only the step instruction; the upstream SWE-Interact markers
    are stable, and the first step always plans. ``step_roles`` wins when set.
    """
    requested = [role.strip().lower() for role in (step_roles or "").split(",") if role.strip()]
    if requested:
        if step_index < 1 or step_index > len(requested):
            raise ValueError(
                f"step_roles has {len(requested)} entries but step {step_index} ran"
            )
        role = requested[step_index - 1]
        if role not in _VALID_ROLES:
            raise ValueError(f"unknown step role {role!r} in step_roles")
        return role
    text = instruction.casefold()
    if any(marker in text for marker in _HANDOFF_MARKERS):
        return _HANDOFF_ROLE
    if any(marker in text for marker in _PLAN_MARKERS):
        return _PLAN_ONLY_ROLE
    if step_index == 1:
        return _PLAN_ONLY_ROLE
    return _EXECUTE_ROLE


def extract_handoff_script(
    instruction: str,
    *,
    logs_dir: str | PurePosixPath = "/logs/agent",
    workspace: str = "/app",
) -> str:
    """Return the shell script a handoff step should execute inside the container."""
    blocks = [
        match.group(1).strip("\n")
        for match in _FENCED_BLOCK.finditer(instruction)
        if match.group(1).strip()
    ]
    if blocks:
        return "set -euo pipefail\n" + "\n\n".join(blocks) + "\n"
    logs = shlex.quote(str(logs_dir))
    return (
        "set -euo pipefail\n"
        f"mkdir -p {logs}\n"
        f"cd {shlex.quote(str(workspace))}\n"
        f"git status --short > {logs}/handoff-status.txt\n"
    )


class OlympusInteractAgent(_AgentBase or _FallbackAgent):
    """Harbor agent that drives ``olympus session`` across SWE-Interact steps."""

    capabilities = (
        _HarborCapabilities(base_url=True, mcp_servers=True)
        if _HarborCapabilities is not None
        else _FallbackCapabilities()
    )
    DEFAULT_SETUP_TIMEOUT_SEC: ClassVar[float] = 900.0

    def __init__(
        self,
        logs_dir: str | Path | None = None,
        model_name: str | None = None,
        logger: logging.Logger | None = None,
        mcp_servers: Any = None,
        skills_dir: str | None = None,
        extra_env: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        raw_env = dict(extra_env or {})
        if extra_env is None and isinstance(kwargs.get("extra_env"), dict):
            raw_env = dict(kwargs.pop("extra_env"))
        self._interact = InteractOptions.from_kwargs(kwargs, model_name=model_name)
        self._step_index = 0
        self._run_id: str | None = None
        self._installed_olympus_bin: str | None = None
        self._installed_pi_bin: str | None = None
        super().__init__(
            logs_dir=logs_dir if logs_dir is not None else Path.cwd() / "logs",
            model_name=model_name,
            logger=logger,
            mcp_servers=mcp_servers,
            skills_dir=skills_dir,
            extra_env=raw_env,
            **kwargs,
        )

    @staticmethod
    def name() -> str:
        return "olympus-interact"

    def version(self) -> str | None:
        return VERSION

    @classmethod
    def import_path(cls) -> str:
        return f"{cls.__module__}:{cls.__name__}"

    @property
    def interact_options(self) -> InteractOptions:
        return self._interact

    @property
    def run_id(self) -> str | None:
        return self._run_id

    @classmethod
    def resolve_model_connection_config(
        cls, model_name: str | None, resolve_env: Any, **options: Any
    ) -> Any:
        """Accept ``base_url`` even though this adapter owns its own provider wiring."""
        try:
            return super().resolve_model_connection_config(
                model_name, resolve_env, **options
            )
        except (AttributeError, ValueError):
            return _empty_model_connection()

    @classmethod
    def resolve_model_connection_for_execution(
        cls, model_name: str | None, resolve_env: Any, **options: Any
    ) -> Any:
        try:
            return super().resolve_model_connection_for_execution(
                model_name, resolve_env, **options
            )
        except (AttributeError, ValueError):
            return _empty_model_connection()

    async def setup(self, environment: Any) -> None:
        """Install Olympus and the bridge scripts inside the task container."""
        options = self._interact
        bridge_dir = PurePosixPath(BRIDGE_DIR)
        logs_dir = PurePosixPath(self.environment_logs_dir)
        await self._exec(
            environment,
            "mkdir -p "
            + " ".join(
                shlex.quote(str(path))
                for path in (
                    bridge_dir,
                    PurePosixPath(options.state_dir),
                    logs_dir,
                )
            ),
            timeout_sec=120,
        )
        await self._install_olympus(environment)
        await self._upload_bridge_scripts(environment, bridge_dir)
        if self._pi_install_requested():
            await self._install_pi(environment)

    async def run(
        self, instruction: str, environment: Any, context: Any
    ) -> None:
        """Run one Harbor step through the bridge or the handoff script."""
        self._step_index += 1
        options = self._interact
        role = classify_step(instruction, self._step_index, options.step_roles)
        mode = {
            _PLAN_ONLY_ROLE: "plan_only",
            _EXECUTE_ROLE: "execute",
            _HANDOFF_ROLE: "handoff",
        }[role]
        self.logger.info(
            "Olympus step %s resolved to %r (mode=%s)", self._step_index, role, mode
        )
        if role == _HANDOFF_ROLE:
            await self._run_handoff(instruction, environment)
            return
        await self._run_bridge_step(role, instruction, environment)

    def populate_context_post_run(self, context: Any) -> None:
        summaries: list[dict[str, Any]] = []
        for path in sorted(Path(self.logs_dir).glob("olympus-interact-step-*.jsonl")):
            turns, status, run_id = 0, None, None
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except OSError:
                continue
            for line in lines:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                event = record.get("event")
                if event == "ask_user":
                    turns += 1
                elif event == "session.envelope":
                    envelope = record.get("envelope") or {}
                    status = envelope.get("status") or status
                    run_id = envelope.get("run_id") or run_id
            summaries.append(
                {"step": path.stem, "turns": turns, "status": status, "run_id": run_id}
            )
        if not summaries:
            return
        metadata = dict(getattr(context, "metadata", None) or {})
        metadata["olympus_interact"] = {
            "steps": summaries,
            "run_id": self._run_id,
        }
        context.metadata = metadata

    def _pi_install_requested(self) -> bool:
        options = self._interact
        if options.install_pi is not None:
            return options.install_pi
        driver = (options.worker_driver or "").strip().lower()
        return driver == "pi" or driver.startswith("pi@")

    async def _install_olympus(self, environment: Any) -> None:
        options = self._interact
        if options.olympus_install_command:
            await self._exec(environment, options.olympus_install_command, timeout_sec=1800)
            return
        if options.olympus_wheel:
            wheel = Path(options.olympus_wheel).expanduser().resolve()
            if not wheel.is_file():
                raise FileNotFoundError(f"olympus wheel not found: {wheel}")
            remote_dir = PurePosixPath("/tmp/olympus-dist")
            await self._exec(environment, f"mkdir -p {shlex.quote(str(remote_dir))}")
            remote = str(remote_dir / wheel.name)
            await environment.upload_file(wheel, remote)
            install_spec = remote
        elif options.olympus_source:
            source = Path(options.olympus_source).expanduser().resolve()
            install_spec = await self._upload_source_tree(environment, source)
        else:
            install_spec = await self._upload_source_tree(environment, self._project_root())
        install_result = await self._exec(
            environment,
            self._pip_install_command(
                install_spec,
                editable=options.olympus_editable,
                olympus_bin=options.olympus_bin,
            ),
            timeout_sec=1800,
        )
        for line in reversed((install_result.stdout or "").splitlines()):
            candidate = line.strip()
            if candidate.startswith("/"):
                self._installed_olympus_bin = candidate
                break
        verify_bin = shlex.quote(self._installed_olympus_bin or options.olympus_bin)
        await self._exec(
            environment,
            f"command -v {verify_bin} >/dev/null 2>&1 || [ -x {verify_bin} ] || "
            'python3 -c "import olympus; print(olympus.__name__)" >/dev/null',
            timeout_sec=120,
        )

    async def _upload_source_tree(self, environment: Any, source_root: Path) -> str:
        pyproject = source_root / "pyproject.toml"
        package_dir = source_root / "src" / "olympus"
        if not pyproject.is_file() or not package_dir.is_dir():
            raise FileNotFoundError(
                f"{source_root} is not an Olympus4SWE source tree "
                "(expected pyproject.toml and src/olympus)"
            )
        remote_root = PurePosixPath("/tmp/olympus-src")
        await self._exec(
            environment,
            f"mkdir -p {shlex.quote(str(remote_root / 'src'))}",
            timeout_sec=120,
        )
        await environment.upload_dir(package_dir, str(remote_root / "src" / "olympus"))
        await environment.upload_file(pyproject, str(remote_root / "pyproject.toml"))
        readme = source_root / "README.md"
        if readme.is_file():
            await environment.upload_file(readme, str(remote_root / "README.md"))
        return str(remote_root)

    @staticmethod
    def _pip_install_command(
        install_spec: str,
        *,
        editable: bool = False,
        olympus_bin: str = "olympus",
    ) -> str:
        """Install Olympus in an isolated Python >= 3.13 venv so task site-packages stay clean."""
        quoted = shlex.quote(install_spec)
        target = f"-e {quoted}" if editable else quoted
        binary = shlex.quote(olympus_bin)
        return "\n".join(
            [
                "set -eu",
                "VENV_DIR=/opt/olympus/venv",
                'if ! mkdir -p /opt/olympus 2>/dev/null || [ ! -w /opt/olympus ]; then '
                "VENV_DIR=/tmp/olympus-venv; mkdir -p /tmp/olympus; fi",
                'if command -v python3 >/dev/null 2>&1; then PY=python3; '
                'elif command -v python >/dev/null 2>&1; then PY=python; '
                'else PY=""; fi',
                'if [ -n "$PY" ] && "$PY" -c '
                "'import sys; raise SystemExit(0 if sys.version_info >= (3, 13) else 1)' "
                '>/dev/null 2>&1 && "$PY" -m venv "$VENV_DIR" >/dev/null 2>&1; then',
                f'  "$VENV_DIR/bin/python" -m pip install --quiet '
                f"--disable-pip-version-check {target}",
                "else",
                '  UV_BIN=""',
                '  UV_BOOT=""',
                '  if command -v uv >/dev/null 2>&1; then UV_BIN=$(command -v uv);',
                '  elif [ -x "$HOME/.local/bin/uv" ]; then UV_BIN="$HOME/.local/bin/uv";',
                '  elif [ -n "$PY" ]; then',
                "    UV_BOOT=/tmp/olympus-uv",
                '    mkdir -p "$UV_BOOT"',
                '    "$PY" -m pip install --quiet --disable-pip-version-check '
                '--target "$UV_BOOT" uv >/dev/null 2>&1 || '
                '"$PY" -m pip install --quiet --disable-pip-version-check '
                '--break-system-packages --target "$UV_BOOT" uv >/dev/null 2>&1',
                "  fi",
                '  if [ -n "$UV_BIN" ]; then',
                '    "$UV_BIN" venv --quiet --python 3.13 "$VENV_DIR"',
                f'    "$UV_BIN" pip install --quiet --python "$VENV_DIR/bin/python" {target}',
                '  elif [ -n "$UV_BOOT" ]; then',
                '    PYTHONPATH="$UV_BOOT${PYTHONPATH:+:$PYTHONPATH}" '
                '"$PY" -m uv venv --quiet --python 3.13 "$VENV_DIR"',
                '    PYTHONPATH="$UV_BOOT${PYTHONPATH:+:$PYTHONPATH}" '
                f'"$PY" -m uv pip install --quiet --python "$VENV_DIR/bin/python" {target}',
                "  elif command -v curl >/dev/null 2>&1; then",
                "    curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1",
                '    "$HOME/.local/bin/uv" venv --quiet --python 3.13 "$VENV_DIR"',
                '    "$HOME/.local/bin/uv" pip install --quiet '
                f'--python "$VENV_DIR/bin/python" {target}',
                "  else",
                '    echo "Python >= 3.13 or uv is required to install olympus4swe" >&2',
                "    exit 1",
                "  fi",
                "fi",
                f'if [ -w /usr/local/bin ] && [ -x "$VENV_DIR/bin"/{binary} ]; then '
                f'ln -sf "$VENV_DIR/bin"/{binary} /usr/local/bin/{binary}; fi',
                f'echo "$VENV_DIR/bin"/{binary}',
            ]
        )

    async def _upload_bridge_scripts(
        self, environment: Any, bridge_dir: PurePosixPath
    ) -> None:
        local_dir = Path(__file__).resolve().parent
        for name in _BRIDGE_SCRIPTS:
            local = local_dir / name
            if not local.is_file():
                raise FileNotFoundError(f"bridge companion script missing: {local}")
            await environment.upload_file(local, str(bridge_dir / name))
        await self._exec(
            environment,
            f"chmod 755 {shlex.quote(str(bridge_dir))}/*.py",
            timeout_sec=60,
        )

    async def _install_pi(self, environment: Any) -> None:
        options = self._interact
        prefix = shlex.quote(PI_PREFIX)
        command = (
            "set -e; "
            "if ! command -v npm >/dev/null 2>&1; then "
            "  if command -v apt-get >/dev/null 2>&1; then "
            "    apt-get update -qq && apt-get install -y -qq nodejs npm; "
            "  else echo 'npm is required to install the pi CLI' >&2; exit 1; fi; "
            "fi; "
            f"npm install -g --ignore-scripts --prefix {prefix} "
            f"{shlex.quote(options.pi_package)} && "
            f"{shlex.quote(PI_PREFIX + '/bin/pi')} --version"
        )
        await self._exec(environment, command, timeout_sec=1800)
        self._installed_pi_bin = f"{PI_PREFIX}/bin/pi"

    @staticmethod
    def _project_root() -> Path:
        return Path(__file__).resolve().parents[3]

    async def _run_handoff(self, instruction: str, environment: Any) -> None:
        script = extract_handoff_script(
            instruction,
            logs_dir=self.environment_logs_dir,
            workspace=self._interact.workspace_path,
        )
        host_path = self._write_artifact(
            f"olympus-interact-step-{self._step_index:02d}-handoff.sh", script
        )
        remote_path = f"{BRIDGE_DIR}/handoff-step-{self._step_index:02d}.sh"
        await environment.upload_file(host_path, remote_path)
        result = await environment.exec(
            f"bash {shlex.quote(remote_path)}", timeout_sec=None
        )
        if result.return_code != 0:
            detail = self._redact((result.stderr or result.stdout or "").strip())[-2000:]
            raise RuntimeError(
                f"handoff step {self._step_index} failed ({result.return_code}): {detail}"
            )

    async def _run_bridge_step(
        self, role: str, instruction: str, environment: Any
    ) -> None:
        step_tag = f"step-{self._step_index:02d}"
        host_instruction = self._write_artifact(
            f"olympus-interact-{step_tag}-instruction.md", instruction
        )
        remote_instruction = f"{BRIDGE_DIR}/{step_tag}-instruction.md"
        if role != _PLAN_ONLY_ROLE:
            await environment.upload_file(host_instruction, remote_instruction)
        trajectory = str(
            PurePosixPath(self.environment_logs_dir)
            / f"olympus-interact-{step_tag}.jsonl"
        )
        argv = self._bridge_argv(role, remote_instruction, trajectory)
        result = await environment.exec(shlex.join(argv), timeout_sec=None)
        summary = self._parse_bridge_summary(result.stdout)
        if result.return_code != 0 or (summary or {}).get("status") != "ok":
            error = (summary or {}).get("error") or (
                result.stderr or result.stdout or ""
            ).strip()
            raise RuntimeError(
                f"olympus bridge {role} step {self._step_index} failed "
                f"(rc={result.return_code}): {self._redact(str(error))[-2000:]}"
            )
        run_id = (summary or {}).get("run_id")
        if run_id:
            self._run_id = str(run_id)

    def _bridge_argv(
        self, role: str, remote_instruction: str, trajectory: str
    ) -> list[str]:
        options = self._interact
        phase = "plan" if role == _PLAN_ONLY_ROLE else "execute"
        argv = [
            "python3",
            f"{BRIDGE_DIR}/interact_bridge.py",
            "--phase",
            phase,
            "--olympus-bin",
            self._installed_olympus_bin or options.olympus_bin,
            "--workspace",
            options.workspace_path,
            "--state-dir",
            options.state_dir,
            "--run-id-file",
            f"{options.state_dir}/run-id.txt",
            "--mcp-url",
            options.mcp_url,
            "--ask-user-tool",
            options.ask_user_tool,
            "--mcp-timeout",
            str(options.mcp_timeout_sec),
            "--max-turns",
            str(options.max_turns_per_step),
            "--trajectory",
            trajectory,
            "--workflow",
            options.workflow,
            "--gate-policy",
            options.gate_policy,
            "--plan-out",
            options.plan_path,
            "--session-timeout",
            str(options.session_timeout_sec),
        ]
        if phase == "plan":
            argv += ["--initial-question", options.initial_question]
        else:
            argv += ["--reply-file", remote_instruction]
        for flag, value in (
            ("--provider", options.provider),
            ("--model", options.model),
            ("--base-url", options.base_url),
            ("--worker-driver", options.worker_driver),
            ("--reviewer-driver", options.reviewer_driver),
            ("--monitor-model", options.monitor_model),
            ("--pi-bin", options.pi_bin or self._installed_pi_bin),
        ):
            if value:
                argv += [flag, str(value)]
        return argv

    def _write_artifact(self, name: str, content: str) -> Path:
        logs_dir = Path(self.logs_dir)
        logs_dir.mkdir(parents=True, exist_ok=True)
        path = logs_dir / name
        path.write_text(content, encoding="utf-8")
        return path

    @staticmethod
    def _parse_bridge_summary(stdout: str | None) -> dict[str, Any] | None:
        for line in reversed((stdout or "").strip().splitlines()):
            candidate = line.strip()
            if not candidate.startswith("{"):
                continue
            try:
                payload = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                return payload
        return None

    async def _exec(
        self,
        environment: Any,
        command: str,
        *,
        timeout_sec: float | None = None,
    ) -> Any:
        result = await environment.exec(command, timeout_sec=timeout_sec)
        if result.return_code != 0:
            detail = self._redact((result.stderr or result.stdout or "").strip())[-2000:]
            raise RuntimeError(
                f"container command failed ({result.return_code}): {command}\n{detail}"
            )
        return result

    def _redact(self, text: str) -> str:
        return redact_text(text, {**os.environ, **self.extra_env})


def _empty_model_connection() -> Any:
    try:
        from harbor.agents.agent_auth_resolution import ResolvedModelConnection
    except Exception:  # pragma: no cover - Harbor is absent in the core test env
        return None
    return ResolvedModelConnection()


__all__ = [
    "InteractOptions",
    "OlympusInteractAgent",
    "classify_step",
    "extract_handoff_script",
]
