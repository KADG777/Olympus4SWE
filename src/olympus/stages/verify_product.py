"""Clean-directory product verification: boot the delivered product and probe it.

The product is materialized into a fresh sandbox from the workspace plus the run's
accepted overlay and deletions (rejecting any symlink escape), then booted in an
allowlisted environment with an isolated ``HOME`` against a brand-new empty data
directory so no host state or secrets can mask a broken delivery. Web apps must
serve ``GET /`` with an HTML ``Content-Type`` and real markup, HTTP APIs must
expose a non-HTML healthy endpoint, the listening socket must belong to the booted
process tree, and both must survive a clean restart against the same data directory.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

from olympus.core.contracts import DeliverableKind, GoalSpec, TaskPlan, new_id
from olympus.harness.delivery import run_delivery_checks
from olympus.harness.worktree import (
    SandboxHandle,
    create_sandbox,
    load_accepted_deletions,
    remove_deleted_paths,
    remove_sandbox,
)

PRODUCT_VERIFIED_EVENT = "product.verified"

_HTTP_KINDS = frozenset({DeliverableKind.WEB_APP, DeliverableKind.REST_API})
_READINESS_PATHS: tuple[str, ...] = ("/", "/api/health", "/health")
_API_SURFACE_PATHS: tuple[str, ...] = ("/api/health", "/health", "/openapi.json", "/api", "/")
_HEALTH_PATHS = frozenset({"/api/health", "/health"})
_ASGI_APP_RE = re.compile(
    r"(?m)^\s*([A-Za-z_]\w*)\s*=\s*(?:FastAPI|Starlette|Quart|Litestar|Sanic)\("
)
_HTML_MEDIA_TYPES = frozenset({"text/html"})
_HTML_MARKERS: tuple[str, ...] = (
    "<!doctype",
    "<html",
    "<head",
    "<body",
    "<main",
    "<div",
    "<h1",
    "<form",
    "<section",
    "<script",
)
_ROUTE_HINT_RE = re.compile(r"\b(GET|POST)\s+(/[A-Za-z0-9_./{}:-]+)")
_PERSISTENCE_HINT_RE = re.compile(
    r"persist|survive|restart|sqlite|database|持久化|重启|不丢失|数据库",
    re.IGNORECASE,
)
_ENV_ALLOWLIST: tuple[str, ...] = (
    "PATH",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TERM",
    "TZ",
    "NODE_PATH",
    "UV_CACHE_DIR",
    "PLAYWRIGHT_BROWSERS_PATH",
    "SYSTEMROOT",
)
_IGNORED_DATA_SUBDIRS = frozenset({"tmp", "home"})


@dataclass(frozen=True)
class ProductVerificationResult:
    passed: bool
    summary: str
    failures: tuple[str, ...]
    report_path: str | None
    boot_log_path: str | None
    screenshot_path: str | None
    checks: list[dict[str, Any]] = field(default_factory=list)

    def as_payload(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "summary": self.summary,
            "failures": list(self.failures),
            "report_path": self.report_path,
            "boot_log_path": self.boot_log_path,
            "screenshot_path": self.screenshot_path,
            "checks": self.checks,
        }


@dataclass(frozen=True)
class _ServerCommand:
    argv: tuple[str, ...]
    label: str


@dataclass(frozen=True)
class _PersistenceProbe:
    path: str
    marker: str


@dataclass(frozen=True)
class _BootOutcome:
    port: int
    url: str
    ready: bool
    surface_ok: bool
    checks: list[dict[str, Any]]
    failures: list[str]
    screenshot_path: str | None = None
    persistence_probe: _PersistenceProbe | None = None
    persistence_outcome: str = "none"
    persistence_detail: str = ""


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _package_start_script(package_json: Path) -> str | None:
    try:
        data = json.loads(_read_text(package_json))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    scripts = data.get("scripts")
    if isinstance(scripts, dict):
        start = scripts.get("start")
        if isinstance(start, str) and start.strip():
            return start
    return None


def _looks_asgi(source: str) -> bool:
    return bool(
        re.search(
            r"(?m)^\s*(?:from\s+fastapi\s+import|import\s+fastapi\b)"
            r"|FastAPI\(|Starlette\(|ASGIApp|Quart\(|Litestar\(",
            source,
        )
    )


def _asgi_app_name(source: str) -> str:
    match = _ASGI_APP_RE.search(source)
    return match.group(1) if match is not None else "app"


def _python_entrypoint(workspace: Path, rel_path: str, module: str | None) -> _ServerCommand:
    source = _read_text(workspace / rel_path)
    if _looks_asgi(source) and importlib.util.find_spec("uvicorn") is not None:
        target_module = module or Path(rel_path).stem
        app_name = _asgi_app_name(source)
        return _ServerCommand(
            (
                sys.executable,
                "-m",
                "uvicorn",
                f"{target_module}:{app_name}",
                "--host",
                "127.0.0.1",
                "--port",
                "{port}",
            ),
            f"uvicorn {target_module}:{app_name}",
        )
    if module is not None:
        package = module.split(".", 1)[0]
        if (workspace / package / "__init__.py").is_file():
            return _ServerCommand((sys.executable, "-m", module), f"python -m {module}")
    return _ServerCommand((sys.executable, rel_path), f"python {rel_path}")


def _discover_server_command(workspace: Path) -> _ServerCommand | None:
    for name in ("run.sh", "start.sh"):
        if (workspace / name).is_file():
            return _ServerCommand(("bash", name), f"bash {name}")
    package_json = workspace / "package.json"
    if package_json.is_file():
        npm = shutil.which("npm")
        if _package_start_script(package_json) is not None and npm is not None:
            return _ServerCommand((npm, "start"), "npm start")
    node = shutil.which("node")
    if node is not None:
        for entry in ("server.js", "app.js", "index.js"):
            if (workspace / entry).is_file():
                return _ServerCommand((node, entry), f"node {entry}")
    for rel_path, module in (
        ("app/main.py", "app.main"),
        ("backend/app.py", "backend.app"),
        ("backend/main.py", "backend.main"),
        ("app.py", None),
        ("main.py", None),
        ("run.py", None),
        ("server.py", None),
    ):
        if (workspace / rel_path).is_file():
            return _python_entrypoint(workspace, rel_path, module)
    return None


def _top_level_listing(workspace: Path) -> str:
    try:
        entries = sorted(workspace.iterdir(), key=lambda item: item.name)
    except OSError:
        return "(unreadable workspace)"
    names = [f"{entry.name}/" if entry.is_dir() else entry.name for entry in entries[:30]]
    return ", ".join(names) if names else "(empty workspace)"


def _ensure_no_symlink_components(root: Path, rel: Path) -> Path:
    """Resolve ``root / rel`` while rejecting any symlink along ``rel`` or root escape."""
    root_resolved = root.resolve()
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"Unsafe overlay path outside workspace: {rel}")
    cursor = root_resolved
    for part in rel.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError(f"Refusing to traverse symlink during materialization: {rel}")
    resolved = cursor.resolve()
    if resolved != root_resolved and root_resolved not in resolved.parents:
        raise ValueError(f"Overlay target escapes sandbox root: {rel}")
    return cursor


def _copy_overlay_safely(overlay_dir: Path, workspace: Path) -> None:
    """Copy ``overlay_dir`` into ``workspace`` without following any symlinks."""
    if overlay_dir.is_symlink():
        raise ValueError(f"Refusing symlinked overlay root: {overlay_dir}")
    if workspace.is_symlink():
        raise ValueError(f"Refusing symlinked workspace root: {workspace}")
    src_root = overlay_dir.resolve()
    dst_root = workspace.resolve()
    for dirpath, dirnames, filenames in os.walk(src_root, followlinks=False):
        current_src = Path(dirpath)
        rel_dir = current_src.relative_to(src_root)
        if rel_dir != Path("."):
            target_dir = _ensure_no_symlink_components(dst_root, rel_dir)
            target_dir.mkdir(parents=True, exist_ok=True)
        for dirname in list(dirnames):
            src_sub = current_src / dirname
            if src_sub.is_symlink():
                raise ValueError(
                    f"Refusing symlink directory in overlay: {src_sub.relative_to(src_root)}"
                )
        for filename in filenames:
            src_file = current_src / filename
            rel_file = src_file.relative_to(src_root)
            if src_file.is_symlink():
                raise ValueError(f"Refusing symlink file in overlay: {rel_file}")
            dst_file = _ensure_no_symlink_components(dst_root, rel_file)
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_file, dst_file, follow_symlinks=False)


def _port_accepting(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.2)
        return probe.connect_ex(("127.0.0.1", port)) == 0


def _free_port() -> int:
    for _ in range(16):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            port = int(sock.getsockname()[1])
        if not _port_accepting(port):
            return port
    raise RuntimeError("Could not allocate a free loopback port")


def _product_env(
    *, workspace: Path, workspace_root: Path, data_dir: Path, port: int
) -> dict[str, str]:
    """Construct an allowlisted, secret-free runtime environment with isolated HOME."""
    ws = workspace.resolve()
    host = workspace_root.resolve()
    env: dict[str, str] = {}
    for key in _ENV_ALLOWLIST:
        val = os.environ.get(key)
        if val is not None:
            env[key] = val

    venv_candidates = [ws / ".venv"]
    if host != ws:
        venv_candidates.append(host / ".venv")
    current_prefix = Path(sys.executable).resolve().parent.parent
    if (current_prefix / "pyvenv.cfg").exists() and current_prefix not in venv_candidates:
        venv_candidates.append(current_prefix)
    for venv_dir in venv_candidates:
        bin_dir = venv_dir / "bin"
        if bin_dir.is_dir():
            env["VIRTUAL_ENV"] = str(venv_dir)
            existing_path = env.get("PATH", "")
            env["PATH"] = f"{bin_dir}{os.pathsep}{existing_path}" if existing_path else str(bin_dir)
            break

    python_roots: list[str] = []
    src_dir = ws / "src"
    if src_dir.is_dir():
        python_roots.append(str(src_dir))
    python_roots.append(str(ws))
    env["PYTHONPATH"] = os.pathsep.join(python_roots)

    home_dir = data_dir / "home"
    tmp_dir = data_dir / "tmp"
    uploads_dir = data_dir / "uploads"
    for directory in (home_dir, tmp_dir, uploads_dir):
        directory.mkdir(parents=True, exist_ok=True)
    db_path = data_dir / "app.db"
    env.update(
        {
            "HOME": str(home_dir),
            "XDG_CONFIG_HOME": str(home_dir / ".config"),
            "XDG_CACHE_HOME": str(home_dir / ".cache"),
            "XDG_DATA_HOME": str(home_dir / ".local" / "share"),
            "PYTHONUNBUFFERED": "1",
            "PORT": str(port),
            "HOST": "127.0.0.1",
            "DATA_DIR": str(data_dir),
            "DB_PATH": str(db_path),
            "DATABASE_PATH": str(db_path),
            "OLYMPUS_DB_PATH": str(data_dir / "olympus.db"),
            "OLYMPUS_UPLOAD_DIR": str(uploads_dir),
            "BLACKNOTE_PORT": str(port),
            "BLACKNOTE_HOST": "127.0.0.1",
            "BLACKNOTE_DB_PATH": str(data_dir / "blacknote.db"),
            "UPLOAD_DIR": str(uploads_dir),
            "TMPDIR": str(tmp_dir),
            "TEMP": str(tmp_dir),
            "TMP": str(tmp_dir),
        }
    )
    return env


def _http_request(
    url: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    timeout: float = 2.0,
) -> tuple[int, str, str] | None:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json"} if payload is not None else {}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            body = response.read(200_000).decode("utf-8", errors="replace")
            return response.status, response.headers.get("content-type", ""), body
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read(200_000).decode("utf-8", errors="replace")
        except OSError:
            body = ""
        return exc.code, exc.headers.get("content-type", "") if exc.headers else "", body
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return None


def _http_get(url: str, *, timeout: float = 2.0) -> tuple[int, str, str] | None:
    return _http_request(url, method="GET", timeout=timeout)


def _media_type(content_type: str) -> str:
    return content_type.split(";", 1)[0].strip().lower()


def _proc_tree_pids(root_pid: int) -> set[int]:
    """Return ``root_pid`` plus every descendant or member of its process group on Linux."""
    proc_root = Path("/proc")
    if not proc_root.is_dir():
        return {root_pid}
    children: dict[int, list[int]] = {}
    same_pgid: set[int] = set()
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            stat_text = (entry / "stat").read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        rparen = stat_text.rfind(")")
        if rparen == -1:
            continue
        fields = stat_text[rparen + 2 :].split()
        if len(fields) < 3:
            continue
        try:
            ppid = int(fields[1])
            pgid = int(fields[2])
        except ValueError:
            continue
        children.setdefault(ppid, []).append(pid)
        if pgid == root_pid:
            same_pgid.add(pid)
    owned: set[int] = {root_pid} | same_pgid
    stack = list(owned)
    while stack:
        current = stack.pop()
        for child in children.get(current, ()):
            if child not in owned:
                owned.add(child)
                stack.append(child)
    return owned


def _listening_socket_inodes(port: int) -> set[str] | None:
    """Return Linux socket inodes listening on ``port``, or ``None`` without ``/proc/net/tcp``."""
    paths = [Path("/proc/net/tcp"), Path("/proc/net/tcp6")]
    readable = [path for path in paths if path.is_file()]
    if not readable:
        return None
    target_hex = f"{port:04X}"
    inodes: set[str] = set()
    for table in readable:
        try:
            lines = table.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line in lines[1:]:
            parts = line.split()
            if len(parts) < 10:
                continue
            local_addr = parts[1]
            state = parts[3]
            inode = parts[9]
            if state != "0A" or ":" not in local_addr:
                continue
            _, port_hex = local_addr.rsplit(":", 1)
            if port_hex.upper() == target_hex and inode.isdigit() and inode != "0":
                inodes.add(inode)
    return inodes


def _port_owned_by_process_tree(root_pid: int, port: int) -> bool | None:
    """Verify ``port`` is listened on by ``root_pid``'s process tree on Linux.

    Returns ``None`` when ``/proc`` socket tables are unavailable (non-Linux).
    """
    inodes = _listening_socket_inodes(port)
    if inodes is None:
        return None
    if not inodes:
        return False
    targets = {f"socket:[{inode}]" for inode in inodes}
    for pid in _proc_tree_pids(root_pid):
        fd_dir = Path("/proc") / str(pid) / "fd"
        try:
            entries = list(fd_dir.iterdir())
        except OSError:
            continue
        for fd in entries:
            try:
                link = os.readlink(fd)
            except OSError:
                continue
            if link in targets:
                return True
    return False


def _stop_process(process: subprocess.Popen[str]) -> None:
    pgid = process.pid
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pgid, signal.SIGTERM)
    if process.poll() is None:
        with contextlib.suppress(subprocess.TimeoutExpired):
            process.wait(timeout=5)
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pgid, signal.SIGKILL)
    with contextlib.suppress(subprocess.TimeoutExpired):
        process.wait(timeout=5)


def _wait_for_readiness(
    *, port: int, process: subprocess.Popen[str], timeout: float
) -> tuple[bool, int, str]:
    deadline = time.monotonic() + timeout
    foreign_listener_seen = False
    while time.monotonic() < deadline:
        if process.poll() is not None:
            code = process.returncode if process.returncode is not None else 1
            return False, code, f"server exited early (exit_code={code})"
        owned = _port_owned_by_process_tree(process.pid, port)
        if owned is False:
            if _port_accepting(port):
                foreign_listener_seen = True
            time.sleep(0.1)
            continue
        for path in _READINESS_PATHS:
            if _http_get(f"http://127.0.0.1:{port}{path}") is not None:
                if _port_owned_by_process_tree(process.pid, port) is False:
                    foreign_listener_seen = True
                    continue
                if process.poll() is not None:
                    code = process.returncode if process.returncode is not None else 1
                    return False, code, f"server exited during readiness probe (exit_code={code})"
                return True, 0, f"server responded on 127.0.0.1:{port}"
        time.sleep(0.1)
    if foreign_listener_seen:
        return (
            False,
            0,
            f"port 127.0.0.1:{port} was served by a different process, not the booted server",
        )
    return False, 0, f"server never became ready within {timeout:g}s (exit_code=0)"


def _spec_route_hints(spec: GoalSpec) -> tuple[list[str], list[str]]:
    """Extract static ``GET`` and ``POST`` routes explicitly named in the frozen GoalSpec."""
    get_paths: list[str] = []
    post_paths: list[str] = []
    sources = list(spec.observable_surfaces) + list(spec.test_seams)
    for req in spec.requirements:
        sources.extend(req.acceptance_criteria)
    for text in sources:
        for method, raw_path in _ROUTE_HINT_RE.findall(text):
            path = raw_path.rstrip(".,;:)'\"")
            if "{" in path or "}" in path or "/:" in path or "*" in path:
                continue
            target = get_paths if method == "GET" else post_paths
            if path not in target:
                target.append(path)
    return get_paths, post_paths


def _spec_requires_persistence(spec: GoalSpec) -> bool:
    texts = [spec.goal, *spec.constraints, *spec.assumptions]
    for req in spec.requirements:
        texts.append(req.description)
        texts.extend(req.acceptance_criteria)
    return any(_PERSISTENCE_HINT_RE.search(text) for text in texts)


def _probe_web_surface(url: str) -> tuple[bool, str]:
    response = _http_get(url)
    if response is None:
        return False, "GET / returned no HTTP response"
    status, content_type, body = response
    if status != 200:
        return False, f"GET / returned HTTP {status} (expected 200)"
    if not body.strip():
        return False, "GET / returned an empty body"
    mtype = _media_type(content_type)
    if mtype not in _HTML_MEDIA_TYPES:
        return (
            False,
            f"GET / did not declare an HTML content-type (content-type={content_type or 'none'})",
        )
    lowered_body = body[:4_000].lower()
    if not any(marker in lowered_body for marker in _HTML_MARKERS):
        return (
            False,
            "GET / did not return an HTML/UI body "
            f"(content-type={content_type or 'none'}, {len(body)} bytes)",
        )
    return True, f"GET / served HTML ({len(body)} bytes, content-type={content_type or 'none'})"


def _is_valid_api_response(path: str, content_type: str, body: str) -> bool:
    mtype = _media_type(content_type)
    if mtype in _HTML_MEDIA_TYPES:
        return False
    stripped = body.strip()
    if not stripped:
        return False
    if mtype == "application/json" or mtype.endswith("+json"):
        try:
            json.loads(stripped)
            return True
        except json.JSONDecodeError:
            return False
    if path in _HEALTH_PATHS and mtype == "text/plain":
        lowered = stripped.lower()
        return not any(marker in lowered for marker in _HTML_MARKERS)
    return False


def _probe_api_surface(base_url: str, spec: GoalSpec) -> tuple[bool, str]:
    spec_gets, _ = _spec_route_hints(spec)
    candidates: list[str] = []
    for path in (*spec_gets, *_API_SURFACE_PATHS):
        if path not in candidates:
            candidates.append(path)
    observed: list[str] = []
    for path in candidates:
        response = _http_get(f"{base_url}{path}")
        if response is None:
            observed.append(f"{path}->no-response")
            continue
        status, content_type, body = response
        mtype = _media_type(content_type) or "none"
        observed.append(f"{path}->{status}({mtype})")
        if status == 200 and _is_valid_api_response(path, content_type, body):
            return True, f"{path} returned HTTP 200 {mtype} ({', '.join(observed)})"
    return False, f"no healthy non-HTML API surface found ({', '.join(observed)})"


def _probe_declared_get_routes(base_url: str, spec: GoalSpec) -> list[str]:
    """Verify static ``GET`` routes declared in ``GoalSpec`` do not 404/405/5xx."""
    spec_gets, _ = _spec_route_hints(spec)
    failures: list[str] = []
    for path in spec_gets:
        if path == "/":
            continue
        response = _http_get(f"{base_url}{path}")
        if response is None:
            failures.append(f"Declared GoalSpec route GET {path} returned no response")
            continue
        status = response[0]
        if status in (404, 405) or status >= 500:
            failures.append(f"Declared GoalSpec route GET {path} returned HTTP {status}")
    return failures


def _attempt_persistence_write(
    base_url: str, spec: GoalSpec
) -> tuple[_PersistenceProbe | None, str, str]:
    """When GoalSpec declares a matching static POST+GET path, write a probe record on boot 1."""
    spec_gets, spec_posts = _spec_route_hints(spec)
    shared = [path for path in spec_posts if path in spec_gets and path != "/"]
    if not shared:
        return None, "none", ""
    marker = f"olympus-persist-{new_id('probe')}"
    payload = {
        "title": marker,
        "name": marker,
        "body": marker,
        "content": marker,
        "text": marker,
    }
    observed: list[str] = []
    saw_auth_gate = False
    for path in shared:
        resp = _http_request(f"{base_url}{path}", method="POST", payload=payload)
        if resp is None:
            observed.append(f"POST {path}->no-response")
            continue
        status = resp[0]
        observed.append(f"POST {path}->{status}")
        if status in (401, 403):
            saw_auth_gate = True
            continue
        if status not in (200, 201):
            continue
        read_back = _http_get(f"{base_url}{path}")
        if read_back is not None and read_back[0] == 200 and marker in read_back[2]:
            return _PersistenceProbe(path=path, marker=marker), "written", (
                f"wrote and read back probe on {path}"
            )
        observed.append(
            f"GET {path}->"
            f"{read_back[0] if read_back is not None else 'no-response'}(missing-marker)"
        )
    if saw_auth_gate and all(
        entry.endswith("->401") or entry.endswith("->403") for entry in observed
    ):
        return None, "auth_gated", f"auth-gated write surface ({', '.join(observed)})"
    return None, "failed", f"persistence write probe did not succeed ({', '.join(observed)})"


def _snapshot_data_dir(data_dir: Path) -> dict[str, tuple[int, str]]:
    """Return ``{rel_path: (size_bytes, sha256)}`` for persistent files under ``data_dir``."""
    snapshot: dict[str, tuple[int, str]] = {}
    if not data_dir.is_dir():
        return snapshot
    for dirpath, dirnames, filenames in os.walk(data_dir, followlinks=False):
        rel_dir = Path(dirpath).relative_to(data_dir)
        if rel_dir == Path("."):
            dirnames[:] = [name for name in dirnames if name not in _IGNORED_DATA_SUBDIRS]
        for filename in filenames:
            path = Path(dirpath) / filename
            if path.is_symlink():
                continue
            try:
                raw = path.read_bytes()
            except OSError:
                continue
            rel = path.relative_to(data_dir).as_posix()
            snapshot[rel] = (len(raw), hashlib.sha256(raw).hexdigest())
    return snapshot


def _capture_screenshot(url: str, screenshot_path: Path) -> str | None:
    if importlib.util.find_spec("playwright") is None:
        return None
    try:
        sync_api = importlib.import_module("playwright.sync_api")
        with sync_api.sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            try:
                page = browser.new_page()
                page.goto(url, wait_until="load", timeout=15_000)
                screenshot_path.parent.mkdir(parents=True, exist_ok=True)
                page.screenshot(path=str(screenshot_path))
            finally:
                browser.close()
    except Exception:  # noqa: BLE001 - screenshots are strictly best-effort
        return None
    return str(screenshot_path)


def _boot_and_probe(
    *,
    argv: tuple[str, ...],
    workspace: Path,
    workspace_root: Path,
    data_dir: Path,
    spec: GoalSpec,
    kind: DeliverableKind,
    timeout: float,
    boot_index: int,
    log: TextIO,
    screenshot_path: Path | None,
    verify_probe: _PersistenceProbe | None = None,
) -> _BootOutcome:
    port = _free_port()
    resolved = tuple(part.replace("{port}", str(port)) for part in argv)
    env = _product_env(
        workspace=workspace, workspace_root=workspace_root, data_dir=data_dir, port=port
    )
    checks: list[dict[str, Any]] = []
    failures: list[str] = []
    log.write(f"\n=== boot {boot_index}: {' '.join(resolved)} (cwd={workspace}) ===\n")
    log.flush()
    process: subprocess.Popen[str] = subprocess.Popen(
        resolved,
        cwd=workspace,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        text=True,
    )
    probe_created: _PersistenceProbe | None = None
    persistence_outcome = "none"
    persistence_detail = ""
    screenshot: str | None = None
    ready = False
    surface_ok = False
    url = f"http://127.0.0.1:{port}"
    try:
        ready, _exit_code, ready_detail = _wait_for_readiness(
            port=port, process=process, timeout=timeout
        )
        checks.append(
            {
                "name": f"boot-{boot_index}",
                "status": "passed" if ready else "failed",
                "detail": ready_detail,
                "port": port,
            }
        )
        if not ready:
            failures.append(f"Boot #{boot_index} failed: {ready_detail}")
            return _BootOutcome(
                port=port,
                url=url,
                ready=False,
                surface_ok=False,
                checks=checks,
                failures=failures,
            )
        if kind == DeliverableKind.WEB_APP:
            surface_ok, detail = _probe_web_surface(f"{url}/")
        else:
            surface_ok, detail = _probe_api_surface(url, spec)
        checks.append(
            {
                "name": f"primary-surface-{boot_index}",
                "status": "passed" if surface_ok else "failed",
                "detail": detail,
            }
        )
        if not surface_ok:
            failures.append(f"Primary user surface failed on boot #{boot_index}: {detail}")
            return _BootOutcome(
                port=port,
                url=url,
                ready=True,
                surface_ok=False,
                checks=checks,
                failures=failures,
            )

        route_failures = _probe_declared_get_routes(url, spec)
        if route_failures:
            checks.append(
                {
                    "name": f"spec-routes-{boot_index}",
                    "status": "failed",
                    "detail": "; ".join(route_failures),
                }
            )
            failures.extend(route_failures)
            return _BootOutcome(
                port=port,
                url=url,
                ready=True,
                surface_ok=False,
                checks=checks,
                failures=failures,
            )

        if boot_index == 1:
            probe_created, persistence_outcome, persistence_detail = _attempt_persistence_write(
                url, spec
            )
            if probe_created is not None:
                checks.append(
                    {
                        "name": "persistence-write-1",
                        "status": "passed",
                        "detail": persistence_detail,
                    }
                )
            elif persistence_outcome == "auth_gated":
                checks.append(
                    {
                        "name": "persistence-write-1",
                        "status": "skipped_auth",
                        "detail": persistence_detail,
                    }
                )
        elif verify_probe is not None:
            after = _http_get(f"{url}{verify_probe.path}")
            survived = (
                after is not None
                and after[0] == 200
                and verify_probe.marker in after[2]
            )
            checks.append(
                {
                    "name": "persistence-readback-2",
                    "status": "passed" if survived else "failed",
                    "detail": (
                        f"probe survived restart on {verify_probe.path}"
                        if survived
                        else f"probe written before restart disappeared on {verify_probe.path}"
                    ),
                }
            )
            if not survived:
                failures.append(
                    f"Restart persistence failed: state written to {verify_probe.path} "
                    "before restart was not returned after restart"
                )

        if screenshot_path is not None:
            screenshot = _capture_screenshot(url, screenshot_path)
            checks.append(
                {
                    "name": "screenshot",
                    "status": "passed" if screenshot else "skipped",
                    "detail": screenshot or "Playwright or browser binaries unavailable",
                }
            )
    finally:
        _stop_process(process)
        log.flush()

    if ready and _port_accepting(port):
        foreign_msg = (
            f"Port 127.0.0.1:{port} remained open after stopping boot #{boot_index}; "
            "listener did not belong to the booted process"
        )
        checks.append(
            {"name": f"listener-closed-{boot_index}", "status": "failed", "detail": foreign_msg}
        )
        failures.append(foreign_msg)
        ready = False
        surface_ok = False

    return _BootOutcome(
        port=port,
        url=url,
        ready=ready,
        surface_ok=surface_ok and not failures,
        checks=checks,
        failures=failures,
        screenshot_path=screenshot,
        persistence_probe=probe_created,
        persistence_outcome=persistence_outcome,
        persistence_detail=persistence_detail,
    )


def _delivery_check_failures(receipt: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    results = receipt.get("results")
    if not isinstance(results, list):
        return ["Delivery checks produced no verifiable results"]
    for index, result in enumerate(results, start=1):
        if not isinstance(result, dict):
            failures.append(f"Delivery check {index} produced an unreadable result")
            continue
        label = (
            f"Delivery check {index} "
            f"({result.get('requirement_id', '?')}#{result.get('criterion_index', '?')})"
        )
        reason = result.get("reason")
        exit_code = result.get("exit_code")
        if isinstance(reason, str) and reason:
            failures.append(f"{label} failed: {reason}")
        elif exit_code != 0:
            failures.append(f"{label} exited with code {exit_code}")
    return failures


def _write_report(
    *,
    report_path: Path,
    spec: GoalSpec,
    plan: TaskPlan,
    checks: list[dict[str, Any]],
    failures: list[str],
    entrypoint: str | None,
    data_dir: Path,
    workspace: Path,
    delivery_receipt: dict[str, Any] | None,
) -> None:
    payload = {
        "passed": not failures,
        "deliverable_kind": spec.deliverable_kind.value if spec.deliverable_kind else None,
        "spec_hash": spec.content_hash,
        "plan_hash": plan.content_hash,
        "entrypoint": entrypoint,
        "data_dir": str(data_dir),
        "materialized_workspace": str(workspace),
        "delivery_receipt_id": (
            delivery_receipt.get("receipt_id") if delivery_receipt is not None else None
        ),
        "checks": checks,
        "failures": failures,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _summary(passed: bool, failures: tuple[str, ...], checks: list[dict[str, Any]]) -> str:
    if passed:
        executed = ", ".join(
            check["name"] for check in checks if check.get("status") == "passed"
        )
        return f"Product verified from a clean data directory ({executed})."
    first = failures[0] if failures else "unknown failure"
    return f"Product verification failed: {first}"


def verify_product_delivery(
    *,
    spec: GoalSpec,
    plan: TaskPlan,
    workspace_root: Path,
    overlay_dir: Path,
    run_dir: Path,
    sandbox_root: Path | str | None = None,
    boot_timeout_seconds: float = 20.0,
    delivery_receipt: dict[str, Any] | None = None,
) -> ProductVerificationResult:
    """Verify the composed delivery boots from nothing and serves its primary surface."""
    artifacts = run_dir / "artifacts" / "verify_product"
    artifacts.mkdir(parents=True, exist_ok=True)
    report_path = artifacts / "report.json"
    boot_log_path = artifacts / "boot.log"
    screenshot_path: str | None = None
    checks: list[dict[str, Any]] = []
    failures: list[str] = []
    entrypoint: str | None = None
    receipt: dict[str, Any] | None = None
    sandbox: SandboxHandle | None = None
    workspace: Path | None = None
    data_dir_ctx = tempfile.TemporaryDirectory(prefix="olympus-product-data-")
    data_dir = Path(data_dir_ctx.name)
    try:
        for name in ("home", "tmp", "uploads"):
            (data_dir / name).mkdir(parents=True, exist_ok=True)
        sandbox = create_sandbox(
            workspace_root,
            label=f"{run_dir.name}-{new_id('product')}",
            sandbox_root=sandbox_root,
        )
        workspace = sandbox.workspace_path
        accepted_deletions = sorted(load_accepted_deletions(run_dir))
        if accepted_deletions:
            remove_deleted_paths(workspace, accepted_deletions)
        if overlay_dir.is_dir():
            _copy_overlay_safely(overlay_dir, workspace)
        checks.append(
            {
                "name": "materialize",
                "status": "passed",
                "detail": f"clean sandbox at {workspace}",
            }
        )

        receipt = delivery_receipt
        if receipt is None and plan.delivery_checks:
            receipt = run_delivery_checks(
                plan,
                workspace_root=workspace_root,
                overlay_dir=overlay_dir,
                run_dir=run_dir,
                sandbox_root=sandbox_root,
            )
        delivery_passed = False
        if receipt is not None:
            delivery_failures = _delivery_check_failures(receipt)
            delivery_passed = not delivery_failures
            failures.extend(delivery_failures)
            checks.append(
                {
                    "name": "delivery_checks",
                    "status": "failed" if delivery_failures else "passed",
                    "detail": (
                        "; ".join(delivery_failures)
                        if delivery_failures
                        else "all declared delivery checks passed"
                    ),
                    "receipt_id": receipt.get("receipt_id"),
                }
            )
        else:
            checks.append(
                {
                    "name": "delivery_checks",
                    "status": "skipped",
                    "detail": "no delivery checks declared",
                }
            )

        if spec.deliverable_kind in _HTTP_KINDS:
            command = _discover_server_command(workspace)
            kind = spec.deliverable_kind
            assert kind is not None
            if command is None:
                detail = (
                    f"No runnable {kind.value} entrypoint discovered. Top-level files: "
                    f"{_top_level_listing(workspace)}"
                )
                checks.append({"name": "discovery", "status": "failed", "detail": detail})
                failures.append(detail)
            else:
                entrypoint = command.label
                checks.append(
                    {"name": "discovery", "status": "passed", "detail": command.label}
                )
                screenshot_target = (
                    artifacts / "home.png" if kind == DeliverableKind.WEB_APP else None
                )
                boot_log_path.touch(exist_ok=True)
                with boot_log_path.open("a", encoding="utf-8") as log:
                    first = _boot_and_probe(
                        argv=command.argv,
                        workspace=workspace,
                        workspace_root=workspace_root,
                        data_dir=data_dir,
                        spec=spec,
                        kind=kind,
                        timeout=boot_timeout_seconds,
                        boot_index=1,
                        log=log,
                        screenshot_path=screenshot_target,
                    )
                    checks.extend(first.checks)
                    failures.extend(first.failures)
                    if screenshot_target is not None:
                        screenshot_path = first.screenshot_path
                    data_after_boot_1 = _snapshot_data_dir(data_dir)
                    if first.ready and first.surface_ok:
                        second = _boot_and_probe(
                            argv=command.argv,
                            workspace=workspace,
                            workspace_root=workspace_root,
                            data_dir=data_dir,
                            spec=spec,
                            kind=kind,
                            timeout=boot_timeout_seconds,
                            boot_index=2,
                            log=log,
                            screenshot_path=None,
                            verify_probe=first.persistence_probe,
                        )
                        checks.extend(second.checks)
                        failures.extend(second.failures)
                        if (
                            first.persistence_outcome == "failed"
                            and not delivery_passed
                            and _spec_requires_persistence(spec)
                        ):
                            persist_msg = (
                                "Restart persistence could not be verified: "
                                f"{first.persistence_detail}"
                            )
                            checks.append(
                                {
                                    "name": "restart-persistence",
                                    "status": "failed",
                                    "detail": persist_msg,
                                }
                            )
                            failures.append(persist_msg)
                        elif (
                            _spec_requires_persistence(spec)
                            and first.persistence_probe is None
                            and not delivery_passed
                            and not data_after_boot_1
                        ):
                            persist_msg = (
                                "GoalSpec requires restart persistence, but clean-dir boot "
                                "wrote no persistent state under DATA_DIR and no delivery "
                                "persistence check passed"
                            )
                            checks.append(
                                {
                                    "name": "restart-persistence",
                                    "status": "failed",
                                    "detail": persist_msg,
                                }
                            )
                            failures.append(persist_msg)
                        elif data_after_boot_1:
                            checks.append(
                                {
                                    "name": "data-dir-state",
                                    "status": "passed",
                                    "detail": (
                                        f"{len(data_after_boot_1)} persistent file(s) "
                                        "created under clean DATA_DIR and reused across restart"
                                    ),
                                }
                            )
                    else:
                        checks.append(
                            {
                                "name": "restart",
                                "status": "skipped",
                                "detail": (
                                    "first boot never became ready"
                                    if not first.ready
                                    else "first boot failed primary surface check"
                                ),
                            }
                        )
        else:
            kind_label = spec.deliverable_kind.value if spec.deliverable_kind else "none"
            checks.append(
                {
                    "name": "http_boot",
                    "status": "skipped_non_http",
                    "detail": f"deliverable_kind={kind_label} has no HTTP surface",
                }
            )
    except (OSError, ValueError, RuntimeError, shutil.Error) as exc:
        failures.append(f"Could not verify the delivery from a clean directory: {exc}")
        checks.append({"name": "materialize", "status": "failed", "detail": str(exc)})
    finally:
        data_dir_ctx.cleanup()
        if sandbox is not None:
            remove_sandbox(sandbox)

    failures_tuple = tuple(failures)
    passed = not failures_tuple
    _write_report(
        report_path=report_path,
        spec=spec,
        plan=plan,
        checks=checks,
        failures=list(failures_tuple),
        entrypoint=entrypoint,
        data_dir=data_dir,
        workspace=workspace if workspace is not None else workspace_root,
        delivery_receipt=receipt,
    )
    return ProductVerificationResult(
        passed=passed,
        summary=_summary(passed, failures_tuple, checks),
        failures=failures_tuple,
        report_path=str(report_path),
        boot_log_path=str(boot_log_path) if boot_log_path.exists() else None,
        screenshot_path=screenshot_path,
        checks=checks,
    )
