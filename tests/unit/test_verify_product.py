"""Clean-directory product verification tests: boot, surface, restart, and skips."""

from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest

from olympus.core.contracts import (
    AcceptanceCheck,
    CheckKind,
    DeliverableKind,
    DeliveryCheck,
    GoalSpec,
    TaskNode,
    TaskPlan,
)
from olympus.stages.verify_product import PRODUCT_VERIFIED_EVENT, verify_product_delivery

_WEB_SERVER = """
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(os.environ["PORT"])
DB_PATH = os.environ["DB_PATH"]
try:
    with open(DB_PATH, "r", encoding="utf-8") as handle:
        BOOT_STATE = handle.read()
except FileNotFoundError:
    BOOT_STATE = "created"
    with open(DB_PATH, "w", encoding="utf-8") as handle:
        handle.write(BOOT_STATE)

BODY = {body}

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            self.send_response({status})
            self.send_header("Content-Type", {content_type!r})
            self.send_header("Content-Length", str(len(BODY)))
            self.end_headers()
            self.wfile.write(BODY)
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'{{"detail":"Not Found"}}')

    def log_message(self, format, *args):
        pass


ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
"""

_API_SERVER = """
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(os.environ["PORT"])
PAYLOAD = json.dumps({"status": "ok"}).encode()

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(PAYLOAD)))
            self.end_headers()
            self.wfile.write(PAYLOAD)
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'{"detail":"Not Found"}')

    def log_message(self, format, *args):
        pass


ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
"""


def _write_web_app(workspace: Path, *, status: int) -> None:
    if status == 200:
        body = '("<html><body><h1>blacknote</h1><p>" + BOOT_STATE + "</p></body></html>").encode()'
        content_type = "text/html"
    else:
        body = "b'{\"detail\":\"Not Found\"}'"
        content_type = "application/json"
    source = _WEB_SERVER.format(body=body, status=status, content_type=content_type)
    (workspace / "app.py").write_text(source, encoding="utf-8")


def _spec(kind: DeliverableKind | None) -> GoalSpec:
    return GoalSpec(
        spec_id="spec-product",
        goal="Ship the blacknote product",
        requirements=[
            {
                "id": "R1",
                "description": "Serve the primary product surface",
                "acceptance_criteria": ["GET / returns 200"],
            }
        ],
        deliverable_kind=kind,
    )


def _plan(
    spec: GoalSpec, *, delivery_checks: list[DeliveryCheck] | None = None
) -> TaskPlan:
    return TaskPlan(
        spec_id=spec.spec_id,
        spec_hash=spec.content_hash,
        tasks=[
            TaskNode(
                task_id="T1",
                title="Serve the product",
                objective="Deliver the server entrypoint",
                requirement_refs=["R1"],
                write_scope=["app.py"],
                acceptance_checks=[
                    AcceptanceCheck(
                        id="C1",
                        kind=CheckKind.COMMAND,
                        description="Entrypoint exists",
                        command="test -f app.py",
                    )
                ],
            )
        ],
        delivery_checks=delivery_checks or [],
    )


def _paths(tmp_path: Path) -> tuple[Path, Path]:
    workspace = tmp_path / "repo"
    workspace.mkdir(exist_ok=True)
    run_dir = tmp_path / "run"
    (run_dir / "accepted_overlay").mkdir(parents=True, exist_ok=True)
    return workspace, run_dir


def _verify(tmp_path: Path, *, kind: DeliverableKind | None):
    workspace, run_dir = _paths(tmp_path)
    spec = _spec(kind)
    plan = _plan(spec)
    return verify_product_delivery(
        spec=spec,
        plan=plan,
        workspace_root=workspace,
        overlay_dir=run_dir / "accepted_overlay",
        run_dir=run_dir,
        sandbox_root=tmp_path / "sandboxes",
        boot_timeout_seconds=8.0,
    )


@pytest.mark.parametrize(
    ("kind", "source"),
    [
        (DeliverableKind.WEB_APP, None),
        (DeliverableKind.REST_API, _API_SERVER),
    ],
)
def test_http_kind_boots_primary_surface_and_clean_restart(
    tmp_path: Path,
    kind: DeliverableKind,
    source: str | None,
) -> None:
    workspace, _ = _paths(tmp_path)
    if source is None:
        _write_web_app(workspace, status=200)
    else:
        (workspace / "app.py").write_text(source, encoding="utf-8")

    result = _verify(tmp_path, kind=kind)

    assert result.passed is True, result.failures
    assert result.boot_log_path is not None
    report = json.loads(Path(result.report_path or "").read_text(encoding="utf-8"))
    assert report["passed"] is True and report["deliverable_kind"] == kind.value
    boots = [check for check in report["checks"] if check["name"].startswith("boot-")]
    surfaces = [
        check for check in report["checks"] if check["name"].startswith("primary-surface-")
    ]
    assert len(boots) == 2 and len(surfaces) == 2
    assert all(check["status"] == "passed" for check in boots + surfaces)
    assert boots[0]["port"] != boots[1]["port"]
    for boot in boots:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.settimeout(0.5)
            assert probe.connect_ex(("127.0.0.1", boot["port"])) != 0
    # The temp data directory is removed and nothing leaks into the repo.
    assert not Path(report["data_dir"]).exists()
    assert not list(workspace.rglob("*.db"))


def test_web_app_serving_404_on_root_fails_loud_with_artifacts(tmp_path: Path) -> None:
    workspace, _ = _paths(tmp_path)
    _write_web_app(workspace, status=404)

    result = _verify(tmp_path, kind=DeliverableKind.WEB_APP)

    assert result.passed is False
    assert any("GET /" in failure and "404" in failure for failure in result.failures)
    assert result.report_path is not None and Path(result.report_path).is_file()
    assert result.boot_log_path is not None and Path(result.boot_log_path).is_file()
    report = json.loads(Path(result.report_path).read_text(encoding="utf-8"))
    assert report["passed"] is False
    assert any(
        check["name"] == "primary-surface-1" and check["status"] == "failed"
        for check in report["checks"]
    )


def test_web_app_without_entrypoint_fails_loud_listing_files(tmp_path: Path) -> None:
    workspace, _ = _paths(tmp_path)
    (workspace / "README.md").write_text("# nothing to boot\n", encoding="utf-8")

    result = _verify(tmp_path, kind=DeliverableKind.WEB_APP)

    assert result.passed is False
    assert any(
        "No runnable web_app entrypoint" in failure and "README.md" in failure
        for failure in result.failures
    )


@pytest.mark.parametrize("kind", [None, DeliverableKind.CLI, DeliverableKind.LIBRARY])
def test_non_http_deliverables_skip_boot_cleanly(
    tmp_path: Path, kind: DeliverableKind | None
) -> None:
    result = _verify(tmp_path, kind=kind)

    assert result.passed is True
    assert result.boot_log_path is None
    assert any(
        check["name"] == "http_boot" and check["status"] == "skipped_non_http"
        for check in result.checks
    )
    report = json.loads(Path(result.report_path or "").read_text(encoding="utf-8"))
    assert report["deliverable_kind"] == (kind.value if kind else None)


def test_precomputed_delivery_receipt_is_not_executed_twice(tmp_path: Path) -> None:
    workspace, run_dir = _paths(tmp_path)
    spec = _spec(DeliverableKind.CLI)
    failing_check = DeliveryCheck(
        requirement_id="R1", criterion_index=1, command="exit 1"
    )
    plan = _plan(spec, delivery_checks=[failing_check])
    receipt = {
        "receipt_id": "delivery-preseeded",
        "results": [
            {
                "requirement_id": "R1",
                "criterion_index": 1,
                "delivery_version": "sha256:" + "0" * 64,
                "exit_code": 0,
                "log_hash": "sha256:" + "1" * 64,
            }
        ],
    }

    result = verify_product_delivery(
        spec=spec,
        plan=plan,
        workspace_root=workspace,
        overlay_dir=run_dir / "accepted_overlay",
        run_dir=run_dir,
        sandbox_root=tmp_path / "sandboxes",
        boot_timeout_seconds=2.0,
        delivery_receipt=receipt,
    )

    # The declared command would fail if re-executed; the supplied receipt is authoritative.
    assert result.passed is True, result.failures
    assert any(
        check["name"] == "delivery_checks" and check.get("receipt_id") == "delivery-preseeded"
        for check in result.checks
    )
    assert not (run_dir / "artifacts" / "delivery-preseeded").exists()


def test_failing_delivery_receipt_blocks_product_pass(tmp_path: Path) -> None:
    workspace, run_dir = _paths(tmp_path)
    spec = _spec(DeliverableKind.CLI)
    plan = _plan(spec, delivery_checks=[DeliveryCheck(
        requirement_id="R1", criterion_index=1, command="exit 1"
    )])
    receipt = {
        "receipt_id": "delivery-failed",
        "results": [
            {
                "requirement_id": "R1",
                "criterion_index": 1,
                "delivery_version": "sha256:" + "0" * 64,
                "exit_code": 1,
            }
        ],
    }

    result = verify_product_delivery(
        spec=spec,
        plan=plan,
        workspace_root=workspace,
        overlay_dir=run_dir / "accepted_overlay",
        run_dir=run_dir,
        sandbox_root=tmp_path / "sandboxes",
        boot_timeout_seconds=2.0,
        delivery_receipt=receipt,
    )

    assert result.passed is False
    assert any("Delivery check 1" in failure and "exit" in failure for failure in result.failures)


def test_product_verified_event_name_is_stable() -> None:
    assert PRODUCT_VERIFIED_EVENT == "product.verified"


def test_symlink_in_overlay_or_workspace_is_rejected_without_escape(tmp_path: Path) -> None:
    workspace, run_dir = _paths(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("SAFE", encoding="utf-8")
    (workspace / "escape.txt").symlink_to(outside)
    (run_dir / "accepted_overlay" / "escape.txt").write_text("PWNED", encoding="utf-8")

    spec = _spec(DeliverableKind.CLI)
    plan = _plan(spec)
    result = verify_product_delivery(
        spec=spec,
        plan=plan,
        workspace_root=workspace,
        overlay_dir=run_dir / "accepted_overlay",
        run_dir=run_dir,
        sandbox_root=tmp_path / "sandboxes",
        boot_timeout_seconds=4.0,
    )

    assert result.passed is False
    assert outside.read_text(encoding="utf-8") == "SAFE"
    assert any("symlink" in failure.lower() for failure in result.failures)

    # Also reject a symlinked overlay_dir root itself.
    real_overlay = tmp_path / "real_overlay"
    real_overlay.mkdir()
    (real_overlay / "x.txt").write_text("hi", encoding="utf-8")
    link_overlay = tmp_path / "link_overlay"
    link_overlay.symlink_to(real_overlay)
    ws2 = tmp_path / "repo2"
    ws2.mkdir()
    run2 = tmp_path / "run2"
    run2.mkdir()
    res_link = verify_product_delivery(
        spec=spec,
        plan=plan,
        workspace_root=ws2,
        overlay_dir=link_overlay,
        run_dir=run2,
        sandbox_root=tmp_path / "sandboxes",
        boot_timeout_seconds=4.0,
    )
    assert res_link.passed is False
    assert any("symlink" in failure.lower() for failure in res_link.failures)


def test_product_boot_uses_allowlisted_env_and_isolated_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OLYMPUS_TEST_HOST_SECRET", "super-secret-token")
    workspace, _ = _paths(tmp_path)
    (workspace / "app.py").write_text(
        """
import json, os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(os.environ["PORT"])
LEAKED = "OLYMPUS_TEST_HOST_SECRET" in os.environ
HOME = os.environ.get("HOME", "")
DATA_DIR = os.environ.get("DATA_DIR", "")
BODY = (
    f"<html><body><h1>ok</h1><p>leaked={LEAKED}</p></body></html>"
    if (not LEAKED and HOME.startswith(DATA_DIR))
    else "LEAKED"
).encode()
STATUS = 200 if (not LEAKED and HOME.startswith(DATA_DIR)) else 500

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(STATUS)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(BODY)))
        self.end_headers()
        self.wfile.write(BODY)
    def log_message(self, format, *args):
        pass

ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
""",
        encoding="utf-8",
    )

    result = _verify(tmp_path, kind=DeliverableKind.WEB_APP)

    assert result.passed is True, result.failures


def test_json_with_html_substring_fails_web_app_and_html_fails_rest_api(tmp_path: Path) -> None:
    web_dir = tmp_path / "web_case"
    web_dir.mkdir()
    ws_web, _ = _paths(web_dir)
    (ws_web / "app.py").write_text(
        """
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
PORT = int(os.environ["PORT"])
BODY = b'{"html": "<div>not a page</div>"}'
class H(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(BODY)))
        self.end_headers()
        self.wfile.write(BODY)
    def log_message(self, format, *args): pass
ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
""",
        encoding="utf-8",
    )
    web_res = _verify(web_dir, kind=DeliverableKind.WEB_APP)
    assert web_res.passed is False
    assert any("content-type" in f.lower() for f in web_res.failures)

    api_dir = tmp_path / "api_case"
    api_dir.mkdir()
    ws_api, _ = _paths(api_dir)
    _write_web_app(ws_api, status=200)
    api_res = _verify(api_dir, kind=DeliverableKind.REST_API)
    assert api_res.passed is False
    assert any("non-html" in f.lower() for f in api_res.failures)

    xhtml_dir = tmp_path / "xhtml_case"
    xhtml_dir.mkdir()
    ws_xhtml, _ = _paths(xhtml_dir)
    (ws_xhtml / "app.py").write_text(
        """
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
PORT = int(os.environ["PORT"])
BODY = b"<html><body><h1>xhtml</h1></body></html>"
class H(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/xhtml+xml")
        self.send_header("Content-Length", str(len(BODY)))
        self.end_headers()
        self.wfile.write(BODY)
    def log_message(self, format, *args): pass
ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
""",
        encoding="utf-8",
    )
    xhtml_res = _verify(xhtml_dir, kind=DeliverableKind.WEB_APP)
    assert xhtml_res.passed is False
    assert any("content-type" in f.lower() for f in xhtml_res.failures)


def test_sleep_entrypoint_cannot_piggyback_on_foreign_listener(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from olympus.stages import verify_product as vp

    class ForeignHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = b"<html><body><h1>foreign</h1></body></html>"
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            pass

    foreign = ThreadingHTTPServer(("127.0.0.1", 0), ForeignHandler)
    foreign_port = int(foreign.server_address[1])
    thread = threading.Thread(target=foreign.serve_forever, daemon=True)
    thread.start()
    try:
        workspace, run_dir = _paths(tmp_path)
        (workspace / "app.py").write_text("import time\ntime.sleep(30)\n", encoding="utf-8")
        monkeypatch.setattr(vp, "_free_port", lambda: foreign_port)
        spec = _spec(DeliverableKind.WEB_APP)
        plan = _plan(spec)
        result = verify_product_delivery(
            spec=spec,
            plan=plan,
            workspace_root=workspace,
            overlay_dir=run_dir / "accepted_overlay",
            run_dir=run_dir,
            sandbox_root=tmp_path / "sandboxes",
            boot_timeout_seconds=1.2,
        )
        assert result.passed is False
        assert any("different process" in f or "did not belong" in f for f in result.failures)
    finally:
        foreign.shutdown()
        foreign.server_close()


def test_restart_persistence_probe_rejects_in_memory_only_app_and_passes_persisted(
    tmp_path: Path,
) -> None:
    def _stateful_spec() -> GoalSpec:
        return GoalSpec(
            spec_id="spec-persist",
            goal="Store items persistently across restarts",
            requirements=[
                {
                    "id": "R1",
                    "description": "Persist items across server restart",
                    "acceptance_criteria": ["POST /items and GET /items survive restart"],
                }
            ],
            deliverable_kind=DeliverableKind.WEB_APP,
            observable_surfaces=["GET /", "POST /items", "GET /items"],
        )

    def _app_source(persist: bool) -> str:
        return f"""
import json, os
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(os.environ["PORT"])
DB = Path(os.environ["DB_PATH"])
ITEMS = json.loads(DB.read_text("utf-8")) if ({persist!r} and DB.exists()) else []

class H(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            body = b"<html><body><h1>items</h1></body></html>"
            ctype = "text/html"
        elif self.path == "/items":
            body = json.dumps({{"items": ITEMS}}).encode()
            ctype = "application/json"
        else:
            self.send_response(404); self.end_headers(); return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path != "/items":
            self.send_response(404); self.end_headers(); return
        raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        ITEMS.append(json.loads(raw))
        if {persist!r}:
            DB.write_text(json.dumps(ITEMS), encoding="utf-8")
        body = json.dumps({{"ok": True}}).encode()
        self.send_response(201)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args): pass

ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
"""

    mem_dir = tmp_path / "mem"
    mem_dir.mkdir()
    ws_mem, run_mem = _paths(mem_dir)
    (ws_mem / "app.py").write_text(_app_source(False), encoding="utf-8")
    spec = _stateful_spec()
    mem_res = verify_product_delivery(
        spec=spec,
        plan=_plan(spec),
        workspace_root=ws_mem,
        overlay_dir=run_mem / "accepted_overlay",
        run_dir=run_mem,
        sandbox_root=mem_dir / "sandboxes",
        boot_timeout_seconds=6.0,
    )
    assert mem_res.passed is False
    assert any("restart persistence failed" in f.lower() for f in mem_res.failures)

    disk_dir = tmp_path / "disk"
    disk_dir.mkdir()
    ws_disk, run_disk = _paths(disk_dir)
    (ws_disk / "app.py").write_text(_app_source(True), encoding="utf-8")
    disk_res = verify_product_delivery(
        spec=spec,
        plan=_plan(spec),
        workspace_root=ws_disk,
        overlay_dir=run_disk / "accepted_overlay",
        run_dir=run_disk,
        sandbox_root=disk_dir / "sandboxes",
        boot_timeout_seconds=6.0,
    )
    assert disk_res.passed is True, disk_res.failures

    # A declared POST /items returning 422 must fail persistence even if boot writes a stamp file.
    bad_dir = tmp_path / "bad_post"
    bad_dir.mkdir()
    ws_bad, run_bad = _paths(bad_dir)
    (ws_bad / "app.py").write_text(
        """
import os
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
PORT = int(os.environ["PORT"])
Path(os.environ["DB_PATH"]).write_text("boot-stamp", encoding="utf-8")
class H(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            body, ctype = b"<html><body><h1>items</h1></body></html>", "text/html"
        elif self.path == "/items":
            body, ctype = b'{"items": []}', "application/json"
        else:
            self.send_response(404); self.end_headers(); return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def do_POST(self):
        self.send_response(422)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"{}")
    def log_message(self, format, *args): pass
ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
""",
        encoding="utf-8",
    )
    bad_res = verify_product_delivery(
        spec=spec,
        plan=_plan(spec),
        workspace_root=ws_bad,
        overlay_dir=run_bad / "accepted_overlay",
        run_dir=run_bad,
        sandbox_root=bad_dir / "sandboxes",
        boot_timeout_seconds=6.0,
    )
    assert bad_res.passed is False
    assert any("422" in f for f in bad_res.failures)
