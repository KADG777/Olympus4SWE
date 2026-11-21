"""Lightweight stdlib HTTP + SSE server for the Ariadne Web Console (Milestone 4)."""

from __future__ import annotations

import contextlib
import json
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import olympus.web.acceptance as delivery_review
from olympus.core.contracts import new_id
from olympus.core.control import RunControl
from olympus.core.store import ConcurrencyConflictError, NotFoundError, RunStore
from olympus.graph.loader import load_workflow
from olympus.web.demos import list_benchmark_presets, materialize_benchmark_demo, seed_demo_runs
from olympus.web.projection import build_run_projection, read_recorded_attempt
from olympus.web.supervisor import RunSupervisor

_STATIC_DIR = Path(__file__).resolve().parent / "static"

# Only these extensions are ever served from /static/; everything else is a 404.
_STATIC_MIME_TYPES: dict[str, str] = {
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".html": "text/html; charset=utf-8",
    ".svg": "image/svg+xml",
}


class AriadneApp:
    """Pure-Python request dispatcher backing the Ariadne HTTP/SSE console."""

    def __init__(
        self,
        *,
        store: RunStore,
        workspace_root: Path | str,
        artifacts_root: Path | None = None,
        supervisor: RunSupervisor | None = None,
    ) -> None:
        self.store = store
        self.control = RunControl(self.store)
        self.workspace_root = Path(workspace_root).resolve()
        self.artifacts_root = (
            artifacts_root
            if artifacts_root is not None
            else self.workspace_root / ".olympus" / "runs"
        )
        self.supervisor = (
            supervisor
            if supervisor is not None
            else RunSupervisor(
                store=self.store,
                workspace_root=self.workspace_root,
                artifacts_root=self.artifacts_root,
            )
        )
        self._retry_guard = threading.Lock()
        self._retry_locks: dict[str, threading.Lock] = {}

    def _retry_lock_for(self, operation_id: str) -> threading.Lock:
        with self._retry_guard:
            return self._retry_locks.setdefault(operation_id, threading.Lock())

    def list_runs_payload(self, *, limit: int = 50) -> dict[str, Any]:
        runs = self.store.list_runs(limit=limit)
        return {
            "ok": True,
            "runs": runs,
            "default_run_id": runs[0]["run_id"] if runs else None,
        }

    def _workspace_for(self, run_id: str) -> Path:
        return self.supervisor.resolve_workspace(run_id)

    def get_run_payload(self, run_id: str) -> dict[str, Any]:
        proj = build_run_projection(
            self.store,
            self._workspace_for(run_id),
            run_id,
            artifacts_root=self.artifacts_root,
        )
        return {"ok": True, "projection": proj}

    def get_events_payload(self, run_id: str, *, after_seq: int = 0) -> dict[str, Any]:
        snap = self.store.get_run(run_id)
        events = [
            e.model_dump(mode="json") for e in self.store.list_events(run_id, after_seq=after_seq)
        ]
        return {
            "ok": True,
            "run_id": run_id,
            "state_revision": snap["state_revision"],
            "status": snap["status"].value,
            "events": events,
        }

    def get_attempt_payload(self, run_id: str, task_id: str, attempt: int) -> dict[str, Any]:
        plan = self.store.get_run(run_id)["plan"]
        artifacts = read_recorded_attempt(self.artifacts_root / run_id, plan, task_id, attempt)
        if artifacts is None:
            raise NotFoundError("Recorded task attempt not found")
        return {"ok": True, "artifacts": artifacts}

    def execute_command(self, run_id: str, body: dict[str, Any]) -> dict[str, Any]:
        """Execute an idempotent HITL command and return updated projection."""
        action = str(body.get("action", "")).strip()
        if not action:
            raise ValueError("Missing required 'action' field")

        operation_id = str(body.get("operation_id") or new_id("op"))
        actor = str(body.get("actor") or "zeus")
        snap = self.store.get_run(run_id)
        ws = self._workspace_for(run_id)
        expected_rev_raw = body.get("expected_state_revision")
        expected_rev = (
            int(expected_rev_raw) if expected_rev_raw is not None else int(snap["state_revision"])
        )
        target_hash: str | None = None
        if "target_hash" in body:
            raw_hash = body["target_hash"]
            target_hash = str(raw_hash) if raw_hash is not None else ""
        auto_advance = bool(body.get("auto_advance", True))
        sync_flag = bool(body.get("sync", False))
        # Omit default flags so legacy command rows replay cleanly while non-default
        # flag changes under the same operation_id still raise a payload conflict.
        dispatch_meta: dict[str, Any] = {}
        if not auto_advance:
            dispatch_meta["auto_advance"] = False
        if sync_flag:
            dispatch_meta["sync"] = True

        if action in ("pause", "resume"):
            cmd_result = self.control.set_paused(
                run_id,
                paused=action == "pause",
                operation_id=operation_id,
                expected_state_revision=expected_rev,
                actor=actor,
            )
        elif action == "approve_spec":
            spec_result = self.control.confirm_spec(
                run_id=run_id,
                approver=actor,
                accept_risk=bool(body.get("accept_risk", False)),
                reason=str(body.get("reason", "")),
                expected_revision=expected_rev,
                expected_target_hash=target_hash,
                operation_id=operation_id,
                command_meta=dispatch_meta,
            )
            cmd_result = (
                spec_result
                if isinstance(spec_result, dict)
                else {
                    "run_id": run_id,
                    "spec_id": spec_result.spec_id,
                    "spec_hash": spec_result.content_hash,
                    "status": spec_result.status.value,
                }
            )
        elif action == "approve_design":
            design_result = self.control.approve_design(
                run_id=run_id,
                approver=actor,
                reason=str(body.get("reason", "")),
                expected_revision=expected_rev,
                expected_target_hash=target_hash,
                operation_id=operation_id,
                command_meta=dispatch_meta,
            )
            cmd_result = (
                design_result
                if isinstance(design_result, dict)
                else design_result.model_dump(mode="json")
            )
        elif action == "approve_plan":
            plan_result = self.control.approve_plan(
                run_id=run_id,
                approver=actor,
                reason=str(body.get("reason", "")),
                expected_revision=expected_rev,
                expected_target_hash=target_hash,
                operation_id=operation_id,
                command_meta=dispatch_meta,
            )
            cmd_result = (
                plan_result
                if isinstance(plan_result, dict)
                else plan_result.model_dump(mode="json")
            )
        elif action == "clarify":
            clarify_result = self.control.clarify_spec(
                run_id=run_id,
                answers=dict(body.get("answers") or {}),
                resolutions=dict(body.get("resolutions") or {}),
                extra_out_of_scope=list(body.get("out_of_scope") or []),
                actor=actor,
                expected_revision=expected_rev,
                operation_id=operation_id,
                command_meta=dispatch_meta,
            )
            cmd_result = (
                clarify_result
                if isinstance(clarify_result, dict)
                else {
                    "run_id": run_id,
                    "version": clarify_result.version,
                    "spec_hash": clarify_result.content_hash,
                }
            )
        elif action == "retry_task":
            task_id = str(body.get("task_id", "")).strip()
            if not task_id:
                raise ValueError("retry_task requires non-empty 'task_id'")
            cmd_result = self.control.retry_task(
                run_id,
                task_id,
                operation_id=operation_id,
                expected_state_revision=expected_rev,
                hint=str(body.get("hint", "")).strip(),
                actor=actor,
                command_meta=dispatch_meta,
            )
        elif action == "accept_risk":
            task_id = str(body.get("task_id", "")).strip()
            reason = str(body.get("reason", "")).strip()
            if not task_id:
                raise ValueError("accept_risk requires non-empty 'task_id'")
            if not reason:
                raise ValueError("accept_risk requires a non-empty 'reason'")
            cmd_result = self.control.accept_task_risk(
                run_id,
                task_id,
                operation_id=operation_id,
                expected_state_revision=expected_rev,
                reason=reason,
                actor=actor,
            )
        elif action == "reassign_human":
            task_id = str(body.get("task_id", "")).strip()
            if not task_id:
                raise ValueError("reassign_human requires non-empty 'task_id'")
            cmd_result = self.control.reassign_task_human(
                run_id,
                task_id,
                operation_id=operation_id,
                expected_state_revision=expected_rev,
                note=str(body.get("note", "")).strip(),
                actor=actor,
            )
        elif action == "retry_workflow":
            retry_payload: dict[str, Any] = {"action": "retry_workflow"}
            if sync_flag:
                retry_payload["sync"] = True
            with self._retry_lock_for(operation_id):
                replay = self.store.lookup_command(
                    operation_id=operation_id,
                    run_id=run_id,
                    action="retry_workflow",
                    payload=retry_payload,
                )
                if replay is not None:
                    cmd_result = replay
                else:
                    pre_snap = self.store.get_run(run_id, expected_revision=expected_rev)
                    pre_rev = int(pre_snap["state_revision"])
                    workflow = self.supervisor.resume_run(run_id, sync=sync_flag)
                    cmd_result = self.control.complete_workflow_retry(
                        run_id,
                        operation_id=operation_id,
                        workflow=workflow,
                        pre_dispatch_revision=pre_rev,
                        actor=actor,
                        sync=sync_flag,
                    )
        elif action == "promote_to_execute":
            target_workflow = str(body.get("workflow_name") or "full_sdlc_tdd").strip()
            target_def = load_workflow(target_workflow)
            if target_def.stop_after != "execute" or target_def.node("execute") is None:
                raise ValueError(
                    f"Target workflow '{target_workflow}' must support stop_after='execute'"
                )
            cur_def = load_workflow(snap["workflow_name"])
            max_llm_calls = (
                int(body["max_llm_calls"]) if body.get("max_llm_calls") is not None else None
            )
            cmd_result = self.control.promote_to_execute(
                run_id,
                target_workflow=target_workflow,
                current_stop_after=cur_def.stop_after,
                operation_id=operation_id,
                expected_state_revision=expected_rev,
                max_llm_calls=max_llm_calls,
                auto_advance=auto_advance,
                sync=sync_flag,
                actor=actor,
            )
        else:
            raise ValueError(f"Unsupported HITL action: {action}")

        if (
            action
            in (
                "clarify",
                "approve_spec",
                "approve_design",
                "approve_plan",
                "promote_to_execute",
                "retry_task",
            )
            and auto_advance
            and not cmd_result.get("deduplicated")
            and (action == "promote_to_execute" or self.supervisor.is_supervised(run_id))
        ):
            cmd_result = {
                **cmd_result,
                "workflow": self.supervisor.resume_run(run_id, sync=sync_flag),
            }

        projection = build_run_projection(
            self.store,
            ws,
            run_id,
            artifacts_root=self.artifacts_root,
        )
        return {
            "ok": True,
            "command_result": cmd_result,
            "projection": projection,
        }


def create_http_server(
    app: AriadneApp,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> ThreadingHTTPServer:
    """Create a ThreadingHTTPServer bound to (host, port) serving the Ariadne console."""

    class AriadneRequestHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _send_json(self, status: int, payload: dict[str, Any]) -> None:
            raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(raw)

        def _send_bytes(self, content_type: str, body: bytes) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_html(self, html_bytes: bytes) -> None:
            self._send_bytes("text/html; charset=utf-8", html_bytes)

        def _serve_static(self, relpath: str) -> None:
            """Serve one whitelisted asset from static/, refusing any escape of the root."""
            root = _STATIC_DIR.resolve()
            requested = unquote(relpath)
            try:
                candidate: Path | None = (root / requested).resolve()
            except (OSError, ValueError):
                candidate = None
            if (
                candidate is None
                or not candidate.is_relative_to(root)
                or not candidate.is_file()
            ):
                self._send_json(
                    HTTPStatus.NOT_FOUND,
                    {"ok": False, "error": f"Unknown static asset: {requested}"},
                )
                return
            content_type = _STATIC_MIME_TYPES.get(candidate.suffix.lower())
            if content_type is None:
                self._send_json(
                    HTTPStatus.NOT_FOUND,
                    {"ok": False, "error": f"Unsupported static type: {candidate.suffix}"},
                )
                return
            self._send_bytes(content_type, candidate.read_bytes())

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            path = parsed.path.rstrip("/") or "/"
            qs = parse_qs(parsed.query)

            try:
                if path.startswith("/static/"):
                    self._serve_static(path[len("/static/") :])
                    return

                if path in ("/", "/index.html"):
                    index_file = _STATIC_DIR / "index.html"
                    if not index_file.exists():
                        self._send_json(
                            HTTPStatus.NOT_FOUND,
                            {"ok": False, "error": "Console static/index.html not found"},
                        )
                        return
                    self._send_html(index_file.read_bytes())
                    return

                if path == "/favicon.ico":
                    self._send_bytes("image/x-icon", b"")
                    return

                if path == "/healthz":
                    self._send_json(
                        HTTPStatus.OK,
                        {"ok": True, "service": "olympus-ariadne", "version": "0.1.0"},
                    )
                    return

                if path == "/api/runs":
                    limit = int(qs.get("limit", ["50"])[0])
                    self._send_json(HTTPStatus.OK, app.list_runs_payload(limit=limit))
                    return

                if path == "/api/providers":
                    self._send_json(
                        HTTPStatus.OK,
                        {"ok": True, "providers": app.supervisor.list_providers_status()},
                    )
                    return

                if path == "/api/workflows":
                    self._send_json(
                        HTTPStatus.OK,
                        {"ok": True, "workflows": app.supervisor.list_workflows()},
                    )
                    return

                if path == "/api/demos":
                    self._send_json(
                        HTTPStatus.OK,
                        {"ok": True, "demos": list_benchmark_presets()},
                    )
                    return

                if path.startswith("/api/runs/"):
                    parts = [p for p in path.split("/") if p]
                    # parts: ['api', 'runs', '<run_id>', ...]
                    if len(parts) == 3:
                        run_id = parts[2]
                        self._send_json(HTTPStatus.OK, app.get_run_payload(run_id))
                        return
                    if len(parts) == 4 and parts[3] == "events":
                        run_id = parts[2]
                        after_seq = int(qs.get("after_seq", ["0"])[0])
                        self._send_json(
                            HTTPStatus.OK,
                            app.get_events_payload(run_id, after_seq=after_seq),
                        )
                        return
                    if len(parts) == 6 and parts[3] == "delivery-log":
                        run_id, event_seq, position = parts[2], parts[4], parts[5]
                        projection = app.get_run_payload(run_id)["projection"]
                        review = projection["delivery_acceptance"] or {}
                        for row in review.get("criteria", []):
                            evidence = row["evidence"]
                            if (
                                evidence
                                and str(evidence["event_seq"]) == event_seq
                                and str(evidence["check_position"]) == position
                            ):
                                log = delivery_review.read_delivery_log(
                                    app.artifacts_root / run_id, evidence
                                )
                                if log is not None:
                                    self._send_bytes("text/plain; charset=utf-8", log)
                                    return
                        raise NotFoundError("No matching current delivery evidence")
                    if len(parts) == 4 and parts[3] == "stream":
                        run_id = parts[2]
                        after_seq = int(qs.get("after_seq", ["0"])[0])
                        once = qs.get("once", ["0"])[0] == "1"
                        self._handle_sse(run_id, after_seq=after_seq, once=once)
                        return
                    if len(parts) == 6 and parts[3] == "task-artifacts" and parts[5].isdigit():
                        self._send_json(
                            HTTPStatus.OK,
                            app.get_attempt_payload(parts[2], unquote(parts[4]), int(parts[5])),
                        )
                        return

                self._send_json(
                    HTTPStatus.NOT_FOUND,
                    {"ok": False, "error": f"Unknown route: {path}"},
                )
            except NotFoundError as exc:
                self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": str(exc)})
            except Exception as exc:
                self._send_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"ok": False, "error": str(exc)},
                )

        def _handle_sse(self, run_id: str, *, after_seq: int = 0, once: bool = False) -> None:
            # Validate run exists before starting stream
            payload = app.get_events_payload(run_id, after_seq=after_seq)
            self.close_connection = True
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()

            def _write_sse(event_name: str, data: dict[str, Any]) -> None:
                chunk = f"event: {event_name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
                self.wfile.write(chunk.encode("utf-8"))
                self.wfile.flush()

            try:
                _write_sse("state", payload)
                if once:
                    return

                last_rev = int(payload["state_revision"])
                events = payload["events"]
                cur_seq = int(events[-1]["event_seq"]) if events else after_seq
                ticks = 0
                while True:
                    time.sleep(0.5)
                    ticks += 1
                    nxt = app.get_events_payload(run_id, after_seq=cur_seq)
                    if int(nxt["state_revision"]) != last_rev or nxt["events"]:
                        last_rev = int(nxt["state_revision"])
                        if nxt["events"]:
                            cur_seq = int(nxt["events"][-1]["event_seq"])
                        _write_sse("state", nxt)
                    elif ticks % 10 == 0:
                        self.wfile.write(b": heartbeat\n\n")
                        self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                return
            finally:
                self.close_connection = True

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            path = parsed.path.rstrip("/")
            try:
                content_len = int(self.headers.get("Content-Length", "0"))
                raw_body = self.rfile.read(content_len) if content_len > 0 else b"{}"
                body = json.loads(raw_body.decode("utf-8") or "{}")

                if path == "/api/seed-demo":
                    seeded = seed_demo_runs(
                        app.store,
                        app.workspace_root,
                        artifacts_root=app.artifacts_root,
                    )
                    self._send_json(
                        HTTPStatus.OK,
                        {
                            "ok": True,
                            "seeded": seeded,
                            **app.list_runs_payload(),
                        },
                    )
                    return

                parts = [p for p in path.split("/") if p]
                if len(parts) == 2 and parts[0] == "api" and parts[1] == "runs":
                    demo_id = str(body.get("demo_id") or "").strip()
                    preset: dict[str, Any] | None = None
                    resolved_ws: Path | str | None = body.get("workspace_root")
                    if demo_id:
                        preset, demo_ws = materialize_benchmark_demo(
                            demo_id,
                            workspace_parent=app.workspace_root / ".olympus" / "workspaces",
                        )
                        if not resolved_ws:
                            resolved_ws = demo_ws
                    elif resolved_ws:
                        ws_candidate = Path(str(resolved_ws)).resolve()
                        if not ws_candidate.is_dir():
                            raise FileNotFoundError(
                                f"workspace_root not found: {resolved_ws}"
                            )
                        resolved_ws = ws_candidate

                    default_goal = str(preset["raw_input"]) if preset else ""
                    raw_input = str(
                        body.get("raw_input") or body.get("goal") or default_goal
                    ).strip()
                    if not raw_input:
                        self._send_json(
                            HTTPStatus.BAD_REQUEST,
                            {
                                "ok": False,
                                "error": "POST /api/runs requires a non-empty 'raw_input'",
                            },
                        )
                        return
                    raw_constraints = body.get("constraints")
                    if isinstance(raw_constraints, list):
                        constraints = [
                            str(item).strip()
                            for item in raw_constraints
                            if str(item).strip()
                        ]
                    elif preset:
                        constraints = list(preset.get("constraints") or [])
                    else:
                        constraints = []
                    default_title = str(preset["title"]) if preset else raw_input[:60]
                    default_workflow = (
                        str(preset["default_workflow"]) if preset else "product_blueprint"
                    )
                    started = app.supervisor.start_run(
                        title=str(body.get("title") or default_title),
                        raw_input=raw_input,
                        workflow_name=str(body.get("workflow_name") or default_workflow),
                        provider_config=dict(body.get("provider_config") or {}),
                        auto_approve=bool(body.get("auto_approve", False)),
                        sync=bool(body.get("sync", False)),
                        constraints=constraints,
                        workspace_root=resolved_ws,
                    )
                    self._send_json(HTTPStatus.OK, {"ok": True, **started})
                    return

                if (
                    len(parts) == 4
                    and parts[0] == "api"
                    and parts[1] == "runs"
                    and parts[3] == "command"
                ):
                    run_id = parts[2]
                    res = app.execute_command(run_id, body)
                    self._send_json(HTTPStatus.OK, res)
                    return

                self._send_json(
                    HTTPStatus.NOT_FOUND,
                    {"ok": False, "error": f"Unknown POST route: {path}"},
                )
            except NotFoundError as exc:
                self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": str(exc)})
            except FileNotFoundError as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})
            except ConcurrencyConflictError as exc:
                conflict_payload: dict[str, Any] = {
                    "ok": False,
                    "conflict": True,
                    "error": str(exc),
                }
                req_parts = [p for p in path.split("/") if p]
                if len(req_parts) == 4 and req_parts[0] == "api" and req_parts[1] == "runs":
                    with contextlib.suppress(Exception):
                        conflict_payload["projection"] = app.get_run_payload(
                            req_parts[2]
                        )["projection"]
                self._send_json(HTTPStatus.CONFLICT, conflict_payload)
            except ValueError as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})
            except Exception as exc:
                self._send_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"ok": False, "error": str(exc)},
                )

    return ThreadingHTTPServer((host, port), AriadneRequestHandler)


def serve_ariadne(
    *,
    store: RunStore,
    workspace_root: Path | str,
    host: str = "127.0.0.1",
    port: int = 8765,
    seed_demo: bool = False,
) -> None:
    """Run the Ariadne console HTTP server in the foreground."""
    ws = Path(workspace_root).resolve()
    if seed_demo:
        seed_demo_runs(store, ws)
    app = AriadneApp(store=store, workspace_root=ws)
    app.supervisor.recover_orphaned_runs()
    httpd = create_http_server(app, host=host, port=port)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
