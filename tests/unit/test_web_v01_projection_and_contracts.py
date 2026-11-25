"""v0.1 read-only projection contracts: whitelists, post-DAG rounds, artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from olympus.core.contracts import (
    AcceptanceCheck,
    ApprovalDecision,
    ApprovalRecord,
    ApprovalTarget,
    CheckKind,
    GoalSpec,
    RequirementItem,
    RunStatus,
    SpecStatus,
    TaskNode,
    TaskPlan,
    TaskStatus,
)
from olympus.core.store import RunStore
from olympus.web.projection import build_run_projection
from olympus.web.schemas import RunProjection
from olympus.web.server import AriadneApp, create_fastapi_app


def _seed_approved_run(
    store: RunStore,
    *,
    status: RunStatus = RunStatus.AWAITING_HUMAN,
    task_status: TaskStatus = TaskStatus.ACCEPTED,
) -> str:
    run_id = store.create_run(
        title="Ariadne v0.1 contract run",
        workflow_name="full_sdlc_tdd",
        provider_config={"provider": "mock"},
    )
    spec = GoalSpec(
        spec_id="spec-v01",
        version=1,
        goal="交付一个可被端到端验证的候补递补服务。",
        requirements=[
            RequirementItem(
                id="R1",
                description="取消报名后自动递补最早的有效候补者",
                acceptance_criteria=["递补在同一事务内完成"],
                source_ref="REQ-V01",
            )
        ],
        status=SpecStatus.CONFIRMED,
    )
    store.save_spec(run_id, spec, actor="align")
    store.record_approval(
        ApprovalRecord(
            run_id=run_id,
            target=ApprovalTarget.GOAL_SPEC,
            target_id=spec.spec_id,
            target_hash=spec.content_hash,
            approver="zeus",
            decision=ApprovalDecision.APPROVED,
            reason="spec confirmed",
        )
    )
    plan = TaskPlan(
        plan_id="plan-v01",
        spec_id=spec.spec_id,
        spec_hash=spec.content_hash,
        version=1,
        tasks=[
            TaskNode(
                task_id="T01",
                title="递补服务",
                objective="实现事务内递补。",
                requirement_refs=["R1"],
                write_scope=["src/waitlist/service.py"],
                acceptance_checks=[
                    AcceptanceCheck(
                        id="T01-cmd",
                        kind=CheckKind.COMMAND,
                        description="运行递补单测",
                        command="pytest tests/test_waitlist.py -q",
                        passed=True,
                    )
                ],
                status=task_status,
                attempt=1,
            )
        ],
    )
    store.save_plan(run_id, plan, actor="orchestrate")
    store.record_approval(
        ApprovalRecord(
            run_id=run_id,
            target=ApprovalTarget.TASK_PLAN,
            target_id=plan.plan_id,
            target_hash=plan.content_hash,
            approver="zeus",
            decision=ApprovalDecision.APPROVED,
            reason="plan approved",
        )
    )
    store.set_run_status(run_id, status, actor="chronos", stage="verify")
    return run_id


def _append_product_verified(
    store: RunStore, run_id: str, *, passed: bool, extra: dict[str, object] | None = None
) -> None:
    payload: dict[str, object] = {
        "passed": passed,
        "summary": "产品端到端验证通过" if passed else "产品端到端验证未通过",
        "failures": [] if passed else ["boot.log 显示启动失败"],
        "report_path": "/tmp/secret/report.json",
        "boot_log_path": "/tmp/secret/boot.log",
        "screenshot_path": None,
        "checks": [
            {
                "name": "http_boot",
                "status": "passed" if passed else "failed",
                "detail": "GET / 200",
                "port": 8123,
                "unexpected": {"nested": True},
            }
        ],
        "unexpected_top_level": "ignored",
    }
    if extra:
        payload.update(extra)
    store.append_event(
        run_id=run_id, stage="verify", event_type="product.verified", actor="chronos",
        payload=payload,
    )


def test_product_verification_and_post_dag_whitelists_drop_unknown_fields(
    tmp_path: Path,
) -> None:
    store = RunStore(tmp_path / "runs.db")
    run_id = _seed_approved_run(store)
    verify_dir = tmp_path / ".olympus" / "runs" / run_id / "artifacts" / "verify_product"
    verify_dir.mkdir(parents=True)
    (verify_dir / "report.json").write_text('{"ok": true}', encoding="utf-8")

    _append_product_verified(store, run_id, passed=False)
    store.append_event(
        run_id=run_id,
        stage="verify",
        event_type="post_dag.repro_rechecked",
        actor="chronos",
        payload={
            "repair_round": 1,
            "checked_count": 2,
            "still_failing_count": 1,
            "unexpected": "ignored",
        },
    )
    store.append_event(
        run_id=run_id,
        stage="verify",
        event_type="review.verdict_recorded",
        actor="argus",
        payload={
            "task_id": "post-dag",
            "attempt": 1,
            "review_cycle": 1,
            "passed": False,
            "important_count": 1,
            "nit_count": 0,
            "unexpected": 42,
        },
    )

    projection = build_run_projection(store, tmp_path, run_id)
    RunProjection.model_validate(projection)

    acceptance = projection["delivery_acceptance"]
    assert acceptance is not None
    verification = acceptance["product_verification"]
    assert verification["passed"] is False
    assert verification["failures"] == ["boot.log 显示启动失败"]
    assert verification["checks"] == [
        {"name": "http_boot", "status": "failed", "detail": "GET / 200", "port": 8123}
    ]
    assert verification["has_report"] is True
    assert verification["has_boot_log"] is False
    assert verification["has_screenshot"] is False
    assert "/tmp/secret" not in json.dumps(verification, ensure_ascii=False)

    outputs = [item for item in projection["key_outputs"] if item["artifact_id"] == "product"]
    assert outputs and outputs[0]["kind"] == "verification"
    assert outputs[0]["current"] is False
    assert outputs[0]["detail_available"] is False
    assert "未通过" in outputs[0]["summary"]

    post_dag = projection["post_dag"]
    assert post_dag is not None
    assert post_dag["total_rounds"] == 3
    assert [row["sequence"] for row in post_dag["rounds"]] == [1, 2, 3]
    assert [row["phase"] for row in post_dag["rounds"]] == ["verify", "recheck", "review"]
    assert post_dag["rounds"][1]["repro_checked"] == 2
    assert post_dag["rounds"][1]["repro_failed"] == 1
    assert post_dag["status"] == "awaiting_human"


def test_post_dag_sequences_stay_monotonic_across_human_resume(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs.db")
    run_id = _seed_approved_run(store)
    for repair_round, event_type in (
        (1, "post_dag.repair_turn_completed"),
        (2, "post_dag.repair_applied"),
    ):
        store.append_event(
            run_id=run_id,
            stage="orchestrate",
            event_type=event_type,
            actor="chronos",
            payload={
                "repair_round": repair_round,
                "completed": True,
                "stop_reason": "done",
                "changed_paths": ["src/waitlist/service.py"],
            },
        )
    store.set_run_status(run_id, RunStatus.AWAITING_HUMAN, actor="chronos", stage="verify")
    store.set_run_status(run_id, RunStatus.RUNNING, actor="zeus", stage="verify")
    store.append_event(
        run_id=run_id,
        stage="orchestrate",
        event_type="post_dag.repair_turn_completed",
        actor="chronos",
        payload={"repair_round": 1, "completed": True, "stop_reason": "done"},
    )

    post_dag = build_run_projection(store, tmp_path, run_id)["post_dag"]
    assert post_dag is not None
    rounds = post_dag["rounds"]
    assert [row["sequence"] for row in rounds] == [1, 2, 3]
    assert [row["reported_round"] for row in rounds] == [1, 2, 1]
    assert len({row["event_seq"] for row in rounds}) == 3


def test_review_delivery_banner_surfaces_failed_product_verification(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs.db")
    run_id = _seed_approved_run(store)
    _append_product_verified(store, run_id, passed=False)

    projection = build_run_projection(store, tmp_path, run_id)
    banner = projection["hitl_banner"]
    assert banner is not None
    assert banner["kind"] == "review_delivery"
    assert banner["tone"] == "danger"
    assert banner["actions"] == ["retry_workflow"]
    assert "交付验收未通过" in banner["headline"]
    assert "boot.log 显示启动失败" in banner["detail"]
    assert projection["post_dag"]["status"] == "awaiting_human"


def test_review_delivery_banner_is_warn_when_delivery_is_clean(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs.db")
    run_id = _seed_approved_run(store)
    _append_product_verified(store, run_id, passed=True)

    banner = build_run_projection(store, tmp_path, run_id)["hitl_banner"]
    assert banner is not None
    assert banner["kind"] == "review_delivery"
    assert banner["tone"] == "warn"
    assert banner["actions"] == ["retry_workflow"]


def test_product_artifact_endpoint_serves_allowlisted_files_only(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs.db")
    run_id = _seed_approved_run(store)
    app = AriadneApp(store=store, workspace_root=tmp_path)
    verify_dir = tmp_path / ".olympus" / "runs" / run_id / "artifacts" / "verify_product"
    verify_dir.mkdir(parents=True)
    (verify_dir / "report.json").write_text('{"ok": true}', encoding="utf-8")
    (verify_dir / "boot.log").write_text("ready\n", encoding="utf-8")
    png = b"\x89PNG\r\n\x1a\nfake"
    (verify_dir / "home.png").write_bytes(png)

    with TestClient(create_fastapi_app(app)) as client:
        proj = client.get(f"/api/runs/{run_id}").json()["projection"]
        _append_product_verified(
            store,
            run_id,
            passed=True,
            extra={"screenshot_path": str(verify_dir / "home.png")},
        )
        proj = client.get(f"/api/runs/{run_id}").json()["projection"]
        assert proj["delivery_acceptance"]["product_verification"]["has_screenshot"] is True

        report = client.get(f"/api/runs/{run_id}/product-artifact/report")
        assert report.status_code == 200
        assert report.headers["content-type"].startswith("application/json")
        assert report.headers["cache-control"] == "no-store"
        assert report.json() == {"ok": True}

        boot = client.get(f"/api/runs/{run_id}/product-artifact/boot_log")
        assert boot.status_code == 200
        assert boot.headers["content-type"].startswith("text/plain")
        assert boot.headers["cache-control"] == "no-store"
        assert boot.text == "ready\n"

        shot = client.get(f"/api/runs/{run_id}/product-artifact/screenshot")
        assert shot.status_code == 200
        assert shot.headers["content-type"] == "image/png"
        assert shot.headers["cache-control"] == "no-store"
        assert shot.content == png

        assert client.get(f"/api/runs/{run_id}/product-artifact/secrets").status_code == 400
        assert client.get("/api/runs/run-missing/product-artifact/report").status_code == 404

        (verify_dir / "boot.log").unlink()
        missing = client.get(f"/api/runs/{run_id}/product-artifact/boot_log")
        assert missing.status_code == 404
        assert missing.headers["cache-control"] == "no-store"

        outside = tmp_path / "outside.log"
        outside.write_text("secret\n", encoding="utf-8")
        (verify_dir / "boot.log").symlink_to(outside)
        symlinked = client.get(f"/api/runs/{run_id}/product-artifact/boot_log")
        assert symlinked.status_code == 404
        assert "secret" not in symlinked.text

        real_dir = verify_dir.with_name("verify_product_real")
        verify_dir.rename(real_dir)
        verify_dir.symlink_to(real_dir)
        assert client.get(f"/api/runs/{run_id}/product-artifact/report").status_code == 404

        # Symlinked run_dir itself must also be rejected against artifacts_root.
        run_root = tmp_path / ".olympus" / "runs" / run_id
        real_run = run_root.with_name(f"{run_id}-real")
        verify_dir.unlink()
        real_dir.rename(verify_dir)
        # FIFO at an artifact path must not block projection or download (404 immediately).
        import os

        if hasattr(os, "mkfifo"):
            (verify_dir / "boot.log").unlink(missing_ok=True)
            os.mkfifo(verify_dir / "boot.log")
            assert client.get(f"/api/runs/{run_id}/product-artifact/boot_log").status_code == 404
            proj_fifo = client.get(f"/api/runs/{run_id}").json()["projection"]
            assert proj_fifo["delivery_acceptance"]["product_verification"]["has_boot_log"] is False
            (verify_dir / "boot.log").unlink()

        run_root.rename(real_run)
        run_root.symlink_to(real_run)
        assert client.get(f"/api/runs/{run_id}/product-artifact/report").status_code == 404


def test_dag_node_review_deferred_and_escape_hatch_lifecycle(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs.db")
    run_id = _seed_approved_run(store, status=RunStatus.RUNNING)

    def node() -> dict[str, object]:
        return build_run_projection(store, tmp_path, run_id)["dag"]["nodes"][0]

    assert node()["review_deferred"] is False
    store.append_event(
        run_id=run_id, stage="verify", event_type="task.review_deferred", actor="chronos",
        payload={"task_id": "T01", "attempt": 1, "review_mode": "post_dag"},
    )
    projection = build_run_projection(store, tmp_path, run_id)
    assert projection["dag"]["nodes"][0]["review_deferred"] is True
    assert projection["recent_progress"]["event_type"] == "task.review_deferred"

    store.append_event(
        run_id=run_id, stage="verify", event_type="task.deferred_review_passed", actor="argus",
        payload={"task_id": "T01"},
    )
    assert node()["review_deferred"] is False

    store.append_event(
        run_id=run_id, stage="verify", event_type="task.review_deferred", actor="chronos",
        payload={"task_id": "T01", "attempt": 1, "review_mode": "post_dag"},
    )
    store.append_event(
        run_id=run_id, stage="orchestrate", event_type="task.leased", actor="scheduler",
        payload={"task_id": "T01", "attempt": 2, "lease_generation": 2},
    )
    assert node()["review_deferred"] is False

    violations = [
        {
            "file_path": "src/app.py",
            "line": index + 1,
            "rule": f"rule-{index}",
            "message": "x" * 300,
        }
        for index in range(12)
    ]
    store.append_event(
        run_id=run_id, stage="verify", event_type="task.escape_hatch_rejected", actor="chronos",
        payload={
            "task_id": "T01",
            "attempt": 2,
            "violations": [
                {"file_path": "src/app.py", "line": 1, "rule": "no-sleep", "message": "blocked"}
            ],
        },
    )
    store.append_event(
        run_id=run_id, stage="verify", event_type="task.escape_hatch_rejected", actor="chronos",
        payload={"task_id": "T01", "attempt": 2, "violations": violations},
    )
    flagged = node()
    assert flagged["escape_hatch_rejections"] == 2
    findings = flagged["escape_hatch_findings"]
    assert len(findings) == 10
    assert all(len(item) <= 240 for item in findings)
    assert all("rule-" in item for item in findings)

    store.append_event(
        run_id=run_id, stage="verify", event_type="task.escape_hatch_rejected", actor="chronos",
        payload={
            "task_id": "T01",
            "attempt": 2,
            "violations": [
                {"file_path": "src/app.py", "line": 9, "rule": "no-sleep", "message": "latest"}
            ],
        },
    )
    reflagged = node()
    assert reflagged["escape_hatch_rejections"] == 3
    assert reflagged["escape_hatch_findings"] == ["src/app.py:9 no-sleep: latest"]


def test_waterfall_is_honest_without_recorded_receipts(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs.db")
    run_id = _seed_approved_run(store, status=RunStatus.RUNNING)

    projection = build_run_projection(store, tmp_path, run_id)
    turns = projection["waterfall_turns"]
    assert turns
    assert all(turn["duration_ms"] == 0 for turn in turns)
    assert all(turn["receipt"] is None for turn in turns)
    serialized = json.dumps(turns, ensure_ascii=False)
    for fake in ("18420", "9360", "12680"):
        assert fake not in serialized
    assert "指出最早候补者已取消边界并给出事务归因" not in serialized
    offsets = [turn["offset_pct"] for turn in turns]
    assert offsets == sorted(offsets)


def test_malformed_events_and_receipts_never_500_projection(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs.db")
    run_id = _seed_approved_run(store, status=RunStatus.RUNNING)
    store.append_event(
        run_id=run_id,
        stage="verify",
        event_type="product.verified",
        actor="chronos",
        payload={"passed": True, "node": [], "summary": "ok"},
    )
    store.append_event(
        run_id=run_id,
        stage="verify",
        event_type="post_dag.repro_rechecked",
        actor="chronos",
        payload={"repair_round": 1, "to": 1, "still_failing_count": False},
    )
    store.append_event(
        run_id=run_id,
        stage="verify",
        event_type="task.review_deferred",
        actor="chronos",
        payload={"task_id": []},
    )
    store.append_event(
        run_id=run_id,
        stage="orchestrate",
        event_type="turn.receipt",
        actor="hephaestus:T01",
        payload={"task_id": "T01", "broken": True},
    )

    projection = build_run_projection(store, tmp_path, run_id)
    RunProjection.model_validate(projection)
    recheck = next(r for r in projection["post_dag"]["rounds"] if r["phase"] == "recheck")
    assert recheck["outcome"] == "failed"


def test_post_dag_status_waterfall_and_review_delivery_failure_priority(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs.db")
    run_id = _seed_approved_run(store, status=RunStatus.RUNNING)
    _append_product_verified(store, run_id, passed=True)
    # While RUNNING before Argus finishes, post_dag must not report passed prematurely.
    running_proj = build_run_projection(store, tmp_path, run_id)
    assert running_proj["post_dag"]["status"] == "verifying"

    store.set_run_status(run_id, RunStatus.PAUSED, actor="zeus", stage="verify")
    assert build_run_projection(store, tmp_path, run_id)["post_dag"]["status"] == "paused"

    store.set_run_status(run_id, RunStatus.FAILED, actor="chronos", stage="verify")
    assert build_run_projection(store, tmp_path, run_id)["post_dag"]["status"] == "failed"

    # Repro failure while RUNNING makes Argus waterfall turn blocked, never accepted.
    store.set_run_status(run_id, RunStatus.RUNNING, actor="zeus", stage="verify")
    store.append_event(
        run_id=run_id,
        stage="verify",
        event_type="post_dag.repro_rechecked",
        actor="chronos",
        payload={"repair_round": 1, "checked_count": 1, "still_failing_count": 1},
    )
    repro_proj = build_run_projection(store, tmp_path, run_id)
    argus_turn = next(t for t in repro_proj["waterfall_turns"] if t["actor"] == "Argus")
    assert argus_turn["status"] == "blocked"

    # When paused after a failed round, both post_dag and Argus waterfall report paused.
    store.set_run_status(run_id, RunStatus.PAUSED, actor="zeus", stage="verify")
    paused_proj = build_run_projection(store, tmp_path, run_id)
    assert paused_proj["post_dag"]["status"] == "paused"
    paused_argus = next(t for t in paused_proj["waterfall_turns"] if t["actor"] == "Argus")
    assert paused_argus["status"] == "paused"

    # Non-HTTP COMPLETED run with failed product verification reports failed, never verifying.
    store.set_run_status(run_id, RunStatus.COMPLETED, actor="chronos", stage="verify")
    completed_fail = build_run_projection(store, tmp_path, run_id)
    assert completed_fail["post_dag"]["status"] == "failed"
    completed_argus = next(t for t in completed_fail["waterfall_turns"] if t["actor"] == "Argus")
    assert completed_argus["status"] == "blocked"

    # Argus invalid output transitions to AWAITING_HUMAN with concrete failure detail.
    store.set_run_status(
        run_id,
        RunStatus.AWAITING_HUMAN,
        actor="chronos",
        stage="verify",
        detail={"reason": "review_output_invalid", "failure": "Argus JSON parse failed"},
    )
    awaiting_proj = build_run_projection(store, tmp_path, run_id)
    banner = awaiting_proj["hitl_banner"]
    assert banner is not None
    assert banner["kind"] == "review_delivery"
    assert banner["tone"] == "danger"
    assert banner["detail"].startswith("Argus JSON parse failed")


def test_waterfall_uses_latest_worker_receipt_and_clears_stale_review_deferred(
    tmp_path: Path,
) -> None:
    from olympus.harness.receipt import build_turn_receipt

    store = RunStore(tmp_path / "runs.db")
    run_id = _seed_approved_run(store, status=RunStatus.RUNNING, task_status=TaskStatus.READY)
    store.append_event(
        run_id=run_id,
        stage="verify",
        event_type="task.review_deferred",
        actor="chronos",
        payload={"task_id": "T01", "attempt": 1},
    )
    first = build_turn_receipt(
        turn_id="turn-t01-a1",
        actor="hephaestus:T01",
        task_id="T01",
        events=[{"kind": "tool_end", "tool": "run_bash", "is_error": True, "duration_ms": 100}],
    )
    second = build_turn_receipt(
        turn_id="turn-t01-a2",
        actor="hephaestus:T01",
        task_id="T01",
        events=[{"kind": "tool_end", "tool": "run_bash", "is_error": False, "duration_ms": 900}],
    )
    store.append_event(
        run_id=run_id, stage="orchestrate", event_type="turn.receipt", actor="hephaestus:T01",
        payload=first,
    )
    store.append_event(
        run_id=run_id, stage="orchestrate", event_type="turn.receipt", actor="hephaestus:T01",
        payload=second,
    )
    proj = build_run_projection(store, tmp_path, run_id)
    assert proj["dag"]["nodes"][0]["review_deferred"] is False
    t01_turn = next(t for t in proj["waterfall_turns"] if t["task_id"] == "T01")
    assert t01_turn["turn_id"] == "turn-t01-a2"
    assert t01_turn["duration_ms"] == 900


def test_product_verification_hides_stale_screenshot_when_latest_round_skips_it(
    tmp_path: Path,
) -> None:
    store = RunStore(tmp_path / "runs.db")
    run_id = _seed_approved_run(store, status=RunStatus.COMPLETED)
    art_dir = tmp_path / ".olympus" / "runs" / run_id / "artifacts" / "verify_product"
    art_dir.mkdir(parents=True, exist_ok=True)
    (art_dir / "home.png").write_bytes(b"\x89PNG\r\n\x1a\nold")

    store.append_event(
        run_id=run_id,
        stage="verify",
        event_type="product.verified",
        actor="verify_product",
        payload={
            "passed": False,
            "summary": "Round 1 failed",
            "screenshot_path": str(art_dir / "home.png"),
            "checks": [{"name": "screenshot", "status": "passed", "detail": "ok"}],
        },
    )
    first = build_run_projection(store, tmp_path, run_id)
    assert first["delivery_acceptance"]["product_verification"]["has_screenshot"] is True

    store.append_event(
        run_id=run_id,
        stage="verify",
        event_type="product.verified",
        actor="verify_product",
        payload={
            "passed": True,
            "summary": "Round 2 passed without screenshot",
            "screenshot_path": None,
            "checks": [
                {"name": "http_boot", "status": "skipped_non_http", "detail": "cli"},
                {"name": "screenshot", "status": "skipped", "detail": "none"},
            ],
        },
    )
    second = build_run_projection(store, tmp_path, run_id)
    assert second["delivery_acceptance"]["product_verification"]["has_screenshot"] is False
