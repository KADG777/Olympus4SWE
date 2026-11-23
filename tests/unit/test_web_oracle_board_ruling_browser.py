"""Playwright E2E tests for Ticket T4 (#105): Scoped Issue Oracle modal & Quick Ruling Drawer.

Run with:
    TMPDIR=/tmp uv run --with playwright pytest \\
        tests/unit/test_web_oracle_board_ruling_browser.py -q
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import httpx
import pytest
import uvicorn

from olympus.core.contracts import (
    AcceptanceCheck,
    AmbiguityItem,
    ApprovalDecision,
    ApprovalRecord,
    ApprovalTarget,
    CheckKind,
    DesignSpec,
    GoalSpec,
    RequirementItem,
    RunStatus,
    TaskNode,
    TaskPlan,
)
from olympus.core.store import RunStore
from olympus.providers.base import LLMResponse, MockLLMProvider
from olympus.web.demos import seed_demo_runs
from olympus.web.server import AriadneApp, create_fastapi_app
from olympus.web.supervisor import RunSupervisor

playwright = pytest.importorskip("playwright.sync_api")


@pytest.fixture
def board_server(tmp_path: Path):
    workspace = tmp_path / "main-repo"
    workspace.mkdir()
    (workspace / "src").mkdir()
    (workspace / "src" / "app.py").write_text("print('ok')\n", encoding="utf-8")

    store = RunStore(tmp_path / "runs.db")
    seed_demo_runs(store, workspace)
    supervisor = RunSupervisor(
        store=store,
        workspace_root=workspace,
        provider_builder=lambda **kwargs: MockLLMProvider(),
    )
    app = AriadneApp(store=store, workspace_root=workspace, supervisor=supervisor)
    server = uvicorn.Server(
        uvicorn.Config(
            create_fastapi_app(app),
            host="127.0.0.1",
            port=0,
            log_level="warning",
        )
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10.0
    while not server.started:
        if time.monotonic() > deadline:
            raise RuntimeError("uvicorn server did not start within 10s")
        time.sleep(0.01)
    assert server.servers
    port = server.servers[0].sockets[0].getsockname()[1]
    try:
        yield store, workspace, f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=10.0)
        supervisor.shutdown()


@pytest.fixture
def browser():
    existing = sorted(Path.home().glob(".cache/ms-playwright/chromium-*/chrome-linux64/chrome"))
    executable = os.environ.get("CHROME_PATH") or (str(existing[-1]) if existing else None)
    with playwright.sync_playwright() as runtime:
        browser = runtime.chromium.launch(executable_path=executable)
        try:
            yield browser
        finally:
            browser.close()


def test_issue_oracle_modal_scoped_creation_and_error_resilience(
    board_server, browser, tmp_path: Path
):
    store, workspace, url = board_server
    default_ws = httpx.get(f"{url}/api/workspaces").json()["workspaces"][0]

    populated_dir = tmp_path / "already-populated"
    populated_dir.mkdir()
    (populated_dir / "README.md").write_text("existing files\n", encoding="utf-8")
    fresh_greenfield = tmp_path / "voice-cli-new"

    page = browser.new_page(viewport={"width": 1440, "height": 900})
    expect = playwright.expect
    page.goto(f"{url}/")

    # Select the default workspace on the board so + Issue Oracle pre-fills it.
    page.locator("#workspace-switcher").select_option(default_ws["workspace_id"])
    page.locator("#btn-issue-oracle").click()
    dialog = page.locator("#new-run-dialog")
    expect(dialog).to_be_visible()

    # Step 1: Brownfield is default and workspace is pre-filled from selectedWorkspaceId.
    expect(dialog.locator("[data-scope-mode='brownfield']")).to_have_attribute(
        "aria-pressed", "true"
    )
    expect(dialog.locator("#new-run-workspace")).to_have_value(default_ws["workspace_id"])

    # Step 2: Selecting Fix shows auto-selected repo_fast_patch workflow summary.
    dialog.locator("[data-task-type='fix']").click()
    expect(dialog.locator("#new-run-workflow-meta")).to_contain_text("repo_fast_patch")

    # Fill optional base_ref, scope_paths, title, and intent, then submit.
    dialog.locator("#new-run-base-ref").fill("main")
    dialog.locator("#new-run-scope-paths").fill("src/app.py")
    dialog.locator("#new-run-title").fill("Fix write_scope guard")
    dialog.locator("#new-run-input").fill("Ensure write_scope rejects path traversal cleanly.")
    with page.expect_response(
        lambda res: res.url.endswith("/api/runs") and res.request.method == "POST"
    ) as resp_info:
        dialog.locator("#new-run-submit").click()
    assert resp_info.value.status == 200
    brownfield_run_id = resp_info.value.json()["run_id"]
    expect(dialog).to_be_hidden()

    # Card appears on Oracle Board with Brownfield / Fix metadata; backend captured scope fields.
    card = page.locator(f".oracle-card[data-run-id='{brownfield_run_id}']")
    expect(card).to_be_visible()
    expect(card).to_have_attribute("data-mode", "brownfield")
    expect(card).to_contain_text("Fix")
    snap = store.get_run(brownfield_run_id)
    assert snap["workspace_mode"] == "brownfield"
    assert snap["task_type"] == "fix"
    assert snap["base_ref"] == "main"
    assert snap["scope_paths"] == ["src/app.py"]
    assert snap["workflow_name"] == "repo_fast_patch"

    # Re-open modal and verify error resilience on invalid brownfield custom path (400).
    page.locator("#btn-issue-oracle").click()
    expect(dialog).to_be_visible()
    dialog.locator("#new-run-workspace").select_option("__custom__")
    dialog.locator("#new-run-workspace-root").fill(str(tmp_path / "missing-repo"))
    dialog.locator("#new-run-title").fill("Preserve Draft Title")
    draft_prompt = "Preserve this exact prompt across 400 validation errors."
    dialog.locator("#new-run-input").fill(draft_prompt)
    with page.expect_response(
        lambda res: res.url.endswith("/api/runs") and res.request.method == "POST"
    ) as bad_brownfield:
        dialog.locator("#new-run-submit").click()
    assert bad_brownfield.value.status == 400
    expect(dialog).to_be_visible()
    expect(dialog.locator("#new-run-error")).to_be_visible()
    expect(dialog.locator("#new-run-error")).to_contain_text("not found")
    expect(dialog.locator("#new-run-title")).to_have_value("Preserve Draft Title")
    expect(dialog.locator("#new-run-input")).to_have_value(
        "Preserve this exact prompt across 400 validation errors."
    )

    # Switch to Greenfield with a populated directory -> 400 inline error, draft still preserved.
    dialog.locator("[data-scope-mode='greenfield']").click()
    dialog.locator("[data-task-type='feature']").click()
    expect(dialog.locator("#new-run-workflow-meta")).to_contain_text("full_sdlc_tdd")
    dialog.locator("#new-run-greenfield-path").fill(str(populated_dir))
    with page.expect_response(
        lambda res: res.url.endswith("/api/runs") and res.request.method == "POST"
    ) as bad_greenfield:
        dialog.locator("#new-run-submit").click()
    assert bad_greenfield.value.status == 400
    expect(dialog).to_be_visible()
    expect(dialog.locator("#new-run-error")).to_contain_text("must be empty")
    expect(dialog.locator("#new-run-title")).to_have_value("Preserve Draft Title")
    expect(dialog.locator("#new-run-input")).to_have_value(
        "Preserve this exact prompt across 400 validation errors."
    )

    # Fix target path to a brand-new empty directory -> succeeds, auto-registers workspace.
    dialog.locator("#new-run-greenfield-path").fill(str(fresh_greenfield))
    with page.expect_response(
        lambda res: res.url.endswith("/api/runs") and res.request.method == "POST"
    ) as ok_greenfield:
        dialog.locator("#new-run-submit").click()
    assert ok_greenfield.value.status == 200
    greenfield_run_id = ok_greenfield.value.json()["run_id"]
    expect(dialog).to_be_hidden()
    assert fresh_greenfield.is_dir()

    gf_card = page.locator(f".oracle-card[data-run-id='{greenfield_run_id}']")
    expect(gf_card).to_be_visible()
    expect(gf_card).to_have_attribute("data-mode", "greenfield")
    ws_names = [ws["name"] for ws in httpx.get(f"{url}/api/workspaces").json()["workspaces"]]
    assert "voice-cli-new" in ws_names

    # Spec-Only + Blueprint -> defaults to product_blueprint without mutating workspace.
    page.locator("#workspace-switcher").select_option("")
    page.locator("#btn-issue-oracle").click()
    expect(dialog).to_be_visible()
    dialog.locator("[data-scope-mode='spec_only']").click()
    dialog.locator("[data-task-type='blueprint']").click()
    expect(dialog.locator("#new-run-workflow-meta")).to_contain_text("product_blueprint")
    dialog.locator("#new-run-title").fill("Billing API Spec")
    dialog.locator("#new-run-input").fill("Draft a read-only billing API specification.")
    with page.expect_response(
        lambda res: res.url.endswith("/api/runs") and res.request.method == "POST"
    ) as ok_spec:
        dialog.locator("#new-run-submit").click()
    assert ok_spec.value.status == 200
    spec_run_id = ok_spec.value.json()["run_id"]
    expect(dialog).to_be_hidden()
    spec_card = page.locator(f".oracle-card[data-run-id='{spec_run_id}']")
    expect(spec_card).to_be_visible()
    expect(spec_card).to_have_attribute("data-mode", "spec_only")
    assert store.get_run(spec_run_id)["workflow_name"] == "product_blueprint"
    page.close()


def test_issue_oracle_benchmark_sample_disclosure_and_advanced_overrides(board_server, browser):
    _, _, url = board_server
    page = browser.new_page(viewport={"width": 1440, "height": 900})
    expect = playwright.expect
    page.goto(f"{url}/")

    page.locator("#btn-issue-oracle").click()
    dialog = page.locator("#new-run-dialog")
    expect(dialog).to_be_visible()

    # Both disclosures start collapsed so Scope -> Task Type -> Intent stay primary.
    advanced = dialog.locator("#new-run-advanced")
    benchmark = dialog.locator("#new-run-benchmark-disclosure")
    expect(advanced).to_have_js_property("open", False)
    expect(benchmark).to_have_js_property("open", False)

    # Expand Benchmark disclosure and load B-001 preset cleanly.
    benchmark.locator("summary").click()
    expect(benchmark).to_have_js_property("open", True)
    dialog.locator("#new-run-demo").select_option("B-001")
    expect(dialog.locator("#new-run-demo-banner")).to_be_visible()
    expect(dialog.locator("#new-run-demo-note")).to_be_visible()
    assert dialog.locator("#new-run-title").input_value().strip() != ""
    assert dialog.locator("#new-run-input").input_value().strip() != ""

    # Clear sample returns cleanly to scoped mode.
    dialog.locator("#btn-clear-demo").click()
    expect(dialog.locator("#new-run-demo-banner")).to_have_count(0)
    expect(dialog.locator("#new-run-workspace")).to_be_visible()

    # Expand Advanced Runtime Overrides to verify power-user controls remain accessible.
    advanced.locator("summary").click()
    expect(advanced).to_have_js_property("open", True)
    for control_id in (
        "#new-run-workflow",
        "#new-run-provider",
        "#new-run-model",
        "#new-run-worker-driver",
        "#new-run-reviewer-driver",
        "#new-run-monitor-model",
    ):
        expect(dialog.locator(control_id)).to_be_visible()
    page.close()


def test_quick_ruling_drawer_clarify_approve_conflict_resync_without_opening_tabs(
    tmp_path: Path, browser
):
    workspace = tmp_path / "ruling-repo"
    workspace.mkdir()
    store = RunStore(tmp_path / "ruling.db")
    seed_demo_runs(store, workspace)

    # Gate the mock design stage so we can deterministically observe In Flight -> Approve Design.
    allow_design = threading.Event()
    design_started = threading.Event()
    sample_design = DesignSpec.model_validate(
        {
            "architecture_summary": "Generate monthly CSV invoices via a billing exporter service.",
            "modules": [
                {
                    "id": "BILLING",
                    "name": "Billing Exporter",
                    "responsibility": "Render CSV invoice summaries",
                    "requirement_refs": ["R1"],
                    "target_paths": ["src/billing.py"],
                }
            ],
        }
    ).model_dump_json()

    def gated_responder(messages, tools):
        design_started.set()
        allow_design.wait(timeout=5.0)
        return LLMResponse(content=sample_design)

    supervisor = RunSupervisor(
        store=store,
        workspace_root=workspace,
        provider_builder=lambda **kwargs: MockLLMProvider(responder=gated_responder),
    )
    app = AriadneApp(store=store, workspace_root=workspace, supervisor=supervisor)
    server = uvicorn.Server(
        uvicorn.Config(
            create_fastapi_app(app),
            host="127.0.0.1",
            port=0,
            log_level="warning",
        )
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10.0
    while not server.started:
        if time.monotonic() > deadline:
            raise RuntimeError("uvicorn server did not start within 10s")
        time.sleep(0.01)
    port = server.servers[0].sockets[0].getsockname()[1]
    url = f"http://127.0.0.1:{port}"

    try:
        # Seed an existing run that the user already opened as an L2 Ariadne tab.
        existing_tab_id = store.create_run(title="Pre-Opened Inspection Tab")
        store.set_run_status(existing_tab_id, RunStatus.RUNNING)

        # 1. Oracle awaiting clarification (Prometheus -> You).
        clarify_id = store.create_run(title="Clarify Rate Limiter")
        store.save_spec(
            clarify_id,
            GoalSpec(
                goal="Add API rate limiting",
                requirements=[
                    RequirementItem(
                        id="R1",
                        description="Throttle excessive requests",
                        acceptance_criteria=["429"],
                    )
                ],
                open_ambiguities=[
                    AmbiguityItem(
                        id="A1",
                        question="Which backend stores rate-limit counters?",
                        impact="Persistence across restarts",
                        options=["SQLite table", "In-memory sliding window"],
                        blocking=True,
                    )
                ],
            ),
        )
        store.set_run_status(clarify_id, RunStatus.AWAITING_HUMAN)

        # 2. Supervised blueprint run paused at gate_spec (Prometheus -> You).
        spec_id = store.create_run(
            title="Approve Billing GoalSpec",
            workflow_name="product_blueprint",
            workspace_mode="spec_only",
            task_type="blueprint",
        )
        store.append_event(
            run_id=spec_id,
            stage="graph",
            event_type="graph.started",
            actor="supervisor",
            payload={"workflow": "product_blueprint"},
        )
        store.save_spec(
            spec_id,
            GoalSpec(
                goal="Generate monthly invoice summaries",
                requirements=[
                    RequirementItem(
                        id="R1",
                        description="Export CSV invoice",
                        acceptance_criteria=["Valid CSV"],
                    )
                ],
            ),
        )
        store.set_run_status(spec_id, RunStatus.AWAITING_HUMAN)

        # 3. Supervised blueprint run paused at gate_plan (Athena -> You) for 409 conflict resync.
        plan_id = store.create_run(title="Approve Export Plan", workflow_name="product_blueprint")
        store.append_event(
            run_id=plan_id,
            stage="graph",
            event_type="graph.started",
            actor="supervisor",
            payload={"workflow": "product_blueprint"},
        )
        plan_spec = GoalSpec(
            goal="Export audit events",
            requirements=[
                RequirementItem(
                    id="R1", description="Audit CSV export", acceptance_criteria=["Contains actor"]
                )
            ],
        )
        store.save_spec(plan_id, plan_spec)
        store.record_approval(
            ApprovalRecord(
                run_id=plan_id,
                target=ApprovalTarget.GOAL_SPEC,
                target_id=plan_spec.spec_id,
                target_hash=plan_spec.content_hash,
                approver="zeus",
                decision=ApprovalDecision.APPROVED,
            )
        )
        plan_design = DesignSpec.model_validate(
            {
                "architecture_summary": "Single audit exporter module.",
                "modules": [
                    {
                        "id": "AUDIT",
                        "name": "Audit Exporter",
                        "responsibility": "Write audit CSV",
                        "requirement_refs": ["R1"],
                        "target_paths": ["src/audit.py"],
                    }
                ],
            }
        )
        store.save_design(plan_id, plan_design)
        store.record_approval(
            ApprovalRecord(
                run_id=plan_id,
                target=ApprovalTarget.DESIGN_SPEC,
                target_id=plan_design.design_id,
                target_hash=plan_design.content_hash,
                approver="zeus",
                decision=ApprovalDecision.APPROVED,
            )
        )
        plan_v1 = TaskPlan(
            spec_id=plan_spec.spec_id,
            spec_hash=plan_spec.content_hash,
            tasks=[
                TaskNode(
                    task_id="T1",
                    title="Implement CSV writer",
                    objective="Write audit events to CSV",
                    write_scope=["src/audit.py"],
                    acceptance_checks=[
                        AcceptanceCheck(
                            id="rev",
                            kind=CheckKind.INDEPENDENT_REVIEW,
                            description="Review CSV format",
                        )
                    ],
                )
            ],
        )
        store.save_plan(plan_id, plan_v1)
        store.set_run_status(plan_id, RunStatus.AWAITING_HUMAN)

        page = browser.new_page(viewport={"width": 1440, "height": 900})
        expect = playwright.expect
        page.goto(f"{url}/")

        # Open existing_tab_id as an L2 tab, then return to Oracle Board.
        page.locator(f".oracle-card[data-run-id='{existing_tab_id}'] .oracle-card-trigger").click()
        expect(page.locator("#open-tabs .ariadne-tab")).to_have_count(1)
        page.locator("#nav-oracle-board").click()
        expect(page.locator("body")).to_have_attribute("data-surface", "board")

        # --- A. Clarify inside Quick Ruling Drawer ---
        clarify_cta = page.locator(f"[data-quick-rule='{clarify_id}']")
        expect(clarify_cta).to_contain_text("Answer in Drawer")
        clarify_cta.click()

        drawer = page.locator("#quick-ruling-drawer")
        expect(drawer).to_be_visible()
        expect(drawer).to_have_attribute("data-run-id", clarify_id)
        expect(page.locator("body")).to_have_attribute("data-surface", "board")
        expect(page.locator("#open-tabs .ariadne-tab")).to_have_count(1)

        drawer.get_by_role("button", name="SQLite table", exact=True).click()
        with page.expect_response(f"{url}/api/runs/{clarify_id}/command") as clarify_res:
            drawer.locator("#btn-clarify-submit").click()
        assert clarify_res.value.status == 200
        expect(drawer).to_be_hidden()
        assert store.get_run(clarify_id)["spec"].version == 2
        expect(page.locator(f"[data-quick-rule='{clarify_id}']")).to_contain_text("Review Spec")
        expect(page.locator("#open-tabs .ariadne-tab")).to_have_count(1)

        # --- B. Approve GoalSpec inside Quick Ruling Drawer -> moves live to In Flight ---
        spec_cta = page.locator(f"[data-quick-rule='{spec_id}']")
        expect(spec_cta).to_contain_text("Review Spec")
        spec_cta.click()
        expect(drawer).to_be_visible()
        expect(drawer.locator("[data-review-action='approve_spec']")).to_be_visible()
        with page.expect_response(f"{url}/api/runs/{spec_id}/command") as spec_res:
            drawer.locator("[data-approve='approve_spec']").click()
        assert spec_res.value.status == 200
        expect(drawer).to_be_hidden()
        assert design_started.wait(timeout=5.0)
        # Card moved out of Awaiting Your Ruling into In Flight; OPEN ARIADNE TABS untouched!
        expect(
            page.locator(f"[data-column='agent'] .oracle-card[data-run-id='{spec_id}']")
        ).to_be_visible()
        expect(
            page.locator(f"[data-column='human'] .oracle-card[data-run-id='{spec_id}']")
        ).to_have_count(0)
        expect(page.locator("#open-tabs .ariadne-tab")).to_have_count(1)
        expect(
            page.locator(f"#open-tabs .ariadne-tab[data-run-id='{existing_tab_id}']")
        ).to_be_visible()
        allow_design.set()

        # --- C. Approve Plan with 409 optimistic-lock conflict resync inside Drawer ---
        plan_cta = page.locator(f"[data-quick-rule='{plan_id}']")
        expect(plan_cta).to_contain_text("Approve Plan")
        plan_cta.click()
        expect(drawer).to_be_visible()
        expect(drawer.get_by_role("heading", name="任务计划 · v1")).to_be_visible()

        plan_v2 = plan_v1.model_copy(
            update={
                "version": 2,
                "tasks": [plan_v1.tasks[0].model_copy(update={"title": "Implement CSV writer v2"})],
            }
        )

        def bump_before_approve(route):
            store.save_plan(plan_id, plan_v2)
            route.continue_()

        page.route(f"{url}/api/runs/{plan_id}/command", bump_before_approve, times=1)
        with page.expect_response(f"{url}/api/runs/{plan_id}/command") as stale_res:
            drawer.locator("[data-approve='approve_plan']").click()
        assert stale_res.value.status == 409
        expect(drawer).to_be_visible()
        expect(drawer.locator("#quick-drawer-notice")).to_contain_text("请确认后重试")
        expect(drawer.get_by_role("heading", name="任务计划 · v2")).to_be_visible()

        # Re-confirm on v2 -> 200 succeeds, drawer closes, card moves to Delivered & Verified!
        with page.expect_response(f"{url}/api/runs/{plan_id}/command") as fresh_res:
            drawer.locator("[data-approve='approve_plan']").click()
        assert fresh_res.value.status == 200
        expect(drawer).to_be_hidden()
        expect(
            page.locator(f"[data-column='done'] .oracle-card[data-run-id='{plan_id}']")
        ).to_be_visible()
        expect(
            page.locator(f"[data-column='human'] .oracle-card[data-run-id='{plan_id}']")
        ).to_have_count(0)
        expect(page.locator("#open-tabs .ariadne-tab")).to_have_count(1)
        page.close()
    finally:
        allow_design.set()
        server.should_exit = True
        thread.join(timeout=10.0)
        supervisor.shutdown()


def test_quick_ruling_drawer_blocker_resolution_and_open_cockpit_escape_hatch(
    board_server, browser
):
    store, _, url = board_server
    blocked_id = "run-campus-waitlist"

    # Seed an extra human-held Oracle to test the "Open Full Ariadne Cockpit ↗" button.
    escape_id = store.create_run(title="Inspect Full Spec In Cockpit")
    store.save_spec(
        escape_id,
        GoalSpec(
            goal="Verify cockpit escape hatch",
            requirements=[
                RequirementItem(id="R1", description="Open L2 tab", acceptance_criteria=["Works"])
            ],
        ),
    )
    store.set_run_status(escape_id, RunStatus.AWAITING_HUMAN)

    page = browser.new_page(viewport={"width": 1440, "height": 900})
    expect = playwright.expect
    page.goto(f"{url}/")

    # Resolve blocker directly from Quick Ruling Drawer without opening an Ariadne tab.
    blocker_cta = page.locator(f"[data-quick-rule='{blocked_id}']")
    expect(blocker_cta).to_contain_text("Resolve Blocker")
    blocker_cta.click()

    drawer = page.locator("#quick-ruling-drawer")
    expect(drawer).to_be_visible()
    expect(drawer.locator("#quick-drawer-blocker")).to_be_visible()
    with page.expect_response(f"{url}/api/runs/{blocked_id}/command") as retry_res:
        drawer.locator("#btn-drawer-blocker-retry").click()
    assert retry_res.value.status == 200
    expect(drawer).to_be_hidden()
    expect(page.locator("#open-tabs .ariadne-tab")).to_have_count(0)

    # Open Quick Ruling Drawer on escape_id and click "Open Full Ariadne Cockpit ↗".
    page.locator(f"[data-quick-rule='{escape_id}']").click()
    expect(drawer).to_be_visible()
    drawer.locator("#btn-drawer-open-cockpit").click()
    expect(drawer).to_have_count(0)
    expect(page.locator("body")).to_have_attribute("data-surface", "oracle")
    expect(page.locator(f"#open-tabs .ariadne-tab[data-run-id='{escape_id}']")).to_have_attribute(
        "data-active", "true"
    )
    page.close()
