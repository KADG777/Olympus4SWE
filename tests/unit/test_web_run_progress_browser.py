"""Run with uv run --with playwright pytest tests/unit/test_web_run_progress_browser.py."""

import os
import sqlite3
import threading
from pathlib import Path

import httpx
import pytest

from olympus.core.contracts import (
    AcceptanceCheck,
    ApprovalDecision,
    ApprovalRecord,
    ApprovalTarget,
    CheckKind,
    DeliveryCheck,
    DesignSpec,
    GoalSpec,
    RequirementItem,
    TaskNode,
    TaskPlan,
)
from olympus.core.store import RunStore
from olympus.web.demos import seed_demo_runs
from olympus.web.server import AriadneApp, create_http_server

playwright = pytest.importorskip("playwright.sync_api")


@pytest.fixture
def workbench(tmp_path):
    store = RunStore(tmp_path / "runs.db")
    run_id = "run-campus-waitlist"
    assert run_id in seed_demo_runs(store, tmp_path)
    app = AriadneApp(store=store, workspace_root=tmp_path)
    httpd = create_http_server(app, host="127.0.0.1", port=0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield store, run_id, f"http://127.0.0.1:{httpd.server_port}"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join()


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


def test_goal_rereview_shows_permission_impact_after_live_update(workbench, browser):
    store, _, url = workbench
    run_id = store.create_run(title="导出权限返审")
    approved = GoalSpec(
        goal="导出报表",
        requirements=[RequirementItem(
            id="EXPORT", description="仅管理员可以导出",
            acceptance_criteria=["导出包含时间戳"],
        )],
        out_of_scope=["定时导出"],
    )
    store.save_spec(run_id, approved)
    store.record_approval(ApprovalRecord(
        run_id=run_id, target=ApprovalTarget.GOAL_SPEC, target_id=approved.spec_id,
        target_hash=approved.content_hash, approver="human", decision=ApprovalDecision.APPROVED,
    ))

    page = browser.new_page()
    page.goto(f"{url}/?run={run_id}")
    expect = playwright.expect
    expect(page.get_by_text("导出权限返审", exact=True).first).to_be_visible()

    revised = approved.model_copy(update={
        "version": 2, "requirements": [approved.requirements[0].model_copy(update={
            "description": "所有成员都可以导出",
        })],
    })
    store.save_spec(run_id, revised)
    review = page.get_by_role("article", name="GoalSpec · v2")
    expect(review.get_by_text("所有成员都可以导出", exact=True)).to_be_visible()
    expect(review.get_by_text("仅管理员可以导出", exact=True)).to_be_visible()
    expect(review.get_by_text("导出包含时间戳", exact=True)).to_be_visible()
    unchanged = review.get_by_text("查看已确认未变的条目", exact=True)
    expect(review.get_by_text("定时导出", exact=True)).to_be_hidden()
    unchanged.click()
    expect(review.get_by_text("定时导出", exact=True)).to_be_visible()
    review.get_by_text("查看上次批准的 GoalSpec", exact=True).click()
    expect(review.locator("[data-review-baseline] pre")).to_contain_text(approved.content_hash)


def test_legacy_rereview_shows_candidate_when_the_approved_original_is_missing(workbench, browser):
    store, _, url = workbench
    run_id = store.create_run(title="旧库返审")
    approved = GoalSpec(
        goal="导出报表",
        requirements=[RequirementItem(
            id="EXPORT", description="仅管理员可以导出", acceptance_criteria=["导出保留时间戳"],
        )],
        out_of_scope=["定时导出"],
    )
    store.save_spec(run_id, approved)
    store.record_approval(ApprovalRecord(
        run_id=run_id, target=ApprovalTarget.GOAL_SPEC, target_id=approved.spec_id,
        target_hash=approved.content_hash, approver="human", decision=ApprovalDecision.APPROVED,
    ))
    store.save_spec(run_id, approved.model_copy(update={"version": 2, "goal": "导出当前报表"}))
    with sqlite3.connect(store.db_path) as conn:
        conn.execute("DROP TABLE approval_artifacts")
    RunStore(store.db_path)

    page = browser.new_page()
    page.goto(f"{url}/?run={run_id}")
    expect = playwright.expect
    review = page.get_by_role("article", name="GoalSpec · v2")
    expect(review.get_by_text("缺少上次批准版本的可靠原文", exact=False)).to_be_visible()
    for commitment in ("导出当前报表", "仅管理员可以导出", "导出保留时间戳", "定时导出"):
        expect(review.get_by_text(commitment, exact=True)).to_be_visible()
    expect(review.locator("[data-review-baseline]")).to_have_count(0)


def test_design_approval_shows_tradeoffs_and_reopens_review_after_conflict(workbench, browser):
    store, _, url = workbench
    run_id = store.create_run(title="架构审阅", provider_config={"provider": "mock"})
    design = DesignSpec.model_validate(
        {
            "architecture_summary": "保留现有入口。写入切到新表。迁移步骤见接口。",
            "modules": [
                {
                    "id": "API",
                    "name": "报名入口",
                    "responsibility": "接收请求",
                    "requirement_refs": ["R1"],
                    "target_paths": ["src/api.py"],
                },
                {
                    "id": "STORE",
                    "name": "席位存储",
                    "responsibility": "写入席位",
                    "requirement_refs": ["R1"],
                    "target_paths": ["src/store.py"],
                },
            ],
            "data_models": [{"name": "Seat", "fields": ["id"], "owner_module_id": "STORE"}],
            "interfaces": [
                {
                    "id": "IF-SEAT",
                    "provider_module_id": "STORE",
                    "consumer_module_ids": ["API"],
                    "signature": "reserve_seat(id: str) -> Seat",
                }
            ],
            "risks_and_tradeoffs": ["迁移窗口长"],
        }
    )
    store.save_design(run_id, design)
    page = browser.new_page()
    page.goto(f"{url}/?run={run_id}")
    expect = playwright.expect
    review = page.locator('[data-review-action="approve_design"]')
    expect(review.get_by_text("规格未记录结构化取舍", exact=False)).to_be_visible()
    expect(review.get_by_text("迁移窗口长", exact=False)).to_be_hidden()

    candidate = DesignSpec.model_validate(
        {
            **design.model_dump(),
            "version": 2,
            "design_tradeoffs": [
                {"impact": "rollback", "choice": "保留旧列", "cost": "延后清理旧数据"}
            ],
        }
    )
    store.save_design(run_id, candidate)
    expect(review.get_by_role("heading", name="DesignSpec · v2")).to_be_visible()
    expect(review.get_by_text("保留现有入口。写入切到新表。", exact=True)).to_be_visible()
    full_summary = "保留现有入口。写入切到新表。迁移步骤见接口。"
    expect(page.get_by_text(full_summary, exact=True)).to_be_hidden()
    expect(review.get_by_text("选择：保留旧列", exact=True)).to_be_visible()
    expect(review.get_by_text("代价：延后清理旧数据", exact=True)).to_be_visible()
    structure = review.get_by_text("查看模块、数据模型与接口", exact=True)
    modules = review.get_by_role("table", name="模块图 · 模块边界与独占写入路径")
    expect(structure.locator("..")).to_have_js_property("open", False)
    structure.click()
    expect(
        modules.get_by_role("row", name="API 报名入口 接收请求 R1 src/api.py STORE")
    ).to_be_visible()
    expect(review.get_by_role("heading", name="数据模型")).to_be_visible()
    expect(review.get_by_role("heading", name="接口契约")).to_be_visible()
    expect(review.get_by_text("reserve_seat(id: str) -> Seat", exact=True)).to_be_visible()
    review.get_by_role("button", name="定位 回滚").click()
    expect(review.get_by_text("/design_tradeoffs/0", exact=True)).to_be_focused()
    expect(review.locator("pre")).to_contain_text(candidate.content_hash)

    newer = candidate.model_copy(
        update={
            "version": 3,
            "architecture_summary": "先验证再切流。",
            "modules": [
                candidate.modules[0].model_copy(update={"name": "新版报名入口"}),
                candidate.modules[1],
            ],
        }
    )

    def supersede_before_command(route):
        assert route.request.post_data_json["target_hash"] == candidate.content_hash
        store.save_design(run_id, newer)
        route.continue_()

    page.route(f"{url}/api/runs/{run_id}/command", supersede_before_command, times=1)
    with page.expect_response(
        lambda response: response.url.endswith("/command") and response.status == 409
    ):
        review.get_by_role("button", name="批准 DesignSpec").click()
    expect(page.get_by_text("请确认后重试", exact=False)).to_be_visible()
    expect(review.get_by_role("heading", name="DesignSpec · v3")).to_be_visible()
    expect(review.get_by_text("先验证再切流。", exact=True)).to_be_visible()
    expect(structure.locator("..")).to_have_js_property("open", True)
    expect(modules.get_by_text("新版报名入口", exact=True)).to_be_visible()
    expect(modules.get_by_text("报名入口", exact=True)).to_have_count(0)
    expect(review.get_by_text("查看完整 DesignSpec", exact=True).locator("..")).to_have_js_property(
        "open", True
    )
    expect(review.locator("pre")).to_contain_text(newer.content_hash)


@pytest.fixture
def export_plan(workbench):
    store, _, url = workbench
    run_id = store.create_run(title="演示 · 先核对权限，再开放导出")
    spec = GoalSpec(
        goal="只有通过权限检查的成员才能导出报表",
        requirements=[RequirementItem(
            id="EXPORT", description="按成员权限开放导出",
            acceptance_criteria=["无权限成员无法导出报表"],
        )],
    )
    store.save_spec(run_id, spec)
    store.record_approval(ApprovalRecord(
        run_id=run_id, target=ApprovalTarget.GOAL_SPEC, target_id=spec.spec_id,
        target_hash=spec.content_hash, approver="zeus", decision=ApprovalDecision.APPROVED,
    ))
    review = AcceptanceCheck(id="review", kind=CheckKind.INDEPENDENT_REVIEW, description="复核实现")
    plan = TaskPlan(
        spec_id=spec.spec_id, spec_hash=spec.content_hash,
        tasks=[
            TaskNode(
                task_id="EXPORT", title="开放导出入口", objective="向有权限成员开放导出",
                depends_on=["AUTH"], write_scope=["export.py"], deliverables=["报表导出入口"],
                acceptance_checks=[AcceptanceCheck(
                    id="member-preview", kind=CheckKind.HUMAN, description="人工试用成员导出入口",
                )],
            ),
            TaskNode(
                task_id="AUTH", title="校验成员权限", objective="拦截无权限成员的导出请求",
                write_scope=["auth.py"], deliverables=["成员权限检查"], acceptance_checks=[review],
            ),
            TaskNode(
                task_id="NOTES", title="整理内部记录", objective="整理开发记录",
                depends_on=["AUTH"], write_scope=["notes.md"], acceptance_checks=[review],
            ),
        ],
        delivery_checks=[DeliveryCheck(
            requirement_id="EXPORT", criterion_index=1,
            command="pytest tests/test_export.py -k denied_member",
        )],
    )
    store.save_plan(run_id, plan)
    return store, run_id, url, plan


def test_plan_review_explains_export_batches_and_opens_evidence(export_plan, browser):
    _, run_id, url, plan = export_plan
    page = browser.new_page()
    expect = playwright.expect
    page.goto(f"{url}/?run={run_id}")

    review = page.get_by_role("article", name="任务计划 · v1")
    batches = review.get_by_role("heading", name="交付批次（按任务依赖）").locator("..")
    expect(batches.get_by_text("第 1 批 · AUTH 校验成员权限")).to_be_visible()
    expect(batches.get_by_text("第 2 批 · EXPORT 开放导出入口")).to_be_visible()
    expect(batches.get_by_text("成员权限检查", exact=False)).to_be_visible()
    expect(batches.get_by_text("报表导出入口", exact=False)).to_be_visible()
    expect(batches.get_by_text("未声明交付物", exact=False)).to_be_visible()

    deps = review.get_by_role("heading", name="影响交付的依赖").locator("..")
    expect(deps).to_contain_text("验收「AUTH 校验成员权限」后，才能开始「EXPORT 开放导出入口」")
    expect(deps).not_to_contain_text("NOTES")
    human = review.get_by_role("heading", name="人工介入点").locator("..")
    expect(human).to_contain_text("人工试用成员导出入口")
    expect(human).not_to_contain_text("复核实现")

    graph = page.get_by_role("heading", name="任务依赖图")
    expect(graph).to_be_hidden()
    expect(review.locator("pre")).to_be_hidden()
    deps.get_by_role("button", name="定位 影响交付的依赖").click()
    expect(review.get_by_text("/tasks/0/depends_on/0", exact=True)).to_be_focused()
    expect(review.locator("pre")).to_contain_text(plan.content_hash)
    expect(review.locator("pre")).to_contain_text(plan.delivery_checks[0].command)
    human.get_by_role("button", name="定位 EXPORT 开放导出入口").click()
    expect(review.get_by_text("/tasks/0/acceptance_checks/0", exact=True)).to_be_focused()

    page.get_by_role("link", name="架构规格").click()
    review.get_by_role("button", name="查看任务依赖图").click()
    expect(graph).to_be_focused()
    page.get_by_role("button", name="任务 EXPORT", exact=False).click()
    expect(page.get_by_role("heading", name="EXPORT · 开放导出入口")).to_be_visible()
    expect(page.locator("#dock-body").get_by_text("AUTH", exact=True)).to_be_visible()


def test_plan_review_rejects_stale_approval_and_retains_approved_checks(export_plan, browser):
    store, run_id, url, plan = export_plan
    page = browser.new_page()
    expect = playwright.expect
    page.goto(f"{url}/?run={run_id}&nosse=1")

    old_review = page.get_by_role("article", name="任务计划 · v1")
    expect(old_review).to_be_visible()
    old_revision = store.get_run(run_id)["state_revision"]
    new_command = "pytest tests/test_export.py -k denied_and_allowed_members"
    revised = plan.model_copy(update={
        "version": 2,
        "delivery_checks": [plan.delivery_checks[0].model_copy(update={"command": new_command})],
    })
    store.save_plan(run_id, revised)

    endpoint = f"{url}/api/runs/{run_id}/command"
    approve = page.get_by_role("button", name="批准任务计划", exact=True)
    with page.expect_response(endpoint) as stale:
        approve.click()
    assert stale.value.status == 409
    assert stale.value.request.post_data_json["target_hash"] == plan.content_hash
    assert stale.value.request.post_data_json["expected_state_revision"] == old_revision
    expect(page.get_by_text("请确认后重试", exact=False)).to_be_visible()
    assert not store.is_approved(
        run_id=run_id, target=ApprovalTarget.TASK_PLAN,
        target_id=revised.plan_id, target_hash=revised.content_hash,
    )

    current = page.get_by_role("article", name="任务计划 · v2")
    current.get_by_text("查看完整 任务计划", exact=True).click()
    expect(current.locator("pre")).to_contain_text(new_command)
    expect(current.locator("pre")).not_to_contain_text(plan.delivery_checks[0].command)
    with page.expect_response(endpoint) as approved:
        approve.click()
    assert approved.value.status == 200
    assert approved.value.request.post_data_json["target_hash"] == revised.content_hash
    assert approved.value.json()["projection"]["plan_approved"]
    expect(approve).to_be_hidden()

    page.get_by_text("查看完整任务计划 · v2", exact=True).click()
    full_plan = page.get_by_role("region", name="任务依赖图").locator("pre")
    expect(full_plan).to_contain_text(revised.content_hash)
    expect(full_plan).to_contain_text(new_command)


def test_design_and_plan_rereview_show_previous_tradeoff_and_dependency(export_plan, browser):
    store, run_id, url, plan = export_plan
    design = DesignSpec.model_validate({
        "architecture_summary": "同步导出。",
        "modules": [{
            "id": "EXPORT", "name": "导出", "responsibility": "管理权限与导出",
            "target_paths": ["src/export.py"], "requirement_refs": ["EXPORT"],
        }],
        "design_tradeoffs": [{"impact": "rollback", "choice": "保留旧列", "cost": "占用空间"}],
    })
    store.save_design(run_id, design)
    for target, artifact_id, fingerprint in (
        (ApprovalTarget.DESIGN_SPEC, design.design_id, design.content_hash),
        (ApprovalTarget.TASK_PLAN, plan.plan_id, plan.content_hash),
    ):
        store.record_approval(ApprovalRecord(
            run_id=run_id, target=target, target_id=artifact_id, target_hash=fingerprint,
            approver="human", decision=ApprovalDecision.APPROVED,
        ))
    store.save_design(run_id, design.model_copy(update={"version": 2, "design_tradeoffs": []}))
    store.save_plan(run_id, plan.model_copy(update={
        "version": 2, "tasks": [task.model_copy(update={"depends_on": []}) for task in plan.tasks],
    }))

    page = browser.new_page()
    page.goto(f"{url}/?run={run_id}")
    expect = playwright.expect
    design_review = page.get_by_role("article", name="DesignSpec · v2")
    old_choice = design_review.get_by_text("选择：保留旧列", exact=True)
    expect(old_choice).to_be_visible()
    expect(old_choice.locator("xpath=ancestor::li").get_by_text("移除", exact=True)).to_be_visible()
    design_review.get_by_role("button", name="定位 上次批准的 回滚").click()
    expect(design_review.get_by_text("/design_tradeoffs/0", exact=True)).to_be_focused()
    expect(design_review.locator("[data-review-baseline] pre")).to_contain_text(design.content_hash)
    plan_review = page.get_by_role("article", name="任务计划 · v2")
    dependency = plan_review.get_by_text("才能开始", exact=False)
    expect(dependency.first).to_be_visible()
    expect(dependency.first.locator("xpath=ancestor::li").get_by_text("移除", exact=True)
           ).to_be_visible()
    expect(plan_review.get_by_role("button", name="查看任务依赖图")).to_be_visible()


def test_plan_history_survives_task_graph_and_updates_over_sse(export_plan, browser):
    store, _, url, plan = export_plan
    run_id = store.create_run(title="等待计划")
    page = browser.new_page()
    page.goto(f"{url}/?run={run_id}")
    expect = playwright.expect
    expect(page.locator("#run-recent-progress")).to_have_text("最近确认 · 已接收编排请求")
    expect(page.locator("#stat-band")).to_be_empty()
    expect(page.get_by_role("heading", name="任务依赖图")).to_be_hidden()
    store.save_plan(run_id, plan)
    graph = page.get_by_role("heading", name="任务依赖图")
    expect(page.locator("#run-output-summary")).to_contain_text("任务计划 v1")
    expect(graph).to_be_hidden()
    page.locator("#run-output-detail").click()
    expect(graph).to_be_visible()
    history = page.get_by_text("产物历史", exact=True)
    history.click()
    history.focus()

    store.save_plan(run_id, plan.model_copy(update={"version": 2}))

    expect(page.locator("#run-output-summary")).to_contain_text("任务计划 v2 已就绪，包含 3 个任务")
    expect(history).to_be_focused()
    expect(history.locator("..")).to_have_attribute("open", "")
    expect(page.get_by_text(plan.content_hash)).to_be_visible()
    expect(graph).to_be_hidden()
    page.locator("#run-output-detail").click()
    expect(graph).to_be_focused()


def test_reconnect_keeps_last_confirmation_and_rejects_late_snapshot(workbench, browser):
    store, run_id, url = workbench
    context = browser.new_context()
    page = context.new_page()
    snapshot_url = f"{url}/api/runs/{run_id}"
    expect = playwright.expect
    page.goto(f"{url}/?run={run_id}")
    time = page.locator("#run-progress-time")
    expect(time).to_be_visible()
    last_time = time.get_attribute("datetime")
    plan = store.get_run(run_id)["plan"]

    context.set_offline(True)
    expect(page.get_by_role("status").filter(has_text="正在重连")).to_be_visible()
    expect(time).to_have_attribute("datetime", last_time)
    store.save_plan(run_id, plan.model_copy(update={"version": 2}))
    context.set_offline(False)
    expect(page.locator("#run-output-summary")).to_contain_text("任务计划 v2")
    expect(page.get_by_text("正在重连", exact=False)).to_be_hidden()

    held = []
    store.save_plan(run_id, plan.model_copy(update={"version": 3}))
    old_snapshot = httpx.get(snapshot_url).json()

    def delay_snapshot(route):
        if not held:
            held.append(route)
            page.evaluate("window.snapshotHeld = true")
        else:
            route.continue_()

    page.route(snapshot_url, delay_snapshot)
    page.wait_for_function("window.snapshotHeld")
    assert held
    expect(page.locator("#run-output-summary")).to_contain_text("任务计划 v2")
    store.save_plan(run_id, plan.model_copy(update={"version": 4}))
    expect(page.locator("#run-output-summary")).to_contain_text("任务计划 v4")
    with page.expect_response(snapshot_url):
        held[0].fulfill(json=old_snapshot)
    page.evaluate("new Promise(done => requestAnimationFrame(() => requestAnimationFrame(done)))")
    expect(page.locator("#run-output-summary")).to_contain_text("任务计划 v4")
    context.close()


@pytest.mark.parametrize("motion", ["no-preference", "reduce"])
def test_progress_motion_and_reading_position(workbench, browser, motion):
    store, run_id, url = workbench
    page = browser.new_page(viewport={"width": 1440, "height": 700}, reduced_motion=motion)
    page.add_init_script("""
        window.progressAnimations = [];
        const animate = Element.prototype.animate;
        Element.prototype.animate = function (...args) {
          const animation = animate.apply(this, args);
          if (["run-current-stage", "run-key-output"].includes(this.id)) {
            window.progressAnimations.push(this.id);
          }
          return animation;
        };
    """)
    page.goto(f"{url}/?run={run_id}")
    expect = playwright.expect
    expect(page.locator("#run-current-stage")).to_contain_text("有任务受阻")
    page.get_by_text("产物历史", exact=True).click()
    target = page.locator("#dag-graph [data-task-id='T01']")
    target.evaluate("""node => {
      const stage = document.querySelector("#stage");
      stage.scrollTop += document.querySelector("#topology-title").getBoundingClientRect().top
        - document.querySelector(".stage-head").getBoundingClientRect().bottom - 12;
      node.focus({preventScroll: true});
    }""")
    anchor = page.locator("#topology-title")
    top = anchor.evaluate("node => node.getBoundingClientRect().top")
    assert page.locator("#stage").evaluate("node => node.scrollTop") > 100
    plan = store.get_run(run_id)["plan"]
    store.save_plan(run_id, plan.model_copy(update={"version": 2}))
    expect(page.locator("#run-output-summary")).to_contain_text("任务计划 v2")
    expect(target).to_be_focused()
    assert abs(anchor.evaluate("node => node.getBoundingClientRect().top") - top) <= 2

    store.save_plan(run_id, store.get_run(run_id)["plan"])
    store.append_event(run_id=run_id, stage="system", event_type="budget.settled", actor="test")
    last = store.list_events(run_id)[-1].event_seq
    expect(page.locator("#foot-note")).to_contain_text(f"#{last}")
    before = page.evaluate("window.progressAnimations.slice()")
    assert before.count("run-key-output") == (1 if motion == "no-preference" else 0)

    store.append_event(
        run_id=run_id, stage="graph", event_type="graph.node_started", actor="test",
        payload={"node": "execute"},
    )
    expect(page.locator("#run-current-stage")).to_contain_text("并发执行")
    after = page.evaluate("window.progressAnimations.slice()")
    expected = before + (["run-current-stage"] if motion == "no-preference" else [])
    assert after == expected
    page.close()


def test_submit_feedback_precedes_http_response(workbench, browser):
    _, run_id, url = workbench
    page = browser.new_page()
    page.goto(f"{url}/?run={run_id}")
    page.get_by_role("button", name="+ 新建编排").click()
    page.get_by_label("需求原文").fill("检查提交反馈")
    held = []
    page.route(f"{url}/api/runs", lambda route: held.append(route))
    page.get_by_role("button", name="创建并启动").click()
    expect = playwright.expect
    expect(page.locator("#run-connection")).to_have_text("正在提交…")
    assert len(held) == 1 and held[0].request.method == "POST"
    held[0].fulfill(json={"ok": False, "error": "测试用失败回执"})
    expect(page.get_by_text("创建请求失败：测试用失败回执")).to_be_visible()
    page.close()


def test_historical_execution_attempt_opens_recorded_patch(workbench, browser):
    store, run_id, url = workbench
    root = store.db_path.parent / ".olympus" / "runs" / run_id
    attempt = root / "artifacts" / "T03-R1" / "attempt-1"
    attempt.mkdir(parents=True, exist_ok=True)
    (attempt / "diff.patch").write_text("+recorded attempt one\n", encoding="utf-8")
    store.append_event(
        run_id=run_id, stage="orchestrate", event_type="task.leased", actor="test",
        payload={"task_id": "T03-R1", "lease_generation": 1, "attempt": 1},
    )
    store.append_event(
        run_id=run_id, stage="orchestrate", event_type="task.verifying", actor="test",
        payload={"task_id": "T03-R1", "lease_generation": 1},
    )
    page = browser.new_page()
    page.goto(f"{url}/?run={run_id}")
    page.get_by_role("button", name="查看代码变更").click(timeout=2200)
    expect = playwright.expect
    expect(page.get_by_text("+recorded attempt one", exact=True)).to_be_visible()
    store.append_event(
        run_id=run_id, stage="verify", event_type="review.verdict_recorded", actor="argus",
        payload={"task_id": "T03-R1", "attempt": 2, "passed": False},
    )
    expect(page.locator("#run-output-summary")).to_have_text("T03-R1 · 独立复核 · 未通过")
    page.locator("#run-output-detail").click()
    expect(page.locator("#dock").get_by_role("heading", name="Argus 独立复核")).to_be_visible()
    page.close()


def test_eventsource_error_reopens_from_snapshot(workbench, browser):
    store, run_id, url = workbench
    page = browser.new_page()
    requests = []

    def close_first_stream(route):
        requests.append(route.request.url)
        if len(requests) == 1:
            route.fulfill(status=503, body="stream interrupted")
        else:
            route.continue_()

    page.route("**/stream?after_seq=*", close_first_stream)
    page.goto(f"{url}/?run={run_id}")
    expect = playwright.expect
    expect(page.locator("#run-connection")).to_contain_text("正在重连")
    plan = store.get_run(run_id)["plan"]
    store.save_plan(run_id, plan.model_copy(update={"version": 2}))
    latest = store.list_events(run_id)[-1].event_seq
    expect(page.locator("#run-output-summary")).to_contain_text("任务计划 v2")
    expect(page.locator("#run-connection")).to_be_hidden()
    assert len(requests) == 2 and requests[1].endswith(f"after_seq={latest}")
    page.close()


# ---------------------------------------------------------------------------
# Task Dependency Graph island (React Flow + ELK) — real-browser geometry seams
# ---------------------------------------------------------------------------

SKIP_DEPS = {
    "A": [], "B": [],
    "C": ["A"], "D": ["B"],
    "E": ["C", "A"], "F": ["D"],
    "G": ["E", "B", "D"], "H": ["F"],
}
SKIP_WAVES = {"A": 0, "B": 0, "C": 1, "D": 1, "E": 2, "F": 2, "G": 3, "H": 3}

GRAPH_GEOMETRY = """(host) => {
  const screen = (m, p) => [m.a * p.x + m.c * p.y + m.e, m.b * p.x + m.d * p.y + m.f];
  const cards = Array.from(host.querySelectorAll('[data-task-id]')).map((card) => {
    const r = card.getBoundingClientRect();
    return {id: card.dataset.taskId, x: r.left, y: r.top, w: r.width, h: r.height};
  });
  const edges = Array.from(host.querySelectorAll('.ag-edge')).map((g) => {
    const path = g.querySelector('path');
    const m = path.getScreenCTM();
    const total = path.getTotalLength();
    const points = [];
    for (let s = 0; s <= total; s += 2) points.push(screen(m, path.getPointAtLength(s)));
    const ref = (path.getAttribute('marker-end') || '').match(/#([^)'"]+)/);
    const marker = ref && document.getElementById(ref[1]);
    const style = getComputedStyle(path);
    return {
      source: g.dataset.edgeSource, target: g.dataset.edgeTarget, tone: g.dataset.tone,
      points, end: screen(m, path.getPointAtLength(total)),
      stroke: style.stroke, dash: style.strokeDasharray,
      markerUnits: marker && marker.getAttribute('markerUnits'),
      markerFill: marker && getComputedStyle(marker.querySelector('path')).fill,
    };
  });
  const waves = Array.from(host.querySelectorAll('[data-wave]')).map((w) => {
    const r = w.getBoundingClientRect();
    return {layer: Number(w.dataset.wave), text: w.textContent, x: r.left, bottom: r.bottom};
  });
  const viewport = host.querySelector('.react-flow__viewport');
  return {
    cards, edges, waves,
    transform: viewport ? getComputedStyle(viewport).transform : null,
    hostWidth: host.getBoundingClientRect().width,
  };
}"""

TOKEN_COLOUR = """(name) => {
  const probe = document.createElement('span');
  probe.style.color = `var(${name})`;
  document.body.append(probe);
  const value = getComputedStyle(probe).color;
  probe.remove();
  return value;
}"""


def _token(page, name):
    return page.evaluate(TOKEN_COLOUR, name)


def _wait_for_layout(page, mode="elk"):
    island = page.locator(f"#dag-graph .ag-island[data-layout='{mode}']")
    playwright.expect(island).to_be_visible()
    page.wait_for_function(
        "() => document.querySelectorAll('#dag-graph .ag-edge path').length > 0"
    )
    return page.locator("#dag-graph").evaluate(GRAPH_GEOMETRY)


def _inside(point, card, inset=1.0):
    x, y = point
    return (card["x"] + inset < x < card["x"] + card["w"] - inset
            and card["y"] + inset < y < card["y"] + card["h"] - inset)


@pytest.fixture
def skip_dag(workbench):
    store, _, url = workbench
    run_id = store.create_run(title="跨层依赖 DAG")
    spec = GoalSpec(goal="跨层依赖", requirements=[RequirementItem(
        id="R1", description="多层跳跃依赖", acceptance_criteria=["路径不穿过卡片"],
    )])
    store.save_spec(run_id, spec)
    store.record_approval(ApprovalRecord(
        run_id=run_id, target=ApprovalTarget.GOAL_SPEC, target_id=spec.spec_id,
        target_hash=spec.content_hash, approver="human", decision=ApprovalDecision.APPROVED,
    ))
    review = AcceptanceCheck(id="review", kind=CheckKind.INDEPENDENT_REVIEW, description="复核")
    plan = TaskPlan(spec_id=spec.spec_id, spec_hash=spec.content_hash, tasks=[
        TaskNode(task_id=task_id, title=f"任务 {task_id}", objective="跨层", depends_on=deps,
                 write_scope=[f"{task_id.lower()}.py"], acceptance_checks=[review])
        for task_id, deps in SKIP_DEPS.items()
    ])
    store.save_plan(run_id, plan)
    store.record_approval(ApprovalRecord(
        run_id=run_id, target=ApprovalTarget.TASK_PLAN, target_id=plan.plan_id,
        target_hash=plan.content_hash, approver="human", decision=ApprovalDecision.APPROVED,
    ))
    return store, run_id, url


def test_task_dag_island_keeps_fixed_scale_and_routes_around_cards(skip_dag, browser):
    _, run_id, url = skip_dag
    page = browser.new_page(viewport={"width": 1600, "height": 1000})
    page.goto(f"{url}/?run={run_id}")
    geo = _wait_for_layout(page)
    cards = {card["id"]: card for card in geo["cards"]}
    assert set(cards) == set(SKIP_DEPS)

    # Fixed 1:1 physical card size, never stretched to the (much wider) frame.
    assert geo["transform"] in ("none", "matrix(1, 0, 0, 1, 0, 0)") or geo["transform"].startswith(
        "matrix(1, 0, 0, 1,"
    ), geo["transform"]
    for card in cards.values():
        assert abs(card["w"] - 184) <= 0.5 and abs(card["h"] - 76) <= 0.5, card
    assert geo["hostWidth"] > 4 * 184 + 3 * 96

    # Wave columns: one shared x per wave, ordered left to right, under a matching header.
    wave_x = {}
    for task_id, wave in SKIP_WAVES.items():
        wave_x.setdefault(wave, cards[task_id]["x"])
        assert abs(cards[task_id]["x"] - wave_x[wave]) <= 0.5, (task_id, wave)
    assert [wave_x[w] for w in sorted(wave_x)] == sorted(wave_x.values())
    headers = {header["layer"]: header for header in geo["waves"]}
    assert sorted(headers) == [0, 1, 2, 3]
    for wave, header in headers.items():
        assert header["text"] == f"Wave {wave}"
        assert abs(header["x"] - wave_x[wave]) <= 1
        assert header["bottom"] <= min(cards[t]["y"] for t, w in SKIP_WAVES.items() if w == wave)

    # Every dependency renders; no edge passes through a non-endpoint card.
    edges = {(edge["source"], edge["target"]): edge for edge in geo["edges"]}
    assert set(edges) == {(dep, task) for task, deps in SKIP_DEPS.items() for dep in deps}
    for (source, target), edge in edges.items():
        for card_id, card in cards.items():
            if card_id in (source, target):
                continue
            hits = [p for p in edge["points"] if _inside(p, card)]
            assert not hits, f"{source}->{target} crosses {card_id} at {hits[:3]}"

    # Multi-parent G: incoming heads land on G's west side at separated y coordinates.
    incoming = [edges[(dep, "G")]["end"] for dep in SKIP_DEPS["G"]]
    for x, _ in incoming:
        assert abs(x - cards["G"]["x"]) <= 2
    ys = sorted(y for _, y in incoming)
    assert all(b - a >= 6 for a, b in zip(ys, ys[1:], strict=False)), ys

    # Fixed-size heads whose fill matches each edge's stroke tone.
    ink, rule = _token(page, "--text-ink"), _token(page, "--rule-hard")
    for edge in edges.values():
        assert edge["markerUnits"] == "userSpaceOnUse"
        assert edge["markerFill"] == edge["stroke"]
        assert edge["stroke"] == (ink if edge["tone"] == "critical" else rule)
    assert edges[("E", "G")]["tone"] == "critical"
    assert edges[("B", "G")]["tone"] == "default"
    page.close()


def test_task_dag_island_separates_selection_from_critical_path(workbench, browser):
    _, run_id, url = workbench
    page = browser.new_page(viewport={"width": 1600, "height": 1000})
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.goto(f"{url}/?run={run_id}")
    expect = playwright.expect
    geo = _wait_for_layout(page)
    edges = {(edge["source"], edge["target"]): edge for edge in geo["edges"]}

    ink, bad, live, accent = (
        _token(page, name) for name in ("--text-ink", "--bad", "--live", "--accent")
    )
    assert accent != ink
    assert edges[("T01", "T03")]["tone"] == "critical"
    assert edges[("T01", "T03")]["stroke"] == ink
    blocked = edges[("T03-R1", "T06")]
    assert blocked["tone"] == "blocked" and blocked["stroke"] == bad
    assert blocked["dash"] not in ("none", "") and blocked["markerFill"] == bad
    running = edges[("T03", "T05")]
    assert running["tone"] == "running" and running["stroke"] == live

    def ring(task_id):
        return expect(page.locator(f"#dag-graph [data-task-id='{task_id}']"))

    blocked_card = page.locator("#dag-graph [data-task-id='T03-R1']")
    expect(blocked_card).to_have_attribute("aria-selected", "true")
    expect(blocked_card).to_have_attribute("data-status", "blocked")
    expect(blocked_card).to_have_attribute("role", "button")
    expect(blocked_card).to_have_attribute("tabindex", "0")
    ring("T03-R1").to_have_css("outline-color", accent)
    ring("T03-R1").to_have_css("outline-style", "solid")
    ring("T03-R1").to_have_css("outline-width", "2px")
    ring("T01").not_to_have_css("outline-color", accent)

    card = page.locator("#dag-graph [data-task-id='T05']")
    card.focus()
    page.keyboard.press("Enter")
    expect(card).to_have_attribute("aria-selected", "true")
    expect(blocked_card).to_have_attribute("aria-selected", "false")
    expect(page.locator("body")).to_have_attribute("data-dock", "open")
    expect(page.locator("#dock-title")).to_contain_text("T05")
    expect(card).to_be_focused()

    first = page.locator("#dag-graph [data-task-id='T01']")
    first.focus()
    page.keyboard.press(" ")
    expect(first).to_have_attribute("aria-selected", "true")
    expect(page.locator("#dock-title")).to_contain_text("T01")
    page.locator("#dag-graph [data-task-id='T02']").click()
    expect(page.locator("#dock-title")).to_contain_text("T02")
    ring("T02").to_have_css("outline-color", accent)
    ring("T03").not_to_have_css("outline-color", accent)

    # Dark theme: cards and edges re-read tokens in place, without remounting.
    light_surface = first.evaluate("node => getComputedStyle(node).backgroundColor")
    page.locator("#theme-toggle").click()
    expect(page.locator("html")).to_have_attribute("data-theme", "dark")
    assert _token(page, "--ok-soft") != light_surface
    expect(first).to_have_css("background-color", _token(page, "--ok-soft"))
    dark = page.locator("#dag-graph").evaluate(GRAPH_GEOMETRY)
    dark_edges = {(edge["source"], edge["target"]): edge for edge in dark["edges"]}
    assert dark_edges[("T01", "T03")]["stroke"] == _token(page, "--text-ink") != ink
    ring("T02").to_have_css("outline-color", _token(page, "--accent"))
    assert len(dark["cards"]) == len(geo["cards"])
    assert first.evaluate("node => node.isConnected")
    assert not errors, errors
    page.close()


def test_task_dag_island_preserves_viewport_on_live_updates(workbench, browser):
    store, run_id, url = workbench
    page = browser.new_page(viewport={"width": 1600, "height": 1000})
    page.goto(f"{url}/?run={run_id}")
    _wait_for_layout(page)
    host = page.locator("#dag-graph")
    viewport = host.locator(".react-flow__viewport")
    before = viewport.evaluate("node => getComputedStyle(node).transform")
    host.scroll_into_view_if_needed()
    box = host.bounding_box()
    page.mouse.move(box["x"] + 12, box["y"] + box["height"] - 12)
    page.mouse.down()
    page.mouse.move(box["x"] + 72, box["y"] + box["height"] - 32, steps=6)
    page.mouse.up()
    panned = viewport.evaluate("node => getComputedStyle(node).transform")
    assert panned != before
    card = host.locator("[data-task-id='T05']").element_handle()

    plan = store.get_run(run_id)["plan"]
    store.save_plan(run_id, plan.model_copy(update={"version": 2}))
    playwright.expect(page.locator("#run-output-summary")).to_contain_text("任务计划 v2")
    assert viewport.evaluate("node => getComputedStyle(node).transform") == panned
    assert card.evaluate("node => node.isConnected")
    page.close()


def test_task_dag_island_empty_cycle_fallback_and_missing_bundle(workbench, browser):
    _, run_id, url = workbench
    page = browser.new_page(viewport={"width": 1400, "height": 900})
    page.goto(f"{url}/?run={run_id}")
    _wait_for_layout(page)
    expect = playwright.expect

    page.evaluate("""() => {
      const make = (id) => {
        const host = document.createElement('div');
        host.id = id;
        host.style.width = '900px';
        document.querySelector('#view-topology').prepend(host);
        return host;
      };
      const view = (id, layer) => ({
        id, layer, status: 'waiting', tone: 'muted', meta: 'engineer · implement',
        statusText: '◷ 等待', ariaLabel: `任务 ${id} ◷ 等待`,
      });
      window.emptyHandle = window.AriadneGraph.mountDag(make('graph-empty'), {
        runId: 'empty', nodes: [], edges: [], emptyMessage: '当前运行尚未生成任务 DAG',
      });
      window.cycleHandle = window.AriadneGraph.mountDag(make('graph-cycle'), {
        runId: 'cycle',
        nodes: [view('X', 0), view('Y', 1), view('Z', 2)],
        edges: [
          {source: 'X', target: 'Y'}, {source: 'Y', target: 'Z'}, {source: 'Z', target: 'X'},
          {source: 'Z', target: 'ghost'},
        ],
      });
    }""")
    empty = page.locator("#graph-empty")
    expect(empty.get_by_text("当前运行尚未生成任务 DAG", exact=True)).to_be_visible()
    expect(empty.locator("[data-task-id]")).to_have_count(0)

    cycle = page.locator("#graph-cycle")
    expect(cycle.locator(".ag-island[data-layout='fallback']")).to_be_visible()
    expect(cycle.locator("[data-task-id]")).to_have_count(3)
    expect(cycle.locator(".ag-edge path")).to_have_count(3)
    for task_id in ("X", "Y", "Z"):
        box = cycle.locator(f"[data-task-id='{task_id}']").bounding_box()
        assert abs(box["width"] - 184) <= 0.5 and abs(box["height"] - 76) <= 0.5

    page.evaluate("""() => window.cycleHandle.update({
      runId: 'cycle', selectedId: 'Y',
      nodes: ['X', 'Y', 'Z'].map((id, layer) => ({
        id, layer, status: 'running', tone: 'live', meta: 'engineer · implement',
        statusText: '▶ 运行中', ariaLabel: `任务 ${id} ▶ 运行中`,
      })),
      edges: [{source: 'X', target: 'Y'}, {source: 'Y', target: 'Z'}, {source: 'Z', target: 'X'}],
    })""")
    expect(cycle.locator("[data-task-id='Y']")).to_have_attribute("aria-selected", "true")
    expect(cycle.locator("[data-task-id='Y']")).to_have_attribute("data-status", "running")
    page.evaluate("() => { window.cycleHandle.destroy(); window.emptyHandle.destroy(); }")
    expect(cycle.locator(".ag-island")).to_have_count(0)
    page.close()

    broken = browser.new_page()
    broken.route("**/static/graph/ariadne-graph.js", lambda route: route.abort())
    broken.goto(f"{url}/?run={run_id}")
    expect(broken.locator("#dag-graph")).to_contain_text("任务依赖图组件未能加载")
    expect(broken.locator("#workflow-svg [data-workflow-node]").first).to_be_visible()
    expect(broken.locator("#run-current-stage")).to_contain_text("有任务受阻")
    broken.close()
