"""Run with uv run --with playwright pytest tests/unit/test_web_run_progress_browser.py."""

import os
import threading
from pathlib import Path

import httpx
import pytest

from olympus.core.contracts import DesignSpec
from olympus.core.store import RunStore
from olympus.web.projection import seed_demo_runs
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


def test_plan_history_survives_task_graph_and_updates_over_sse(workbench, browser):
    store, demo_id, url = workbench
    plan = store.get_run(demo_id)["plan"]
    run_id = store.create_run(title="等待计划")
    page = browser.new_page()
    page.goto(f"{url}/?run={run_id}")
    expect = playwright.expect
    expect(page.locator("#run-recent-progress")).to_have_text("最近确认 · 已接收编排请求")
    expect(page.locator("#stat-band")).to_be_empty()
    expect(page.get_by_role("heading", name="任务依赖图")).to_be_hidden()
    store.save_plan(run_id, plan)
    expect(page.get_by_role("heading", name="任务依赖图")).to_be_visible()
    history = page.get_by_text("产物历史", exact=True)
    history.click()
    history.focus()

    store.save_plan(run_id, plan.model_copy(update={"version": 2}))

    expect(page.get_by_text("任务计划 v2 已就绪，包含 9 个任务").first).to_be_visible()
    expect(history).to_be_focused()
    expect(history.locator("..")).to_have_attribute("open", "")
    expect(page.get_by_text(plan.content_hash)).to_be_visible()
    page.locator("#run-output-detail").click()
    expect(page.get_by_role("heading", name="任务依赖图")).to_be_focused()


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
    target = page.locator("#dag-svg [data-task-id='T01']")
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

    # Replay the identical version, then make a normal event-sequence-only update.
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
