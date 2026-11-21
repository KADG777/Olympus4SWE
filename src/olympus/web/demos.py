"""Seeded demo runs and isolated benchmark fixtures for Ariadne walkthroughs."""

from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from olympus.core.contracts import (
    AcceptanceCheck,
    AmbiguityItem,
    ApprovalDecision,
    ApprovalRecord,
    ApprovalTarget,
    BudgetPolicy,
    CheckKind,
    CritiqueNote,
    CritiquePerspective,
    CritiqueSeverity,
    DeliveryCheck,
    GoalSpec,
    RequirementItem,
    ReviewFinding,
    ReviewSeverity,
    ReviewVerdict,
    RunMode,
    RunStatus,
    SpecStatus,
    TaskNode,
    TaskPlan,
    TaskStatus,
    new_id,
)
from olympus.core.store import RunStore
from olympus.harness.delivery import DELIVERY_EVENT, run_delivery_checks
from olympus.harness.receipt import build_turn_receipt
from olympus.web.projection import _diff_line_stats, _synthesize_tool_stream

_FALLBACK_B001: dict[str, Any] = {
    "case_id": "B-001",
    "benchmark_class": "B",
    "title": "Rate limiting with hidden storage decision",
    "raw_goal": "Add per-API-key rate limiting to the HTTP layer.",
    "initial_constraints": [
        "不得新增第三方依赖",
        "保持现有 HTTP 路由不变",
    ],
    "hidden_slots": [
        {
            "slot_id": "B-001-S1",
            "answer": (
                "仓库已使用 SQLite；不要引入 Redis，直接复用现有 SQLite 存储并限制为 100 req/min。"
            ),
        },
        {
            "slot_id": "B-001-S2",
            "answer": "仅当请求头包含 X-Admin-Bypass: true 时豁免，禁止按角色豁免。",
        },
    ],
    "repo_files": [
        {
            "path": "src/api/server.py",
            "content": (
                '"""Minimal HTTP handler used by the rate-limit fixture."""\n'
                "from collections.abc import Mapping\n\n\n"
                "def handle(request: Mapping[str, str]) -> dict[str, str]:\n"
                '    return {"status": "ok", "path": request.get("path", "/")}\n'
            ),
        },
        {
            "path": "src/api/store.py",
            "content": (
                '"""SQLite-backed persistence used by the API layer."""\n'
                "import sqlite3\n\n\n"
                "def connect(path: str) -> sqlite3.Connection:\n"
                "    return sqlite3.connect(path)\n"
            ),
        },
        {
            "path": "tests/test_api.py",
            "content": "def test_placeholder() -> None:\n    assert True\n",
        },
    ],
}

_DEMO_METADATA: dict[str, dict[str, str]] = {
    "B-001": {
        "label": "B-001 · API-Key 速率限制（含隐藏 SQLite 冲突）",
        "run_title": "Benchmark B-001 · API-Key 速率限制",
        "default_workflow": "full_sdlc_tdd",
        "workspace_summary": (
            "独立 4 文件 git 仓库（src/api/server.py · store.py · tests/test_api.py）"
        ),
    }
}

_PYPROJECT_TOML = """[project]
name = "b001-rate-limiter"
version = "0.1.0"
requires-python = ">=3.13"

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
"""


def _load_eval_case(case_id: str) -> dict[str, Any] | None:
    dataset = (
        Path(__file__).resolve().parents[3]
        / "tests"
        / "evals"
        / "datasets"
        / "stage1_4class.jsonl"
    )
    if dataset.is_file():
        for raw_line in dataset.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            item = json.loads(line)
            if str(item.get("case_id")) == case_id:
                return dict(item)
    if case_id == "B-001":
        return copy.deepcopy(_FALLBACK_B001)
    return None


def list_benchmark_presets() -> list[dict[str, Any]]:
    """Return UI-safe summaries for curated runnable benchmark demo presets."""
    presets: list[dict[str, Any]] = []
    for demo_id, meta in _DEMO_METADATA.items():
        case = _load_eval_case(demo_id)
        if case is None:
            continue
        presets.append(
            {
                "id": demo_id,
                "case_id": demo_id,
                "label": meta["label"],
                "title": meta["run_title"],
                "raw_input": str(case["raw_goal"]),
                "constraints": list(case.get("initial_constraints") or []),
                "default_workflow": meta["default_workflow"],
                "workspace_summary": meta["workspace_summary"],
                "clarification_hints": [
                    str(slot["answer"])
                    for slot in case.get("hidden_slots") or []
                    if slot.get("answer")
                ],
            }
        )
    return presets


def materialize_benchmark_demo(
    demo_id: str,
    *,
    workspace_parent: Path,
    slug: str | None = None,
) -> tuple[dict[str, Any], Path]:
    """Create a fresh isolated git repo containing the benchmark fixture files."""
    presets = {item["id"]: item for item in list_benchmark_presets()}
    preset = presets.get(demo_id)
    case = _load_eval_case(demo_id)
    if preset is None or case is None:
        raise ValueError(f"Unknown benchmark demo preset: '{demo_id}'")

    safe_slug = "".join(c if c.isalnum() or c in "-_" else "-" for c in (slug or new_id("ws")))
    target = (workspace_parent / f"demo-{demo_id.lower()}-{safe_slug}").resolve()
    target.mkdir(parents=True, exist_ok=True)

    for entry in case.get("repo_files") or []:
        rel = str(entry["path"]).lstrip("/")
        dest = (target / rel).resolve()
        if not dest.is_relative_to(target):
            raise ValueError(f"Refusing path outside demo workspace: {rel}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(str(entry.get("content") or ""), encoding="utf-8")

    init_py = target / "src" / "api" / "__init__.py"
    if (target / "src" / "api").is_dir() and not init_py.exists():
        init_py.write_text('"""API package."""\n', encoding="utf-8")

    pyproject = target / "pyproject.toml"
    if not pyproject.exists():
        pyproject.write_text(_PYPROJECT_TOML, encoding="utf-8")

    gitignore = target / ".gitignore"
    if not gitignore.exists():
        gitignore.write_text(
            ".olympus/\n__pycache__/\n.pytest_cache/\n*.pyc\n",
            encoding="utf-8",
        )

    if not (target / ".git").exists():
        subprocess.run(["git", "init"], cwd=target, check=True, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "olympus@example.invalid"],
            cwd=target,
            check=True,
        )
        subprocess.run(["git", "config", "user.name", "Olympus"], cwd=target, check=True)
        subprocess.run(["git", "add", "."], cwd=target, check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", f"Seed benchmark {demo_id} fixture"],
            cwd=target,
            check=True,
            capture_output=True,
        )

    return preset, target


DELIVERY_DEMO_ID = "run-delivery-acceptance"


def seed_delivery_acceptance_demo(store: RunStore, artifacts_root: Path) -> None:
    """Seed a small delivery with real command logs and an explicit accepted risk."""
    run_id = store.create_run(
        title="演示 · 逐条核对交付验收", mode=RunMode.SIMULATION, run_id=DELIVERY_DEMO_ID
    )
    spec = GoalSpec(
        goal="发布可供用户查看的公告",
        requirements=[
            RequirementItem(
                id="ANNOUNCEMENT",
                description="公告内容",
                acceptance_criteria=["公告文本包含 Ready", "公告文本包含支持邮箱"],
            ),
            RequirementItem(
                id="MOBILE",
                description="移动端兼容性",
                acceptance_criteria=["窄屏浏览公告时无需横向滚动"],
            ),
        ],
    )
    plan = TaskPlan(
        spec_id=spec.spec_id,
        spec_hash=spec.content_hash,
        delivery_checks=[
            DeliveryCheck(
                requirement_id="ANNOUNCEMENT",
                criterion_index=1,
                command="grep -Fx Ready announcement.txt",
            ),
            DeliveryCheck(
                requirement_id="ANNOUNCEMENT",
                criterion_index=2,
                command="grep -Fx support@example.com announcement.txt",
            ),
        ],
        tasks=[
            TaskNode(
                task_id="ANNOUNCEMENT",
                title="交付示例公告",
                objective="生成示例公告文件",
                requirement_refs=["ANNOUNCEMENT"],
                write_scope=["announcement.txt"],
                status=TaskStatus.ACCEPTED,
                acceptance_checks=[
                    AcceptanceCheck(
                        id="file",
                        kind=CheckKind.COMMAND,
                        description="公告文件存在",
                        command="test -f announcement.txt",
                    ),
                ],
            ),
            TaskNode(
                task_id="MOBILE",
                title="检查公告窄屏布局",
                objective="在移动端视口打开公告",
                requirement_refs=["MOBILE"],
                write_scope=["mobile.css"],
                status=TaskStatus.ACCEPTED,
                accepted_risk_reason="本次演示缺少移动端布局，先审阅公告内容",
                acceptance_checks=[
                    AcceptanceCheck(id="viewport", kind=CheckKind.HUMAN, description="窄屏检查"),
                ],
            ),
        ],
    )
    store.save_spec(run_id, spec)
    store.save_plan(run_id, plan)
    for target, target_id, target_hash in (
        (ApprovalTarget.GOAL_SPEC, spec.spec_id, spec.content_hash),
        (ApprovalTarget.TASK_PLAN, plan.plan_id, plan.content_hash),
    ):
        store.record_approval(
            ApprovalRecord(
                run_id=run_id,
                target=target,
                target_id=target_id,
                target_hash=target_hash,
                approver="demo",
                decision=ApprovalDecision.APPROVED,
                reason="演示夹具",
            )
        )

    run_dir = artifacts_root / run_id
    with TemporaryDirectory(prefix="olympus-delivery-demo-") as demo_workspace:
        workspace = Path(demo_workspace)
        (workspace / "announcement.txt").write_text("Ready\n", encoding="utf-8")
        receipt = run_delivery_checks(
            plan,
            workspace_root=workspace,
            overlay_dir=workspace / "accepted_overlay",
            run_dir=run_dir,
        )
    store.append_event(
        run_id=run_id, stage="verify", event_type=DELIVERY_EVENT, actor="chronos", payload=receipt
    )
    store.set_run_status(run_id, RunStatus.COMPLETED, actor="chronos", stage="orchestrate")


def seed_demo_runs(
    store: RunStore,
    workspace_root: Path,
    *,
    artifacts_root: Path | None = None,
) -> list[str]:
    """Seed the delivery acceptance and campus waitlist walkthroughs if missing."""
    existing_ids = {r["run_id"] for r in store.list_runs(limit=100)}
    seeded: list[str] = []
    art_base = (
        artifacts_root
        if artifacts_root is not None
        else Path(workspace_root).resolve() / ".olympus" / "runs"
    )

    if DELIVERY_DEMO_ID not in existing_ids:
        seed_delivery_acceptance_demo(store, art_base)
        seeded.append(DELIVERY_DEMO_ID)

    demo_id = "run-campus-waitlist"
    if demo_id not in existing_ids:
        store.create_run(
            title="社团活动报名与候补自动递补系统 (I001-v2)",
            mode=RunMode.EXECUTION,
            budget_policy=BudgetPolicy(
                max_llm_calls=120,
                max_total_tokens=2_000_000,
                max_concurrent_tasks=3,
            ),
            run_id=demo_id,
        )
        spec = GoalSpec(
            spec_id="spec-i001-v2",
            version=2,
            goal="实现社团活动报名与满员候补自动递补工具，保证并发名额不超限且取消后跳过已退出记录自动递补。",
            requirements=[
                RequirementItem(
                    id="FR-001",
                    description="并发争抢最后名额时不超活动容量上限",
                    acceptance_criteria=["同一活动事务级加锁校验余量，满员自动转入 WAITLISTED"],
                    source_ref="I001-v2 / D-CON-01",
                ),
                RequirementItem(
                    id="FR-002",
                    description="同一账号在同一活动内不重复占位",
                    acceptance_criteria=["活动 ID + 账号 ID 唯一约束，重复提交返回幂等状态"],
                    source_ref="H-01 / D-DATA-01",
                ),
                RequirementItem(
                    id="FR-003",
                    description="已报名者取消后自动递补最早的有效候补者",
                    acceptance_criteria=[
                        "按服务端入队序号递补首个状态仍为 WAITLISTED 的记录，跳过已取消记录"
                    ],
                    source_ref="I001-v2 / D-SM-01",
                ),
                RequirementItem(
                    id="FR-004",
                    description="候补者取消后立即退出有效候补队列",
                    acceptance_criteria=["状态迁移至 CANCELLED，后续名额释放时不再被递补"],
                    source_ref="I001-v2 / D-SM-01",
                ),
                RequirementItem(
                    id="FR-005",
                    description="支持主动查询当前报名/候补排位并展示无推送提示",
                    acceptance_criteria=[
                        "查询接口返回状态与排位，界面固定展示“仅能主动查询，不推送通知”"
                    ],
                    source_ref="I001-v2 / D-UI-01",
                ),
            ],
            out_of_scope=["不接入真实统一身份认证", "不收报名费", "不发送短信或邮件推送通知"],
            constraints=["报名、取消与自动递补必须在同一 SQLite 事务内锁定活动记录"],
            assumptions=["单活动并发请求峰值 <= 200 QPS"],
            open_ambiguities=[
                AmbiguityItem(
                    id="H-01",
                    question="同一账号取消后是否允许重新报名同一活动？",
                    impact="影响唯一索引与状态迁移表设计",
                    options=["允许重新入队并重新排号", "禁止再次报名"],
                    blocking=True,
                    answer="允许重新入队并重新排号，但同一时刻仅保留一条有效记录",
                )
            ],
            minority_notes=[
                CritiqueNote(
                    id="CRIT-QA-01",
                    perspective=CritiquePerspective.QA,
                    severity=CritiqueSeverity.WARNING,
                    summary="Momus 指出：最早候补者若已取消，递补游标必须跳过 CANCELLED 记录",
                    evidence_refs=["src/waitlist/service.py"],
                    resolved=True,
                    resolution_note="已纳入 FR-003 与 T03-R1 验收条件",
                )
            ],
            status=SpecStatus.CONFIRMED,
        )
        store.save_spec(demo_id, spec, actor="align")
        store.record_approval(
            ApprovalRecord(
                run_id=demo_id,
                target=ApprovalTarget.GOAL_SPEC,
                target_id=spec.spec_id,
                target_hash=spec.content_hash,
                approver="zeus",
                decision=ApprovalDecision.APPROVED,
                reason="需求规格 I001-v2 已核对确认",
            )
        )

        tasks = [
            TaskNode(
                task_id="T01",
                title="接口类型与测试夹具",
                objective=(
                    "定义活动、报名记录状态机 (CONFIRMED/WAITLISTED/CANCELLED) 与 SQLite 事务夹具。"
                ),
                requirement_refs=["FR-001", "FR-002"],
                depends_on=[],
                write_scope=["src/waitlist/models.py", "tests/conftest.py"],
                acceptance_checks=[
                    AcceptanceCheck(
                        id="T01-cmd",
                        kind=CheckKind.COMMAND,
                        description="校验数据模型与枚举导入",
                        command="pytest tests/test_models.py -q",
                        passed=True,
                    ),
                    AcceptanceCheck(
                        id="T01-rev",
                        kind=CheckKind.INDEPENDENT_REVIEW,
                        description="Argus 独立复核接口契约",
                        passed=True,
                    ),
                ],
                status=TaskStatus.ACCEPTED,
                attempt=1,
                review_cycle=0,
            ),
            TaskNode(
                task_id="T02",
                title="报名与容量锁服务",
                objective="实现报名注册、同账号唯一校验与容量上限事务锁。",
                requirement_refs=["FR-001", "FR-002"],
                depends_on=["T01"],
                write_scope=["src/waitlist/registration.py"],
                acceptance_checks=[
                    AcceptanceCheck(
                        id="T02-cmd",
                        kind=CheckKind.COMMAND,
                        description="运行报名与唯一性检查",
                        command="pytest tests/test_registration.py -q",
                        passed=True,
                    )
                ],
                status=TaskStatus.ACCEPTED,
                attempt=1,
                review_cycle=0,
            ),
            TaskNode(
                task_id="T03",
                title="候补入队与基础递补服务",
                objective="满员时按服务端时间戳进入候补队列，并在名额释放时触发递补。",
                requirement_refs=["FR-003", "FR-004"],
                depends_on=["T01"],
                write_scope=["src/waitlist/promotion.py"],
                acceptance_checks=[
                    AcceptanceCheck(
                        id="T03-cmd",
                        kind=CheckKind.COMMAND,
                        description="运行基础候补递补单测",
                        command="pytest tests/test_promotion_basic.py -q",
                        passed=True,
                    )
                ],
                status=TaskStatus.ACCEPTED,
                attempt=1,
                review_cycle=0,
            ),
            TaskNode(
                task_id="T04",
                title="主动状态查询与免推送提示",
                objective="实现报名状态与候补顺位查询接口，附带固定提示“仅能主动查询，不推送通知”。",
                requirement_refs=["FR-005"],
                depends_on=["T01"],
                write_scope=["src/waitlist/query.py"],
                acceptance_checks=[
                    AcceptanceCheck(
                        id="T04-cmd",
                        kind=CheckKind.COMMAND,
                        description="校验查询返回与免推送提示文案",
                        command="pytest tests/test_query.py -q",
                        passed=True,
                    )
                ],
                status=TaskStatus.ACCEPTED,
                attempt=1,
                review_cycle=0,
            ),
            TaskNode(
                task_id="T05",
                title="并发抢报与恢复交互联调",
                objective="基于契约夹具执行多线程并发争抢最后 1 个名额与异常恢复校验。",
                requirement_refs=["FR-001", "FR-002"],
                depends_on=["T03"],
                write_scope=["src/waitlist/concurrency.py"],
                acceptance_checks=[
                    AcceptanceCheck(
                        id="T05-cmd",
                        kind=CheckKind.COMMAND,
                        description="20 线程并发压测最后名额",
                        command="pytest tests/test_concurrency.py -q",
                        passed=False,
                    )
                ],
                status=TaskStatus.RUNNING,
                attempt=1,
                review_cycle=0,
            ),
            TaskNode(
                task_id="T03-R1",
                title="修复已取消候补者重复递补缺陷",
                objective="确保递补查询在同一事务内过滤 CANCELLED 记录并原子更新被递补者状态。",
                requirement_refs=["FR-003", "FR-004"],
                depends_on=["T03"],
                write_scope=["src/waitlist/waitlist.py"],
                max_cycles=2,
                acceptance_checks=[
                    AcceptanceCheck(
                        id="T03-R1-cmd",
                        kind=CheckKind.COMMAND,
                        description="验证候补取消与连续递补边界用例",
                        command="pytest tests/test_waitlist.py",
                        passed=False,
                    ),
                    AcceptanceCheck(
                        id="T03-R1-rev",
                        kind=CheckKind.INDEPENDENT_REVIEW,
                        description="Argus 审查事务边界",
                        passed=False,
                    ),
                ],
                status=TaskStatus.BLOCKED,
                attempt=2,
                review_cycle=2,
                last_error=(
                    "同一候补者被递补 2 次\n递补未在同一事务内校验队列（Momus 建议，待人工判断）"
                ),
            ),
            TaskNode(
                task_id="T07",
                title="报名与并发异常验收测试",
                objective="覆盖 FR-001、FR-002 与 FR-005 的端到端独立验收测试集。",
                requirement_refs=["FR-001", "FR-002", "FR-005"],
                depends_on=["T02", "T03", "T04", "T05"],
                write_scope=["tests/test_acceptance_reg.py"],
                acceptance_checks=[
                    AcceptanceCheck(
                        id="T07-cmd",
                        kind=CheckKind.COMMAND,
                        description="执行报名与查询验收套件",
                        command="pytest tests/test_acceptance_reg.py -q",
                        passed=False,
                    )
                ],
                status=TaskStatus.WAITING,
                attempt=0,
                review_cycle=0,
            ),
            TaskNode(
                task_id="T06",
                title="候补取消与递补链验收测试",
                objective="覆盖 FR-003 与 FR-004 的端到端候补递补与取消跳过验收测试。",
                requirement_refs=["FR-003", "FR-004"],
                depends_on=["T02", "T03", "T03-R1"],
                write_scope=["tests/test_waitlist.py"],
                acceptance_checks=[
                    AcceptanceCheck(
                        id="T06-cmd",
                        kind=CheckKind.COMMAND,
                        description="执行候补与递补验收测试",
                        command="pytest tests/test_waitlist.py -q",
                        passed=False,
                    )
                ],
                status=TaskStatus.WAITING,
                attempt=0,
                review_cycle=0,
            ),
            TaskNode(
                task_id="T08",
                title="全量集成与追踪矩阵合拢",
                objective="汇总 T06 与 T07 验收产物，生成最终交付包与追踪证据。",
                requirement_refs=["FR-001", "FR-002", "FR-003", "FR-004", "FR-005"],
                depends_on=["T06", "T07"],
                write_scope=["src/waitlist/__init__.py"],
                acceptance_checks=[
                    AcceptanceCheck(
                        id="T08-cmd",
                        kind=CheckKind.COMMAND,
                        description="全量回归测试",
                        command="pytest -q",
                        passed=False,
                    )
                ],
                status=TaskStatus.WAITING,
                attempt=0,
                review_cycle=0,
            ),
        ]
        plan = TaskPlan(
            plan_id="plan-i001-v1",
            spec_id=spec.spec_id,
            spec_hash=spec.content_hash,
            version=1,
            tasks=tasks,
        )
        store.save_plan(demo_id, plan, actor="orchestrate")
        store.record_approval(
            ApprovalRecord(
                run_id=demo_id,
                target=ApprovalTarget.TASK_PLAN,
                target_id=plan.plan_id,
                target_hash=plan.content_hash,
                approver="zeus",
                decision=ApprovalDecision.APPROVED,
                reason="计划 DAG 与 write_scope 静态校验通过",
            )
        )
        store.set_run_status(
            demo_id,
            RunStatus.AWAITING_HUMAN,
            actor="chronos",
            stage="orchestrate",
            detail={"blocked_task": "T03-R1", "accepted": 4, "total": 9},
        )

        # Record explicit swimlane events matching the walkthrough diagram
        swimlane_events = [
            ("Hephaestus-1", "T02", "T02", "accepted", 1, "报名与容量锁服务已验收"),
            ("Hephaestus-1", "T03", "T03", "accepted", 1, "候补入队基础服务已验收"),
            ("Hephaestus-1", "T03-R1", "T03-R1 ×2", "blocked", 2, "同一候补者被递补 2 次"),
            ("Hephaestus-2", "T04", "T04", "accepted", 1, "状态查询与免推送提示已验收"),
            ("Hephaestus-2", "T05", "T05 运行中", "running", 1, "20 线程并发争抢最后名额压测中"),
            (
                "Momus",
                "T03-R1",
                "审查 DS-v1",
                "accepted",
                1,
                "指出最早候补者已取消边界并给出事务归因",
            ),
        ]
        for lane, tid, label, st, att, summ in swimlane_events:
            store.append_event(
                run_id=demo_id,
                stage="orchestrate",
                event_type="agent.swimlane",
                actor=lane,
                payload={
                    "lane": lane,
                    "task_id": tid,
                    "label": label,
                    "status": st,
                    "attempts": att,
                    "summary": summ,
                },
            )

        # Populate realistic budget usage (41 / 120 calls)
        for _ in range(41):
            store.reserve_llm_call(demo_id, actor="hephaestus:T03-R1", purpose="coding_step")
            store.settle_llm_call(
                demo_id, actor="hephaestus:T03-R1", input_tokens=2850, output_tokens=620
            )

        # Write realistic artifacts for T03-R1 attempt-2
        t03_art = art_base / demo_id / "artifacts" / "T03-R1" / "attempt-2"
        t03_art.mkdir(parents=True, exist_ok=True)
        t03_diff = (
            "--- a/src/waitlist/waitlist.py\n"
            "+++ b/src/waitlist/waitlist.py\n"
            "@@ -42,8 +42,14 @@ def promote_next_waitlisted(\n"
            "     conn: sqlite3.Connection, activity_id: str\n"
            " ) -> str | None:\n"
            "-    row = conn.execute(\n"
            '-        "SELECT account_id FROM registrations "\n'
            '-        "WHERE activity_id = ? ORDER BY seq ASC LIMIT 1",\n'
            "-        (activity_id,),\n"
            "-    ).fetchone()\n"
            "+    row = conn.execute(\n"
            '+        "SELECT id, account_id FROM registrations "\n'
            "+        \"WHERE activity_id = ? AND status = 'WAITLISTED' \"\n"
            '+        "ORDER BY seq ASC LIMIT 1",\n'
            "+        (activity_id,),\n"
            "+    ).fetchone()\n"
            "     if row is None:\n"
            "         return None\n"
            "+    # BUG: missing atomic status update inside same IMMEDIATE transaction\n"
            "     return str(row[1])\n"
        )
        (t03_art / "diff.patch").write_text(t03_diff, encoding="utf-8")
        (t03_art / "check-T03-R1-cmd.log").write_text(
            "$ pytest tests/test_waitlist.py\n"
            "============================= test session starts ==============================\n"
            "collected 4 items\n\n"
            "tests/test_waitlist.py ..F.                                              [100%]\n\n"
            "=================================== FAILURES ===================================\n"
            "_________________ test_concurrent_cancel_promotes_once_only ____________________\n"
            "AssertionError: 同一候补者被递补 2 次 "
            "(expected 1 promotion for account 'stu-042', got 2)\n"
            "=========================== short test summary info ============================\n"
            "FAILED tests/test_waitlist.py::test_concurrent_cancel_promotes_once_only\n"
            "1 failed, 3 passed in 0.19s\n",
            encoding="utf-8",
        )
        verdict = ReviewVerdict(
            task_id="T03-R1",
            attempt=2,
            passed=False,
            summary="递补查询虽过滤了 WAITLISTED，但未在同一事务内将命中记录原子更新为 CONFIRMED。",
            findings=[
                ReviewFinding(
                    id="F-01",
                    severity=ReviewSeverity.IMPORTANT,
                    file_path="src/waitlist/waitlist.py",
                    line=49,
                    message=(
                        "SELECT 与 UPDATE 分属两个事务，并发取消两个名额时同一候补者被递补 2 次。"
                    ),
                    suggestion=(
                        "在 BEGIN IMMEDIATE 事务内完成 SELECT ... WHERE status='WAITLISTED' "
                        "与 UPDATE status='CONFIRMED'。"
                    ),
                )
            ],
        )
        (t03_art / "verdict-cycle-2.json").write_text(
            verdict.model_dump_json(indent=2),
            encoding="utf-8",
        )

        def _append_receipt(
            *,
            turn_id: str,
            actor: str,
            task_id: str | None,
            tools: list[tuple[str, bool, int]],
            calls: int,
            input_tokens: int,
            output_tokens: int,
            diff_lines: tuple[int, int] = (0, 0),
            checks_passed: int = 0,
            checks_total: int = 0,
        ) -> dict[str, Any]:
            turn_events: list[dict[str, Any]] = [
                {
                    "kind": "llm",
                    "calls": calls,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                }
            ]
            for tool_name, is_error, duration_ms in tools:
                turn_events.append(
                    {
                        "kind": "tool_end",
                        "tool": tool_name,
                        "is_error": is_error,
                        "duration_ms": duration_ms,
                    }
                )
            receipt = build_turn_receipt(
                turn_id=turn_id,
                actor=actor,
                task_id=task_id,
                events=turn_events,
                diff_lines=diff_lines,
                checks_passed=checks_passed,
                checks_total=checks_total,
            )
            store.append_event(
                run_id=demo_id,
                stage="orchestrate",
                event_type="turn.receipt",
                actor=actor,
                payload=receipt,
            )
            return receipt

        # Real worker trace so the dock renders recorded tool steps, not synthetic ones
        worker_trace = [
            (
                1,
                "read_file",
                "src/waitlist/waitlist.py",
                "ok",
                384,
                "def promote_next_waitlisted(conn, activity_id): ...",
            ),
            (
                2,
                "read_file",
                "tests/test_waitlist.py",
                "ok",
                296,
                "def test_concurrent_cancel_promotes_once_only(): ...",
            ),
            (
                3,
                "edit_file",
                "src/waitlist/waitlist.py",
                "ok",
                1420,
                "patched promote_next_waitlisted: added status='WAITLISTED' filter",
            ),
            (
                4,
                "run_bash",
                "pytest tests/test_waitlist.py",
                "error",
                2210,
                "1 failed, 3 passed in 0.19s",
            ),
            (
                5,
                "read_file",
                "src/waitlist/models.py",
                "ok",
                212,
                "class RegistrationStatus(StrEnum): ...",
            ),
        ]
        (t03_art / "worker.jsonl").write_text(
            "\n".join(
                json.dumps(
                    {
                        "event": "tool_call",
                        "step": step,
                        "tool": tool,
                        "target": target,
                        "status": status,
                        "duration_ms": duration,
                        "output": output,
                    },
                    ensure_ascii=False,
                )
                for step, tool, target, status, duration, output in worker_trace
            )
            + "\n",
            encoding="utf-8",
        )

        t03_receipt = _append_receipt(
            turn_id="turn-t03-r1-a2",
            actor="hephaestus:T03-R1",
            task_id="T03-R1",
            tools=[
                (tool, status != "ok", duration)
                for _, tool, _, status, duration, _ in worker_trace
            ],
            calls=3,
            input_tokens=12840,
            output_tokens=2410,
            diff_lines=_diff_line_stats(t03_diff),
            checks_passed=0,
            checks_total=2,
        )
        t03_added, t03_removed = _diff_line_stats(t03_diff)
        (t03_art / "MANIFEST.md").write_text(
            "\n".join(
                [
                    "# MANIFEST — T03-R1 (attempt 2)",
                    "",
                    "- Task: T03-R1 — 修复已取消候补者重复递补缺陷",
                    "- Outcome: checks_failed",
                    "- Task status: blocked",
                    "- Review cycle: 2",
                    "- Checks: 0/2 passed",
                    f"- Diff: +{t03_added}/-{t03_removed} lines",
                    f"- Turn: {t03_receipt['turn_id']}",
                    f"- Receipt: {t03_receipt['one_line']}",
                    "",
                    "## Objective",
                    "",
                    "确保递补查询在同一事务内过滤 CANCELLED 记录并原子更新被递补者状态。",
                    "",
                    "## Review verdict",
                    "",
                    "- passed: False",
                    (
                        "- summary: 递补查询虽过滤了 WAITLISTED，"
                        "但未在同一事务内将命中记录原子更新为 CONFIRMED。"
                    ),
                    "- findings: 1",
                    "",
                    "## Acceptance checks",
                    "",
                    "- T03-R1-cmd (command): not passed — 验证候补取消与连续递补边界用例",
                    "- T03-R1-rev (independent_review): not passed — Argus 审查事务边界",
                    "",
                ]
            ),
            encoding="utf-8",
        )

        # Align turn receipt: the one non-worker actor in the waterfall
        _append_receipt(
            turn_id="turn-align-1",
            actor="align",
            task_id=None,
            tools=[("read_file", False, 412), ("read_file", False, 508)],
            calls=2,
            input_tokens=8400,
            output_tokens=2150,
        )
        # Worker turn receipts for every task that actually executed
        for order, task in enumerate(tasks):
            if task.attempt <= 0 or task.task_id == "T03-R1":
                continue
            synth_stream = _synthesize_tool_stream(task)
            if not synth_stream:
                continue
            _append_receipt(
                turn_id=f"turn-{task.task_id.lower()}-a{max(1, task.attempt)}",
                actor=f"hephaestus:{task.task_id}",
                task_id=task.task_id,
                tools=[
                    (str(step["tool"]), step["status"] != "ok", int(step["duration_ms"]))
                    for step in synth_stream
                ],
                calls=1,
                input_tokens=2480 + 140 * order,
                output_tokens=560 + 40 * order,
                diff_lines=(0, 0),
                checks_passed=sum(1 for c in task.acceptance_checks if c.passed),
                checks_total=len(task.acceptance_checks),
            )

        seeded.append(demo_id)

    return seeded
