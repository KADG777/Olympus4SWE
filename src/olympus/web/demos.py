"""Isolated benchmark fixture presets for Ariadne live walkthroughs."""

from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path
from typing import Any

from olympus.core.contracts import (
    AcceptanceCheck,
    ApprovalDecision,
    ApprovalRecord,
    ApprovalTarget,
    CheckKind,
    DeliveryCheck,
    GoalSpec,
    RequirementItem,
    RunMode,
    RunStatus,
    TaskNode,
    TaskPlan,
    TaskStatus,
    new_id,
)
from olympus.core.store import RunStore
from olympus.harness.delivery import DELIVERY_EVENT, run_delivery_checks

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
    workspace = run_dir / "demo_workspace"
    workspace.mkdir(parents=True)
    (workspace / "announcement.txt").write_text("Ready\n", encoding="utf-8")
    receipt = run_delivery_checks(
        plan, workspace_root=workspace, overlay_dir=run_dir / "accepted_overlay", run_dir=run_dir
    )
    store.append_event(
        run_id=run_id, stage="verify", event_type=DELIVERY_EVENT, actor="chronos", payload=receipt
    )
    store.set_run_status(run_id, RunStatus.COMPLETED, actor="chronos", stage="orchestrate")
