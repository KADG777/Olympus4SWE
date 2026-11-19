"""Presentation contract for the existing Ariadne approval gates."""

import re
from typing import Literal, NotRequired, TypedDict

from olympus.core.contracts import CheckKind, DesignSpec, GoalSpec, TaskPlan

ArtifactKey = Literal["spec", "design", "plan"]
APPROVAL_ARTIFACTS: dict[str, ArtifactKey] = {
    "approve_spec": "spec",
    "approve_design": "design",
    "approve_plan": "plan",
}


class ReviewEntry(TypedDict):
    """A displayed commitment with a JSON Pointer into the snapshot's candidate."""

    ref: str
    text: str
    label: NotRequired[str]
    cost: NotRequired[str]
    provenance: NotRequired[Literal["agent_high_impact", "unknown"]]


class ReviewSection(TypedDict):
    title: str
    entries: list[ReviewEntry]
    empty_text: NotRequired[str]


class ApprovalReview(TypedDict):
    """A gate's review sections and candidate key in the same run snapshot."""

    artifact: ArtifactKey
    target_hash: str
    sections: list[ReviewSection]


def plan_review_sections(plan: TaskPlan, layers: dict[str, int]) -> list[ReviewSection]:
    tasks = plan.tasks
    by_id = {task.task_id: task for task in tasks}
    labels = {task.task_id: f"{task.task_id} {task.title}" for task in tasks}
    ordered = sorted(enumerate(tasks), key=lambda item: layers[item[1].task_id])
    batches: list[ReviewEntry] = []
    dependencies: list[ReviewEntry] = []
    human_checks: list[ReviewEntry] = []

    delivery_lineage: set[str] = set()
    prerequisites = [
        task.task_id for task in tasks if any(item.strip() for item in task.deliverables)
    ]
    while prerequisites:
        task_id = prerequisites.pop()
        if task_id not in delivery_lineage:
            delivery_lineage.add(task_id)
            prerequisites.extend(dep for dep in by_id[task_id].depends_on if dep in by_id)

    for i, task in ordered:
        label = labels[task.task_id]
        batches.append({
            "ref": f"/tasks/{i}/deliverables",
            "label": f"第 {layers[task.task_id] + 1} 批 · {label}",
            "text": "、".join(item for item in task.deliverables if item.strip()) or "未声明交付物",
        })
        if task.task_id in delivery_lineage:
            for j, dep in enumerate(task.depends_on):
                text = (
                    f"验收「{labels[dep]}」后，才能开始「{label}」"
                    if dep in labels else f"「{label}」依赖 {dep}，计划中缺少该任务"
                )
                dependencies.append({"ref": f"/tasks/{i}/depends_on/{j}", "text": text})

        human_checks.extend(
            {"ref": f"/tasks/{i}/acceptance_checks/{j}", "label": label, "text": check.description}
            for j, check in enumerate(task.acceptance_checks)
            if check.kind == CheckKind.HUMAN
        )

    return [
        {"title": "交付批次（按任务依赖）", "entries": batches},
        {
            "title": "影响交付的依赖",
            "empty_text": "已声明的交付任务无前置依赖" if delivery_lineage else "计划未声明交付物",
            "entries": dependencies,
        },
        {"title": "人工介入点", "empty_text": "计划未声明人工检查", "entries": human_checks},
    ]


def goal_review_sections(spec: GoalSpec) -> list[ReviewSection]:
    agent_assumptions = set(spec.agent_high_impact_assumptions)
    return [
        {"title": "目标", "entries": [{"ref": "/goal", "text": spec.goal}]},
        {
            "title": "假设",
            "entries": [
                {
                    "ref": f"/assumptions/{i}",
                    "text": text,
                    "provenance": "agent_high_impact" if text in agent_assumptions else "unknown",
                }
                for i, text in enumerate(spec.assumptions)
            ],
        },
        {
            "title": "范围",
            "entries": [
                {"ref": f"/requirements/{i}/description", "label": req.id, "text": req.description}
                for i, req in enumerate(spec.requirements)
            ],
        },
        {
            "title": "排除项",
            "empty_text": "规格未声明",
            "entries": [
                {"ref": f"/out_of_scope/{i}", "text": text}
                for i, text in enumerate(spec.out_of_scope)
            ],
        },
        {
            "title": "验收条件",
            "entries": [
                {
                    "ref": f"/requirements/{i}/acceptance_criteria/{j}",
                    "label": f"{req.id} · 条件 {j + 1}",
                    "text": text,
                }
                for i, req in enumerate(spec.requirements)
                for j, text in enumerate(req.acceptance_criteria)
            ],
        },
    ]


def design_review_sections(design: DesignSpec) -> list[ReviewSection]:
    summary = design.architecture_summary.strip()
    sentence_ends = list(re.finditer(r"[。！？]+|[.!?]+(?=\s|$)", summary))
    if len(sentence_ends) >= 2:
        summary = summary[: sentence_ends[1].end()]
    impacts = {"existing_system": "现有系统", "deployment": "部署", "rollback": "回滚"}
    return [
        {"title": "架构概述", "entries": [{"ref": "/architecture_summary", "text": summary}]},
        {
            "title": "关键取舍",
            "empty_text": "规格未记录结构化取舍，可查看完整 DesignSpec 核对。",
            "entries": [
                {
                    "ref": f"/design_tradeoffs/{i}",
                    "label": impacts[item.impact],
                    "text": item.choice,
                    "cost": item.cost,
                }
                for i, item in enumerate(design.design_tradeoffs)
            ],
        },
    ]
