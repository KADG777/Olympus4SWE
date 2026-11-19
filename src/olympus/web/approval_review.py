"""Presentation contract for the existing Ariadne approval gates."""

import re
from typing import Literal, NotRequired, TypedDict

from olympus.core.contracts import DesignSpec, GoalSpec

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
