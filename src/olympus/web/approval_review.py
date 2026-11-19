"""Presentation contract for the existing Ariadne approval gates."""

from typing import Literal, NotRequired, TypedDict

from olympus.core.contracts import GoalSpec

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
