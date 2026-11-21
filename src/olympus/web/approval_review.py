"""Presentation contract for the existing Ariadne approval gates."""

import json
import re
from collections import defaultdict
from collections.abc import Callable
from typing import Any, Literal, NotRequired, TypedDict

from olympus.core.contracts import CheckKind, DesignSpec, GoalSpec, TaskNode, TaskPlan
from olympus.core.store import ApprovalArtifact

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
    change: NotRequired[Literal["changed", "added", "removed", "affected"]]
    previous: NotRequired["ReviewEntry"]
    related: NotRequired[bool]


class ReviewSection(TypedDict):
    title: str
    entries: list[ReviewEntry]
    empty_text: NotRequired[str]


class ApprovalReview(TypedDict):
    """A gate's review sections and candidate key in the same run snapshot."""

    artifact: ArtifactKey
    target_hash: str
    sections: list[ReviewSection]
    baseline: NotRequired[dict[str, Any]]
    unchanged_sections: NotRequired[list[ReviewSection]]
    comparison_gap: NotRequired[str]


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


def _review_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _field_entries(
    fields: dict[str, Any], labels: dict[str, str], prefix: str = ""
) -> list[ReviewEntry]:
    return [
        {"ref": f"{prefix}/{key}", "label": label, "text": _review_value(fields[key])}
        for key, label in labels.items()
        if fields[key] not in (None, "", [])
    ]


def _additional_sections(
    artifact: ApprovalArtifact, raw: dict[str, Any], sections: list[ReviewSection]
) -> list[ReviewSection]:
    if isinstance(artifact, GoalSpec):
        scope = _field_entries(raw, {
            "actors": "角色", "constraints": "约束", "deliverable_kind": "交付类型",
            "observable_surfaces": "可观测入口",
        })
        for i, requirement in enumerate(raw["requirements"]):
            scope.extend(_field_entries(
                requirement, {
                    "actor": f"{requirement['id']} · 角色",
                    "source_ref": f"{requirement['id']} · 来源",
                    "ears_criteria": f"{requirement['id']} · 结构化验收条件",
                },
                f"/requirements/{i}",
            ))
        return [
            {"title": "角色与约束", "entries": scope},
            {"title": "澄清与评审记录", "entries": _field_entries(raw, {
                "open_ambiguities": "澄清项", "minority_notes": "少数意见",
            })},
        ]
    if isinstance(artifact, DesignSpec):
        structures: list[ReviewSection] = [
            {"title": title, "entries": [
                {"ref": f"/{key}/{i}", "label": item[id_field], "text": _review_value(item)}
                for i, item in enumerate(raw[key])
            ]}
            for key, title, id_field in (
                ("modules", "模块", "id"), ("interfaces", "接口", "id"),
                ("data_models", "数据模型", "name"), ("shared_invariants", "全局约束", "id"),
            )
        ]
        structures.append({"title": "方案约束与绑定", "entries": _field_entries(raw, {
            "spec_hash": "GoalSpec 指纹", "shared_contract_paths": "共享契约路径",
            "execution_strategy": "执行策略", "strategy_rationale": "策略依据",
            "risks_and_tradeoffs": "风险与取舍",
        })})
        return structures

    definitions: list[ReviewEntry] = []
    checks: list[ReviewEntry] = []
    dependencies: list[ReviewEntry] = []
    covered = {entry["ref"] for section in sections for entry in section["entries"]}
    for i, task in enumerate(raw["tasks"]):
        rows = _field_entries(task, {
            "objective": "目标", "requirement_refs": "需求引用", "design_refs": "设计引用",
            "write_scope": "写入范围", "read_scope": "读取范围",
            "kind": "类型", "role": "角色", "max_cycles": "允许轮数",
        }, f"/tasks/{i}")
        for entry in rows:
            entry["label"] = f"{task['task_id']} · {entry['label']}"
        definitions.extend(rows)
        checks.extend({
            "ref": f"/tasks/{i}/acceptance_checks/{j}",
            "label": f"{task['task_id']} · {check['id']}",
            "text": f"{check['description']} · {check['command'] or check['kind']}",
        } for j, check in enumerate(task["acceptance_checks"]) if check["kind"] != CheckKind.HUMAN)
        for j, dep in enumerate(task["depends_on"]):
            ref = f"/tasks/{i}/depends_on/{j}"
            if ref not in covered:
                dependencies.append({"ref": ref, "text": f"{task['task_id']} 依赖 {dep}"})
    checks.extend({
        "ref": f"/delivery_checks/{i}",
        "label": f"{check.requirement_id} · 条件 {check.criterion_index}", "text": check.command,
    } for i, check in enumerate(artifact.delivery_checks))
    return [
        {"title": "任务定义", "entries": definitions},
        {"title": "其他任务依赖", "entries": dependencies},
        {"title": "验收检查", "entries": checks},
        {"title": "计划绑定", "entries": _field_entries(raw, {
            "spec_hash": "GoalSpec 指纹", "design_hash": "DesignSpec 指纹",
        })},
    ]


def expand_candidate_review(review: ApprovalReview, candidate: ApprovalArtifact) -> None:
    """Show all available commitments when a previous approval cannot be reconstructed."""
    review["sections"].extend(
        _additional_sections(candidate, candidate.model_dump(mode="json"), review["sections"])
    )


def _entry_key(raw: dict[str, Any], entry: ReviewEntry) -> tuple[str, ...]:
    field, *path = entry["ref"].strip("/").split("/")
    if field == "requirements":
        return (field, raw[field][int(path[0])]["id"], path[1])
    if field == "tasks":
        task = raw[field][int(path[0])]
        if path[1] == "depends_on":
            return (field, task["task_id"], path[1], task[path[1]][int(path[2])])
        if path[1] == "acceptance_checks":
            return (field, task["task_id"], path[1], task[path[1]][int(path[2])]["id"])
        return (field, task["task_id"], path[1])
    if field == "delivery_checks":
        check = raw[field][int(path[0])]
        return (field, check["requirement_id"], str(check["criterion_index"]))
    id_fields = {
        "modules": "id", "interfaces": "id", "data_models": "name",
        "design_tradeoffs": "impact", "shared_invariants": "id",
    }
    if field in id_fields:
        return (field, raw[field][int(path[0])][id_fields[field]])
    if field in ("assumptions", "out_of_scope"):
        return (field, entry["text"])
    return (field,)


def _entry_content(raw: dict[str, Any], entry: ReviewEntry) -> tuple[Any, ...]:
    parts = entry["ref"].strip("/").split("/")
    value: Any = raw
    for part in parts:
        value = value[int(part)] if isinstance(value, list) else value[part]
    if parts[-2:-1] == ["acceptance_checks"]:
        value = {key: value[key] for key in ("id", "kind", "description", "command")}
    label = None if "acceptance_criteria" in parts else entry.get("label")
    dependency = entry["text"] if "depends_on" in parts else None
    return value, label, entry.get("provenance"), dependency


def _compare_entries(
    current: list[ReviewEntry],
    previous: list[ReviewEntry],
    new_raw: dict[str, Any],
    old_raw: dict[str, Any],
    affected: Callable[[tuple[str, ...]], bool],
) -> tuple[list[ReviewEntry], list[ReviewEntry]]:
    new_groups: dict[tuple[str, ...], list[ReviewEntry]] = defaultdict(list)
    old_groups: dict[tuple[str, ...], list[ReviewEntry]] = defaultdict(list)
    for entry in current:
        new_groups[_entry_key(new_raw, entry)].append(entry)
    for entry in previous:
        old_groups[_entry_key(old_raw, entry)].append(entry)
    visible: list[ReviewEntry] = []
    unchanged: list[ReviewEntry] = []
    for key in dict.fromkeys([*new_groups, *old_groups]):
        old = old_groups[key].copy()
        unmatched: list[ReviewEntry] = []
        for new in new_groups[key]:
            value = _entry_content(new_raw, new)
            match = next((i for i, item in enumerate(old)
                          if _entry_content(old_raw, item) == value), None)
            if match is None:
                unmatched.append(new)
            else:
                old.pop(match)
                if affected(key):
                    visible.append({**new, "change": "affected"})
                else:
                    unchanged.append(new)
        if len(unmatched) == len(old) == 1:
            before, after = old[0], unmatched[0]
            if after["ref"] == "/architecture_summary":
                before = {**before, "text": old_raw["architecture_summary"]}
                after = {**after, "text": new_raw["architecture_summary"]}
            visible.append({**after, "change": "changed", "previous": before})
        else:
            visible.extend({**entry, "change": "added"} for entry in unmatched)
            visible.extend({**entry, "change": "removed"} for entry in old)
    return visible, unchanged


def _changes(before: dict[str, Any], after: dict[str, Any]) -> set[str]:
    return {key for key in before.keys() | after.keys() if before.get(key) != after.get(key)}


def _design_impact(
    before: DesignSpec, after: DesignSpec
) -> tuple[set[str], set[str]]:
    modules = {item.id: item for item in before.modules}
    revised = {item.id: item for item in after.modules}
    changed_modules = _changes(modules, revised)
    interfaces = {item.id: item for item in before.interfaces}
    revised_interfaces = {item.id: item for item in after.interfaces}
    changed_interfaces = _changes(interfaces, revised_interfaces)
    models = {item.name: item for item in before.data_models}
    revised_models = {item.name: item for item in after.data_models}
    changed_models = _changes(models, revised_models)
    for name in changed_models:
        changed_modules.update(
            model.owner_module_id for model in (models.get(name), revised_models.get(name))
            if model is not None
        )
    for key, interface in [*interfaces.items(), *revised_interfaces.items()]:
        if key in changed_interfaces:
            changed_modules.update([interface.provider_module_id, *interface.consumer_module_ids])
    for key, interface in [*interfaces.items(), *revised_interfaces.items()]:
        if changed_modules.intersection(
            [interface.provider_module_id, *interface.consumer_module_ids]
        ) or changed_models.intersection(interface.data_entity_refs):
            changed_interfaces.add(key)
    requirements = {
        ref for key, module in [*modules.items(), *revised.items()]
        if key in changed_modules for ref in module.requirement_refs
    }
    return requirements, changed_interfaces


def _plan_impact(before: TaskPlan, after: TaskPlan) -> tuple[set[str], set[str]]:
    old_tasks = {task.task_id: task.definition() for task in before.tasks}
    new_tasks = {task.task_id: task.definition() for task in after.tasks}
    impacted = _changes(old_tasks, new_tasks)
    dependents: dict[str, set[str]] = defaultdict(set)
    for task in [*before.tasks, *after.tasks]:
        for dep in task.depends_on:
            dependents[dep].add(task.task_id)
    pending = list(impacted)
    while pending:
        downstream = dependents[pending.pop()] - impacted
        pending.extend(downstream)
        impacted.update(downstream)
    requirements = {
        ref for task in [*before.tasks, *after.tasks] if task.task_id in impacted
        for ref in task.requirement_refs
    }
    return requirements, impacted


def _bound_goal(
    artifact: DesignSpec | TaskPlan,
    goals: list[GoalSpec],
    designs: list[DesignSpec],
) -> GoalSpec | None:
    bindings = [(artifact.spec_id, artifact.spec_hash)]
    if isinstance(artifact, TaskPlan):
        bindings.extend(
            (design.spec_id, design.spec_hash) for design in designs
            if (design.design_id, design.content_hash) == (artifact.design_id, artifact.design_hash)
        )
    return next((goal for binding in bindings for goal in goals
                 if (goal.spec_id, goal.content_hash) == binding), None)


def _related_criteria(
    requirements: set[str],
    check_keys: set[tuple[str, int]],
    goals: list[GoalSpec],
) -> list[ReviewEntry]:
    entries: list[ReviewEntry] = []
    for goal in goals:
        for requirement in goal.requirements:
            for index, text in enumerate(requirement.acceptance_criteria, start=1):
                if requirement.id in requirements or (requirement.id, index) in check_keys:
                    entries.append({
                        "ref": "", "related": True, "change": "affected", "text": text,
                        "label": f"GoalSpec v{goal.version} · {requirement.id} · 条件 {index}",
                    })
    resolved = {requirement.id for goal in goals for requirement in goal.requirements}
    for requirement_id in sorted(requirements - resolved):
        entries.append({
            "ref": "", "related": True, "change": "affected", "label": requirement_id,
            "text": "当前缺少绑定版本的验收条件原文。",
        })
    for requirement_id, index in sorted(check_keys):
        if not any(req.id == requirement_id and len(req.acceptance_criteria) >= index
                   for goal in goals for req in goal.requirements):
            entries.append({
                "ref": "", "related": True, "change": "affected",
                "label": f"{requirement_id} · 条件 {index}",
                "text": "当前缺少绑定版本的验收条件原文。",
            })
    return entries


def compare_approval_review(
    review: ApprovalReview,
    candidate: ApprovalArtifact,
    baseline: ApprovalArtifact,
    *,
    plan_layers: Callable[[list[TaskNode]], dict[str, int]],
    goals: list[GoalSpec],
    designs: list[DesignSpec],
) -> None:
    before, after = baseline.model_dump(mode="json"), candidate.model_dump(mode="json")
    requirements: set[str] = set()
    check_keys: set[tuple[str, int]] = set()
    impacted: set[str] = set()
    all_criteria = False
    if isinstance(baseline, GoalSpec) and isinstance(candidate, GoalSpec):
        previous_sections = goal_review_sections(baseline)
        requirements = _changes(
            {item.id: item for item in baseline.requirements},
            {item.id: item for item in candidate.requirements},
        )
        all_criteria = any(before[key] != after[key] for key in (
            "goal", "actors", "constraints", "assumptions", "out_of_scope",
            "deliverable_kind", "observable_surfaces",
        ))
    elif isinstance(baseline, DesignSpec) and isinstance(candidate, DesignSpec):
        previous_sections = design_review_sections(baseline)
        requirements, impacted = _design_impact(baseline, candidate)
        all_criteria = any(before[key] != after[key] for key in (
            "architecture_summary", "design_tradeoffs", "execution_strategy",
            "shared_invariants", "shared_contract_paths", "spec_hash",
        ))
    elif isinstance(baseline, TaskPlan) and isinstance(candidate, TaskPlan):
        previous_sections = plan_review_sections(baseline, plan_layers(baseline.tasks))
        requirements, impacted = _plan_impact(baseline, candidate)
        old_checks = {(check.requirement_id, check.criterion_index): check.command
                      for check in baseline.delivery_checks}
        new_checks = {(check.requirement_id, check.criterion_index): check.command
                      for check in candidate.delivery_checks}
        check_keys = {key for key in old_checks.keys() | new_checks.keys()
                      if old_checks.get(key) != new_checks.get(key)}
        all_criteria = any(before[key] != after[key] for key in ("spec_hash", "design_hash"))
    else:
        raise TypeError("Approval comparison requires two artifacts of the same type")

    def affected(key: tuple[str, ...]) -> bool:
        if key[0] == "requirements" and key[-1] == "acceptance_criteria":
            return all_criteria or key[1] in requirements
        if key[0] == "tasks":
            return key[2] in ("deliverables", "depends_on", "acceptance_checks") and (
                key[1] in impacted or (key[2] == "depends_on" and key[3] in impacted)
            )
        return key[0] == "interfaces" and key[1] in impacted

    visible: list[ReviewSection] = []
    unchanged: list[ReviewSection] = []
    sections = review["sections"]
    current_sections = sections + _additional_sections(candidate, after, sections)
    old_sections = previous_sections + _additional_sections(baseline, before, previous_sections)
    for current, old in zip(current_sections, old_sections, strict=True):
        changed, equal = _compare_entries(
            current["entries"], old["entries"], after, before, affected,
        )
        if changed:
            visible.append({**current, "entries": changed})
        if equal or (not changed and current.get("empty_text") and not current["entries"]):
            unchanged.append({**current, "entries": equal})

    upstream_goals = [
        _bound_goal(artifact, goals, designs) for artifact in (baseline, candidate)
        if isinstance(artifact, (DesignSpec, TaskPlan))
    ]
    if upstream_goals:
        bound = {goal.content_hash: goal for goal in upstream_goals if goal is not None}
        if all_criteria:
            requirements.update(req.id for goal in bound.values() for req in goal.requirements)
        related = _related_criteria(requirements, check_keys, list(bound.values()))
        if related:
            visible.append({"title": "关联验收条件", "entries": related})

    review["baseline"] = before
    review["sections"] = visible
    review["unchanged_sections"] = unchanged
