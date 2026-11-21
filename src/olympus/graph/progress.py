"""Single source of truth for workflow gate, resume, and completion state.

The loader, the run supervisor, and the read-only Ariadne projection all derive
gate decisions from this module so a paused gate, an approved artifact, and a
resumed entry point can never disagree.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, TypedDict

from olympus.core.contracts import ApprovalTarget, RunEvent, RunStatus, TaskPlan, TaskStatus
from olympus.core.control import is_design_bound_to_spec, is_plan_bound_to_upstream
from olympus.core.store import RunStore

if TYPE_CHECKING:
    from olympus.graph.loader import WorkflowDefinition


class GateSpec(TypedDict):
    """Canonical metadata for one human approval gate."""

    node: str
    snapshot_key: str
    target: ApprovalTarget
    id_attr: str
    label: str


GATE_SPECS: dict[str, GateSpec] = {
    "spec": {
        "node": "gate_spec",
        "snapshot_key": "spec",
        "target": ApprovalTarget.GOAL_SPEC,
        "id_attr": "spec_id",
        "label": "GoalSpec",
    },
    "design": {
        "node": "gate_design",
        "snapshot_key": "design",
        "target": ApprovalTarget.DESIGN_SPEC,
        "id_attr": "design_id",
        "label": "DesignSpec",
    },
    "plan": {
        "node": "gate_plan",
        "snapshot_key": "plan",
        "target": ApprovalTarget.TASK_PLAN,
        "id_attr": "plan_id",
        "label": "TaskPlan",
    },
}


def is_artifact_approved(
    store: RunStore,
    run_id: str,
    gate: str,
    snap: Mapping[str, Any],
) -> bool:
    """Return whether the run's current artifact for ``gate`` carries a valid approval."""
    spec = GATE_SPECS[gate]
    artifact = snap.get(spec["snapshot_key"])
    if artifact is None:
        return False
    return store.is_approved(
        run_id=run_id,
        target=spec["target"],
        target_id=getattr(artifact, spec["id_attr"]),
        target_hash=artifact.content_hash,
    )


def resolve_resume_node(
    definition: WorkflowDefinition,
    snap: Mapping[str, Any],
    *,
    store: RunStore,
) -> str | None:
    """Pick the graph entry point implied by the run's confirmed artifacts."""
    names = {node.name for node in definition.nodes}
    run_id = str(snap["run_id"])

    if (
        snap["plan"] is not None
        and is_artifact_approved(store, run_id, "plan", snap)
        and is_plan_bound_to_upstream(store, run_id, snap)
    ):
        if definition.stop_after == "execute" and "execute" in names:
            return "execute"
        # Re-enter the terminal gate so the graph records its final routed/END events.
        if "gate_plan" in names:
            return "gate_plan"
        return None

    if snap["design"] is not None:
        if is_design_bound_to_spec(store, run_id, snap):
            return "plan" if "plan" in names else None
        spec = snap.get("spec")
        design = snap["design"]
        design_stale_for_spec = (
            spec is not None
            and is_artifact_approved(store, run_id, "spec", snap)
            and design.spec_hash is not None
            and design.spec_hash != spec.content_hash
        )
        if design_stale_for_spec:
            for candidate in ("design", "plan"):
                if candidate in names:
                    return candidate
            return None
        for candidate in ("gate_design", "validate_design", "design"):
            if candidate in names:
                return candidate
        return None

    if snap["spec"] is not None:
        if is_artifact_approved(store, run_id, "spec", snap):
            for candidate in ("design", "plan"):
                if candidate in names:
                    return candidate
            return None
        if "gate_spec" in names:
            return "gate_spec"
        return "align" if "align" in names else None

    return "intake" if "intake" in names else ("align" if "align" in names else None)


_VISITED_STATUSES = frozenset({"running", "completed", "paused", "error", "failed"})


def _declared_edges(description: Mapping[str, Any] | None) -> list[tuple[str, str]]:
    """Return the workflow graph's declared ``(source, target)`` pairs in order."""
    if not description:
        return []
    declared: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for edge in description.get("edges") or []:
        if not isinstance(edge, Mapping):
            continue
        source, target = edge.get("source"), edge.get("target")
        if not isinstance(source, str) or not isinstance(target, str) or source == target:
            continue
        if (source, target) in seen:
            continue
        seen.add((source, target))
        declared.append((source, target))
    return declared


def _node_names(value: Any) -> list[str]:
    """Normalize a ``graph.routed`` ``from``/``to`` payload into node names."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence):
        return [item for item in value if isinstance(item, str)]
    return []


def _derive_traversed_edges(
    events: Sequence[RunEvent],
    declared: Sequence[tuple[str, str]],
) -> list[dict[str, str]]:
    """Collect the executed trajectory against the declared workflow edges.

    Live runs persist one ``graph.routed`` event per wave, so the ``from`` x
    ``to`` product is the exact trajectory. Legacy and seeded runs never wrote
    them; for those, consecutive ``graph.node_finished`` -> ``graph.node_started``
    transitions are matched against the declared edges instead.
    """
    allowed = set(declared)
    traversed: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def record(source: str, target: str) -> None:
        if (source, target) not in allowed or (source, target) in seen:
            return
        seen.add((source, target))
        traversed.append({"source": source, "target": target})

    if any(event.event_type == "graph.routed" for event in events):
        for event in events:
            if event.event_type != "graph.routed":
                continue
            for source in _node_names(event.payload.get("from")):
                for target in _node_names(event.payload.get("to")):
                    record(source, target)
        return traversed

    last_finished: str | None = None
    for event in events:
        node = event.payload.get("node")
        if not isinstance(node, str):
            continue
        if event.event_type == "graph.node_finished":
            last_finished = node
        elif event.event_type == "graph.node_started" and last_finished is not None:
            record(last_finished, node)
    return traversed


def _forward_reachable(start: str, adjacency: Mapping[str, Sequence[str]]) -> set[str]:
    """Return every node reachable downstream of ``start`` via declared edges."""
    reached: set[str] = set()
    stack = [start]
    while stack:
        for target in adjacency.get(stack.pop(), ()):
            if target == start or target in reached:
                continue
            reached.add(target)
            stack.append(target)
    return reached


def _derive_skipped(
    statuses: Mapping[str, str],
    declared: Sequence[tuple[str, str]],
    traversed: Sequence[Mapping[str, str]],
) -> set[str]:
    """Return never-visited nodes the executed trajectory passed straight over."""
    known = set(statuses)
    adjacency: dict[str, list[str]] = {name: [] for name in statuses}
    for source, target in declared:
        if source in known and target in known and target not in adjacency[source]:
            adjacency[source].append(target)

    visited = {name for name, status in statuses.items() if status in _VISITED_STATUSES}
    if not visited:
        return set()

    reach = {name: _forward_reachable(name, adjacency) for name in known}
    downstream_of_visited: set[str] = set()
    for name in visited:
        downstream_of_visited |= reach[name]
    skip_edges = [
        (str(edge["source"]), str(edge["target"]))
        for edge in traversed
        if edge["source"] in known and edge["target"] in known
    ]

    skipped: set[str] = set()
    for name, status in statuses.items():
        if status != "pending":
            continue
        between_visited = name in downstream_of_visited and bool(reach[name] & visited)
        bypassed = any(
            name in reach[source] and target in reach[name] for source, target in skip_edges
        )
        if between_visited or bypassed:
            skipped.add(name)
    return skipped


def build_workflow_view(
    workflow_name: str,
    events: Sequence[RunEvent],
    run_status: RunStatus,
    *,
    spec_approved: bool = False,
    design_approved: bool = False,
    plan_approved: bool = False,
) -> dict[str, Any]:
    """Describe the run's workflow topology plus live per-node statuses from graph.* events."""
    from olympus.graph.loader import build_graph, load_workflow

    topology: str | None = None
    stop_after: str | None = None
    description: dict[str, Any] | None = None
    node_names: list[str] = []
    try:
        definition = load_workflow(workflow_name)
        description = build_graph(definition).describe()
        topology = definition.topology
        stop_after = definition.stop_after
        node_names = [str(n["name"]) for n in description.get("nodes", [])]
    except Exception:  # noqa: BLE001 - a legacy/unknown workflow must not break the console
        description = None

    statuses: dict[str, str] = {name: "pending" for name in node_names}
    current_node: str | None = None
    paused: dict[str, Any] | None = None
    failed_node: str | None = None
    last_error: str | None = None
    finished = False
    for event in events:
        payload = event.payload
        node = payload.get("node")
        if event.event_type == "graph.started":
            finished = False
        elif event.event_type == "graph.node_started" and isinstance(node, str):
            finished = False
            paused = None
            failed_node = None
            last_error = None
            current_node = node
            if node in statuses:
                statuses[node] = "running"
        elif event.event_type == "graph.node_finished" and isinstance(node, str):
            if node in statuses:
                statuses[node] = "completed"
        elif event.event_type == "graph.paused":
            paused = {
                "node": node,
                "reason": payload.get("reason"),
            }
            current_node = None
            if isinstance(node, str) and node in statuses:
                statuses[node] = "paused"
        elif event.event_type == "graph.finished":
            current_node = None
            paused = None
            finished = True
        elif event.event_type == "graph.node_error" and isinstance(node, str) and node in statuses:
            statuses[node] = "error"
            failed_node = node
            if payload.get("error"):
                last_error = str(payload.get("error"))
        elif event.event_type == "graph.failed":
            last_error = str(payload.get("error") or "工作流执行异常中断")
            failed_node = current_node or (node if isinstance(node, str) else None)
            if failed_node and failed_node in statuses:
                statuses[failed_node] = "failed"
            current_node = None

    # Reconcile gates that were approved before resuming directly downstream.
    approved_by_gate = {
        "spec": spec_approved,
        "design": design_approved,
        "plan": plan_approved,
    }
    for gate, gate_spec in GATE_SPECS.items():
        if approved_by_gate[gate] and statuses.get(gate_spec["node"]) == "paused":
            statuses[gate_spec["node"]] = "completed"

    if run_status == RunStatus.COMPLETED:
        paused = None
        for gate_spec in GATE_SPECS.values():
            if statuses.get(gate_spec["node"]) == "paused":
                statuses[gate_spec["node"]] = "completed"

    if run_status == RunStatus.FAILED:
        for name, status in list(statuses.items()):
            if status == "running":
                statuses[name] = "failed"
                failed_node = failed_node or name
        if last_error is None:
            last_error = "工作流执行异常中断"

    declared_edges = _declared_edges(description)
    traversed_edges = _derive_traversed_edges(events, declared_edges)
    for name in _derive_skipped(statuses, declared_edges, traversed_edges):
        statuses[name] = "skipped"

    return {
        "name": workflow_name,
        "topology": topology,
        "stop_after": stop_after,
        "graph": description,
        "nodes": [{"name": name, "status": status} for name, status in statuses.items()],
        "node_statuses": dict(statuses),
        "traversed_edges": traversed_edges,
        "current_node": current_node,
        "paused": paused,
        "failed_node": failed_node,
        "last_error": last_error,
        "finished": finished or run_status == RunStatus.COMPLETED,
    }


def is_blueprint_completed(
    *,
    plan: TaskPlan | None,
    plan_approved: bool,
    run_status: RunStatus,
    stop_after: str | None,
) -> bool:
    """Return whether a blueprint-only workflow froze a plan without executing code."""
    tasks = plan.tasks if plan is not None else []
    all_terminal = bool(tasks) and all(
        task.status in (TaskStatus.ACCEPTED, TaskStatus.CANCELLED) for task in tasks
    )
    return bool(
        plan is not None
        and plan_approved
        and run_status == RunStatus.COMPLETED
        and (stop_after or "execute") != "execute"
        and not all_terminal
    )


__all__ = [
    "GATE_SPECS",
    "GateSpec",
    "build_workflow_view",
    "is_artifact_approved",
    "is_blueprint_completed",
    "resolve_resume_node",
]