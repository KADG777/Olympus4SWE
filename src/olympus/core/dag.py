"""Deterministic DAG validation, write_scope conflict detection, and ready-task scheduling."""

from __future__ import annotations

import re
from collections import deque
from pathlib import Path, PurePosixPath

from olympus.core.contracts import CheckKind, GoalSpec, TaskNode, TaskPlan, TaskStatus


class PlanValidationError(ValueError):
    """Raised when a TaskPlan violates structural, dependency, coverage, or scope invariants."""

    def __init__(self, violations: list[str]) -> None:
        self.violations = violations
        super().__init__("; ".join(violations))


def normalize_scope_path(raw: str) -> str:
    """Normalize a repo-relative POSIX scope pattern and reject escapes."""
    cleaned = raw.strip().replace("\\", "/")
    if not cleaned:
        raise ValueError("Empty write_scope path is not allowed")
    if cleaned.startswith("/"):
        raise ValueError(f"write_scope must be repo-relative, got absolute path: {raw}")
    is_dir_hint = cleaned.endswith("/")
    pure = PurePosixPath(cleaned.rstrip("/") or ".")
    if ".." in pure.parts:
        raise ValueError(f"write_scope cannot contain '..': {raw}")
    norm = pure.as_posix()
    if norm == ".":
        return "."
    return f"{norm}/" if is_dir_hint else norm


def paths_overlap(a_raw: str, b_raw: str) -> bool:
    """Return True if two normalized repo-relative paths or directory prefixes can overlap."""
    a = normalize_scope_path(a_raw).rstrip("/")
    b = normalize_scope_path(b_raw).rstrip("/")
    if a == "." or b == ".":
        return True
    if a == b:
        return True
    return a.startswith(f"{b}/") or b.startswith(f"{a}/")


def scopes_overlap(scope_a: list[str], scope_b: list[str]) -> list[tuple[str, str]]:
    """Return all conflicting pairs between two write_scope lists."""
    conflicts: list[tuple[str, str]] = []
    for pa in scope_a:
        for pb in scope_b:
            if paths_overlap(pa, pb):
                conflicts.append((pa, pb))
    return conflicts


def _build_ancestors(tasks: list[TaskNode]) -> dict[str, set[str]]:
    by_id = {t.task_id: t for t in tasks}
    ancestors: dict[str, set[str]] = {}

    def visit(tid: str, visiting: set[str]) -> set[str]:
        if tid in ancestors:
            return ancestors[tid]
        if tid in visiting:
            return set()
        visiting.add(tid)
        acc: set[str] = set()
        for dep in by_id[tid].depends_on:
            if dep in by_id:
                acc.add(dep)
                acc |= visit(dep, visiting)
        visiting.remove(tid)
        ancestors[tid] = acc
        return acc

    for t in tasks:
        visit(t.task_id, set())
    return ancestors


def validate_plan(
    plan: TaskPlan,
    spec: GoalSpec | None = None,
    *,
    workspace_root: Path | str | None = None,
) -> list[str]:
    """Validate structural DAG properties, spec coverage, and concurrent write_scope isolation."""
    violations: list[str] = []
    host_root_pattern: re.Pattern[str] | None = None
    host_root_display = ""
    if workspace_root is not None:
        host_root_display = str(Path(workspace_root).resolve())
        host_root_pattern = re.compile(
            rf"(?:^|[\s\"'=:;]){re.escape(host_root_display)}(?:$|[\s\"'/:;])"
        )

    by_id: dict[str, TaskNode] = {}
    for task in plan.tasks:
        if task.task_id in by_id:
            violations.append(f"Duplicate task_id: {task.task_id}")
        by_id[task.task_id] = task
        for scope in task.write_scope:
            try:
                normalize_scope_path(scope)
            except ValueError as exc:
                violations.append(f"Task {task.task_id} invalid write_scope ({scope}): {exc}")
        if host_root_pattern is not None:
            for check in task.acceptance_checks:
                if (
                    check.kind == CheckKind.COMMAND
                    and check.command
                    and host_root_pattern.search(check.command)
                ):
                    violations.append(
                        f"Task {task.task_id} check {check.id} hardcodes host workspace path "
                        f"'{host_root_display}'; command checks run inside isolated sandbox cwd "
                        "and must use relative paths."
                    )

    # Dependency existence & self-loops
    in_degree: dict[str, int] = {tid: 0 for tid in by_id}
    children: dict[str, list[str]] = {tid: [] for tid in by_id}
    for task in plan.tasks:
        for dep in task.depends_on:
            if dep == task.task_id:
                violations.append(f"Task {task.task_id} cannot depend on itself")
            elif dep not in by_id:
                violations.append(f"Task {task.task_id} depends on unknown task: {dep}")
            else:
                in_degree[task.task_id] += 1
                children[dep].append(task.task_id)

    # Kahn topological sort for cycle detection
    queue = deque(tid for tid, deg in in_degree.items() if deg == 0)
    visited = 0
    while queue:
        cur = queue.popleft()
        visited += 1
        for nxt in children[cur]:
            in_degree[nxt] -= 1
            if in_degree[nxt] == 0:
                queue.append(nxt)
    if visited != len(by_id):
        violations.append("TaskPlan contains a dependency cycle")

    # Spec traceability coverage
    if spec is not None:
        valid_reqs = {r.id for r in spec.requirements}
        covered_reqs: set[str] = set()
        for task in plan.tasks:
            for ref in task.requirement_refs:
                if ref not in valid_reqs:
                    violations.append(f"Task {task.task_id} references unknown requirement: {ref}")
                else:
                    covered_reqs.add(ref)
        missing_reqs = sorted(valid_reqs - covered_reqs)
        if missing_reqs:
            violations.append(f"Requirements not covered by any task: {', '.join(missing_reqs)}")

    # Independent sibling tasks must have disjoint write_scopes
    if visited == len(by_id):
        ancestors = _build_ancestors(plan.tasks)
        tasks_list = list(by_id.values())
        for i in range(len(tasks_list)):
            for j in range(i + 1, len(tasks_list)):
                ta, tb = tasks_list[i], tasks_list[j]
                ordered = tb.task_id in ancestors[ta.task_id] or ta.task_id in ancestors[tb.task_id]
                if ordered:
                    continue
                try:
                    conflicts = scopes_overlap(ta.write_scope, tb.write_scope)
                except ValueError:
                    conflicts = []
                if conflicts:
                    pairs = ", ".join(f"{a}<->{b}" for a, b in conflicts)
                    violations.append(
                        f"Concurrent-eligible tasks {ta.task_id} and {tb.task_id} "
                        f"have overlapping write_scope: {pairs}"
                    )

    return violations


def assert_valid_plan(
    plan: TaskPlan,
    spec: GoalSpec | None = None,
    *,
    workspace_root: Path | str | None = None,
) -> None:
    violations = validate_plan(plan, spec=spec, workspace_root=workspace_root)
    if violations:
        raise PlanValidationError(violations)


def compute_dispatchable_tasks(
    plan: TaskPlan,
    *,
    max_concurrent: int = 3,
) -> list[TaskNode]:
    """Return tasks ready to run now without write_scope or concurrency conflicts."""
    by_id = {t.task_id: t for t in plan.tasks}
    active = [t for t in plan.tasks if t.status in (TaskStatus.RUNNING, TaskStatus.VERIFYING)]
    available_slots = max(0, max_concurrent - len(active))
    if available_slots == 0:
        return []

    active_scopes = [s for t in active for s in t.write_scope]
    dispatchable: list[TaskNode] = []

    for task in plan.tasks:
        if task.status not in (TaskStatus.WAITING, TaskStatus.READY):
            continue
        deps_accepted = all(
            dep in by_id and by_id[dep].status == TaskStatus.ACCEPTED for dep in task.depends_on
        )
        if not deps_accepted:
            continue
        if scopes_overlap(task.write_scope, active_scopes):
            continue
        dispatchable.append(task)
        active_scopes.extend(task.write_scope)
        if len(dispatchable) >= available_slots:
            break

    return dispatchable
