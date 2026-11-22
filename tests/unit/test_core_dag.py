"""Unit tests for DAG validation, write_scope conflict detection, and dispatch logic."""

from __future__ import annotations

import pytest

from olympus.core.contracts import (
    AcceptanceCheck,
    CheckKind,
    DesignModule,
    DesignSpec,
    GoalSpec,
    RequirementItem,
    TaskKind,
    TaskNode,
    TaskPlan,
    TaskStatus,
)
from olympus.core.dag import (
    PlanValidationError,
    assert_valid_plan,
    compute_dispatchable_tasks,
    normalize_scope_path,
    scopes_overlap,
    validate_plan,
)


def _check(cid: str = "C1", command: str = "pytest -q") -> AcceptanceCheck:
    return AcceptanceCheck(
        id=cid,
        kind=CheckKind.COMMAND,
        description="Run pytest",
        command=command,
    )


def _spec() -> GoalSpec:
    return GoalSpec(
        spec_id="spec-1",
        goal="Build module",
        requirements=[
            RequirementItem(id="R1", description="Core logic", acceptance_criteria=["Tests pass"]),
            RequirementItem(id="R2", description="CLI", acceptance_criteria=["CLI works"]),
        ],
    )


def test_normalize_scope_rejects_escape_and_absolute_paths() -> None:
    assert normalize_scope_path("src/olympus/") == "src/olympus/"
    assert normalize_scope_path("./src/a.py") == "src/a.py"
    with pytest.raises(ValueError):
        normalize_scope_path("/etc/passwd")
    with pytest.raises(ValueError):
        normalize_scope_path("../outside")


def test_scopes_overlap_detects_nested_and_root_conflicts() -> None:
    assert scopes_overlap(["src/core"], ["src/core/store.py"])
    assert scopes_overlap(["."], ["tests/test_a.py"])
    assert not scopes_overlap(["src/core/"], ["src/cli/"])


def test_validate_plan_detects_cycles_and_missing_coverage() -> None:
    spec = _spec()
    cyclic = TaskPlan(
        spec_id=spec.spec_id,
        spec_hash=spec.content_hash,
        tasks=[
            TaskNode(
                task_id="T1",
                title="A",
                objective="A",
                requirement_refs=["R1"],
                depends_on=["T2"],
                write_scope=["src/a.py"],
                acceptance_checks=[_check()],
            ),
            TaskNode(
                task_id="T2",
                title="B",
                objective="B",
                requirement_refs=["R1"],
                depends_on=["T1"],
                write_scope=["src/b.py"],
                acceptance_checks=[_check()],
            ),
        ],
    )
    with pytest.raises(PlanValidationError) as exc:
        assert_valid_plan(cyclic, spec=spec)
    msg = str(exc.value)
    assert "cycle" in msg.lower()
    assert "R2" in msg


def test_validate_plan_forbids_overlapping_write_scope_for_concurrent_siblings() -> None:
    spec = _spec()
    conflicting = TaskPlan(
        spec_id=spec.spec_id,
        spec_hash=spec.content_hash,
        tasks=[
            TaskNode(
                task_id="T1",
                title="A",
                objective="A",
                requirement_refs=["R1"],
                write_scope=["src/app/"],
                acceptance_checks=[_check()],
            ),
            TaskNode(
                task_id="T2",
                title="B",
                objective="B",
                requirement_refs=["R2"],
                write_scope=["src/app/cli.py"],
                acceptance_checks=[_check()],
            ),
        ],
    )
    with pytest.raises(PlanValidationError) as exc:
        assert_valid_plan(conflicting, spec=spec)
    assert "overlapping write_scope" in str(exc.value)

    # Making T2 depend on T1 serializes them, so shared write_scope becomes valid
    serialized = conflicting.model_copy(
        update={
            "tasks": [
                conflicting.tasks[0],
                conflicting.tasks[1].model_copy(update={"depends_on": ["T1"]}),
            ]
        }
    )
    assert_valid_plan(serialized, spec=spec)


def test_compute_dispatchable_tasks_respects_deps_slots_and_active_scopes() -> None:
    spec = _spec()
    plan = TaskPlan(
        spec_id=spec.spec_id,
        spec_hash=spec.content_hash,
        tasks=[
            TaskNode(
                task_id="T1",
                title="A",
                objective="A",
                requirement_refs=["R1"],
                write_scope=["src/a.py"],
                acceptance_checks=[_check()],
                status=TaskStatus.ACCEPTED,
            ),
            TaskNode(
                task_id="T2",
                title="B",
                objective="B",
                requirement_refs=["R2"],
                depends_on=["T1"],
                write_scope=["src/b.py"],
                acceptance_checks=[_check()],
                status=TaskStatus.RUNNING,
            ),
            TaskNode(
                task_id="T3",
                title="C",
                objective="C",
                requirement_refs=["R2"],
                depends_on=["T1"],
                write_scope=["src/c.py"],
                acceptance_checks=[_check()],
                status=TaskStatus.WAITING,
            ),
            TaskNode(
                task_id="T4",
                title="D",
                objective="D",
                requirement_refs=["R2"],
                depends_on=["T2"],
                write_scope=["src/d.py"],
                acceptance_checks=[_check()],
                status=TaskStatus.WAITING,
            ),
        ],
    )
    # T1 accepted -> T3 ready; T4 blocked on T2 (RUNNING)
    ready = compute_dispatchable_tasks(plan, max_concurrent=2)
    assert [t.task_id for t in ready] == ["T3"]
    # When max_concurrent=1, T2 already consumes the only slot
    assert compute_dispatchable_tasks(plan, max_concurrent=1) == []


def test_validate_plan_rejects_hardcoded_host_workspace_path_in_command_check() -> None:
    spec = _spec()
    bad_plan = TaskPlan(
        spec_id=spec.spec_id,
        spec_hash=spec.content_hash,
        tasks=[
            TaskNode(
                task_id="T1",
                title="Bad",
                objective="Bad command path",
                requirement_refs=["R1", "R2"],
                write_scope=["src/a.py"],
                acceptance_checks=[_check("C1", "cd /tmp/host_repo && pytest -q")],
            )
        ],
    )
    violations = validate_plan(bad_plan, spec=spec, workspace_root="/tmp/host_repo")
    assert any("hardcodes host workspace path" in v for v in violations)

    good_plan = bad_plan.model_copy(
        update={
            "tasks": [
                bad_plan.tasks[0].model_copy(
                    update={"acceptance_checks": [_check("C1", "pytest -q")]}
                )
            ]
        }
    )
    assert validate_plan(good_plan, spec=spec, workspace_root="/tmp/host_repo") == []


def test_validate_plan_rejects_broken_inline_python_syntax_and_accepts_multiline() -> None:
    spec = _spec()
    bad_try_cmd = (
        'python -c "from src.api.cfg import Config; c = Config(); '
        "try: c.limit = 99; assert False; except AttributeError: pass\""
    )
    bad_plan = TaskPlan(
        spec_id=spec.spec_id,
        spec_hash=spec.content_hash,
        tasks=[
            TaskNode(
                task_id="T1",
                title="Contract",
                objective="Verify config immutability",
                requirement_refs=["R1", "R2"],
                write_scope=["src/api/cfg.py"],
                acceptance_checks=[
                    _check("C-try", bad_try_cmd),
                    _check("C-quote", 'python -c "unclosed'),
                ],
            )
        ],
    )
    violations = validate_plan(bad_plan, spec=spec)
    assert any("C-try" in v and "invalid Python syntax" in v for v in violations)
    assert any("C-quote" in v and "invalid shell syntax" in v for v in violations)

    good_multiline_cmd = (
        'python -c "\nfrom src.api.cfg import Config\nc = Config()\n'
        'try:\n    c.limit = 99\n    assert False\nexcept AttributeError:\n    pass\n"'
    )
    good_plan = bad_plan.model_copy(
        update={
            "tasks": [
                bad_plan.tasks[0].model_copy(
                    update={
                        "acceptance_checks": [
                            _check("C-multi", good_multiline_cmd),
                            _check("C-pytest", "python -m pytest -c pyproject.toml -q"),
                        ]
                    }
                )
            ]
        }
    )
    assert validate_plan(good_plan, spec=spec) == []


def test_validate_plan_rejects_non_ancestor_sibling_import_and_accepts_dependency() -> None:
    spec = _spec()
    mw_cmd = (
        "python -c 'from src.api.limiter import FixedWindowLimiter; "
        "from src.api import server; assert FixedWindowLimiter'"
    )
    parallel_bad = TaskPlan(
        spec_id=spec.spec_id,
        spec_hash=spec.content_hash,
        tasks=[
            TaskNode(
                task_id="T1-limiter",
                title="Limiter",
                objective="Limiter core",
                requirement_refs=["R1"],
                write_scope=["src/api/limiter.py"],
                acceptance_checks=[_check("CL", "python -c 'import src.api.limiter'")],
            ),
            TaskNode(
                task_id="T1-middleware",
                title="Middleware",
                objective="Rate limit middleware",
                requirement_refs=["R2"],
                depends_on=[],
                write_scope=["src/api/middleware.py"],
                acceptance_checks=[_check("CM", mw_cmd)],
            ),
            TaskNode(
                task_id="T2-integrate",
                title="Export",
                objective="Export package symbol",
                requirement_refs=["R2"],
                depends_on=["T1-limiter", "T1-middleware"],
                write_scope=["src/api/__init__.py"],
                acceptance_checks=[_check("CI", "pytest -q")],
            ),
        ],
    )
    violations = validate_plan(parallel_bad, spec=spec)
    assert len(violations) == 1
    assert "T1-middleware" in violations[0]
    assert "T1-limiter" in violations[0]

    ordered_ok = parallel_bad.model_copy(
        update={
            "tasks": [
                parallel_bad.tasks[0],
                parallel_bad.tasks[1].model_copy(update={"depends_on": ["T1-limiter"]}),
                parallel_bad.tasks[2],
            ]
        }
    )
    assert validate_plan(ordered_ok, spec=spec) == []

    # Stdlib / third-party imports must never false-positive even when a sibling owns `src/`.
    stdlib_plan = TaskPlan(
        spec_id=spec.spec_id,
        spec_hash=spec.content_hash,
        tasks=[
            TaskNode(
                task_id="T1-src",
                title="Core",
                objective="Core src",
                requirement_refs=["R1"],
                write_scope=["src/"],
                acceptance_checks=[_check("C1", "python -c 'pass'")],
            ),
            TaskNode(
                task_id="T2-docs",
                title="Docs",
                objective="Verify docs with pathlib and pytest",
                requirement_refs=["R2"],
                write_scope=["docs/"],
                acceptance_checks=[
                    _check("C2", "python -c 'import pathlib, pytest; assert pathlib.Path'")
                ],
            ),
        ],
    )
    assert validate_plan(stdlib_plan, spec=spec) == []


def _vertical_spec() -> GoalSpec:
    return GoalSpec(
        spec_id="spec-vertical",
        goal="Ship inventory capabilities in vertical slices",
        requirements=[
            RequirementItem(
                id=f"R{index}",
                description=f"Capability {index}",
                acceptance_criteria=["Works end to end"],
            )
            for index in range(1, 5)
        ],
    )


def _vertical_design(spec: GoalSpec) -> DesignSpec:
    return DesignSpec(
        design_id="design-vertical",
        spec_id=spec.spec_id,
        spec_hash=spec.content_hash,
        architecture_summary="One deep storage core with two feature modules.",
        modules=[
            DesignModule(
                id="M1",
                name="Core",
                responsibility="Storage and entrypoint",
                requirement_refs=["R1"],
                target_paths=["src/core.py"],
            ),
            DesignModule(
                id="M2",
                name="Items",
                responsibility="Item capability",
                requirement_refs=["R2", "R3"],
                target_paths=["src/items.py"],
            ),
            DesignModule(
                id="M3",
                name="Count",
                responsibility="Count capability",
                requirement_refs=["R4"],
                target_paths=["src/count.py"],
            ),
        ],
    )


def _vertical_slices_plan(spec: GoalSpec, design: DesignSpec) -> TaskPlan:
    """Slice 1 skeleton, then two parallel vertical feature slices."""
    return TaskPlan(
        spec_id=spec.spec_id,
        spec_hash=spec.content_hash,
        design_id=design.design_id,
        design_hash=design.content_hash,
        tasks=[
            TaskNode(
                task_id="T1-skeleton",
                title="Runnable skeleton",
                objective="Wire storage into one observable seam",
                requirement_refs=["R1"],
                design_refs=["M1"],
                write_scope=["src/core.py", "src/app.py", "tests/test_app.py"],
                acceptance_checks=[_check("C1", "python -m pytest tests/test_app.py -q")],
            ),
            TaskNode(
                task_id="T2-items",
                title="Items slice",
                objective="Deliver the item capability end to end",
                requirement_refs=["R2", "R3"],
                design_refs=["M2"],
                depends_on=["T1-skeleton"],
                write_scope=["src/items.py", "tests/test_items.py"],
                acceptance_checks=[_check("C2", "python -m pytest tests/test_items.py -q")],
            ),
            TaskNode(
                task_id="T3-count",
                title="Count slice",
                objective="Deliver the count capability end to end",
                requirement_refs=["R4"],
                design_refs=["M3"],
                depends_on=["T1-skeleton"],
                write_scope=["src/count.py", "tests/test_count.py"],
                acceptance_checks=[_check("C3", "python -m pytest tests/test_count.py -q")],
            ),
        ],
    )


@pytest.mark.parametrize("topology", ["tdd_first", "vertical_slices"])
@pytest.mark.parametrize("kind", [TaskKind.IMPLEMENT, TaskKind.INTEGRATE])
def test_validate_plan_vertical_gate_requires_executable_command_per_slice(
    topology: str, kind: TaskKind
) -> None:
    spec = _spec()
    review_only = AcceptanceCheck(
        id="C1", kind=CheckKind.INDEPENDENT_REVIEW, description="Cold review only"
    )
    blank_command = AcceptanceCheck(
        id="C1", kind=CheckKind.COMMAND, description="Blank command", command="   "
    )
    comment_only = AcceptanceCheck(
        id="C1", kind=CheckKind.COMMAND, description="TODO placeholder", command="# seam test TODO"
    )
    human_on_engineer = AcceptanceCheck(
        id="C1", kind=CheckKind.HUMAN, description="Manual signoff on coding slice"
    )
    for checks in ([review_only], [blank_command], [comment_only], [human_on_engineer]):
        plan = TaskPlan(
            spec_id=spec.spec_id,
            spec_hash=spec.content_hash,
            tasks=[
                TaskNode(
                    task_id="T1-slice",
                    title="Horizontal code wave",
                    objective="Implement without a seam check",
                    requirement_refs=["R1", "R2"],
                    kind=kind,
                    write_scope=["src/slice.py"],
                    acceptance_checks=checks,
                )
            ],
        )
        violations = validate_plan(plan, spec=spec, topology=topology)
        assert any("T1-slice" in violation and "command" in violation for violation in violations)

    human_gate = TaskPlan(
        spec_id=spec.spec_id,
        spec_hash=spec.content_hash,
        tasks=[
            TaskNode(
                task_id="T1-human",
                title="Human approval gate",
                objective="Review release notes",
                requirement_refs=["R1", "R2"],
                role="human",
                kind=kind,
                write_scope=["docs/release.md"],
                acceptance_checks=[human_on_engineer],
            )
        ],
    )
    assert validate_plan(human_gate, spec=spec, topology=topology) == []


def test_validate_plan_vertical_gate_rejects_horizontal_test_only_deferral() -> None:
    spec = _spec()
    plan = TaskPlan(
        spec_id=spec.spec_id,
        spec_hash=spec.content_hash,
        tasks=[
            TaskNode(
                task_id="T1-code",
                title="Code wave",
                objective="Ship code without a seam test",
                requirement_refs=["R1"],
                write_scope=["src/code.py"],
                acceptance_checks=[
                    AcceptanceCheck(
                        id="C1", kind=CheckKind.INDEPENDENT_REVIEW, description="Cold review only"
                    )
                ],
            ),
            TaskNode(
                task_id="T2-tests",
                title="Bulk test wave",
                objective="Backfill all tests after the code wave",
                requirement_refs=["R2"],
                kind=TaskKind.TEST,
                depends_on=["T1-code"],
                write_scope=["tests/"],
                acceptance_checks=[_check("C2")],
            ),
        ],
    )
    violations = validate_plan(plan, spec=spec, topology="tdd_first")
    assert any(
        "T2-tests" in violation and "horizontally defers verification" in violation
        for violation in violations
    )
    assert any("T1-code" in violation and "command" in violation for violation in violations)


def test_validate_plan_vertical_gate_rejects_monolithic_multi_module_sink() -> None:
    spec = _vertical_spec()
    design = _vertical_design(spec)
    plan = TaskPlan(
        spec_id=spec.spec_id,
        spec_hash=spec.content_hash,
        design_id=design.design_id,
        design_hash=design.content_hash,
        tasks=[
            TaskNode(
                task_id="T1-everything",
                title="Catch-all implementation",
                objective="Build every module and requirement in one ticket",
                requirement_refs=["R1", "R2", "R3", "R4"],
                design_refs=["M1", "M2", "M3"],
                write_scope=["src/core.py", "src/items.py", "src/count.py", "src/api.py"],
                acceptance_checks=[_check()],
            )
        ],
    )
    violations = validate_plan(plan, spec=spec, design=design, topology="vertical_slices")
    assert any(
        "T1-everything" in violation and "monolithic" in violation for violation in violations
    )


@pytest.mark.parametrize("topology", ["tdd_first", "vertical_slices"])
def test_validate_plan_vertical_gate_accepts_clean_tracer_bullet_dag(topology: str) -> None:
    spec = _vertical_spec()
    design = _vertical_design(spec)
    assert (
        validate_plan(
            _vertical_slices_plan(spec, design), spec=spec, design=design, topology=topology
        )
        == []
    )


@pytest.mark.parametrize("topology", [None, "minimal_patch", "modular_product"])
def test_validate_plan_vertical_gate_is_noop_for_legacy_topologies(
    topology: str | None,
) -> None:
    spec = _vertical_spec()
    design = _vertical_design(spec)
    monolithic = TaskPlan(
        spec_id=spec.spec_id,
        spec_hash=spec.content_hash,
        design_id=design.design_id,
        design_hash=design.content_hash,
        tasks=[
            TaskNode(
                task_id="T1-everything",
                title="Catch-all implementation",
                objective="Build every module and requirement in one ticket",
                requirement_refs=["R1", "R2", "R3", "R4"],
                design_refs=["M1", "M2", "M3"],
                write_scope=["src/core.py", "src/items.py", "src/count.py", "src/api.py"],
                acceptance_checks=[_check()],
            )
        ],
    )
    assert validate_plan(monolithic, spec=spec, design=design) == []
    assert validate_plan(monolithic, spec=spec, design=design, topology=topology) == []
    with pytest.raises(PlanValidationError):
        assert_valid_plan(monolithic, spec=spec, design=design, topology="tdd_first")


def test_validate_plan_monolithic_sink_catches_bypasses_and_allows_thin_tracer() -> None:
    spec = _vertical_spec()
    design = _vertical_design(spec)

    root_monolith = TaskPlan(
        spec_id=spec.spec_id,
        spec_hash=spec.content_hash,
        design_id=design.design_id,
        design_hash=design.content_hash,
        tasks=[
            TaskNode(
                task_id="T-root",
                title="Repo-root monolith",
                objective="Implement everything under repo root",
                requirement_refs=["R1", "R2", "R3", "R4"],
                design_refs=["M1", "M2", "M3"],
                write_scope=["."],
                acceptance_checks=[_check()],
            )
        ],
    )
    assert any(
        "T-root" in v and "monolithic" in v
        for v in validate_plan(root_monolith, spec=spec, design=design, topology="tdd_first")
    )

    dir_scopes = ["storage/", "logic/", "ui/", "tests/"]
    multi_dir_design = design.model_copy(
        update={
            "modules": [
                mod.model_copy(update={"target_paths": [scope]})
                for mod, scope in zip(design.modules, dir_scopes[:3], strict=True)
            ]
        }
    )
    omitted_refs_plan = TaskPlan(
        spec_id=spec.spec_id,
        spec_hash=spec.content_hash,
        design_id=multi_dir_design.design_id,
        design_hash=multi_dir_design.content_hash,
        tasks=[
            TaskNode(
                task_id="T-boot",
                title="Bootstrap",
                objective="Seed bootstrap",
                requirement_refs=["R1", "R2", "R3"],
                design_refs=["M1", "M2", "M3"],
                write_scope=["bootstrap/"],
                acceptance_checks=[_check("C1")],
            ),
            TaskNode(
                task_id="T-sink",
                title="Catch-all without design_refs",
                objective="Write every module scope at once",
                requirement_refs=["R4"],
                design_refs=[],
                depends_on=["T-boot"],
                write_scope=dir_scopes,
                acceptance_checks=[_check("C2")],
            ),
        ],
    )
    assert any(
        "T-sink" in v and "monolithic" in v
        for v in validate_plan(
            omitted_refs_plan, spec=spec, design=multi_dir_design, topology="tdd_first"
        )
    )

    design_only_sink = TaskPlan(
        design_id=multi_dir_design.design_id,
        design_hash=multi_dir_design.content_hash,
        tasks=[
            TaskNode(
                task_id="T-design-only",
                title="Design-only catch-all",
                objective="Implement all modules in one task",
                requirement_refs=["R1", "R2", "R3", "R4"],
                design_refs=["M1", "M2", "M3"],
                write_scope=dir_scopes,
                acceptance_checks=[_check()],
            )
        ],
    )
    assert any(
        "T-design-only" in v and "monolithic" in v
        for v in validate_plan(design_only_sink, design=multi_dir_design, topology="tdd_first")
    )

    clean_plan = _vertical_slices_plan(spec, design)
    thin_tracer = clean_plan.tasks[0].model_copy(
        update={
            "requirement_refs": ["R1"],
            "design_refs": ["M1", "M2", "M3"],
            "write_scope": ["src/core.py", "src/items.py", "src/count.py", "src/app.py"],
        }
    )
    tracer_plan = clean_plan.model_copy(
        update={"tasks": [thin_tracer, clean_plan.tasks[1], clean_plan.tasks[2]]}
    )
    assert validate_plan(tracer_plan, spec=spec, design=design, topology="tdd_first") == []
