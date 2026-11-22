"""Seam-first tests for deep-module design guidance and the shallow-duplicate gate."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from olympus.core.contracts import (
    AcceptanceCriterion,
    DataEntity,
    DesignSpec,
    EarsPattern,
    ExecutionStrategy,
    GoalSpec,
    RequirementItem,
    SpecStatus,
)
from olympus.core.dag import validate_design_spec
from olympus.core.store import RunStore
from olympus.providers.base import ChatMessage, LLMResponse, MockLLMProvider, ToolSpec
from olympus.stages.design import DesignStage


def _goal_spec() -> GoalSpec:
    return GoalSpec(
        spec_id="spec-deep",
        goal="Operators manage notes through a REST API",
        deliverable_kind="rest_api",
        observable_surfaces=["HTTP API"],
        actors=["operator"],
        requirements=[
            RequirementItem(
                id="R1",
                description="Create a note through POST /notes",
                actor="operator",
                ears_criteria=[
                    AcceptanceCriterion(
                        id="R1-C1",
                        pattern=EarsPattern.EVENT_DRIVEN,
                        condition="an operator posts a note",
                        outcome="return HTTP 201 within 200 ms",
                    )
                ],
            ),
            RequirementItem(
                id="R2",
                description="Reject malformed payloads",
                actor="operator",
                ears_criteria=[
                    AcceptanceCriterion(
                        id="R2-C1",
                        pattern=EarsPattern.UNWANTED_BEHAVIOR,
                        condition="the name field is missing",
                        outcome="return HTTP 422 and never write a row",
                    )
                ],
            ),
        ],
        out_of_scope=["multi-tenant isolation"],
        constraints=["python 3.13"],
    )


def _interface(**overrides: object) -> dict[str, object]:
    data: dict[str, object] = {
        "id": "IF1",
        "provider_module_id": "M2",
        "consumer_module_ids": ["M1"],
        "signature": "create_note(name: str) -> Note",
        "input_schema": [{"name": "name", "type_expr": "str", "required": True}],
        "output_schema": [{"name": "id", "type_expr": "str", "required": True}],
        "data_entity_refs": ["Note"],
        "error_cases": ["duplicate name"],
    }
    data.update(overrides)
    return data


def _payload(spec: GoalSpec, **overrides: object) -> dict[str, object]:
    data: dict[str, object] = {
        "architecture_summary": "One deep notes module behind the HTTP seam",
        "spec_id": spec.spec_id,
        "spec_hash": spec.content_hash,
        "shared_invariants": [
            {
                "id": "INV-ERR",
                "category": "error_envelope",
                "rule": "Non-2xx responses use {error: {code, message}}.",
            }
        ],
        "modules": [
            {
                "id": "M1",
                "name": "Notes core",
                "responsibility": "Validation, storage, and the HTTP seam for notes",
                "requirement_refs": ["R1", "R2"],
                "target_paths": ["src/notes/"],
            }
        ],
        "data_models": [{"name": "Note", "fields": ["id", "name"], "owner_module_id": "M1"}],
        "interfaces": [],
        "execution_strategy": ExecutionStrategy.SERIAL_HANDOFF.value,
        "strategy_rationale": "One cohesive module; sequential delivery has no cross-module seam.",
    }
    data.update(overrides)
    return data


def test_design_stage_prompt_teaches_deep_modules_and_accepts_lean_design(
    tmp_path: Path,
) -> None:
    spec = _goal_spec()
    store = RunStore(tmp_path / ".olympus" / "olympus.db")
    run_id = store.create_run(title="deep design")
    store.save_spec(run_id, spec.model_copy(update={"status": SpecStatus.CONFIRMED}))
    payload = _payload(spec)

    def responder(messages: Sequence[ChatMessage], tools: Sequence[ToolSpec]) -> LLMResponse:
        del tools
        assert "White-Box Solution Architect" in messages[0].content
        return LLMResponse(content=json.dumps(payload))

    provider = MockLLMProvider(responder=responder)
    design = DesignStage(store, provider, tmp_path).propose_design(run_id)

    assert len(design.modules) == 1
    assert design.execution_strategy is ExecutionStrategy.SERIAL_HANDOFF
    assert design.shared_contract_paths == []

    system_prompt = provider.recorded_requests[0][0][0].content
    assert "1-3" in system_prompt
    assert "Deletion Test" in system_prompt
    assert ">= 2 real callers/adapters" in system_prompt
    assert "Real vs. hypothetical seams" in system_prompt
    assert "test_seams" in system_prompt
    assert "2-6" not in system_prompt


def test_validate_design_spec_accepts_lean_serial_handoff_without_wave0_contracts() -> None:
    spec = _goal_spec()

    single = DesignSpec.model_validate(_payload(spec))
    assert validate_design_spec(single, spec, strict_v2=True) == []

    two = DesignSpec.model_validate(
        _payload(
            spec,
            architecture_summary="HTTP edge over a deep notes store",
            modules=[
                {
                    "id": "M1",
                    "name": "HTTP edge",
                    "responsibility": "Routing and request validation",
                    "requirement_refs": ["R1", "R2"],
                    "target_paths": ["src/api/"],
                },
                {
                    "id": "M2",
                    "name": "Notes store",
                    "responsibility": "Persistence and retrieval",
                    "requirement_refs": ["R1"],
                    "target_paths": ["src/store/"],
                },
            ],
            data_models=[{"name": "Note", "fields": ["id", "name"], "owner_module_id": "M2"}],
            interfaces=[_interface()],
        )
    )
    assert two.shared_contract_paths == []
    assert validate_design_spec(two, spec, strict_v2=True) == []

    three = DesignSpec.model_validate(
        _payload(
            spec,
            architecture_summary="HTTP edge over storage and validation cores",
            modules=[
                {
                    "id": "M1",
                    "name": "HTTP edge",
                    "responsibility": "Routing and request validation",
                    "requirement_refs": ["R1", "R2"],
                    "target_paths": ["src/api/"],
                },
                {
                    "id": "M2",
                    "name": "Notes store",
                    "responsibility": "Persistence and retrieval",
                    "requirement_refs": ["R1"],
                    "target_paths": ["src/store/"],
                },
                {
                    "id": "M3",
                    "name": "Validation rules",
                    "responsibility": "Payload validation rules",
                    "requirement_refs": ["R2"],
                    "target_paths": ["src/rules/"],
                },
            ],
            data_models=[{"name": "Note", "fields": ["id", "name"], "owner_module_id": "M2"}],
            interfaces=[
                _interface(),
                _interface(
                    id="IF2",
                    provider_module_id="M3",
                    signature="validate_note(name: str) -> bool",
                    output_schema=[{"name": "ok", "type_expr": "bool", "required": True}],
                    data_entity_refs=[],
                    error_cases=["missing name"],
                ),
            ],
        )
    )
    assert validate_design_spec(three, spec, strict_v2=True) == []


def test_validate_design_spec_rejects_shallow_duplicate_modules() -> None:
    spec = _goal_spec()
    duplicate_payload = _payload(
        spec,
        architecture_summary="Storage core shadowed by a pass-through facade",
        modules=[
            {
                "id": "M1",
                "name": "Notes core",
                "responsibility": "Persistence",
                "requirement_refs": ["R1", "R2"],
                "target_paths": ["src/core/"],
            },
            {
                "id": "M2",
                "name": "Notes facade",
                "responsibility": "Single-caller pass-through facade",
                "requirement_refs": ["R1", "R2"],
                "target_paths": ["src/facade/"],
            },
        ],
        data_models=[{"name": "Note", "fields": ["id", "name"], "owner_module_id": "M1"}],
        interfaces=[
            _interface(provider_module_id="M1", consumer_module_ids=["M2"]),
        ],
    )
    duplicate = DesignSpec.model_validate(duplicate_payload)
    violations = validate_design_spec(duplicate, spec, strict_v2=True)
    assert any("Deletion Test" in violation and "M2" in violation for violation in violations)
    assert validate_design_spec(duplicate, spec) == []

    distinct_owner = duplicate.model_copy(
        update={
            "data_models": [
                *duplicate.data_models,
                DataEntity(name="FacadeAudit", fields=["id"], owner_module_id="M2"),
            ]
        }
    )
    assert validate_design_spec(distinct_owner, spec, strict_v2=True) == []

    for surface in ("HTTP /api/notes", "CLI blacknote"):
        boundary_spec = _goal_spec().model_copy(update={"observable_surfaces": [surface]})
        boundary = DesignSpec.model_validate(
            _payload(
                boundary_spec,
                architecture_summary="Domain core paired with a dedicated public boundary",
                modules=[
                    {
                        "id": "M1",
                        "name": "Notes core",
                        "responsibility": "Persist notes and enforce domain invariants",
                        "requirement_refs": ["R1", "R2"],
                        "target_paths": ["src/core/"],
                    },
                    {
                        "id": "M2",
                        "name": "Public boundary",
                        "responsibility": (
                            f"Own {surface}, authenticate requests, decode input, "
                            "and map domain errors to public responses"
                        ),
                        "requirement_refs": ["R1", "R2"],
                        "target_paths": ["src/edge/"],
                    },
                ],
                data_models=[{"name": "Note", "fields": ["id", "name"], "owner_module_id": "M1"}],
                interfaces=[_interface(provider_module_id="M1", consumer_module_ids=["M2"])],
            )
        )
        assert validate_design_spec(boundary, boundary_spec, strict_v2=True) == []

    case_spec = _goal_spec().model_copy(update={"observable_surfaces": ["bin/BlackNote"]})
    case_wrapper = DesignSpec.model_validate(
        _payload(
            case_spec,
            architecture_summary="Case-mismatched thin wrapper around binary entrypoint",
            modules=[
                {
                    "id": "M1",
                    "name": "Notes core",
                    "responsibility": "Persist notes and ship the binary entrypoint",
                    "requirement_refs": ["R1", "R2"],
                    "target_paths": ["src/core/", "bin/BlackNote"],
                },
                {
                    "id": "M2",
                    "name": "Forwarding shim",
                    "responsibility": "Single-caller forwarding wrapper delegating straight to M1",
                    "requirement_refs": ["R1", "R2"],
                    "target_paths": ["bin/blacknote"],
                },
            ],
            data_models=[{"name": "Note", "fields": ["id", "name"], "owner_module_id": "M1"}],
            interfaces=[_interface(provider_module_id="M1", consumer_module_ids=["M2"])],
        )
    )
    case_violations = validate_design_spec(case_wrapper, case_spec, strict_v2=True)
    assert any("M2" in violation and "Deletion Test" in violation for violation in case_violations)


def test_validate_design_spec_keeps_contract_first_parallel_multimodule_designs_valid() -> None:
    spec = _goal_spec()
    parallel = DesignSpec.model_validate(
        _payload(
            spec,
            architecture_summary="HTTP edge and notes store lock Wave 0 contracts",
            shared_contract_paths=["src/contracts/"],
            modules=[
                {
                    "id": "M1",
                    "name": "HTTP edge",
                    "responsibility": "Routing and request validation",
                    "requirement_refs": ["R1", "R2"],
                    "target_paths": ["src/api/"],
                },
                {
                    "id": "M2",
                    "name": "Notes store",
                    "responsibility": "Persistence and retrieval",
                    "requirement_refs": ["R1"],
                    "target_paths": ["src/store/"],
                },
            ],
            data_models=[{"name": "Note", "fields": ["id", "name"], "owner_module_id": "M2"}],
            interfaces=[_interface()],
            execution_strategy=ExecutionStrategy.CONTRACT_FIRST_PARALLEL.value,
            strategy_rationale="M1 and M2 build independently against src/contracts/ from Wave 0.",
        )
    )
    assert validate_design_spec(parallel, spec, strict_v2=True) == []

    shared_ownership = DesignSpec.model_validate(
        _payload(
            spec,
            architecture_summary="Two owned cores over identical requirements",
            shared_contract_paths=["src/contracts/"],
            modules=[
                {
                    "id": "M1",
                    "name": "Notes core",
                    "responsibility": "Note persistence",
                    "requirement_refs": ["R1", "R2"],
                    "target_paths": ["src/core/"],
                },
                {
                    "id": "M2",
                    "name": "Audit core",
                    "responsibility": "Audit persistence",
                    "requirement_refs": ["R1", "R2"],
                    "target_paths": ["src/audit/"],
                },
            ],
            data_models=[
                {"name": "Note", "fields": ["id", "name"], "owner_module_id": "M1"},
                {"name": "Audit", "fields": ["id"], "owner_module_id": "M2"},
            ],
            interfaces=[
                _interface(
                    provider_module_id="M1",
                    consumer_module_ids=["M2"],
                    signature="create_note(name: str) -> Note",
                    data_entity_refs=["Note"],
                )
            ],
            execution_strategy=ExecutionStrategy.CONTRACT_FIRST_PARALLEL.value,
            strategy_rationale=(
                "Both cores own distinct entities and build against Wave 0 contracts."
            ),
        )
    )
    assert validate_design_spec(shared_ownership, spec, strict_v2=True) == []
