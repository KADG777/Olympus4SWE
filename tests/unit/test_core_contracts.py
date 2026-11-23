"""Unit tests for core contracts, fingerprinting, and budget/review invariants."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from olympus.core.contracts import (
    AcceptanceCheck,
    ApprovalDecision,
    ApprovalRecord,
    ApprovalTarget,
    BudgetLedger,
    BudgetPolicy,
    CheckKind,
    GoalSpec,
    RequirementItem,
    ReviewFinding,
    ReviewSeverity,
    ReviewVerdict,
    SpecStatus,
)


def _sample_spec(**overrides: object) -> GoalSpec:
    data: dict[str, object] = {
        "spec_id": "spec-1",
        "version": 1,
        "goal": "Build auth service",
        "requirements": [
            RequirementItem(
                id="REQ-1",
                description="Support login token issue",
                acceptance_criteria=["POST /login returns JWT"],
            )
        ],
        "out_of_scope": ["OAuth social login"],
        "constraints": ["Python 3.13"],
    }
    data.update(overrides)
    return GoalSpec(**data)


_LEGACY_GOAL_SPEC_JSON = """{
  "spec_id": "spec-legacy",
  "version": 1,
  "goal": "Build auth service",
  "requirements": [
    {
      "id": "R1",
      "description": "Support login token issue",
      "acceptance_criteria": ["POST /login returns JWT"]
    }
  ],
  "out_of_scope": ["OAuth social login"],
  "constraints": ["Python 3.13"]
}"""

_LEGACY_GOAL_SPEC_HASH = "sha256:2640ccd1bb0c4b8b5312297df31a2ce01ea922df5c85358f5e41760498eee678"


def test_goal_spec_hash_ignores_status_and_detects_content_changes() -> None:
    draft = _sample_spec(status=SpecStatus.DRAFT)
    confirmed = _sample_spec(status=SpecStatus.CONFIRMED)
    assert draft.content_hash.startswith("sha256:")
    assert draft.content_hash == confirmed.content_hash

    edited = _sample_spec(out_of_scope=["OAuth social login", "SMS OTP"])
    assert edited.content_hash != draft.content_hash


def test_legacy_goal_spec_json_without_test_seams_keeps_its_content_hash() -> None:
    spec = GoalSpec.model_validate_json(_LEGACY_GOAL_SPEC_JSON)
    assert spec.test_seams == []
    assert spec.content_hash == _LEGACY_GOAL_SPEC_HASH


def test_goal_spec_test_seams_fingerprint_content_and_render_markdown() -> None:
    legacy = GoalSpec.model_validate_json(_LEGACY_GOAL_SPEC_JSON)
    seamed = legacy.model_copy(update={"test_seams": ["POST /login", "GET /health"]})

    assert seamed.content_hash != legacy.content_hash
    assert "## Test seams" not in legacy.to_markdown()
    markdown = seamed.to_markdown()
    assert "## Test seams" in markdown
    assert "- POST /login" in markdown
    assert "- GET /health" in markdown


def test_review_verdict_blocks_passing_with_important_finding() -> None:
    with pytest.raises(ValidationError):
        ReviewVerdict(
            task_id="T1",
            attempt=1,
            passed=True,
            summary="Looks fine",
            findings=[
                ReviewFinding(
                    id="F1",
                    severity=ReviewSeverity.IMPORTANT,
                    file_path="src/app.py",
                    line=10,
                    message="SQL injection risk",
                )
            ],
        )

    ok = ReviewVerdict(
        task_id="T1",
        attempt=1,
        passed=True,
        summary="Only minor style nit",
        findings=[
            ReviewFinding(
                id="F2",
                severity=ReviewSeverity.NIT,
                file_path="src/app.py",
                line=4,
                message="Rename variable",
            )
        ],
    )
    assert ok.passed is True


def test_acceptance_check_requires_command_when_kind_is_command() -> None:
    with pytest.raises(ValidationError):
        AcceptanceCheck(id="C1", kind=CheckKind.COMMAND, description="Run tests")


def test_approval_accept_risk_requires_non_empty_reason() -> None:
    with pytest.raises(ValidationError):
        ApprovalRecord(
            run_id="run-1",
            target=ApprovalTarget.FINAL_ACCEPTANCE,
            target_id="plan-1",
            target_hash="sha256:abc",
            approver="human",
            decision=ApprovalDecision.ACCEPT_RISK,
            reason="   ",
        )


def test_budget_ledger_is_monotonic_and_enforces_limit() -> None:
    policy = BudgetPolicy(max_llm_calls=2, max_total_tokens=1000)
    ledger = BudgetLedger()

    ledger = ledger.reserve(policy)
    assert ledger.llm_calls_reserved == 1
    ledger = ledger.settle(input_tokens=300, output_tokens=200, unknown=True)
    assert ledger.llm_calls_used == 1
    assert ledger.llm_calls_reserved == 0
    assert ledger.unknown_outcome_calls == 1
    assert ledger.total_tokens == 500

    ledger = ledger.reserve(policy)
    assert not ledger.can_reserve(policy)
    with pytest.raises(ValueError):
        ledger.reserve(policy)


def test_budget_ledger_accounts_cache_tokens_and_reads_legacy_json() -> None:
    legacy = BudgetLedger.model_validate_json('{"input_tokens": 300, "output_tokens": 200}')
    assert (legacy.cache_read_tokens, legacy.cache_write_tokens) == (0, 0)
    assert legacy.total_tokens == 500

    settled = legacy.reserve(BudgetPolicy(max_total_tokens=2000)).settle(
        input_tokens=100,
        output_tokens=50,
        cache_read_tokens=250,
        cache_write_tokens=150,
    )
    assert settled.total_tokens == 1050
    assert (settled.cache_read_tokens, settled.cache_write_tokens) == (250, 150)
    # Cached tokens count against the true total even when input/output stay small.
    assert settled.can_reserve(BudgetPolicy(max_total_tokens=1050)) is False
    assert settled.can_reserve(BudgetPolicy(max_total_tokens=1051)) is True
