"""Strong-typed domain contracts for Stage 1 (Align), Stage 2 (Orchestrate), and audit state."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Self
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from olympus.core.fingerprint import content_hash


def utc_now() -> datetime:
    return datetime.now(UTC)


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex[:12]}"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FingerprintedModel(StrictModel):
    """Base model whose content_hash stays consistent across validation and model_copy."""

    content_hash: str = ""

    def _hash_payload(self) -> dict[str, Any]:
        raise NotImplementedError

    def _refresh_hash(self) -> Self:
        object.__setattr__(self, "content_hash", content_hash(self._hash_payload()))
        return self

    @model_validator(mode="after")
    def _compute_hash_after_validate(self) -> Self:
        return self._refresh_hash()

    def model_copy(
        self,
        *,
        update: Mapping[str, Any] | None = None,
        deep: bool = False,
    ) -> Self:
        copied = super().model_copy(update=update, deep=deep)
        return copied._refresh_hash()


class SpecStatus(StrEnum):
    DRAFT = "draft"
    AWAITING_HUMAN = "awaiting_human"
    CONFIRMED = "confirmed"
    SUPERSEDED = "superseded"


class CritiquePerspective(StrEnum):
    PRODUCT = "product"
    ENGINEERING = "engineering"
    QA = "qa"


class CritiqueSeverity(StrEnum):
    BLOCKING = "blocking"
    WARNING = "warning"


class RequirementItem(StrictModel):
    id: str
    description: str
    acceptance_criteria: list[str] = Field(min_length=1)
    source_ref: str | None = None


class AmbiguityItem(StrictModel):
    id: str
    question: str
    impact: str
    options: list[str] = Field(default_factory=list)
    default_assumption: str | None = None
    blocking: bool = True
    answer: str | None = None


class CritiqueNote(StrictModel):
    id: str
    perspective: CritiquePerspective
    severity: CritiqueSeverity
    summary: str
    evidence_refs: list[str] = Field(default_factory=list)
    resolved: bool = False
    resolution_note: str | None = None


class GoalSpec(FingerprintedModel):
    spec_id: str = Field(default_factory=lambda: new_id("spec"))
    version: int = Field(default=1, ge=1)
    goal: str = Field(min_length=1)
    requirements: list[RequirementItem] = Field(min_length=1)
    out_of_scope: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    open_ambiguities: list[AmbiguityItem] = Field(default_factory=list)
    minority_notes: list[CritiqueNote] = Field(default_factory=list)
    status: SpecStatus = SpecStatus.DRAFT

    def _hash_payload(self) -> dict[str, Any]:
        return self.model_dump(exclude={"status", "content_hash"}, mode="python")

    def unresolved_blockers(self) -> list[str]:
        blockers: list[str] = []
        for amb in self.open_ambiguities:
            if amb.blocking and not amb.answer:
                blockers.append(f"ambiguity:{amb.id}")
        for note in self.minority_notes:
            if note.severity == CritiqueSeverity.BLOCKING and not note.resolved:
                blockers.append(f"critique:{note.id}")
        return blockers


class TaskStatus(StrEnum):
    WAITING = "waiting"
    READY = "ready"
    RUNNING = "running"
    VERIFYING = "verifying"
    ACCEPTED = "accepted"
    FAILED = "failed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"
    STALE = "stale"


class CheckKind(StrEnum):
    COMMAND = "command"
    INDEPENDENT_REVIEW = "independent_review"
    HUMAN = "human"


class AcceptanceCheck(StrictModel):
    id: str
    kind: CheckKind
    description: str = Field(min_length=1)
    command: str | None = None
    passed: bool = False
    evidence_ref: str | None = None

    @model_validator(mode="after")
    def _validate_command_check(self) -> AcceptanceCheck:
        if self.kind == CheckKind.COMMAND and not self.command:
            raise ValueError("command check requires non-empty command")
        return self


class ReviewSeverity(StrEnum):
    IMPORTANT = "important"
    NIT = "nit"
    PRE_EXISTING = "pre_existing"


class ReviewFinding(StrictModel):
    id: str
    severity: ReviewSeverity
    file_path: str
    line: int | None = Field(default=None, ge=1)
    message: str = Field(min_length=1)
    suggestion: str | None = None


class ReviewVerdict(StrictModel):
    verdict_id: str = Field(default_factory=lambda: new_id("verdict"))
    task_id: str
    attempt: int = Field(ge=1)
    passed: bool
    summary: str = Field(min_length=1)
    findings: list[ReviewFinding] = Field(default_factory=list)
    checked_commit: str | None = None
    incremental_only: bool = False

    @model_validator(mode="after")
    def _enforce_important_blocker(self) -> ReviewVerdict:
        has_important = any(f.severity == ReviewSeverity.IMPORTANT for f in self.findings)
        if self.passed and has_important:
            raise ValueError("ReviewVerdict cannot pass while Important findings remain")
        return self


class TaskNode(StrictModel):
    task_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    objective: str = Field(min_length=1)
    requirement_refs: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)
    write_scope: list[str] = Field(min_length=1)
    read_scope: list[str] = Field(default_factory=list)
    acceptance_checks: list[AcceptanceCheck] = Field(min_length=1)
    max_cycles: int = Field(default=3, ge=1, le=5)
    status: TaskStatus = TaskStatus.WAITING
    attempt: int = Field(default=0, ge=0)
    review_cycle: int = Field(default=0, ge=0)
    lease_generation: int = Field(default=0, ge=0)
    worktree_path: str | None = None
    commit_sha: str | None = None
    last_error: str | None = None
    accepted_risk_reason: str | None = None


class TaskPlan(FingerprintedModel):
    plan_id: str = Field(default_factory=lambda: new_id("plan"))
    spec_id: str
    spec_hash: str
    version: int = Field(default=1, ge=1)
    tasks: list[TaskNode] = Field(min_length=1)

    def _hash_payload(self) -> dict[str, Any]:
        # Hash structural plan definition (not mutable runtime execution counters)
        structural_tasks = [
            t.model_dump(
                include={
                    "task_id",
                    "title",
                    "objective",
                    "requirement_refs",
                    "depends_on",
                    "write_scope",
                    "read_scope",
                    "max_cycles",
                },
                mode="python",
            )
            | {
                "acceptance_checks": [
                    c.model_dump(include={"id", "kind", "description", "command"}, mode="python")
                    for c in t.acceptance_checks
                ]
            }
            for t in self.tasks
        ]
        return {
            "plan_id": self.plan_id,
            "spec_id": self.spec_id,
            "spec_hash": self.spec_hash,
            "version": self.version,
            "tasks": structural_tasks,
        }


class RunMode(StrEnum):
    SIMULATION = "simulation"
    EXECUTION = "execution"


class RunStatus(StrEnum):
    CREATED = "created"
    ALIGNING = "aligning"
    AWAITING_HUMAN = "awaiting_human"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class BudgetPolicy(StrictModel):
    max_llm_calls: int = Field(default=100, ge=1)
    max_total_tokens: int = Field(default=2_000_000, ge=1)
    max_concurrent_tasks: int = Field(default=3, ge=1)


class BudgetLedger(StrictModel):
    llm_calls_used: int = Field(default=0, ge=0)
    llm_calls_reserved: int = Field(default=0, ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    unknown_outcome_calls: int = Field(default=0, ge=0)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def can_reserve(self, policy: BudgetPolicy) -> bool:
        return (
            self.llm_calls_used + self.llm_calls_reserved < policy.max_llm_calls
            and self.total_tokens < policy.max_total_tokens
        )

    def reserve(self, policy: BudgetPolicy) -> BudgetLedger:
        if not self.can_reserve(policy):
            raise ValueError("Budget exhausted: cannot reserve additional LLM call")
        return self.model_copy(update={"llm_calls_reserved": self.llm_calls_reserved + 1})

    def settle(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
        *,
        unknown: bool = False,
    ) -> BudgetLedger:
        if self.llm_calls_reserved <= 0:
            raise ValueError("No reserved LLM call to settle")
        return self.model_copy(
            update={
                "llm_calls_reserved": self.llm_calls_reserved - 1,
                "llm_calls_used": self.llm_calls_used + 1,
                "input_tokens": self.input_tokens + max(0, input_tokens),
                "output_tokens": self.output_tokens + max(0, output_tokens),
                "unknown_outcome_calls": self.unknown_outcome_calls + (1 if unknown else 0),
            }
        )


class ApprovalTarget(StrEnum):
    GOAL_SPEC = "goal_spec"
    TASK_PLAN = "task_plan"
    HIGH_RISK_ACTION = "high_risk_action"
    FINAL_ACCEPTANCE = "final_acceptance"
    RISK_ACCEPTANCE = "risk_acceptance"


class ApprovalDecision(StrEnum):
    APPROVED = "approved"
    REJECTED = "rejected"
    ACCEPT_RISK = "accept_risk"


class ApprovalRecord(StrictModel):
    approval_id: str = Field(default_factory=lambda: new_id("appr"))
    run_id: str
    target: ApprovalTarget
    target_id: str
    target_hash: str
    commit_sha: str | None = None
    approver: str = Field(min_length=1)
    decision: ApprovalDecision
    reason: str = ""
    created_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def _validate_risk_reason(self) -> ApprovalRecord:
        if self.decision == ApprovalDecision.ACCEPT_RISK and not self.reason.strip():
            raise ValueError("ACCEPT_RISK requires an explicit non-empty reason")
        return self


class RunEvent(StrictModel):
    event_seq: int = Field(ge=1)
    event_id: str = Field(default_factory=lambda: new_id("evt"))
    run_id: str
    stage: str
    event_type: str
    actor: str
    payload: dict[str, Any] = Field(default_factory=dict)
    state_revision: int = Field(ge=0)
    created_at: datetime = Field(default_factory=utc_now)
