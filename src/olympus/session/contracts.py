"""Stable headless session envelopes shared by ``olympus session start/reply``."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from olympus.core.contracts import StrictModel


class SessionStatus(StrEnum):
    """Terminal-ish state a headless caller observes after one invocation."""

    NEEDS_INPUT = "needs_input"
    COMPLETED = "completed"
    FAILED = "failed"


class PauseKind(StrEnum):
    """Why the session yielded control back to the external caller."""

    CLARIFY = "clarify"
    APPROVAL = "approval"
    BLOCKED = "blocked"
    PLAN_HANDOFF = "plan_handoff"


class StepMode(StrEnum):
    """How far one invocation may drive the workflow."""

    PLAN_ONLY = "plan_only"
    EXECUTE = "execute"


class GatePolicyName(StrEnum):
    """Gate austerity modes exposed by the session CLI."""

    INTERACTIVE_MINIMAL = "interactive_minimal"
    DEFAULT = "default"


class ActorTokenUsage(StrictModel):
    """Token totals attributed to a single actor (e.g. ``align:proposer``)."""

    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    total_tokens: int = 0


class TokenUsageSummary(StrictModel):
    """Run-wide token, cache, and reasoning accounting for the session."""

    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    total_tokens: int = 0
    cache_hit_rate: float = 0.0
    by_actor: dict[str, ActorTokenUsage] = Field(default_factory=dict)


class SessionEnvelope(StrictModel):
    """The single machine-readable contract every session invocation prints."""

    run_id: str
    status: SessionStatus
    pause_kind: PauseKind | None = None
    step_mode: StepMode = StepMode.EXECUTE
    workflow: str = ""
    message: str = ""
    unresolved_blockers: list[str] = Field(default_factory=list)
    changed_files: list[str] = Field(default_factory=list)
    error: str | None = None
    token_usage: TokenUsageSummary = Field(default_factory=TokenUsageSummary)

    @classmethod
    def failure(
        cls,
        *,
        error: str,
        run_id: str = "",
        step_mode: StepMode = StepMode.EXECUTE,
        workflow: str = "",
        message: str | None = None,
        token_usage: TokenUsageSummary | None = None,
    ) -> SessionEnvelope:
        return cls(
            run_id=run_id,
            status=SessionStatus.FAILED,
            pause_kind=None,
            step_mode=step_mode,
            workflow=workflow,
            message=message or error,
            unresolved_blockers=[],
            changed_files=[],
            error=error,
            token_usage=token_usage or TokenUsageSummary(),
        )


__all__ = [
    "ActorTokenUsage",
    "GatePolicyName",
    "PauseKind",
    "SessionEnvelope",
    "SessionStatus",
    "StepMode",
    "TokenUsageSummary",
]
