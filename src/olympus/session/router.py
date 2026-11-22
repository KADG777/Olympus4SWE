"""Free-form human reply routing bounded by deterministic ``RunControl`` actions.

The router is deliberately two-layered:

1. A lightweight LLM pass translates prose into one structured intent.
2. Deterministic sanitisation filters every answer/resolution against the run's
   live artifacts, so a hallucinated or malformed reply can never touch the store.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import ConfigDict

from olympus.core.contracts import CritiqueSeverity, GoalSpec, StrictModel, TaskPlan
from olympus.providers.base import ChatMessage, LLMProvider
from olympus.session.contracts import PauseKind

ReplyAction = Literal["clarify", "approve", "revise", "reopen", "unknown"]

_ROUTER_SYSTEM_PROMPT = """You are the Olympus4SWE Session Reply Router.
Translate one free-form human reply into exactly one routing intent. Never invent
requirements, never answer ambiguities the human did not address, and never mutate state.

Return ONLY one JSON object:
{
  "action": "clarify | approve | revise | reopen | unknown",
  "answers": {"<AMBIGUITY_ID>": "verbatim answer from the human"},
  "resolutions": {"<CRITIQUE_ID>": "how the concern is resolved"},
  "feedback": "revision or follow-up instruction, when the human wants changes",
  "message": "one concise sentence describing the routing decision"
}

Routing rules:
- action=clarify only when the human answers one or more listed ambiguity/critique IDs.
- action=approve only when the human explicitly approves the pending artifact/gate.
- action=revise when the human rejects or asks to change the current plan/spec.
- action=reopen when the run is already completed and the human asks for follow-up work.
- action=unknown when the reply is ambiguous; do not guess.
- Use only the IDs listed in the context; omit answers for anything not addressed."""


class ReplyIntent(StrictModel):
    """A deterministic, sanitised routing decision."""

    model_config = ConfigDict(extra="ignore")

    action: ReplyAction = "unknown"
    answers: dict[str, str] = {}
    resolutions: dict[str, str] = {}
    feedback: str = ""
    message: str = ""


class _RawIntent(ReplyIntent):
    model_config = ConfigDict(extra="ignore")


_APPROVE_TOKENS = (
    "approve",
    "approved",
    "lgtm",
    "looks good",
    "go ahead",
    "proceed",
    "ship it",
    "同意",
    "批准",
    "通过",
    "没问题",
    "继续",
)
_REVISE_TOKENS = (
    "revise",
    "rework",
    "change",
    "instead",
    "reject",
    "not right",
    "redo",
    "修改",
    "调整",
    "重做",
    "不对",
    "改一下",
)


def _blocking_ambiguity_ids(spec: GoalSpec | None) -> set[str]:
    if spec is None:
        return set()
    return {amb.id for amb in spec.open_ambiguities if amb.blocking and not amb.answer}


def _blocking_critique_ids(spec: GoalSpec | None) -> set[str]:
    if spec is None:
        return set()
    return {
        note.id
        for note in spec.minority_notes
        if note.severity == CritiqueSeverity.BLOCKING and not note.resolved
    }


def _reply_context(
    *,
    message: str,
    snapshot: dict[str, Any],
    pause_kind: PauseKind | None,
    gate: str | None,
    completed: bool,
) -> str:
    spec: GoalSpec | None = snapshot.get("spec")
    plan: TaskPlan | None = snapshot.get("plan")
    parts: list[str] = [
        f"Run status: {snapshot['status'].value}",
        f"Pause kind: {pause_kind.value if pause_kind else 'none'}",
        f"Pending gate: {gate or 'none'}",
        f"Run already completed: {str(completed).lower()}",
    ]
    if spec is not None:
        parts.append(f"GoalSpec goal: {spec.goal}")
        ambiguity_lines: list[str] = []
        for amb in spec.open_ambiguities:
            if not amb.blocking or amb.answer:
                continue
            options = (
                f" options={json.dumps(amb.options, ensure_ascii=False)}" if amb.options else ""
            )
            ambiguity_lines.append(f"- {amb.id}: {amb.question}{options}")
        if ambiguity_lines:
            parts.append("Open blocking ambiguities:\n" + "\n".join(ambiguity_lines))
        critique_lines = [
            f"- {note.id}: {note.summary}"
            for note in spec.minority_notes
            if note.severity == CritiqueSeverity.BLOCKING and not note.resolved
        ]
        if critique_lines:
            parts.append("Unresolved blocking critiques:\n" + "\n".join(critique_lines))
    if plan is not None:
        tasks = "; ".join(f"{task.task_id}={task.title}" for task in plan.tasks)
        parts.append(f"Current TaskPlan tasks: {tasks or '(none)'}")
    parts.append(f"Human reply:\n{message}")
    return "\n".join(parts)


def _keyword_fallback(message: str) -> ReplyIntent:
    """Conservative fallback used only when the structured parse fails."""
    text = message.casefold().strip()
    if not text:
        return ReplyIntent(action="unknown", message="Empty reply")
    negative = any(
        token in text
        for token in ("not approved", "don't approve", "do not approve", "拒绝")
    )
    revise = any(token in text for token in _REVISE_TOKENS) or negative
    approve = any(token in text for token in _APPROVE_TOKENS) and not revise
    if approve:
        return ReplyIntent(action="approve", message="Keyword fallback: approval")
    if revise:
        return ReplyIntent(
            action="revise", feedback=message.strip(), message="Keyword fallback: revision"
        )
    return ReplyIntent(action="unknown", message="Could not parse the reply safely")


def _sanitize(
    raw: ReplyIntent,
    *,
    snapshot: dict[str, Any],
    message: str,
) -> ReplyIntent:
    spec: GoalSpec | None = snapshot.get("spec")
    valid_answers = _blocking_ambiguity_ids(spec)
    valid_resolutions = _blocking_critique_ids(spec)
    answers = {
        key.strip(): value.strip()
        for key, value in raw.answers.items()
        if key.strip() in valid_answers and value.strip()
    }
    resolutions = {
        key.strip(): value.strip()
        for key, value in raw.resolutions.items()
        if key.strip() in valid_resolutions and value.strip()
    }
    action = raw.action
    feedback = raw.feedback.strip()
    if answers or resolutions:
        # Human answers always win: never approve/revise while silently dropping them.
        action = "clarify"
    elif action == "clarify":
        action = "unknown"
    if action == "revise" and not feedback:
        feedback = message.strip()
        if not feedback:
            action = "unknown"
    return ReplyIntent(
        action=action,
        answers=answers,
        resolutions=resolutions,
        feedback=feedback,
        message=raw.message.strip(),
    )


def parse_reply(
    *,
    provider: LLMProvider,
    message: str,
    snapshot: dict[str, Any],
    pause_kind: PauseKind | None,
    gate: str | None = None,
    completed: bool = False,
) -> ReplyIntent:
    """Translate prose into a sanitised intent; malformed output degrades safely."""
    user_prompt = _reply_context(
        message=message,
        snapshot=snapshot,
        pause_kind=pause_kind,
        gate=gate,
        completed=completed,
    )
    try:
        response = provider.complete(
            messages=[
                ChatMessage(role="system", content=_ROUTER_SYSTEM_PROMPT),
                ChatMessage(role="user", content=user_prompt),
            ]
        )
        parsed = _RawIntent.model_validate_json(_extract_json(response.content))
        intent = _sanitize(parsed, snapshot=snapshot, message=message)
        if intent.action == "unknown":
            return _keyword_fallback(message)
        return intent
    except Exception:  # noqa: BLE001 - a malformed reply must never corrupt state
        return _keyword_fallback(message)


def _extract_json(content: str) -> str:
    from olympus.harness.loop import extract_json_payload

    return extract_json_payload(content)


__all__ = ["ReplyIntent", "parse_reply"]
