"""Free-form human reply routing bounded by deterministic ``RunControl`` actions.

The router is deliberately two-layered:

1. A lightweight LLM pass translates prose into one structured intent.
2. Deterministic sanitisation filters every answer/resolution against the run's
   live artifacts, so a hallucinated or malformed reply can never touch the store.
"""

from __future__ import annotations

import json
import re
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


_NEGATED_APPROVAL_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        r"\b(?:do\s+not|don'?t|cannot|can'?t|won'?t|will\s+not|shouldn'?t|should\s+not|wouldn'?t|never)\b[^.!?\n]{0,40}\b(?:proceed|approve|approved|go\s+ahead|continue|ship|merge)\b",
        r"\bnot\s+(?:ready|approv(?:e|ed|al)|good\s+to\s+go|ok(?:ay)?|yet)\b",
        r"\b(?:not|never)\s+approv(?:e|ed|al)\b",
        r"\bhold\s+off\b",
        r"\bno\s*[,;:]",
        r"^\s*no\b",
        r"\breject(?:s|ed|ing)?\b",
        r"\bcancel(?:s|led|ing)?\b",
    )
)
_NEGATED_APPROVAL_CJK = ("不要", "不批准", "不同意", "拒绝", "取消", "暂不", "先不")
# ``别`` negates an instruction, but is also a morpheme in 特别/分别/区别/个别 ….
_NEGATED_APPROVAL_CJK_BIE = re.compile(r"(?<![特分区个性差级告])别")
_ACTIONABLE_CHARS = re.compile(r"[0-9A-Za-z\u3400-\u9fff]")


def _has_actionable_text(text: str) -> bool:
    return len(_ACTIONABLE_CHARS.findall(text)) >= 3


def _negated_approval(text: str) -> bool:
    if any(pattern.search(text) for pattern in _NEGATED_APPROVAL_PATTERNS):
        return True
    if any(token in text for token in _NEGATED_APPROVAL_CJK):
        return True
    return _NEGATED_APPROVAL_CJK_BIE.search(text) is not None


def _keyword_fallback(message: str) -> ReplyIntent:
    """Conservative fallback used only when the structured parse fails."""
    text = message.casefold().strip()
    if not text:
        return ReplyIntent(action="unknown", message="Empty reply")
    revise = any(token in text for token in _REVISE_TOKENS)
    if _negated_approval(text):
        # A negated approval is never an approval: treat an explicit revision ask
        # as a revision and everything else as ambiguous, so no gate is confirmed.
        if revise:
            return ReplyIntent(
                action="revise",
                feedback=message.strip(),
                message="Keyword fallback: negated approval, treated as revision",
            )
        return ReplyIntent(
            action="unknown",
            message="Keyword fallback: negated approval needs an explicit revision request",
        )
    if revise:
        return ReplyIntent(
            action="revise", feedback=message.strip(), message="Keyword fallback: revision"
        )
    if any(token in text for token in _APPROVE_TOKENS):
        return ReplyIntent(action="approve", message="Keyword fallback: approval")
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
    if action == "approve" and _negated_approval(message.casefold()):
        revise = any(token in message.casefold() for token in _REVISE_TOKENS)
        action = "revise" if revise else "unknown"
        feedback = message.strip() if revise else ""
    if action in ("reopen", "revise") and not _has_actionable_text(message):
        action = "unknown"
        feedback = ""
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
    response = provider.complete(
        messages=[
            ChatMessage(role="system", content=_ROUTER_SYSTEM_PROMPT),
            ChatMessage(role="user", content=user_prompt),
        ]
    )
    try:
        parsed = _RawIntent.model_validate_json(_extract_json(response.content))
    except Exception:  # noqa: BLE001 -- malformed router JSON falls back to keyword guard
        return _keyword_fallback(message)
    intent = _sanitize(parsed, snapshot=snapshot, message=message)
    if intent.action == "unknown":
        return _keyword_fallback(message)
    return intent


def _extract_json(content: str) -> str:
    from olympus.harness.loop import extract_json_payload

    return extract_json_payload(content)


__all__ = ["ReplyIntent", "parse_reply"]
