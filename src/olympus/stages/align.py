"""Stage 1 (Align): Evidence-backed spec synthesis, optional critique, and human freeze."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pydantic import Field

from olympus.core.contracts import (
    CritiqueNote,
    CritiquePerspective,
    GoalSpec,
    RunEvent,
    RunStatus,
    SpecStatus,
    StrictModel,
)
from olympus.core.control import RunControl
from olympus.core.dag import validate_goal_spec
from olympus.core.store import ConcurrencyConflictError, RunStore
from olympus.harness.driver import DriverRegistry, ExecutionDriversConfig
from olympus.harness.loop import AgentLoop
from olympus.harness.structured import run_structured_turn
from olympus.providers.base import LLMProvider
from olympus.stages.common import format_workspace_inventory, resolve_stage_driver
from olympus.tools.aci import WorkspaceToolkit


class CritiqueOutput(StrictModel):
    notes: list[CritiqueNote] = Field(default_factory=list)


def _scope_directive(
    *,
    workspace_mode: str | None,
    task_type: str | None,
    base_ref: str | None,
    scope_paths: Sequence[str] | None,
) -> str:
    """Render the deterministic Scope x Type contract injected into the Stage 1 prompt."""
    mode = (workspace_mode or "").strip().lower()
    kind = (task_type or "").strip().lower()
    blocks: list[str] = []
    if mode == "greenfield":
        blocks.append(
            "GREENFIELD contract (explicit, authoritative):\n"
            "- The workspace is a brand-new empty directory; there is NO existing codebase.\n"
            "- Never scan, cite, or treat unrelated host repository files as context,\n"
            "  contradictions, or constraints for this goal; synthesize the standalone\n"
            "  GoalSpec directly from Raw user goal."
        )
    elif mode == "brownfield":
        binding = ["Brownfield binding:"]
        if base_ref:
            binding.append(f"- Base git ref: {base_ref}")
        if scope_paths:
            binding.append(
                "- Scoped paths (authoritative, workspace-relative): "
                + ", ".join(scope_paths)
            )
        if len(binding) > 1:
            blocks.append("\n".join(binding))
    if kind == "fix":
        blocks.append(
            "Fix task directive: prioritize reproducible defect behavior, regression "
            "constraints, and a minimal verified patch over broad architectural ceremony."
        )
    elif kind == "refactor":
        blocks.append(
            "Refactor task directive: preserve externally observable behavior and require "
            "regression evidence for every refactored seam."
        )
    if not blocks:
        return ""
    return "\n\n".join(blocks) + "\n\n"


_ALIGN_SYSTEM_PROMPT = """You are the Stage 1 Requirements Alignment Architect of Olympus4SWE.
Inspect the observed workspace context (and use read-only tools when deeper inspection is helpful)
to convert the user's raw goal into a crisp, verifiable GoalSpec JSON object.

Rules:
1. Code-First Inspection (Grill With Docs): For brownfield goals, inspect the relevant repository
   modules, entrypoints, and test layout with read-only tools BEFORE asking the human anything.
   Never ask what the codebase already answers; a blocking ambiguity must survive an honest read
   of the existing code and observed tests.
2. Strict Path Grounding: Cite ONLY file paths listed in Observed workspace files or observed
   via tool output. NEVER invent or mention unverified paths (e.g., `src/.../main.py` or
   `tests/...`) inside `constraints`, `assumptions`, `open_ambiguities`, or `minority_notes`.
   When a task is standalone or target layout is unspecified, refer to behavior/deliverables
   conceptually rather than guessing repository paths.
3. Semantic Audit & Genuine Blockers: Inspect the goal (and existing code via read-only tools)
   across each distinct behavioral dimension — contradictions with the repo, missing high-level
   scope/success criteria (vague goals MUST stay blocked via `open_ambiguities`), undefined value
   domains, empty/boundary/error/status contracts, operation ordering/tie-breakers/idempotency,
   schema/output format/precision/serialization, optimization targets, and persistence/compatibility
   defaults — so no correctness-critical choice is silently guessed. Only genuine, irreversible
   behavioral decisions may be `blocking`; repo-answerable details (paths, layout, language
   version, framework, existing conventions) are never blocking because the code already decides
   them.
4. Question Austerity (NO Shotgun Questioning): Treat clarification budget as precious.
   - If requirements are already clear and verifiable, return `"open_ambiguities": []`.
   - Restrict `blocking: true` ambiguities to 0-3 genuine irreversible behavioral choices; keep
     independent behavioral dimensions separate and merge only truly overlapping points.
   - Every blocking ambiguity MUST carry a concrete, code-grounded `default_assumption` (your
     recommended answer) and, when meaningful, `options`, so a human can accept your defaults
     deterministically instead of restating each answer.
   - NEVER pad with generic boilerplate questions about file paths/layout, language version,
     CLI vs function signature, logging, test framework, or validation of inputs already ruled
     out by stated constraints.
5. Term Collisions & Contradictions: When existing repository behavior contradicts the requested
   change, or a user term collides with an existing domain concept, surface the mismatch
   explicitly inside `assumptions` or `constraints` (quoting only observed paths) instead of
   silently redefining it. Never mutate repository files while aligning.
6. Bounded Scope & Falsifiable Criteria: List explicit `out_of_scope` items to prevent scope
   creep, and give every requirement concrete, verifiable `acceptance_criteria`.
7. Exploration Discipline: Keep read-only inspection focused (typically 2-6 calls on public
   API/module layout and test layout) before returning GoalSpec JSON; leave deep implementation
   reading to Stage 2.
8. Output ONLY a single valid JSON object conforming to GoalSpec. Never emit `$comment`,
   `$schema`, or unlisted extra keys.
9. Greenfield vs. Brownfield: First check whether `Raw user goal` asks to modify the observed
   workspace repository or to build a brand-new standalone product unrelated to the existing
   workspace files (e.g., building a consumer app while the workspace holds an unrelated
   framework). If it is a new standalone product unrelated to the workspace files, treat it as
   GREENFIELD immediately: do NOT call tools to inspect unrelated host repo files or treat
   host files as contradictions; synthesize the standalone GoalSpec directly.
10. Language Mirroring & Conciseness: ALWAYS write all human-facing prose (`goal`, `description`,
   `acceptance_criteria`, `out_of_scope`, `constraints`, `question`, `impact`, `options`,
   `default_assumption`) in the exact same natural language as `Raw user goal` (never switch
   languages; keep IDs and enum values in ASCII). Keep each requirement and criterion crisp and
   concise (1-2 sentences each, not multi-paragraph essays).
11. Assumption Provenance: For assumptions you introduce that materially affect scope or acceptance,
   copy the exact verbatim string from `assumptions` into `agent_high_impact_assumptions` (or leave
   `agent_high_impact_assumptions` as `[]`). Do not mark user decisions or confirmed clarifications
   as Agent assumptions. Ask about genuine blockers.
12. Seam-First Test Boundaries: Populate `test_seams` with the fewest, highest-level public
   boundaries where behavior is verified end to end (HTTP routes, CLI commands, public module
   APIs) — ideal 1-2. Each seam must be exercisable by an external caller in one step; never name
   brittle internal file paths, private helpers, or test-framework internals.
13. Fidelity to Explicit User Constraints: Every concrete domain entity, fixed category/taxonomy,
   user flow, or UX boundary stated in `Raw user goal` or `Explicit constraints` MUST appear
   verbatim as an explicit requirement or criterion. Never downgrade a stated category or boundary
   into a loose example (such as "e.g., #topic1 #topic2").
14. MVP Restraint for Short Greenfield Goals: For a concise greenfield idea, model only the core
   MVP capabilities stated or directly required by the goal (typically 3-5 focused requirements).
   Do NOT invent unrequested subsystems (follow graphs, custom math ranking formulas, moderation
   queues, privacy ACLs, or analytics dashboards); record plausible future expansions under
   `out_of_scope` instead."""

_ALIGN_RESYNTHESIZE_SYSTEM_PROMPT = """You are the Stage 1 Requirements Alignment Architect.
The human has answered all blocking ambiguities and may have added explicit product boundaries or
constraints. Fuse EVERY human clarification and additional instruction into the existing GoalSpec
body so downstream Design and Planning execute the human's exact decisions.

Rules:
1. Authoritative Human Decisions: Every resolved ambiguity answer, critique resolution, and line in
   `Additional human instructions` overrides earlier draft assumptions or loose examples.
2. Body-Level Fusion: Update `goal`, `requirements`, `ears_criteria`, `acceptance_criteria`,
   `constraints`, `assumptions`, `out_of_scope`, `actors`, `observable_surfaces`, and `test_seams`
   so every confirmed requirement, fixed category/taxonomy, auth/storage rule, or UX boundary is
   explicitly stated in the GoalSpec body (never leave human decisions stranded only in
   `open_ambiguities`).
3. Remove Superseded Assumptions: Delete any draft assumption contradicted by a human answer or
   instruction, and do NOT place confirmed human decisions into `agent_high_impact_assumptions`.
4. MVP Restraint & Preservation: Keep the spec tight and faithful to what the human asked for; do
   not invent unrequested subsystems, and do not add new unresolved blocking ambiguities.
5. Language Mirroring & Schema Compliance: Keep the exact natural language of the existing GoalSpec
   (ASCII IDs/enums). Output ONLY one valid JSON object matching GoalSpec with no extra keys."""

_CRITIQUE_SYSTEM_TEMPLATE = """You are a cold-start read-only {perspective} reviewer.
Inspect the draft GoalSpec (and repository files if needed) and report ONLY real omissions,
hidden contradictions, feasibility risks, or untestable acceptance criteria.
Do NOT invent scores or rewrite the spec. If no issues exist, return {{"notes": []}}.
Output ONLY valid JSON matching {{"notes": [...]}} with no `$comment`, `$schema`, or extra keys."""

_MAX_BLOCKING_AMBIGUITIES = 3
_LIST_PREFIX_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)、])\s*")
_ASCII_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_-]{1,}")
_CJK_SEGMENT_RE = re.compile(r"[\u3400-\u9fff]{2,}")
_STOPWORDS = frozenset(
    {
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "from",
        "have",
        "must",
        "should",
        "will",
        "use",
        "using",
        "also",
        "only",
        "not",
        "all",
        "are",
        "was",
        "were",
        "into",
        "onto",
        "about",
        "please",
        "note",
        "notes",
        "rule",
        "rules",
        "补充",
        "说明",
        "注意",
        "边界",
        "核心",
        "如下",
        "要求",
        "支持",
        "包含",
        "必须",
        "需要",
        "以及",
        "并且",
        "或者",
        "默认",
    }
)


def _norm_clause(text: str | None) -> str:
    return " ".join((text or "").casefold().split())


def _distinctive_tokens(line: str) -> set[str]:
    cleaned = _LIST_PREFIX_RE.sub("", line).casefold()
    tokens = {tok for tok in _ASCII_TOKEN_RE.findall(cleaned) if tok not in _STOPWORDS}
    for segment in _CJK_SEGMENT_RE.findall(cleaned):
        if len(segment) == 2:
            if segment not in _STOPWORDS:
                tokens.add(segment)
        else:
            for idx in range(len(segment) - 1):
                bigram = segment[idx : idx + 2]
                if bigram not in _STOPWORDS:
                    tokens.add(bigram)
    return tokens


def _spec_body_snapshot(spec: GoalSpec) -> dict[str, object]:
    return {
        "goal": spec.goal,
        "requirements": [req.model_dump(mode="json") for req in spec.requirements],
        "constraints": list(spec.constraints),
        "assumptions": list(spec.assumptions),
        "out_of_scope": list(spec.out_of_scope),
        "deliverable_kind": spec.deliverable_kind.value if spec.deliverable_kind else None,
        "observable_surfaces": list(spec.observable_surfaces),
        "actors": list(spec.actors),
        "test_seams": list(spec.test_seams),
    }


def _spec_core_body_snapshot(spec: GoalSpec) -> dict[str, object]:
    body = _spec_body_snapshot(spec)
    body.pop("assumptions", None)
    return body


def _spec_body_text(spec: GoalSpec) -> str:
    parts = [
        spec.goal,
        *spec.constraints,
        *spec.out_of_scope,
        *spec.observable_surfaces,
        *spec.actors,
        *spec.test_seams,
    ]
    for req in spec.requirements:
        parts.append(req.description)
        parts.extend(req.acceptance_criteria)
        parts.extend(criterion.to_ears_text() for criterion in req.ears_criteria)
    return _norm_clause(" ".join(parts))


_PROPER_TECH_TOKEN_RE = re.compile(
    r"\b(?:[A-Z]{2,}[A-Za-z0-9_-]*|[A-Z][a-z]+[A-Z][A-Za-z0-9_-]*|[A-Za-z]+\d+[A-Za-z0-9_-]*|"
    r"\d+[A-Za-z][A-Za-z0-9_-]*)\b"
)
_SENTENCE_INITIAL_CAP_RE = re.compile(
    r"(?:^|[.!?。！？\n]\s*|[-*•]\s*|\d+[.)、]\s*)([A-Z][a-z]+)\b"
)
_NEGATION_CLAUSE_RE = re.compile(
    r"(?:\b(?:no|not|never|without|disable|disabled|disabling|off|forbid|forbidden|"
    r"prohibit|prohibited|exclude|excluded|disallow|disallowed)\b|"
    r"禁用|关闭|禁止|不得|不要|不启用|不允许|关闭|无需|不支持)",
    re.IGNORECASE,
)


def _proper_tech_tokens(line: str) -> set[str]:
    cleaned = _LIST_PREFIX_RE.sub("", line)
    tokens = {tok.casefold() for tok in _PROPER_TECH_TOKEN_RE.findall(cleaned)}
    # Also capture capitalized non-sentence-initial named technologies (e.g. Redis, Postgres).
    sentence_starts = {m.start(1) for m in _SENTENCE_INITIAL_CAP_RE.finditer(cleaned)}
    for match in re.finditer(r"\b([A-Z][a-z]{2,})\b", cleaned):
        if match.start(1) in sentence_starts:
            continue
        word = match.group(1).casefold()
        if word not in _STOPWORDS:
            tokens.add(word)
    return {tok for tok in tokens if tok not in _STOPWORDS and len(tok) >= 2}


def _only_negated_mentions(body_text: str, token: str) -> bool:
    """True when every occurrence of ``token`` in ``body_text`` sits inside a negation window."""
    matches = list(re.finditer(re.escape(token), body_text))
    if not matches:
        return False
    for match in matches:
        start = max(0, match.start() - 32)
        end = min(len(body_text), match.end() + 32)
        window = body_text[start:end]
        if not _NEGATION_CLAUSE_RE.search(window):
            return False
    return True


def _has_negated_mention(body_text: str, token: str) -> bool:
    """True when at least one occurrence of ``token`` in ``body_text`` sits in a negation window."""
    for match in re.finditer(re.escape(token), body_text):
        start = max(0, match.start() - 32)
        end = min(len(body_text), match.end() + 32)
        if _NEGATION_CLAUSE_RE.search(body_text[start:end]):
            return True
    return False


def _clause_reflection_violations(
    label: str,
    clause: str,
    *,
    current_body_text: str,
    candidate_body_text: str,
) -> list[str]:
    tokens = _distinctive_tokens(clause)
    if not tokens:
        return []
    novel_tokens = {tok for tok in tokens if tok not in current_body_text}
    check_tokens = novel_tokens or tokens
    matched = {tok for tok in check_tokens if tok in candidate_body_text}
    min_needed = 2 if len(check_tokens) >= 4 else 1
    if len(matched) < min_needed:
        return [
            f"{label} ({clause!r}) was not faithfully reflected in GoalSpec requirements, "
            "acceptance_criteria, or constraints."
        ]

    required_tech = {tok for tok in _proper_tech_tokens(clause) if tok in check_tokens}
    missing_tech = sorted(tok for tok in required_tech if tok not in candidate_body_text)
    if missing_tech:
        return [
            f"{label} ({clause!r}) dropped explicit technical requirement(s): "
            f"{', '.join(missing_tech)}."
        ]

    target_tokens = required_tech or matched
    if _NEGATION_CLAUSE_RE.search(clause):
        unnegated = sorted(
            tok for tok in target_tokens if not _has_negated_mention(candidate_body_text, tok)
        )
        if unnegated and len(unnegated) == len(target_tokens):
            return [
                f"{label} ({clause!r}) was not preserved as a prohibition/disabled rule in "
                f"GoalSpec body: {', '.join(unnegated)}."
            ]
    else:
        negated = sorted(
            tok for tok in target_tokens if _only_negated_mentions(candidate_body_text, tok)
        )
        if negated and len(negated) == len(target_tokens):
            return [
                f"{label} ({clause!r}) was negated in GoalSpec body instead of enabled: "
                f"{', '.join(negated)}."
            ]
    return []


_INTRO_HEADING_RE = re.compile(
    r"^(?:以下是|以下为|针对|关于|补充说明|说明|备注|核心边界|澄清回复|"
    r"(?:here\s+are|the\s+following|notes?|context|summary|constraints?|"
    r"requirements?|clarifications?|boundaries)\b).*[:：]\s*$",
    re.IGNORECASE,
)
_SUBSTANTIVE_DIRECTIVE_RE = re.compile(
    r"(?:\b(?:must|shall|require|required|enable|enabled|support|use|keep|allow)\b|"
    r"必须|启用|支持|强制|要求|保留|禁止|禁用)",
    re.IGNORECASE,
)


def _is_intro_heading(line: str) -> bool:
    """True only for pure introductory headings ending in a single colon."""
    if _LIST_PREFIX_RE.match(line) or _NEGATION_CLAUSE_RE.search(line):
        return False
    if line.count(":") + line.count("：") != 1:
        return False
    if _SUBSTANTIVE_DIRECTIVE_RE.search(line):
        return False
    return bool(_INTRO_HEADING_RE.match(line))


def _resynthesis_fidelity_violations(
    current: GoalSpec,
    candidate: GoalSpec,
    *,
    extra_feedback: str,
) -> list[str]:
    """Deterministically verify that resynthesis fused human clarifications without mislabeling."""
    violations: list[str] = []
    extra_lines = [line.strip() for line in extra_feedback.splitlines() if line.strip()]
    current_body_text = _spec_body_text(current)
    candidate_body_text = _spec_body_text(candidate)

    for amb in current.open_ambiguities:
        if not amb.blocking or not (amb.answer or "").strip():
            continue
        if _norm_clause(amb.answer) == _norm_clause(amb.default_assumption):
            continue
        violations.extend(
            _clause_reflection_violations(
                f"Resolved clarification {amb.id}",
                amb.answer or "",
                current_body_text=current_body_text,
                candidate_body_text=candidate_body_text,
            )
        )

    for note in current.minority_notes:
        if note.severity.value != "blocking" or not note.resolved:
            continue
        violations.extend(
            _clause_reflection_violations(
                f"Resolved critique {note.id}",
                note.resolution_note or "",
                current_body_text=current_body_text,
                candidate_body_text=candidate_body_text,
            )
        )

    human_clauses = [
        clause
        for raw in (
            *(amb.answer for amb in current.open_ambiguities if (amb.answer or "").strip()),
            *(
                note.resolution_note
                for note in current.minority_notes
                if note.resolved and (note.resolution_note or "").strip()
            ),
            *(_LIST_PREFIX_RE.sub("", line) for line in extra_lines),
        )
        if len(clause := _norm_clause(raw)) >= 4
    ]
    for assumption in candidate.agent_high_impact_assumptions:
        norm_assumption = _norm_clause(assumption)
        if not norm_assumption:
            continue
        if any(
            norm_assumption == clause or clause in norm_assumption or norm_assumption in clause
            for clause in human_clauses
        ):
            violations.append(
                f"Confirmed human decision must not be listed in agent_high_impact_assumptions: "
                f"{assumption!r}"
            )

    if extra_lines:
        if _spec_core_body_snapshot(candidate) == _spec_core_body_snapshot(current):
            violations.append(
                "Additional human instructions must be fused into goal, requirements, "
                "acceptance_criteria, constraints, or out_of_scope (not assumptions alone)."
            )
        substantive_lines = [line for line in extra_lines if not _is_intro_heading(line)]
        for line in substantive_lines or extra_lines:
            violations.extend(
                _clause_reflection_violations(
                    "Additional human instruction",
                    line,
                    current_body_text=current_body_text,
                    candidate_body_text=candidate_body_text,
                )
            )
    return violations


def _blocking_discipline_violations(spec: GoalSpec) -> list[str]:
    """Keep blocking questions few and every one resolvable by a recorded default."""
    blocking = [amb for amb in spec.open_ambiguities if amb.blocking and not amb.answer]
    violations: list[str] = []
    if len(blocking) > _MAX_BLOCKING_AMBIGUITIES:
        violations.append(
            f"{len(blocking)} blocking ambiguities exceed the limit of "
            f"{_MAX_BLOCKING_AMBIGUITIES}; keep only genuine irreversible decisions"
        )
    for amb in blocking:
        if not (amb.default_assumption or "").strip():
            violations.append(
                f"Blocking ambiguity {amb.id} lacks a concrete code-grounded "
                "`default_assumption` (the recommended answer)"
            )
    return violations


_CLARIFICATION_WINDOW_RESETS = frozenset(
    {"session.started", "session.continued", "run.revision_requested", "run.reopened"}
)


def _normalize_clarification_text(value: str | None) -> str:
    return " ".join((value or "").casefold().split())


def _needs_clarification_resynthesis(
    spec: GoalSpec,
    extra_notes: list[str],
    *,
    strict_canonical: bool = False,
) -> bool:
    """Return True when clarifications go beyond defaults or a strict spec still needs repair."""
    if any(note.strip() for note in extra_notes):
        return True
    if strict_canonical and validate_goal_spec(spec):
        return True
    for amb in spec.open_ambiguities:
        if not amb.blocking or not (amb.answer or "").strip():
            continue
        if _normalize_clarification_text(amb.answer).rstrip(".!。！？") != (
            _normalize_clarification_text(amb.default_assumption).rstrip(".!。！？")
        ):
            return True
    return any(note.severity.value == "blocking" and note.resolved for note in spec.minority_notes)


def _clarification_window(events: Sequence[RunEvent]) -> list[RunEvent]:
    """Return events since the last session/revision boundary in chronological order."""
    window: list[RunEvent] = []
    for event in reversed(list(events)):
        if event.event_type in _CLARIFICATION_WINDOW_RESETS:
            break
        window.append(event)
    window.reverse()
    return window


def _consumed_note_state(window: Sequence[RunEvent], *, spec_id: str) -> tuple[set[str], int]:
    consumed_ids: set[str] = set()
    legacy_cutoff_seq = 0
    for event in window:
        if event.event_type != "spec.resynthesized":
            continue
        if event.payload.get("spec_id") not in (None, "", spec_id):
            continue
        if "consumed_note_ids" not in event.payload:
            legacy_cutoff_seq = max(legacy_cutoff_seq, event.event_seq)
            continue
        ids = event.payload.get("consumed_note_ids")
        if isinstance(ids, list):
            consumed_ids.update(str(item) for item in ids if item)
    return consumed_ids, legacy_cutoff_seq


def clarification_note_states(
    events: Sequence[RunEvent], *, spec_id: str
) -> list[tuple[RunEvent, bool]]:
    """Return every clarification note for ``spec_id`` with its consumed-by-resynthesis flag."""
    window = _clarification_window(events)
    consumed_ids, legacy_cutoff_seq = _consumed_note_state(window, spec_id=spec_id)
    notes: list[tuple[RunEvent, bool]] = []
    for event in window:
        if event.event_type != "spec.clarification_note":
            continue
        if event.payload.get("spec_id") not in (None, "", spec_id):
            continue
        if not str(event.payload.get("feedback") or "").strip():
            continue
        consumed = event.event_id in consumed_ids or event.event_seq <= legacy_cutoff_seq
        notes.append((event, consumed))
    return notes


def unconsumed_clarification_notes(
    events: Sequence[RunEvent], *, spec_id: str
) -> list[tuple[str, str]]:
    """Return ``(event_id, feedback)`` note pairs not yet consumed by ``spec.resynthesized``."""
    return [
        (event.event_id, str(event.payload.get("feedback")).strip())
        for event, consumed in clarification_note_states(events, spec_id=spec_id)
        if not consumed
    ]


def has_unfulfilled_revision(events: Sequence[RunEvent]) -> bool:
    """True when a persisted revision/reopen has not yet been re-aligned into a saved spec."""
    latest_restart_seq = 0
    for ev in events:
        if ev.event_type in ("run.revision_requested", "run.reopened"):
            latest_restart_seq = max(latest_restart_seq, ev.event_seq)
    if latest_restart_seq == 0:
        return False
    return not any(
        ev.event_seq > latest_restart_seq
        and ev.event_type == "spec.saved"
        and ev.actor == "align"
        for ev in events
    )


def _clarify_event_has_input(payload: dict[str, object]) -> bool:
    """Distinguish real human clarification submissions from empty no-op ``clarify({})`` calls."""
    tracked_keys = ("answered_ids", "resolved_note_ids", "added_out_of_scope")
    if any(key in payload for key in tracked_keys):
        return any(
            isinstance(payload.get(key), list) and bool(payload.get(key))
            for key in tracked_keys
        )
    return True


def has_pending_resynthesis(
    events: Sequence[RunEvent],
    spec: GoalSpec | None,
    *,
    spec_approved: bool,
    strict_canonical: bool = False,
) -> bool:
    """True when an unapproved, unblocked GoalSpec still needs body-level resynthesis."""
    if spec is None or spec_approved or spec.unresolved_blockers():
        return False
    if unconsumed_clarification_notes(events, spec_id=spec.spec_id):
        return True
    saw_clarification = False
    saw_prior_resynthesis = False
    for event in reversed(list(events)):
        if event.event_type in _CLARIFICATION_WINDOW_RESETS:
            break
        if event.event_type == "spec.resynthesized":
            if event.payload.get("spec_id") in (None, "", spec.spec_id):
                saw_prior_resynthesis = True
                break
            continue
        if event.payload.get("spec_id") in (None, "", spec.spec_id) and (
            event.event_type == "spec.clarification_note"
            or (
                event.event_type == "spec.clarified"
                and _clarify_event_has_input(event.payload)
            )
        ):
            saw_clarification = True
    if not saw_clarification:
        return False
    if saw_prior_resynthesis:
        return True
    return _needs_clarification_resynthesis(spec, [], strict_canonical=strict_canonical)


class AlignStage:
    """First-principles requirements alignment: probe -> optional critique -> clarify -> freeze."""

    def __init__(
        self,
        *,
        store: RunStore,
        provider: LLMProvider,
        workspace_root: Path | str,
        trace_dir: Path | None = None,
        execution_drivers: ExecutionDriversConfig | None = None,
        driver_registry: DriverRegistry | None = None,
    ) -> None:
        self.store = store
        self.provider = provider
        self.workspace_root = Path(workspace_root).resolve()
        self.trace_dir = trace_dir
        self.execution_drivers = execution_drivers or ExecutionDriversConfig.resolve()
        self.driver_registry = driver_registry

    def _trace_file(self, run_id: str, name: str) -> Path | None:
        if self.trace_dir is None:
            return None
        return self.trace_dir / run_id / f"{name}.jsonl"

    def propose_spec(
        self,
        *,
        run_id: str,
        raw_goal: str,
        constraints: list[str] | None = None,
        enable_critique: bool = False,
        max_steps: int = 16,
        strict_canonical: bool = False,
        workspace_mode: str | None = None,
        task_type: str | None = None,
        base_ref: str | None = None,
        scope_paths: Sequence[str] | None = None,
    ) -> GoalSpec:
        """Probe repo context, draft GoalSpec, and optionally run cold-start critique.

        ``strict_canonical`` opts into the 5-element + EARS contract: the draft is checked
        with the zero-token gate and repaired once when it fails without blocking
        ambiguities. The default stays ``False`` for full backward compatibility.
        ``workspace_mode``/``task_type``/``base_ref``/``scope_paths`` are explicit
        deterministic inputs: a ``greenfield`` run skips host inventory scanning and a
        ``brownfield`` run binds its scoped paths and git ref into the prompt.
        """
        self.store.set_run_status(run_id, RunStatus.ALIGNING, actor="align", stage="align")
        driver, driver_model = resolve_stage_driver(
            settings=self.execution_drivers,
            registry=self.driver_registry,
            provider=self.provider,
            store=self.store,
            run_id=run_id,
        )
        toolkit = WorkspaceToolkit(self.workspace_root, read_only=True)
        loop = AgentLoop(
            provider=self.provider,
            toolkit=toolkit,
            store=self.store,
            run_id=run_id,
            actor="align:proposer",
            max_steps=max_steps,
            trace_path=self._trace_file(run_id, "align-proposer"),
        )
        greenfield = (workspace_mode or "").strip().lower() == "greenfield"
        inventory = (
            "Observed workspace files: (skipped: greenfield target has no existing repository)"
            if greenfield
            else format_workspace_inventory(toolkit, scope_paths=scope_paths)
        )
        scope_block = _scope_directive(
            workspace_mode=workspace_mode,
            task_type=task_type,
            base_ref=base_ref,
            scope_paths=scope_paths,
        )
        schema_json = json.dumps(GoalSpec.model_json_schema(), ensure_ascii=False)
        strict_block = (
            "\nCanonical contract (required):\n"
            "- deliverable_kind: one of web_app/rest_api/cli/library/desktop/multi_surface\n"
            "- observable_surfaces: the externally visible surfaces that can be exercised\n"
            "- actors: every human/system actor that triggers behavior\n"
            "- test_seams: 1-2 highest-level public seams (HTTP routes, CLI commands, public\n"
            "  APIs) where behavior is verified end to end; never internal file paths or helpers\n"
            "- every requirement sets actor plus 1-2 concise ears_criteria with pattern="
            "ubiquitous|event_driven|state_driven|unwanted_behavior|optional_feature\n"
            "- omit acceptance_criteria when ears_criteria is set (auto-materialized) to keep\n"
            "  JSON compact; acceptance_criteria stay plain strings with measurable thresholds\n"
            "- never leak internal implementation (CREATE TABLE, ALTER TABLE, src/*.py paths)\n"
            if strict_canonical
            else ""
        )
        user_prompt = (
            f"Workspace root: {self.workspace_root}\n"
            f"{inventory}\n"
            f"{scope_block}"
            f"Raw user goal:\n{raw_goal}\n\n"
            f"Explicit constraints: {json.dumps(constraints or [], ensure_ascii=False)}\n"
            f"{strict_block}\n"
            f"Return GoalSpec JSON matching schema:\n{schema_json}"
        )

        gate_calls = {"count": 0}

        def validation_gate(candidate: GoalSpec) -> list[str]:
            # Blocking ambiguities are a human decision, not something the model repairs:
            # only the initial draft may skip the canonical repair pass. A repaired draft is
            # always re-validated so a repair that introduces a blocker is still surfaced.
            # The blocking-discipline contract (count + required default) is always checked.
            gate_calls["count"] += 1
            discipline = _blocking_discipline_violations(candidate)
            if gate_calls["count"] == 1 and candidate.unresolved_blockers():
                return discipline
            if not strict_canonical:
                return discipline
            return [*discipline, *validate_goal_spec(candidate)]

        exhausted_violations: list[str] = []

        def record_validation_failure(candidate: GoalSpec, violations: list[str]) -> GoalSpec:
            exhausted_violations.clear()
            exhausted_violations.extend(violations)
            return candidate

        draft = run_structured_turn(
            loop=loop,
            driver=driver,
            driver_model=driver_model,
            store=self.store,
            run_id=run_id,
            workspace_root=self.workspace_root,
            actor="align:proposer",
            system_prompt=_ALIGN_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            model_cls=GoalSpec,
            trace_path=self._trace_file(run_id, "align-proposer"),
            max_steps=max_steps,
            max_validation_retries=1,
            validation_header="[GOALSPEC VALIDATION FAILED]",
            validation_instruction=(
                "Apply the canonical contract and return a corrected GoalSpec JSON object."
            ),
            validator=validation_gate,
            on_validation_exhausted=record_validation_failure,
        )

        notes: list[CritiqueNote] = list(draft.minority_notes)
        if enable_critique:
            notes.extend(self._run_parallel_critique(run_id=run_id, spec=draft))

        spec = draft.model_copy(
            update={
                "minority_notes": notes,
                "status": SpecStatus.AWAITING_HUMAN,
            }
        )
        self.store.save_spec(run_id, spec, actor="align")
        if exhausted_violations:
            self.store.append_event(
                run_id=run_id,
                stage="align",
                event_type="spec.validation_failed",
                actor="align",
                payload={
                    "violations": list(exhausted_violations),
                    "spec_hash": spec.content_hash,
                },
            )
        self.store.set_run_status(
            run_id,
            RunStatus.AWAITING_HUMAN,
            actor="align",
            stage="align",
            detail={
                "spec_hash": spec.content_hash,
                "blockers": spec.unresolved_blockers(),
            },
        )
        return spec

    def critique_spec(self, *, run_id: str) -> GoalSpec:
        """Run the parallel cold-start critique and persist the merged notes."""
        snap = self.store.get_run(run_id)
        spec: GoalSpec | None = snap["spec"]
        if spec is None:
            raise ValueError(f"Run {run_id} has no GoalSpec to critique")
        notes = [*spec.minority_notes, *self._run_parallel_critique(run_id=run_id, spec=spec)]
        updated = spec.model_copy(
            update={"minority_notes": notes, "status": SpecStatus.AWAITING_HUMAN}
        )
        self.store.save_spec(run_id, updated, actor="align")
        return updated

    def _run_parallel_critique(self, *, run_id: str, spec: GoalSpec) -> list[CritiqueNote]:
        """Run isolated cold-start read-only critique perspectives in parallel."""
        perspectives = (
            CritiquePerspective.PRODUCT,
            CritiquePerspective.ENGINEERING,
            CritiquePerspective.QA,
        )

        critique_schema = json.dumps(CritiqueOutput.model_json_schema(), ensure_ascii=False)

        def run_one(perspective: CritiquePerspective) -> list[CritiqueNote]:
            driver, driver_model = resolve_stage_driver(
                settings=self.execution_drivers,
                registry=self.driver_registry,
                provider=self.provider,
                store=self.store,
                run_id=run_id,
            )
            toolkit = WorkspaceToolkit(self.workspace_root, read_only=True)
            loop = AgentLoop(
                provider=self.provider,
                toolkit=toolkit,
                store=self.store,
                run_id=run_id,
                actor=f"align:critic-{perspective.value}",
                max_steps=10,
                trace_path=self._trace_file(run_id, f"align-critic-{perspective.value}"),
            )
            inventory = format_workspace_inventory(toolkit)
            sys_prompt = _CRITIQUE_SYSTEM_TEMPLATE.format(perspective=perspective.value)
            user_prompt = (
                f"Workspace root: {self.workspace_root}\n"
                f"{inventory}\n"
                f"Draft GoalSpec:\n{spec.model_dump_json(indent=2)}\n\n"
                f"Critique ONLY from the {perspective.value} perspective. "
                f"Set perspective='{perspective.value}' on every note.\n"
                f"Return JSON matching schema:\n{critique_schema}"
            )
            out = run_structured_turn(
                loop=loop,
                driver=driver,
                driver_model=driver_model,
                store=self.store,
                run_id=run_id,
                workspace_root=self.workspace_root,
                actor=f"align:critic-{perspective.value}",
                system_prompt=sys_prompt,
                user_prompt=user_prompt,
                model_cls=CritiqueOutput,
                trace_path=self._trace_file(run_id, f"align-critic-{perspective.value}"),
                max_steps=10,
            )
            return out.notes

        collected: list[CritiqueNote] = []
        with ThreadPoolExecutor(max_workers=len(perspectives)) as pool:
            for notes in pool.map(run_one, perspectives):
                collected.extend(notes)
        return collected

    def resynthesize_after_clarification(
        self,
        *,
        run_id: str,
        raw_goal: str = "",
        constraints: list[str] | None = None,
        extra_feedback: str = "",
        consumed_note_ids: Sequence[str] | None = None,
        strict_canonical: bool | None = None,
        max_steps: int = 8,
    ) -> GoalSpec:
        """Fuse resolved human answers and supplementary clarification lines into GoalSpec body."""
        snap = self.store.get_run(run_id)
        current: GoalSpec | None = snap["spec"]
        if current is None:
            raise ValueError(f"Run {run_id} has no GoalSpec to resynthesize")
        if current.unresolved_blockers():
            raise ValueError(
                f"Cannot resynthesize GoalSpec {current.spec_id} with unresolved blockers: "
                f"{current.unresolved_blockers()}"
            )
        initial_events = self.store.list_events(run_id)
        if snap["status"] in (RunStatus.PAUSED, RunStatus.CANCELLED) or has_unfulfilled_revision(
            initial_events
        ):
            raise ConcurrencyConflictError(
                f"Run {run_id} was revised or interrupted before resynthesis"
            )
        initial_note_ids = [
            ev.event_id
            for ev in initial_events
            if ev.event_type == "spec.clarification_note"
            and ev.payload.get("spec_id") in (None, "", current.spec_id)
        ]
        recorded_consumed_ids = (
            list(consumed_note_ids) if consumed_note_ids is not None else list(initial_note_ids)
        )
        driver, driver_model = resolve_stage_driver(
            settings=self.execution_drivers,
            registry=self.driver_registry,
            provider=self.provider,
            store=self.store,
            run_id=run_id,
        )
        toolkit = WorkspaceToolkit(self.workspace_root, read_only=True)
        loop = AgentLoop(
            provider=self.provider,
            toolkit=toolkit,
            store=self.store,
            run_id=run_id,
            actor="align:resynthesizer",
            max_steps=max_steps,
            trace_path=self._trace_file(run_id, "align-resynthesizer"),
        )
        resolved_ambiguities = [
            {
                "id": amb.id,
                "question": amb.question,
                "default_assumption": amb.default_assumption,
                "answer": amb.answer,
            }
            for amb in current.open_ambiguities
            if (amb.answer or "").strip()
        ]
        resolved_notes = [
            {
                "id": note.id,
                "summary": note.summary,
                "resolution_note": note.resolution_note,
            }
            for note in current.minority_notes
            if note.resolved
        ]
        effective_strict = (
            bool(strict_canonical)
            if strict_canonical is not None
            else (current.deliverable_kind is not None or bool(current.observable_surfaces))
        )
        strict_block = (
            "\nCanonical contract (required):\n"
            "- Set deliverable_kind, observable_surfaces, actors, and 1-2 public test_seams\n"
            "- Every requirement sets actor plus 1-2 concise ears_criteria\n"
            "- Never leak internal implementation paths or SQL DDL\n"
            if effective_strict
            else ""
        )
        schema_json = json.dumps(GoalSpec.model_json_schema(), ensure_ascii=False)
        user_prompt = (
            f"Raw user goal:\n{raw_goal or current.goal}\n\n"
            f"Explicit constraints: {json.dumps(constraints or [], ensure_ascii=False)}\n\n"
            f"Current GoalSpec JSON:\n{current.model_dump_json(indent=2)}\n\n"
            f"Resolved human clarifications:\n"
            f"{json.dumps(resolved_ambiguities, ensure_ascii=False, indent=2)}\n\n"
            f"Resolved critique notes:\n"
            f"{json.dumps(resolved_notes, ensure_ascii=False, indent=2)}\n\n"
            f"Additional human instructions / boundaries:\n"
            f"{extra_feedback.strip() or '(none)'}\n"
            f"{strict_block}\n"
            f"Return updated GoalSpec JSON matching schema:\n{schema_json}"
        )

        def lock_deterministic_fields(candidate: GoalSpec) -> GoalSpec:
            return candidate.model_copy(
                update={
                    "spec_id": current.spec_id,
                    "version": current.version + 1,
                    "status": SpecStatus.AWAITING_HUMAN,
                    "open_ambiguities": list(current.open_ambiguities),
                    "minority_notes": list(current.minority_notes),
                }
            )

        def validation_gate(candidate: GoalSpec) -> list[str]:
            locked = lock_deterministic_fields(candidate)
            violations = _resynthesis_fidelity_violations(
                current, locked, extra_feedback=extra_feedback
            )
            if effective_strict:
                violations.extend(validate_goal_spec(locked))
            return violations

        exhausted_violations: list[str] = []

        def record_validation_failure(candidate: GoalSpec, violations: list[str]) -> GoalSpec:
            exhausted_violations.clear()
            exhausted_violations.extend(violations)
            return candidate

        draft = run_structured_turn(
            loop=loop,
            driver=driver,
            driver_model=driver_model,
            store=self.store,
            run_id=run_id,
            workspace_root=self.workspace_root,
            actor="align:resynthesizer",
            system_prompt=_ALIGN_RESYNTHESIZE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            model_cls=GoalSpec,
            trace_path=self._trace_file(run_id, "align-resynthesizer"),
            max_steps=max_steps,
            max_validation_retries=1,
            validation_header="[GOALSPEC RESYNTHESIS VALIDATION FAILED]",
            validation_instruction=(
                "Fix the validation issues while preserving all human clarifications and return "
                "a valid GoalSpec JSON object."
            ),
            validator=validation_gate,
            on_validation_exhausted=record_validation_failure,
        )
        latest = self.store.get_run(run_id)
        latest_events = self.store.list_events(run_id)
        if latest["status"] in (
            RunStatus.PAUSED,
            RunStatus.CANCELLED,
        ) or has_unfulfilled_revision(latest_events):
            raise ConcurrencyConflictError(
                f"Run {run_id} was revised or interrupted during resynthesis"
            )
        latest_spec: GoalSpec | None = latest["spec"]
        latest_note_ids = [
            ev.event_id
            for ev in latest_events
            if ev.event_type == "spec.clarification_note"
            and ev.payload.get("spec_id") in (None, "", current.spec_id)
        ]
        if (
            latest_spec is None
            or latest_spec.spec_id != current.spec_id
            or latest_spec.version != current.version
            or latest_spec.content_hash != current.content_hash
            or latest_note_ids != initial_note_ids
        ):
            raise ConcurrencyConflictError(
                f"GoalSpec or clarification notes changed concurrently during resynthesis "
                f"for run {run_id}"
            )
        if exhausted_violations:
            self.store.append_event(
                run_id=run_id,
                stage="align",
                event_type="spec.validation_failed",
                actor="align",
                payload={
                    "violations": list(exhausted_violations),
                    "spec_hash": current.content_hash,
                },
                expected_state_revision=int(latest["state_revision"]),
            )
            raise ValueError(
                "GoalSpec resynthesis failed validation: " + "; ".join(exhausted_violations)
            )
        updated = lock_deterministic_fields(draft)
        self.store.save_spec(
            run_id,
            updated,
            actor="align",
            expected_revision=int(latest["state_revision"]),
            extra_events=[
                (
                    "align",
                    "spec.resynthesized",
                    {
                        "spec_id": updated.spec_id,
                        "version": updated.version,
                        "spec_hash": updated.content_hash,
                        "previous_spec_hash": current.content_hash,
                        "had_extra_feedback": bool(extra_feedback.strip()),
                        "consumed_note_ids": recorded_consumed_ids,
                    },
                )
            ],
        )
        return updated

    def apply_clarifications(
        self,
        *,
        run_id: str,
        answers: dict[str, str],
        resolutions: dict[str, str] | None = None,
        extra_out_of_scope: list[str] | None = None,
        actor: str = "zeus",
    ) -> GoalSpec:
        """Apply human answers/resolutions, producing a new spec version (invalidates old hash)."""
        return RunControl(self.store).clarify_spec(
            run_id=run_id,
            answers=answers,
            resolutions=resolutions,
            extra_out_of_scope=extra_out_of_scope,
            actor=actor,
        )

    def confirm_spec(
        self,
        *,
        run_id: str,
        approver: str = "zeus",
        accept_risk: bool = False,
        reason: str = "",
        commit_sha: str | None = None,
        expected_revision: int | None = None,
    ) -> GoalSpec:
        """Freeze and approve the current GoalSpec once all blocking ambiguities are resolved."""
        return RunControl(self.store).confirm_spec(
            run_id=run_id,
            approver=approver,
            accept_risk=accept_risk,
            reason=reason,
            commit_sha=commit_sha,
            expected_revision=expected_revision,
        )
