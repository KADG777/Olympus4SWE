"""Stage 1 (Align): Evidence-backed spec synthesis, optional critique, and human freeze."""

from __future__ import annotations

import json
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pydantic import Field

from olympus.core.contracts import (
    CritiqueNote,
    CritiquePerspective,
    GoalSpec,
    RunStatus,
    SpecStatus,
    StrictModel,
)
from olympus.core.control import RunControl
from olympus.core.dag import validate_goal_spec
from olympus.core.store import RunStore
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
   brittle internal file paths, private helpers, or test-framework internals."""

_CRITIQUE_SYSTEM_TEMPLATE = """You are a cold-start read-only {perspective} reviewer.
Inspect the draft GoalSpec (and repository files if needed) and report ONLY real omissions,
hidden contradictions, feasibility risks, or untestable acceptance criteria.
Do NOT invent scores or rewrite the spec. If no issues exist, return {{"notes": []}}.
Output ONLY valid JSON matching {{"notes": [...]}} with no `$comment`, `$schema`, or extra keys."""

_MAX_BLOCKING_AMBIGUITIES = 3


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
