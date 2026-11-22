"""Gate austerity policy for headless interactive evaluation sessions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from olympus.core.dag import validate_design_spec, validate_plan

if TYPE_CHECKING:
    from olympus.graph.loader import WorkflowRuntime


class InteractiveMinimalPolicy:
    """Pause only on genuine blockers or failed deterministic validation.

    - ``GoalSpec``: pause for clarification only while ``unresolved_blockers()`` is
      non-empty; otherwise auto-confirm so the headless caller is never asked to
      rubber-stamp requirements it never objected to.
    - ``DesignSpec`` / ``TaskPlan``: re-run the zero-token validators; auto-approve
      when they pass, block when they still fail after the stage-level repair loop.
    """

    name = "interactive_minimal"

    def __call__(
        self,
        gate: str,
        artifact: Any,
        runtime: WorkflowRuntime,
        run_id: str,
    ) -> str:
        snap = runtime.store.get_run(run_id)
        if gate == "spec":
            blockers = getattr(artifact, "unresolved_blockers", lambda: [])()
            return "await_human" if blockers else "auto_approve"
        if gate == "design":
            violations = validate_design_spec(artifact, snap.get("spec"))
            return "block" if violations else "auto_approve"
        if gate == "plan":
            violations = validate_plan(
                artifact,
                snap.get("spec"),
                design=snap.get("design"),
                workspace_root=runtime.workspace_root,
            )
            return "block" if violations else "auto_approve"
        return "await_human"


__all__ = ["InteractiveMinimalPolicy"]
