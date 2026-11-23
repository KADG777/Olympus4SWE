"""AgentDriver adapter for the built-in append-only loop."""

from collections.abc import Callable
from dataclasses import replace
from typing import Any

from pydantic import BaseModel

from olympus.core.store import RunStore
from olympus.harness.driver import ActiveTurnHandle, DriverEvent, DriverTurnRequest, TurnOutcome
from olympus.harness.loop import AgentLoop, TurnCancelledError, compose_observers
from olympus.harness.turn_control import NativeTurnControl
from olympus.providers.base import LLMProvider
from olympus.tools.aci import WorkspaceToolkit


class NativeDriver:
    name = "native"

    def __init__(
        self,
        *,
        provider: LLMProvider,
        store: RunStore | None = None,
        run_id: str | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> None:
        self._provider = provider
        self._store = store
        self._run_id = run_id
        self._response_model = response_model

    def run_turn(
        self,
        request: DriverTurnRequest,
        *,
        on_event: DriverEvent | None = None,
        on_handle: Callable[[ActiveTurnHandle], None] | None = None,
    ) -> TurnOutcome:
        control = NativeTurnControl()

        def observe(kind: str, payload: dict[str, Any]) -> None:
            if on_event is None:
                return
            if kind == "tool_start":
                on_event("agent.tool_started", payload)
            elif kind in ("tool_end", "pre_tool_blocked"):
                fields = ("actor", "task_id", "step", "tool", "status", "duration_ms")
                finished = {key: payload[key] for key in fields}
                finished["error_signature"] = payload.get("error_signature", "")
                on_event("agent.tool_finished", finished)
            else:
                on_event(kind, payload)

        try:
            if on_handle is not None:
                on_handle(control)
            toolkit = WorkspaceToolkit(
                request.workspace_path,
                read_only=request.read_only,
                allowed_write_scope=request.write_scope if request.strict_scope else None,
                forbidden_write_scope=request.forbidden_globs,
                env_root=request.env_root,
            )
            loop = AgentLoop(
                provider=self._provider,
                toolkit=toolkit,
                store=self._store,
                run_id=self._run_id,
                actor=request.actor,
                max_steps=request.max_steps,
                trace_path=request.trace_path,
                synthesize_on_max_steps=True,
                observer=compose_observers(observe),
                pre_tool_hook=None if request.read_only else toolkit.as_pre_tool_hook(),
                task_id=request.task_id,
                lease_generation=request.lease_generation,
                control=control,
                normalized_trace=True,
            )
            result = loop.run(system_prompt=request.system_prompt, user_prompt=request.user_prompt)
            content = result.final_content
            if self._response_model is not None and result.stop_reason != "cancelled":
                try:
                    parsed, result = loop.validate_structured_result(
                        result, model_cls=self._response_model
                    )
                    content = parsed.model_dump_json()
                    if not result.completed:
                        result = replace(result, completed=True, stop_reason="structured_salvage")
                except TurnCancelledError as exc:
                    result = exc.last_result
        finally:
            cancelled = control.close()
        return TurnOutcome(
            completed=not cancelled and result.completed,
            stop_reason="cancelled" if cancelled else result.stop_reason,
            final_content="" if cancelled else content,
            steps_taken=result.steps_taken,
            total_input_tokens=result.total_input_tokens,
            total_output_tokens=result.total_output_tokens,
            receipt=result.receipt,
            driver_name=self.name,
        )
