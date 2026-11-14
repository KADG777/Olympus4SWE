"""Transparent Append-Only AgentLoop with KV-prefix stability, budget guard, and trace log."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from olympus.core.contracts import utc_now
from olympus.core.fingerprint import content_hash
from olympus.core.store import RunStore
from olympus.providers.base import ChatMessage, LLMProvider, ToolSpec, canonical_tool_specs
from olympus.tools.aci import WorkspaceToolkit


@dataclass(frozen=True)
class LoopStepRecord:
    step: int
    prefix_hash: str
    message_count: int
    assistant_content: str
    tool_calls: list[dict[str, Any]]
    tool_outputs: list[dict[str, str]]
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class LoopResult:
    completed: bool
    stop_reason: str
    final_content: str
    steps_taken: int
    messages: tuple[ChatMessage, ...] = field(default_factory=tuple)
    total_input_tokens: int = 0
    total_output_tokens: int = 0


class AgentLoop:
    """Append-Only execution loop maintaining static prefix and full error preservation."""

    def __init__(
        self,
        *,
        provider: LLMProvider,
        toolkit: WorkspaceToolkit | None = None,
        store: RunStore | None = None,
        run_id: str | None = None,
        actor: str = "agent",
        max_steps: int = 20,
        trace_path: Path | None = None,
    ) -> None:
        self.provider = provider
        self.toolkit = toolkit
        self.store = store
        self.run_id = run_id
        self.actor = actor
        self.max_steps = max_steps
        self.trace_path = trace_path

    def _append_trace(self, entry: dict[str, Any]) -> None:
        if self.trace_path is None:
            return
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        with self.trace_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")

    def run(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        extra_tools: Sequence[ToolSpec] = (),
        temperature: float = 0.0,
    ) -> LoopResult:
        tool_list: list[ToolSpec] = []
        if self.toolkit is not None:
            tool_list.extend(self.toolkit.tool_specs())
        tool_list.extend(extra_tools)
        frozen_tools = tuple(canonical_tool_specs(tool_list))

        # Static prefix stays invariant across all steps for KV-cache reuse
        messages: list[ChatMessage] = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=user_prompt),
        ]
        prefix_hash = content_hash(
            {
                "system": system_prompt,
                "user": user_prompt,
                "tools": [t.name for t in frozen_tools],
            }
        )

        total_in = 0
        total_out = 0

        for step in range(1, self.max_steps + 1):
            if self.store is not None and self.run_id is not None:
                try:
                    self.store.reserve_llm_call(
                        self.run_id,
                        actor=self.actor,
                        purpose=f"loop_step_{step}",
                    )
                except ValueError as exc:
                    return LoopResult(
                        completed=False,
                        stop_reason=f"budget_exhausted: {exc}",
                        final_content="",
                        steps_taken=step - 1,
                        messages=tuple(messages),
                        total_input_tokens=total_in,
                        total_output_tokens=total_out,
                    )

            try:
                resp = self.provider.complete(
                    messages=messages,
                    tools=frozen_tools,
                    temperature=temperature,
                )
            except Exception as exc:
                if self.store is not None and self.run_id is not None:
                    self.store.settle_llm_call(self.run_id, actor=self.actor, unknown=True)
                self._append_trace(
                    {
                        "timestamp": utc_now().isoformat(),
                        "actor": self.actor,
                        "step": step,
                        "prefix_hash": prefix_hash,
                        "error": str(exc),
                    }
                )
                return LoopResult(
                    completed=False,
                    stop_reason=f"provider_error: {exc}",
                    final_content="",
                    steps_taken=step,
                    messages=tuple(messages),
                    total_input_tokens=total_in,
                    total_output_tokens=total_out,
                )

            if self.store is not None and self.run_id is not None:
                self.store.settle_llm_call(
                    self.run_id,
                    actor=self.actor,
                    input_tokens=resp.usage.input_tokens,
                    output_tokens=resp.usage.output_tokens,
                )
            total_in += resp.usage.input_tokens
            total_out += resp.usage.output_tokens

            # Append-only update: preserve assistant turn + any tool errors
            messages.append(
                ChatMessage(
                    role="assistant",
                    content=resp.content,
                    tool_calls=resp.tool_calls,
                )
            )

            tool_outputs: list[dict[str, str]] = []
            if resp.tool_calls:
                for tc in resp.tool_calls:
                    if self.toolkit is None:
                        out = f"Error: no toolkit attached to execute tool '{tc.name}'."
                    else:
                        out = self.toolkit.execute(tc.name, tc.arguments)
                    tool_outputs.append({"id": tc.id, "name": tc.name, "output": out})
                    messages.append(
                        ChatMessage(
                            role="tool",
                            content=out,
                            tool_call_id=tc.id,
                            name=tc.name,
                        )
                    )

            self._append_trace(
                {
                    "timestamp": utc_now().isoformat(),
                    "actor": self.actor,
                    "step": step,
                    "prefix_hash": prefix_hash,
                    "message_count": len(messages),
                    "assistant_content": resp.content,
                    "tool_calls": [
                        {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                        for tc in resp.tool_calls
                    ],
                    "tool_outputs": tool_outputs,
                    "usage": {
                        "input_tokens": resp.usage.input_tokens,
                        "output_tokens": resp.usage.output_tokens,
                    },
                }
            )

            if not resp.tool_calls:
                return LoopResult(
                    completed=True,
                    stop_reason="final_response",
                    final_content=resp.content,
                    steps_taken=step,
                    messages=tuple(messages),
                    total_input_tokens=total_in,
                    total_output_tokens=total_out,
                )

        return LoopResult(
            completed=False,
            stop_reason="max_steps_exceeded",
            final_content=messages[-1].content if messages else "",
            steps_taken=self.max_steps,
            messages=tuple(messages),
            total_input_tokens=total_in,
            total_output_tokens=total_out,
        )
