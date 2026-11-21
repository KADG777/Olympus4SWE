"""Transparent Append-Only AgentLoop with KV-prefix stability, budget guard, and trace log."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ValidationError

from olympus.core.contracts import ProviderTimeoutRetry, new_id, utc_now
from olympus.core.fingerprint import content_hash
from olympus.core.store import RunStore
from olympus.harness.hooks import PreToolUseHook
from olympus.harness.receipt import build_turn_receipt
from olympus.providers.base import ChatMessage, LLMProvider, ToolSpec, canonical_tool_specs
from olympus.tools.aci import WorkspaceToolkit

Observer = Callable[[str, dict[str, Any]], None]


def compose_observers(*observers: Observer | None) -> Observer:
    """Fan out ``(kind, payload)`` to every observer; UI failures never break the loop."""
    active = tuple(observer for observer in observers if observer is not None)

    def composed(kind: str, payload: dict[str, Any]) -> None:
        for observer in active:
            with suppress(Exception):
                observer(kind, payload)

    return composed


LONG_CHARS = 4000
KEEP_CHARS = 2000
TRIM_NOTE = (
    "\n\n[Cut from {total:,} characters after you read it. "
    "Full output is preserved in this run's trace.]"
)


def shrink_seen(
    messages: list[ChatMessage],
    *,
    long_chars: int = LONG_CHARS,
    keep_chars: int = KEEP_CHARS,
) -> list[ChatMessage]:
    """Shrink oversized tool outputs that were superseded by a later assistant turn.

    The newest tool results (those after the last assistant message) stay byte-for-byte
    intact so the model always sees the evidence it is currently reasoning over.
    """
    last_assistant_idx = -1
    for idx, msg in enumerate(messages):
        if msg.role == "assistant":
            last_assistant_idx = idx
    if last_assistant_idx < 0:
        return list(messages)

    shrunk: list[ChatMessage] = []
    for idx, msg in enumerate(messages):
        if (
            idx < last_assistant_idx
            and msg.role == "tool"
            and len(msg.content) > long_chars
            and "[Cut from " not in msg.content
        ):
            note = TRIM_NOTE.format(total=len(msg.content))
            shrunk.append(replace(msg, content=msg.content[:keep_chars] + note))
        else:
            shrunk.append(msg)
    return shrunk


_OUTER_FENCE_RE = re.compile(r"^\s*```(?:json|[a-zA-Z0-9_-]+)?\s*\n(.*)\n```\s*$", re.DOTALL)
_FENCE_RE = re.compile(r"```([a-zA-Z0-9_-]*)\s*\n?(.*?)\n?```", re.DOTALL)
_LINE_OBJECT_RE = re.compile(r"(?m)^(?:\{|\[)")
_SUBMIT_ARG_KEYS = (
    "plan",
    "spec",
    "design",
    "verdict",
    "result",
    "output",
    "payload",
    "data",
    "json",
)
_SCHEMA_META_KEYS = frozenset(
    {
        "$schema",
        "$comment",
        "$id",
        "$defs",
        "$ref",
        "$anchor",
        "$vocabulary",
        "$dynamicRef",
        "$dynamicAnchor",
    }
)


def strip_schema_meta_keys(value: Any) -> Any:
    """Recursively drop JSON Schema metadata keys while keeping extra='forbid' strictness."""
    if isinstance(value, dict):
        return {
            key: strip_schema_meta_keys(val)
            for key, val in value.items()
            if not (isinstance(key, str) and (key in _SCHEMA_META_KEYS or key.startswith("$")))
        }
    if isinstance(value, list):
        return [strip_schema_meta_keys(item) for item in value]
    return value


def _is_valid_json(candidate: str) -> bool:
    if not candidate:
        return False
    with suppress(json.JSONDecodeError):
        json.loads(candidate)
        return True
    return False


def _largest_decodable_json(text: str) -> str | None:
    """Return the longest top-level JSON object/array span cleanly decodable from text."""
    decoder = json.JSONDecoder()
    best: str | None = None
    idx = 0
    length = len(text)
    while idx < length:
        if text[idx] in ("{", "["):
            try:
                obj, end = decoder.raw_decode(text, idx)
            except json.JSONDecodeError:
                idx += 1
                continue
            if isinstance(obj, (dict, list)):
                candidate = text[idx:end]
                if best is None or len(candidate) >= len(best):
                    best = candidate
            idx = end
            continue
        idx += 1
    return best


def extract_json_payload(raw_text: str) -> str:
    """Extract outer JSON without letting inline prose dicts or inner code fences hijack it."""
    stripped = raw_text.strip()
    if not stripped:
        return ""
    if _is_valid_json(stripped):
        return stripped
    outer_match = _OUTER_FENCE_RE.match(stripped)
    if outer_match:
        outer_body = outer_match.group(1).strip()
        if _is_valid_json(outer_body):
            return outer_body
    largest = _largest_decodable_json(stripped)
    if largest is not None:
        return largest
    fences = [(m.group(1).lower(), m.group(2).strip()) for m in _FENCE_RE.finditer(stripped)]
    for tag, body in fences:
        if tag in ("", "json") and _is_valid_json(body):
            return body
    if outer_match:
        return outer_match.group(1).strip()
    if not stripped.startswith(("{", "[")):
        line_start = _LINE_OBJECT_RE.search(stripped)
        if line_start is not None:
            return stripped[line_start.start() :]
        for tag, body in fences:
            if tag in ("", "json"):
                return body
    start_obj = stripped.find("{")
    end_obj = stripped.rfind("}")
    if start_obj != -1 and end_obj > start_obj:
        return stripped[start_obj : end_obj + 1]
    return stripped


def parse_model_json[T: BaseModel](raw_text: str, model_cls: type[T]) -> T:
    """Extract JSON, strip schema meta-keys ($comment/$schema), and validate strictly."""
    candidate = extract_json_payload(raw_text)
    data = strip_schema_meta_keys(json.loads(candidate))
    return model_cls.model_validate(data)


def _validate_candidate_value[T: BaseModel](raw: Any, model_cls: type[T]) -> T | None:
    if isinstance(raw, str):
        with suppress(json.JSONDecodeError, ValidationError, ValueError):
            return parse_model_json(raw, model_cls)
    elif isinstance(raw, dict):
        with suppress(ValidationError, ValueError):
            return model_cls.model_validate(strip_schema_meta_keys(raw))
    return None


def _salvage_from_messages[T: BaseModel](
    messages: Sequence[ChatMessage],
    model_cls: type[T],
) -> T | None:
    """Recover a valid model payload from the last assistant turn or hallucinated submit tools."""
    last_assistant = next((m for m in reversed(messages) if m.role == "assistant"), None)
    if last_assistant is None:
        return None
    if last_assistant.content.strip():
        parsed = _validate_candidate_value(last_assistant.content, model_cls)
        if parsed is not None:
            return parsed
    for tc in reversed(last_assistant.tool_calls):
        args = tc.arguments
        for key in _SUBMIT_ARG_KEYS:
            if key in args:
                parsed = _validate_candidate_value(args[key], model_cls)
                if parsed is not None:
                    return parsed
        if len(args) == 1:
            only_val = next(iter(args.values()))
            parsed = _validate_candidate_value(only_val, model_cls)
            if parsed is not None:
                return parsed
        parsed = _validate_candidate_value(args, model_cls)
        if parsed is not None:
            return parsed
    return None


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
    receipt: dict[str, Any] = field(default_factory=dict)


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
        synthesize_on_max_steps: bool = False,
        observer: Observer | None = None,
        pre_tool_hook: PreToolUseHook | None = None,
        trim_seen: bool = True,
        task_id: str | None = None,
        lease_generation: int | None = None,
    ) -> None:
        self.provider = provider
        self.toolkit = toolkit
        self.store = store
        self.run_id = run_id
        self.actor = actor
        self.max_steps = max_steps
        self.trace_path = trace_path
        self.synthesize_on_max_steps = synthesize_on_max_steps
        self.observer = observer
        self.pre_tool_hook = pre_tool_hook
        self.trim_seen = trim_seen
        self.task_id = task_id
        self.lease_generation = lease_generation

    def _record_timeout_retry(
        self, retry_id: str, phase: Literal["started", "recovered", "failed"]
    ) -> None:
        if (
            self.store is None
            or self.run_id is None
            or self.task_id is None
            or self.lease_generation is None
        ):
            return
        retry = ProviderTimeoutRetry(
            retry_id=retry_id,
            task_id=self.task_id,
            lease_generation=self.lease_generation,
            phase=phase,
        )
        self.store.append_event(
            run_id=self.run_id,
            stage="orchestrate",
            event_type=retry.event_type,
            actor=self.actor,
            payload=retry.model_dump(mode="json"),
        )

    def _append_trace(self, entry: dict[str, Any]) -> None:
        if self.trace_path is None:
            return
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        with self.trace_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")

    def _call_provider_with_budget(
        self,
        *,
        step: int,
        purpose: str,
        messages: list[ChatMessage],
        tools: Sequence[ToolSpec],
        prefix_hash: str,
        temperature: float,
    ) -> tuple[Any, str | None]:
        """Reserve budget, invoke provider (with 1 transient timeout retry), settle usage."""
        retry_id = None
        for net_attempt in range(2):
            call_purpose = purpose if net_attempt == 0 else f"{purpose}_retry"
            if self.store is not None and self.run_id is not None:
                try:
                    self.store.reserve_llm_call(
                        self.run_id, actor=self.actor, purpose=call_purpose
                    )
                except ValueError as exc:
                    return None, f"budget_exhausted: {exc}"
            if net_attempt == 1:
                retry_id = new_id("retry")
                self._record_timeout_retry(retry_id, "started")
            try:
                resp = self.provider.complete(
                    messages=messages,
                    tools=tools,
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
                        "retry": net_attempt,
                    }
                )
                is_timeout = isinstance(exc, TimeoutError) or "timed out" in str(exc).lower()
                if is_timeout and net_attempt == 0:
                    continue
                if retry_id is not None:
                    self._record_timeout_retry(retry_id, "failed")
                return None, f"provider_error: {exc}"

            if self.store is not None and self.run_id is not None:
                self.store.settle_llm_call(
                    self.run_id,
                    actor=self.actor,
                    input_tokens=resp.usage.input_tokens,
                    output_tokens=resp.usage.output_tokens,
                )
            if retry_id is not None:
                self._record_timeout_retry(retry_id, "recovered")
            return resp, None
        return None, "provider_error: retry exhausted"

    def complete_repair(
        self,
        *,
        prior_messages: Sequence[ChatMessage],
        correction: str,
        step: int = 1,
        temperature: float = 0.0,
    ) -> LoopResult:
        """Run one tool-free repair turn preserving prior exploration messages."""
        messages = [*prior_messages, ChatMessage(role="user", content=correction)]
        prefix_hash = content_hash({"actor": self.actor, "repair": True})
        resp, err = self._call_provider_with_budget(
            step=step,
            purpose=f"repair_step_{step}",
            messages=messages,
            tools=(),
            prefix_hash=prefix_hash,
            temperature=temperature,
        )
        if resp is None:
            return LoopResult(
                completed=False,
                stop_reason=err or "provider_error",
                final_content="",
                steps_taken=0,
                messages=tuple(messages),
            )
        messages.append(ChatMessage(role="assistant", content=resp.content, tool_calls=()))
        self._append_trace(
            {
                "timestamp": utc_now().isoformat(),
                "actor": self.actor,
                "step": step,
                "prefix_hash": prefix_hash,
                "message_count": len(messages),
                "assistant_content": resp.content,
                "tool_calls": [],
                "tool_outputs": [],
                "usage": {
                    "input_tokens": resp.usage.input_tokens,
                    "output_tokens": resp.usage.output_tokens,
                },
            }
        )
        return LoopResult(
            completed=True,
            stop_reason="final_response",
            final_content=resp.content,
            steps_taken=1,
            messages=tuple(messages),
            total_input_tokens=resp.usage.input_tokens,
            total_output_tokens=resp.usage.output_tokens,
        )

    def _run_structured_turn[T: BaseModel](
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model_cls: type[T],
        max_parse_retries: int,
        temperature: float,
    ) -> T:
        """One structured turn: explore, then tool-free repairs until JSON validates."""
        last_err = ""
        res = self.run(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            synthesize_on_max_steps=True,
        )
        for attempt in range(max_parse_retries + 1):
            if not res.completed:
                if res.stop_reason.startswith(("budget_exhausted", "provider_error")):
                    raise RuntimeError(
                        f"AgentLoop stopped before producing final output: {res.stop_reason}"
                    )
                salvaged = _salvage_from_messages(res.messages, model_cls)
                if salvaged is not None:
                    return salvaged
                last_err = f"stopped early ({res.stop_reason})"
                if attempt < max_parse_retries:
                    schema_hint = json.dumps(model_cls.model_json_schema(), ensure_ascii=False)
                    correction = (
                        f"[STEP LIMIT REACHED: {res.stop_reason}]\n"
                        "Stop calling tools immediately. Output ONLY a single raw JSON object "
                        "starting with '{' matching schema "
                        f"(no prose, no $comment/$schema/extra keys):\n{schema_hint}"
                    )
                    res = self.complete_repair(
                        prior_messages=res.messages,
                        correction=correction,
                        step=attempt + 1,
                        temperature=temperature,
                    )
                    continue
                raise RuntimeError(
                    f"AgentLoop stopped before producing final output: {res.stop_reason}"
                )
            try:
                return parse_model_json(res.final_content, model_cls)
            except (json.JSONDecodeError, ValidationError) as exc:
                salvaged = _salvage_from_messages(res.messages, model_cls)
                if salvaged is not None:
                    return salvaged
                last_err = str(exc)
                if attempt < max_parse_retries:
                    schema_hint = json.dumps(model_cls.model_json_schema(), ensure_ascii=False)
                    empty_note = (
                        "Previous output was empty. "
                        if not res.final_content.strip()
                        else f"Error: {last_err}\n"
                    )
                    correction = (
                        "[PREVIOUS OUTPUT FAILED VALIDATION]\n"
                        f"{empty_note}"
                        "Do NOT call any tools and do NOT include leading prose. Return ONLY "
                        "a single valid JSON object starting with '{' matching schema "
                        f"(no $comment, $schema, or extra keys):\n{schema_hint}"
                    )
                    res = self.complete_repair(
                        prior_messages=res.messages,
                        correction=correction,
                        step=attempt + 1,
                        temperature=temperature,
                    )
        raise ValueError(f"Failed to parse {model_cls.__name__} JSON: {last_err}")

    def run_structured[T: BaseModel](
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model_cls: type[T],
        max_parse_retries: int = 2,
        temperature: float = 0.0,
        transform: Callable[[T], T] | None = None,
        validator: Callable[[T], Sequence[str]] | None = None,
        max_validation_retries: int = 0,
        validation_header: str = "[VALIDATION FAILED]",
        validation_instruction: str = "Fix all violations and return valid JSON.",
        on_validation_exhausted: Callable[[T, list[str]], T] | None = None,
    ) -> T:
        """Run the loop and return a validated typed payload.

        ``transform`` stamps deterministic bindings before ``validator`` inspects the
        payload. Domain violations re-run the whole turn from the original prompt plus
        the latest violations; when retries run out, ``on_validation_exhausted`` lets
        each stage decide whether to raise or record and continue. Final-step synthesis
        is requested per call, so ``self.synthesize_on_max_steps`` is never mutated.
        """
        prompt = user_prompt
        for attempt in range(max_validation_retries + 1):
            candidate = self._run_structured_turn(
                system_prompt=system_prompt,
                user_prompt=prompt,
                model_cls=model_cls,
                max_parse_retries=max_parse_retries,
                temperature=temperature,
            )
            if transform is not None:
                candidate = transform(candidate)
            if validator is None:
                return candidate
            violations = list(validator(candidate))
            if not violations:
                return candidate
            if attempt < max_validation_retries:
                prompt = (
                    f"{user_prompt}\n\n{validation_header}\n"
                    f"Violations: {'; '.join(violations)}\n{validation_instruction}"
                )
                continue
            if on_validation_exhausted is not None:
                return on_validation_exhausted(candidate, violations)
            raise ValueError(
                f"{model_cls.__name__} failed validation: {'; '.join(violations)}"
            )
        raise AssertionError("unreachable: validation loop always returns or raises")

    def run(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        extra_tools: Sequence[ToolSpec] = (),
        temperature: float = 0.0,
        synthesize_on_max_steps: bool | None = None,
    ) -> LoopResult:
        """Execute the append-only loop.

        ``synthesize_on_max_steps`` overrides the loop-level flag for this call only,
        so structured-output callers never mutate shared loop state.
        """
        synthesize = (
            self.synthesize_on_max_steps
            if synthesize_on_max_steps is None
            else synthesize_on_max_steps
        )
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
        turn_id = new_id("turn")
        turn_events: list[dict[str, Any]] = []

        def _emit(kind: str, payload: dict[str, Any]) -> None:
            turn_events.append({"kind": kind, **payload})
            if self.observer is not None:
                self.observer(kind, payload)

        def _finalize_result(
            *,
            completed: bool,
            stop_reason: str,
            final_content: str,
            steps_taken: int,
            final_messages: list[ChatMessage],
            total_input_tokens: int,
            total_output_tokens: int,
        ) -> LoopResult:
            receipt = build_turn_receipt(
                turn_id=turn_id,
                actor=self.actor,
                task_id=self.task_id,
                events=turn_events,
            )
            _emit("receipt", receipt)
            store = self.store
            run_id = self.run_id
            if store is not None and run_id is not None:
                # Best-effort: a missing run row must never fail an otherwise valid turn.
                with suppress(Exception):
                    store.append_event(
                        run_id=run_id,
                        stage="orchestrate",
                        event_type="turn.receipt",
                        actor=self.actor,
                        payload=receipt,
                    )
            return LoopResult(
                completed=completed,
                stop_reason=stop_reason,
                final_content=final_content,
                steps_taken=steps_taken,
                messages=tuple(final_messages),
                total_input_tokens=total_input_tokens,
                total_output_tokens=total_output_tokens,
                receipt=receipt,
            )

        for step in range(1, self.max_steps + 1):
            if step > 1 and self.trim_seen:
                messages = shrink_seen(messages)
            _emit(
                "step_start",
                {"actor": self.actor, "task_id": self.task_id, "step": step},
            )

            step_tools = frozen_tools
            if (
                synthesize
                and frozen_tools
                and step == self.max_steps
                and self.max_steps > 1
                and any(msg.role == "tool" for msg in messages)
            ):
                step_tools = ()
                messages.append(
                    ChatMessage(
                        role="user",
                        content=(
                            "[EXPLORATION BUDGET REACHED] Stop calling tools now and output "
                            "your final response using the repository facts already observed."
                        ),
                    )
                )

            resp, err = self._call_provider_with_budget(
                step=step,
                purpose=f"loop_step_{step}",
                messages=messages,
                tools=step_tools,
                prefix_hash=prefix_hash,
                temperature=temperature,
            )
            if resp is None:
                stop = err or "provider_error"
                steps = step - 1 if stop.startswith("budget_exhausted") else step
                return _finalize_result(
                    completed=False,
                    stop_reason=stop,
                    final_content="",
                    steps_taken=steps,
                    final_messages=messages,
                    total_input_tokens=total_in,
                    total_output_tokens=total_out,
                )

            total_in += resp.usage.input_tokens
            total_out += resp.usage.output_tokens
            _emit(
                "llm",
                {
                    "actor": self.actor,
                    "task_id": self.task_id,
                    "step": step,
                    "input_tokens": resp.usage.input_tokens,
                    "output_tokens": resp.usage.output_tokens,
                },
            )

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
                    denied_reason: str | None = None
                    if self.pre_tool_hook is not None:
                        allowed, reason = self.pre_tool_hook(tc.name, tc.arguments)
                        if not allowed:
                            denied_reason = reason
                    if denied_reason is not None:
                        out = f"Error: [PRE_TOOL_HOOK DENIED] {denied_reason}"
                        _emit(
                            "pre_tool_blocked",
                            {
                                "actor": self.actor,
                                "task_id": self.task_id,
                                "step": step,
                                "tool": tc.name,
                                "reason": denied_reason,
                            },
                        )
                    else:
                        _emit(
                            "tool_start",
                            {
                                "actor": self.actor,
                                "task_id": self.task_id,
                                "step": step,
                                "tool": tc.name,
                            },
                        )
                        started = time.monotonic()
                        if self.toolkit is None:
                            out = f"Error: no toolkit attached to execute tool '{tc.name}'."
                        else:
                            out = self.toolkit.execute(tc.name, tc.arguments)
                        elapsed_ms = int((time.monotonic() - started) * 1000)
                        _emit(
                            "tool_end",
                            {
                                "actor": self.actor,
                                "task_id": self.task_id,
                                "step": step,
                                "tool": tc.name,
                                "is_error": out.startswith("Error:"),
                                "duration_ms": elapsed_ms,
                            },
                        )
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
                return _finalize_result(
                    completed=True,
                    stop_reason="final_response",
                    final_content=resp.content,
                    steps_taken=step,
                    final_messages=messages,
                    total_input_tokens=total_in,
                    total_output_tokens=total_out,
                )

        return _finalize_result(
            completed=False,
            stop_reason="max_steps_exceeded",
            final_content=messages[-1].content if messages else "",
            steps_taken=self.max_steps,
            final_messages=messages,
            total_input_tokens=total_in,
            total_output_tokens=total_out,
        )
