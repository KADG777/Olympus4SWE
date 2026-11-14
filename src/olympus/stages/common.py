"""Shared helpers for structured LLM output parsing and role prompts."""

from __future__ import annotations

import json
import re

from pydantic import BaseModel, ValidationError

from olympus.harness.loop import AgentLoop
from olympus.providers.base import LLMProvider

_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?```", re.DOTALL)


def extract_json_payload(raw_text: str) -> str:
    """Extract JSON object/array from plain text or fenced Markdown blocks."""
    stripped = raw_text.strip()
    match = _FENCE_RE.search(stripped)
    if match:
        return match.group(1).strip()
    # Fallback: locate outermost braces
    start_obj = stripped.find("{")
    end_obj = stripped.rfind("}")
    if start_obj != -1 and end_obj > start_obj:
        return stripped[start_obj : end_obj + 1]
    return stripped


def run_structured_loop[T: BaseModel](
    *,
    loop: AgentLoop,
    system_prompt: str,
    user_prompt: str,
    model_cls: type[T],
    max_parse_retries: int = 1,
) -> T:
    """Run an AgentLoop and validate final JSON response against a Pydantic model."""
    prompt = user_prompt
    last_err = ""
    for attempt in range(max_parse_retries + 1):
        res = loop.run(system_prompt=system_prompt, user_prompt=prompt)
        if not res.completed:
            raise RuntimeError(
                f"AgentLoop stopped before producing final output: {res.stop_reason}"
            )
        candidate = extract_json_payload(res.final_content)
        try:
            data = json.loads(candidate)
            return model_cls.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as exc:
            last_err = str(exc)
            if attempt < max_parse_retries:
                schema_hint = json.dumps(model_cls.model_json_schema(), ensure_ascii=False)
                prompt = (
                    f"{user_prompt}\n\n"
                    f"[PREVIOUS OUTPUT FAILED VALIDATION]\n"
                    f"Error: {last_err}\n"
                    f"Return ONLY valid JSON matching schema:\n{schema_hint}"
                )
    raise ValueError(f"Failed to parse {model_cls.__name__} JSON: {last_err}")


def build_read_only_loop(
    *,
    provider: LLMProvider,
    loop_kwargs: dict[str, object],
) -> AgentLoop:
    """Convenience constructor helper preserving strict keyword args."""
    return AgentLoop(provider=provider, **loop_kwargs)  # type: ignore[arg-type]
