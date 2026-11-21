"""Shared stage helpers: workspace inventory plus structured-output compatibility seams."""

from __future__ import annotations

from pydantic import BaseModel

from olympus.harness.loop import (
    AgentLoop,
    extract_json_payload,
    parse_model_json,
    strip_schema_meta_keys,
)
from olympus.tools.aci import WorkspaceToolkit


def format_workspace_inventory(toolkit: WorkspaceToolkit, *, max_files: int = 40) -> str:
    """Format a bounded deterministic inventory of existing workspace paths."""
    files, total = toolkit.list_relative_files(max_files=max_files)
    if not files:
        return "Observed workspace files: (none)"
    suffix = f" (+{total - len(files)} more)" if total > len(files) else ""
    return f"Observed workspace files ({len(files)}/{total}): " + ", ".join(files) + suffix


def run_structured_loop[T: BaseModel](
    *,
    loop: AgentLoop,
    system_prompt: str,
    user_prompt: str,
    model_cls: type[T],
    max_parse_retries: int = 2,
) -> T:
    """Compatibility shim delegating to ``AgentLoop.run_structured``."""
    return loop.run_structured(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model_cls=model_cls,
        max_parse_retries=max_parse_retries,
    )


__all__ = [
    "extract_json_payload",
    "format_workspace_inventory",
    "parse_model_json",
    "run_structured_loop",
    "strip_schema_meta_keys",
]
