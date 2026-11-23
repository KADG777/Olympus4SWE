"""Shared stage helpers: workspace inventory plus structured-output compatibility seams."""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel

from olympus.harness.loop import (
    AgentLoop,
    extract_json_payload,
    parse_model_json,
    strip_schema_meta_keys,
)
from olympus.tools.aci import WorkspaceToolkit


def _matches_scope(rel_path: str, scope_paths: Sequence[str]) -> bool:
    normalized = rel_path.lstrip("./")
    for raw in scope_paths:
        target = raw.strip().strip("/")
        if not target:
            continue
        if normalized == target or normalized.startswith(f"{target}/"):
            return True
    return False


def format_workspace_inventory(
    toolkit: WorkspaceToolkit,
    *,
    max_files: int = 40,
    scope_paths: Sequence[str] | None = None,
) -> str:
    """Format a bounded deterministic inventory of existing workspace paths."""
    cleaned_scopes = [item.strip().strip("/") for item in (scope_paths or ()) if item.strip()]
    if cleaned_scopes:
        all_files, _ = toolkit.list_relative_files(max_files=5000)
        matched = [path for path in all_files if _matches_scope(path, cleaned_scopes)]
        files = matched[:max_files]
        total = len(matched)
    else:
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
