"""Shared stage helpers: workspace inventory plus structured-output compatibility seams."""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel

from olympus.core.store import RunStore
from olympus.harness.driver import (
    AgentDriver,
    DriverRegistry,
    ExecutionDriversConfig,
    default_driver_factories,
    resolve_driver_spec,
)
from olympus.harness.loop import (
    AgentLoop,
    extract_json_payload,
    parse_model_json,
    strip_schema_meta_keys,
)
from olympus.providers.base import LLMProvider
from olympus.tools.aci import WorkspaceToolkit


def _normalize_rel_path(raw: str) -> str:
    cleaned = raw.strip().replace("\\", "/")
    while cleaned.startswith("./"):
        cleaned = cleaned[2:]
    return cleaned.strip("/")


def _matches_scope(rel_path: str, scope_paths: Sequence[str]) -> bool:
    normalized = _normalize_rel_path(rel_path)
    for raw in scope_paths:
        target = _normalize_rel_path(raw)
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
    cleaned_scopes = [
        cleaned
        for item in (scope_paths or ())
        if (cleaned := _normalize_rel_path(item))
    ]
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


def resolve_stage_driver(
    *,
    settings: ExecutionDriversConfig | None,
    registry: DriverRegistry | None,
    provider: LLMProvider,
    store: RunStore,
    run_id: str,
) -> tuple[AgentDriver | None, str | None]:
    """Resolve the planning-stage driver; ``None`` means the native ``AgentLoop``."""
    config = settings or ExecutionDriversConfig.resolve()
    spec = resolve_driver_spec(config.stage_driver, provider=provider)
    if spec.partition("@")[0] == "native":
        return None, None
    resolved_registry = (registry or DriverRegistry()).with_factories(
        default_driver_factories(store=store, run_id=run_id)
    )
    return resolved_registry.resolve_external(spec)


__all__ = [
    "extract_json_payload",
    "format_workspace_inventory",
    "parse_model_json",
    "resolve_stage_driver",
    "run_structured_loop",
    "strip_schema_meta_keys",
]
