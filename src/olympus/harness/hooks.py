"""Deterministic pre-tool-use authorization gates (write_scope enforcement)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from olympus.tools.aci import WorkspacePolicy

PreToolUseHook = Callable[[str, dict[str, Any]], tuple[bool, str]]


def default_write_scope_hook(
    allowed_globs: Sequence[str],
    *,
    forbidden_globs: Sequence[str] = (),
    workspace_root: Path | None = None,
) -> PreToolUseHook:
    """Build a hook that denies writes outside ``allowed_globs`` or inside ``forbidden_globs``."""
    return WorkspacePolicy(
        workspace_root=workspace_root,
        allowed_write_scope=list(allowed_globs),
        forbidden_write_scope=list(forbidden_globs),
    ).as_pre_tool_hook()