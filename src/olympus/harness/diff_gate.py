"""Deterministic post-flight Diff Gate enforcing a task's sandbox write boundary.

The gate inspects every sandbox path the worker mutated during its turn (tracked
edits, renames, deletions, and untracked files), then evaluates each relative path
with the shared :class:`~olympus.tools.aci.WorkspacePolicy` matcher. Violations are
reported — never auto-reverted or silently dropped — so Stage 2 can requeue or
block the attempt with the exact offending paths.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass

from olympus.harness.worktree import (
    SandboxBaseline,
    SandboxHandle,
    collect_sandbox_changes,
)
from olympus.tools.aci import WorkspacePolicy

FORBIDDEN_REASON = "forbidden"
OUTSIDE_SCOPE_REASON = "outside_write_scope"


@dataclass(frozen=True)
class DiffGateViolation:
    path: str
    reason: str
    matched_pattern: str | None = None


@dataclass(frozen=True)
class DiffGateResult:
    changed_paths: tuple[str, ...]
    violations: tuple[DiffGateViolation, ...]
    allowed_write_scope: tuple[str, ...]
    forbidden_globs: tuple[str, ...]
    advisory_out_of_scope: tuple[str, ...] = ()
    strict_scope: bool = True

    @property
    def rejected(self) -> bool:
        return bool(self.violations)

    def rejection_message(self) -> str:
        """Build an actionable ``last_error`` naming every offending relative path."""
        lines = [
            f"Diff Gate rejected the attempt: {len(self.violations)} changed path(s) "
            "violate the task sandbox scope.",
            f"Allowed write_scope: {json.dumps(list(self.allowed_write_scope))}",
            f"Forbidden globs: {json.dumps(list(self.forbidden_globs))}",
            "Violating paths:",
        ]
        for violation in self.violations:
            if violation.reason == FORBIDDEN_REASON:
                detail = f"forbidden: matches {violation.matched_pattern!r}"
            else:
                detail = "outside authorized write_scope"
            lines.append(f"- {violation.path} ({detail})")
        if self.advisory_out_of_scope:
            lines.append(
                "Advisory out-of-declared-scope paths (non-blocking): "
                f"{json.dumps(list(self.advisory_out_of_scope))}"
            )
        lines.append(
            "No changes were reverted. Move these paths back inside the authorized scope "
            "or request a scope change, then retry."
        )
        return "\n".join(lines)


def evaluate_sandbox_changes(
    changed_paths: Sequence[str],
    *,
    allowed_write_scope: Sequence[str],
    forbidden_globs: Sequence[str],
    strict_scope: bool = True,
) -> DiffGateResult:
    """Classify each changed relative path using the shared WorkspacePolicy matcher.

    ``forbidden_globs`` are always hard violations. Paths outside ``allowed_write_scope``
    are hard violations under ``strict_scope=True`` (concurrent waves) and advisory audit
    metadata under ``strict_scope=False`` (single-task/serial execution).
    """
    policy = WorkspacePolicy(
        allowed_write_scope=allowed_write_scope,
        forbidden_write_scope=forbidden_globs,
    )
    violations: list[DiffGateViolation] = []
    advisory: list[str] = []
    for rel in sorted(set(changed_paths)):
        matched = policy.forbidden_match(rel)
        if matched is not None:
            violations.append(
                DiffGateViolation(
                    path=rel,
                    reason=FORBIDDEN_REASON,
                    matched_pattern=matched,
                )
            )
        elif not policy.allows_relative_path(rel):
            if strict_scope:
                violations.append(DiffGateViolation(path=rel, reason=OUTSIDE_SCOPE_REASON))
            else:
                advisory.append(rel)
    return DiffGateResult(
        changed_paths=tuple(changed_paths),
        violations=tuple(violations),
        allowed_write_scope=tuple(allowed_write_scope),
        forbidden_globs=tuple(forbidden_globs),
        advisory_out_of_scope=tuple(advisory),
        strict_scope=strict_scope,
    )


def run_diff_gate(
    handle: SandboxHandle,
    baseline: SandboxBaseline,
    *,
    allowed_write_scope: Sequence[str],
    forbidden_globs: Sequence[str],
    strict_scope: bool = True,
) -> DiffGateResult:
    """Collect worker mutations in the sandbox and evaluate them against its scope."""
    return evaluate_sandbox_changes(
        collect_sandbox_changes(handle, baseline),
        allowed_write_scope=allowed_write_scope,
        forbidden_globs=forbidden_globs,
        strict_scope=strict_scope,
    )
