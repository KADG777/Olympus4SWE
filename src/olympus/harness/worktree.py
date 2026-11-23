"""Isolated git worktree sandbox management and diff/commit inspection."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path, PurePosixPath

from olympus.tools.aci import IGNORED_DIR_NAMES, WorkspacePolicy, sandbox_tmp_path
from olympus.tools.aci import build_sandbox_env as build_sandbox_env

EPHEMERAL_ARTIFACT_PATTERNS: tuple[str, ...] = (
    "*.backup",
    "*.bak",
    "*.orig",
    "*.tmp",
    "*~",
    ".*.swp",
    ".DS_Store",
    "**/__pycache__/**",
    "*.pyc",
    "**/.pytest_cache/**",
)


@dataclass(frozen=True)
class SandboxHandle:
    repo_root: Path
    workspace_path: Path
    base_commit: str | None
    is_git_worktree: bool
    reused: bool = False

    @property
    def tmp_dir(self) -> Path:
        return sandbox_tmp_path(self.workspace_path, self.repo_root)

    @property
    def baseline_path(self) -> Path:
        if not self.workspace_path.resolve().is_relative_to(self.repo_root.resolve()):
            return (
                self.workspace_path.resolve().parent
                / ".sandbox_baselines"
                / f"{self.workspace_path.name}.json"
            )
        return (
            self.repo_root
            / ".olympus"
            / "sandbox_baselines"
            / f"{self.workspace_path.name}.json"
        )


@dataclass(frozen=True)
class SandboxBaseline:
    """Fingerprints of the sandbox state handed to a worker before its turn."""

    is_git: bool
    entries: tuple[tuple[str, str | None], ...]
    base_commit: str | None = None
    preexisting_ephemeral: tuple[str, ...] = ()

    def mapping(self) -> dict[str, str | None]:
        return dict(self.entries)


def _run_git(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


def is_git_repo(path: Path) -> bool:
    proc = _run_git(path, "rev-parse", "--is-inside-work-tree")
    return proc.returncode == 0 and proc.stdout.strip() == "true"


def _safe_label(label: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "-" for c in label)


def _open_retained_sandbox(root: Path, sandbox_dir: Path) -> SandboxHandle | None:
    if is_git_repo(root):
        if not is_git_repo(sandbox_dir):
            return None
        probe = SandboxHandle(root, sandbox_dir, None, is_git_worktree=True, reused=True)
        saved = load_sandbox_baseline(probe)
        if saved is not None and saved.base_commit is not None:
            base_commit = saved.base_commit
        else:
            head_proc = _run_git(sandbox_dir, "rev-parse", "HEAD")
            if head_proc.returncode != 0:
                return None
            base_commit = head_proc.stdout.strip()
        return SandboxHandle(
            root, sandbox_dir, base_commit, is_git_worktree=True, reused=True
        )
    return SandboxHandle(root, sandbox_dir, None, is_git_worktree=False, reused=True)


def create_sandbox(
    repo_root: Path | str,
    *,
    label: str,
    reuse: bool = False,
    sandbox_root: Path | str | None = None,
) -> SandboxHandle:
    """Create an isolated git worktree when inside a git repo, else copy workspace.

    With ``reuse=True`` a healthy retained sandbox is reopened as-is so a retried
    attempt keeps every uncommitted file from its previous turn. ``sandbox_root``
    keeps sandbox worktrees (and their baselines/tmp dirs) outside the target
    repository so headless sessions never pollute the repo with ``.olympus``.
    """
    root = Path(repo_root).resolve()
    safe_label = _safe_label(label)
    sandbox_dir = (
        Path(sandbox_root).resolve() / safe_label
        if sandbox_root is not None
        else root / ".olympus" / "sandboxes" / safe_label
    )
    if sandbox_dir.exists():
        if reuse:
            retained = _open_retained_sandbox(root, sandbox_dir)
            if retained is not None:
                return retained
        remove_sandbox(SandboxHandle(root, sandbox_dir, None, is_git_repo(root)))

    if is_git_repo(root):
        head_proc = _run_git(root, "rev-parse", "HEAD")
        if head_proc.returncode == 0:
            base_commit = head_proc.stdout.strip()
            sandbox_dir.parent.mkdir(parents=True, exist_ok=True)
            add_proc = _run_git(
                root,
                "worktree",
                "add",
                "--detach",
                str(sandbox_dir),
                base_commit,
            )
            if add_proc.returncode == 0:
                return SandboxHandle(
                    repo_root=root,
                    workspace_path=sandbox_dir,
                    base_commit=base_commit,
                    is_git_worktree=True,
                )

    # Fallback for uninitialized or unborn-HEAD repos
    sandbox_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        root,
        sandbox_dir,
        ignore=shutil.ignore_patterns(".git", ".venv", ".olympus", "__pycache__"),
    )
    return SandboxHandle(
        repo_root=root,
        workspace_path=sandbox_dir,
        base_commit=None,
        is_git_worktree=False,
    )


def save_sandbox_baseline(handle: SandboxHandle, baseline: SandboxBaseline) -> None:
    """Persist the task's Attempt-1 baseline next to its retained sandbox."""
    path = handle.baseline_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "is_git": baseline.is_git,
                "base_commit": baseline.base_commit,
                "entries": [list(e) for e in baseline.entries],
                "preexisting_ephemeral": list(baseline.preexisting_ephemeral),
            },
        ),
        encoding="utf-8",
    )


def load_sandbox_baseline(handle: SandboxHandle) -> SandboxBaseline | None:
    """Load the persisted Attempt-1 baseline; ``None`` when absent or unreadable."""
    try:
        payload = json.loads(handle.baseline_path.read_text(encoding="utf-8"))
        entries = tuple(
            (str(rel), None if fingerprint is None else str(fingerprint))
            for rel, fingerprint in payload["entries"]
        )
        raw_commit = payload.get("base_commit")
        base_commit = str(raw_commit) if isinstance(raw_commit, str) and raw_commit else None
        raw_ephemeral = payload.get("preexisting_ephemeral", [])
        preexisting_ephemeral = (
            tuple(str(rel) for rel in raw_ephemeral if isinstance(rel, str) and rel)
            if isinstance(raw_ephemeral, list)
            else ()
        )
        return SandboxBaseline(
            is_git=bool(payload["is_git"]),
            entries=entries,
            base_commit=base_commit,
            preexisting_ephemeral=preexisting_ephemeral,
        )
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def capture_sandbox_diff(
    handle: SandboxHandle, baseline: SandboxBaseline | None = None
) -> str:
    """Return tracked diff plus untracked file contents for cold-start review.

    Supplying the task's pre-worker ``baseline`` restricts the output to paths
    mutated since that snapshot, so unchanged upstream ``accepted_overlay``
    files never leak into a downstream task's diff. Without a baseline the
    legacy ``HEAD`` + untracked behavior is preserved for direct callers.
    """
    if baseline is not None:
        return _baseline_scoped_diff(handle, baseline)
    ws = handle.workspace_path
    if not is_git_repo(ws):
        return "(non-git sandbox; inspect modified files directly)"
    return _head_scoped_diff(ws)


def _head_scoped_diff(ws: Path) -> str:
    tracked = _run_git(ws, "diff", "HEAD").stdout
    untracked_proc = _run_git(ws, "ls-files", "--others", "--exclude-standard")
    parts: list[str] = []
    if tracked.strip():
        parts.append(tracked.rstrip())
    untracked = [line.strip() for line in untracked_proc.stdout.splitlines() if line.strip()]
    for rel in untracked:
        fpath = ws / rel
        if fpath.is_file():
            parts.append(f"--- /dev/null\n+++ b/{rel}\n{_read_text(fpath)}")
    return "\n\n".join(parts) if parts else "(no changes)"


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return "<binary file>"


def _prefixed_content(path: Path) -> tuple[str, int]:
    text = _read_text(path)
    lines = text.splitlines()
    if not lines:
        return "", 0
    return "\n".join(f"+{line}" for line in lines) + "\n", len(lines)


def _is_tracked_at_base(ws: Path, base_commit: str | None, rel: str) -> bool:
    if base_commit is None or not is_git_repo(ws):
        return False
    return _run_git(ws, "cat-file", "-e", f"{base_commit}:{rel}").returncode == 0


def _no_index_new_file_diff(ws: Path, rel: str) -> str:
    proc = _run_git(
        ws,
        "diff",
        "--binary",
        "--no-ext-diff",
        "--no-textconv",
        "--no-index",
        "--",
        "/dev/null",
        rel,
    )
    if proc.returncode in (0, 1) and proc.stdout.strip():
        lines = proc.stdout.rstrip().splitlines()
        if lines and lines[0].startswith("diff --git "):
            lines[0] = f"diff --git a/{rel} b/{rel}"
        return "\n".join(lines)
    body, count = _prefixed_content(ws / rel)
    hunk = f"@@ -0,0 +1,{count} @@\n{body}" if count else ""
    return (
        f"diff --git a/{rel} b/{rel}\n"
        "new file mode 100644\n"
        f"--- /dev/null\n+++ b/{rel}\n"
        f"{hunk}"
    ).rstrip()


def _new_file_block(ws: Path, rel: str) -> str:
    return _no_index_new_file_diff(ws, rel)


def _deleted_file_block(ws: Path, rel: str, base_commit: str | None) -> str:
    if base_commit is not None and is_git_repo(ws):
        tracked = _run_git(
            ws, "diff", "--binary", "--no-ext-diff", "--no-textconv", base_commit, "--", rel
        ).stdout.rstrip()
        if tracked:
            return tracked
    return (
        f"diff --git a/{rel} b/{rel}\n"
        "deleted file mode 100644\n"
        f"--- a/{rel}\n+++ /dev/null"
    )


def _overlay_file_block(ws: Path, rel: str) -> str:
    body, count = _prefixed_content(ws / rel)
    hunk = f"@@ -0,0 +1,{count} @@\n{body}" if count else ""
    return (
        f"diff --git a/{rel} b/{rel}\n"
        f"--- a/{rel}\n+++ b/{rel}\n"
        f"{hunk}"
    ).rstrip()


def _modified_file_block(ws: Path, rel: str, base_commit: str | None) -> str:
    if _is_tracked_at_base(ws, base_commit, rel):
        assert base_commit is not None
        diff_out = _run_git(
            ws, "diff", "--binary", "--no-ext-diff", "--no-textconv", base_commit, "--", rel
        ).stdout.rstrip()
        if diff_out:
            return diff_out
    return _overlay_file_block(ws, rel)


def _baseline_scoped_diff(handle: SandboxHandle, baseline: SandboxBaseline) -> str:
    ws = handle.workspace_path
    before = baseline.mapping()
    base_commit = baseline.base_commit or handle.base_commit
    changed = collect_sandbox_changes(handle, baseline)
    if not changed:
        return "(no changes)"

    def present(rel: str) -> bool:
        candidate = ws / rel
        return candidate.is_file() or candidate.is_symlink()

    added = {
        rel
        for rel in changed
        if rel not in before and not _is_tracked_at_base(ws, base_commit, rel) and present(rel)
    }
    deleted = {
        rel
        for rel in changed
        if (rel in before or _is_tracked_at_base(ws, base_commit, rel)) and not present(rel)
    }
    modified = set(changed) - added - deleted

    renames: dict[str, str] = {}
    for old in sorted(deleted):
        fingerprint = before.get(old)
        if fingerprint is None:
            continue
        match = next(
            (new for new in sorted(added) if _file_fingerprint(ws / new) == fingerprint),
            None,
        )
        if match is not None:
            renames[old] = match
            added.discard(match)

    blocks: list[str] = []
    for rel in sorted(changed):
        if rel in renames:
            new = renames[rel]
            blocks.append(
                f"diff --git a/{rel} b/{new}\n"
                "similarity index 100%\n"
                f"rename from {rel}\nrename to {new}"
            )
        elif rel in renames.values():
            continue
        elif rel in added:
            blocks.append(_new_file_block(ws, rel))
        elif rel in deleted:
            blocks.append(_deleted_file_block(ws, rel, base_commit))
        elif rel in modified:
            blocks.append(_modified_file_block(ws, rel, base_commit))
    return "\n\n".join(blocks) if blocks else "(no changes)"


def archive_sandbox_recovery(handle: SandboxHandle, attempt_dir: Path) -> None:
    """Record recoverable binary and text changes from a paused attempt."""
    patch = attempt_dir / "diff.patch"
    if handle.base_commit is None:
        patch.write_text("Non-git workspace preserved in sandbox/\n", encoding="utf-8")
        shutil.copytree(handle.workspace_path, attempt_dir / "sandbox", dirs_exist_ok=True)
        return

    ws = handle.workspace_path
    diff = ["git", "diff", "--binary", "--no-ext-diff", "--no-textconv"]
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        cwd=ws, capture_output=True, check=True,
    ).stdout
    with patch.open("wb") as output:
        subprocess.run([*diff, handle.base_commit, "--"], cwd=ws, stdout=output, check=True)
        for raw_path in untracked.split(b"\0"):
            if not raw_path:
                continue
            result = subprocess.run(
                [*diff, "--no-index", "--", "/dev/null", os.fsdecode(raw_path)],
                cwd=ws, stdout=output, check=False,
            )
            if result.returncode != 1:
                result.check_returncode()


def _is_ignored_rel(rel: str) -> bool:
    return any(part in IGNORED_DIR_NAMES for part in PurePosixPath(rel).parts)


def _file_fingerprint(path: Path) -> str | None:
    try:
        info = os.lstat(path)
    except OSError:
        return None
    if stat.S_ISLNK(info.st_mode):
        try:
            return f"link:{os.readlink(path)}"
        except OSError:
            return None
    if not stat.S_ISREG(info.st_mode):
        return None
    digest = hashlib.sha256()
    try:
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                digest.update(chunk)
    except OSError:
        return None
    return f"{info.st_mode & 0o777:o}:{digest.hexdigest()}"


def _workspace_fingerprints(root: Path) -> dict[str, str | None]:
    fingerprints: dict[str, str | None] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in IGNORED_DIR_NAMES)
        for name in sorted(filenames):
            candidate = Path(dirpath) / name
            rel = candidate.relative_to(root).as_posix()
            if _is_ignored_rel(rel):
                continue
            fingerprints[rel] = _file_fingerprint(candidate)
    return fingerprints


def _git_changed_fingerprints(ws: Path, base_commit: str) -> dict[str, str | None]:
    proc = _run_git(ws, "diff", "--name-status", "-z", "-M", base_commit)
    if proc.returncode != 0:
        raise RuntimeError(
            f"git diff against {base_commit} failed: {proc.stderr.strip()[:200]}"
        )
    entries: dict[str, str | None] = {}
    tokens = proc.stdout.split("\0")
    index = 0
    while index < len(tokens):
        status = tokens[index]
        index += 1
        if not status:
            continue
        if status[0] in ("R", "C"):
            old_path, new_path = tokens[index], tokens[index + 1]
            index += 2
            if status[0] == "R" and old_path:
                # The rename deletes the old path; the deletion is itself a mutation.
                entries[old_path] = _file_fingerprint(ws / old_path)
            if new_path:
                entries[new_path] = _file_fingerprint(ws / new_path)
            continue
        rel = tokens[index]
        index += 1
        if rel:
            entries[rel] = _file_fingerprint(ws / rel)

    untracked = _run_git(ws, "ls-files", "--others", "--exclude-standard", "-z")
    if untracked.returncode != 0:
        raise RuntimeError(
            f"git ls-files failed: {untracked.stderr.strip()[:200]}"
        )
    for rel in untracked.stdout.split("\0"):
        if rel and not _is_ignored_rel(rel):
            entries[rel] = _file_fingerprint(ws / rel)
    return entries


def snapshot_sandbox_state(handle: SandboxHandle) -> SandboxBaseline:
    """Capture the sandbox state before a worker runs so later mutations are attributable."""
    ws = handle.workspace_path
    if handle.is_git_worktree and handle.base_commit is not None and is_git_repo(ws):
        try:
            entries = _git_changed_fingerprints(ws, handle.base_commit)
        except RuntimeError:
            entries = _workspace_fingerprints(ws)
            base = SandboxBaseline(is_git=False, entries=tuple(sorted(entries.items())))
            ephemeral = tuple(
                sorted(
                    rel.replace("\\", "/")
                    for rel in _untracked_paths(handle, base)
                    if is_ephemeral_artifact(rel)
                )
            )
            return SandboxBaseline(
                is_git=False,
                entries=base.entries,
                preexisting_ephemeral=ephemeral,
            )
        base = SandboxBaseline(
            is_git=True,
            entries=tuple(sorted(entries.items())),
            base_commit=handle.base_commit,
        )
        ephemeral = tuple(
            sorted(
                rel.replace("\\", "/")
                for rel in _untracked_paths(handle, base)
                if is_ephemeral_artifact(rel)
            )
        )
        return SandboxBaseline(
            is_git=True,
            entries=base.entries,
            base_commit=handle.base_commit,
            preexisting_ephemeral=ephemeral,
        )
    base = SandboxBaseline(
        is_git=False, entries=tuple(sorted(_workspace_fingerprints(ws).items()))
    )
    ephemeral = tuple(
        sorted(
            rel.replace("\\", "/")
            for rel in _untracked_paths(handle, base)
            if is_ephemeral_artifact(rel)
        )
    )
    return SandboxBaseline(
        is_git=False,
        entries=base.entries,
        preexisting_ephemeral=ephemeral,
    )


def _current_fingerprints(
    handle: SandboxHandle, baseline: SandboxBaseline | None = None
) -> dict[str, str | None]:
    ws = handle.workspace_path
    use_git = baseline.is_git if baseline is not None else handle.is_git_worktree
    base_commit = (
        baseline.base_commit
        if baseline is not None and baseline.base_commit is not None
        else handle.base_commit
    )
    if use_git and base_commit is not None and is_git_repo(ws):
        try:
            return _git_changed_fingerprints(ws, base_commit)
        except RuntimeError:
            # A tampered sandbox (e.g. deleted .git) fails closed: every file looks changed.
            return _workspace_fingerprints(ws)
    return _workspace_fingerprints(ws)


def collect_sandbox_changes(handle: SandboxHandle, baseline: SandboxBaseline) -> list[str]:
    """Return repo-relative paths mutated since ``baseline`` (adds, edits, deletes, renames)."""
    current = _current_fingerprints(handle, baseline)
    before = baseline.mapping()
    changed = {
        rel
        for rel, fingerprint in current.items()
        if rel not in before or before[rel] != fingerprint
    }
    changed.update(rel for rel in before if rel not in current)
    return sorted(changed)


def is_ephemeral_artifact(rel: str) -> bool:
    """Return True when ``rel`` matches a conservative temporary/backup junk pattern."""
    parts = PurePosixPath(rel.replace("\\", "/")).parts
    if not parts:
        return False
    name = parts[-1]
    for pattern in EPHEMERAL_ARTIFACT_PATTERNS:
        if pattern.startswith("**/") and pattern.endswith("/**"):
            segment = pattern[3:-3]
            if segment and segment in parts[:-1]:
                return True
        elif fnmatchcase(name, pattern):
            return True
    return False


_EPHEMERAL_WALK_SKIP = frozenset({".git", ".venv", ".olympus", "node_modules"})


def _untracked_paths(handle: SandboxHandle, baseline: SandboxBaseline) -> list[str]:
    """List untracked files, including ignored dirs such as ``__pycache__``."""
    ws = handle.workspace_path
    if baseline.is_git and is_git_repo(ws):
        proc = _run_git(ws, "ls-files", "--others", "-z")
        if proc.returncode == 0:
            return [rel for rel in proc.stdout.split("\0") if rel]
    paths: list[str] = []
    for dirpath, dirnames, filenames in os.walk(ws):
        dirnames[:] = sorted(d for d in dirnames if d not in _EPHEMERAL_WALK_SKIP)
        for name in sorted(filenames):
            rel = (Path(dirpath) / name).relative_to(ws).as_posix()
            paths.append(rel)
    return paths


def clean_ephemeral_artifacts(
    handle: SandboxHandle, baseline: SandboxBaseline
) -> list[str]:
    """Delete newly created untracked junk files and return their sorted relative paths.

    Only files absent from the pre-worker ``baseline`` (including gitignored pre-existing
    ephemeral candidates) and the git ``base_commit`` are considered, so tracked files and
    pre-existing sandbox files are never removed.
    """
    before = baseline.mapping()
    preexisting_ephemeral = set(baseline.preexisting_ephemeral)
    base_commit = baseline.base_commit or handle.base_commit
    ws = handle.workspace_path
    removed: list[str] = []
    for raw_rel in sorted(_untracked_paths(handle, baseline)):
        rel = raw_rel.replace("\\", "/")
        if rel in before or rel in preexisting_ephemeral:
            continue
        if _is_tracked_at_base(ws, base_commit, rel):
            continue
        if not is_ephemeral_artifact(rel):
            continue
        target = ws / rel
        try:
            if target.is_symlink() or target.is_file():
                target.unlink()
            else:
                continue
        except OSError:
            continue
        removed.append(rel)
    return removed


def sync_upstream_files(
    handle: SandboxHandle,
    overlay_dir: Path,
    baseline: SandboxBaseline,
    *,
    write_scope: Sequence[str],
    deleted_paths: Sequence[str] = (),
) -> SandboxBaseline:
    """Refresh newly accepted upstream files/deletions without clobbering in-progress work.

    Only paths the worker left exactly at their baseline content are synced. Worker
    mutations outside ``write_scope`` stay untouched so the Diff Gate still audits them,
    and the returned baseline records fingerprints for the synced upstream paths only.
    """
    if not overlay_dir.is_dir() and not deleted_paths:
        return baseline
    policy = WorkspacePolicy(allowed_write_scope=list(write_scope))
    merged = baseline.mapping()
    current_map = _current_fingerprints(handle, baseline)
    base_commit = baseline.base_commit or handle.base_commit
    ws = handle.workspace_path
    synced_rels: list[str] = []
    if overlay_dir.is_dir():
        for src in sorted(overlay_dir.rglob("*")):
            if not src.is_file():
                continue
            rel = src.relative_to(overlay_dir).as_posix()
            if any(
                part in IGNORED_DIR_NAMES or part == ".olympus"
                for part in PurePosixPath(rel).parts
            ):
                continue
            if policy.allows_relative_path(rel):
                continue
            if rel in current_map and current_map[rel] != merged.get(rel):
                continue
            if rel in merged and rel not in current_map:
                continue
            current = _file_fingerprint(ws / rel)
            if rel in merged:
                if current != merged[rel]:
                    continue
            elif (current is not None) != _is_tracked_at_base(ws, base_commit, rel):
                continue
            dst = ws / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            synced_rels.append(rel)
    for rel in sorted(set(deleted_paths)):
        if any(
            part in IGNORED_DIR_NAMES or part == ".olympus"
            for part in PurePosixPath(rel).parts
        ):
            continue
        if policy.allows_relative_path(rel):
            continue
        if rel in current_map and current_map[rel] != merged.get(rel):
            continue
        if rel in merged and rel not in current_map:
            continue
        current = _file_fingerprint(ws / rel)
        if current is None:
            continue
        if rel in merged:
            if current != merged[rel]:
                continue
        elif not _is_tracked_at_base(ws, base_commit, rel):
            continue
        target = ws / rel
        if target.is_symlink() or target.is_file():
            target.unlink()
            synced_rels.append(rel)
    if synced_rels:
        updated_map = _current_fingerprints(handle, baseline)
        for rel in synced_rels:
            if rel in updated_map:
                merged[rel] = updated_map[rel]
            else:
                merged.pop(rel, None)
    return SandboxBaseline(
        is_git=baseline.is_git,
        entries=tuple(sorted(merged.items())),
        base_commit=baseline.base_commit,
        preexisting_ephemeral=baseline.preexisting_ephemeral,
    )


def absorb_verification_artifacts(
    handle: SandboxHandle,
    baseline: SandboxBaseline,
    *,
    write_scope: Sequence[str],
) -> SandboxBaseline:
    """Fold newly created untracked out-of-scope check-command side effects into the baseline.

    Tracked or pre-existing baseline files modified or deleted by a check command are never
    absorbed so tampered inputs remain visible to the Diff Gate on retry.
    """
    policy = WorkspacePolicy(allowed_write_scope=list(write_scope))
    current = _current_fingerprints(handle, baseline)
    merged = baseline.mapping()
    base_commit = baseline.base_commit or handle.base_commit
    ws = handle.workspace_path
    for rel, fingerprint in current.items():
        if policy.allows_relative_path(rel):
            continue
        if rel in merged or _is_tracked_at_base(ws, base_commit, rel):
            continue
        if fingerprint is not None:
            merged[rel] = fingerprint
    return SandboxBaseline(
        is_git=baseline.is_git,
        entries=tuple(sorted(merged.items())),
        base_commit=baseline.base_commit,
        preexisting_ephemeral=baseline.preexisting_ephemeral,
    )


def remove_sandbox(handle: SandboxHandle) -> None:
    """Clean up a worktree or copied sandbox directory plus its temp dir and baseline."""
    if handle.is_git_worktree and is_git_repo(handle.repo_root):
        _run_git(handle.repo_root, "worktree", "remove", "--force", str(handle.workspace_path))
    if handle.workspace_path.exists():
        shutil.rmtree(handle.workspace_path, ignore_errors=True)
    shutil.rmtree(handle.tmp_dir, ignore_errors=True)
    handle.baseline_path.unlink(missing_ok=True)


def remove_run_sandboxes(
    repo_root: Path | str,
    run_id: str,
    *,
    sandbox_root: Path | str | None = None,
) -> None:
    """Reclaim every retained sandbox, baseline, and temp dir belonging to ``run_id``."""
    root = Path(repo_root).resolve()
    prefix = f"{_safe_label(run_id)}-"
    git_host = is_git_repo(root)
    if sandbox_root is not None:
        sandboxes = Path(sandbox_root).resolve()
        external = not sandboxes.is_relative_to(root)
        baselines = (
            sandboxes / ".sandbox_baselines"
            if external
            else root / ".olympus" / "sandbox_baselines"
        )
        tmp_root = sandboxes / ".sandbox_tmp" if external else root / ".olympus" / "sandbox_tmp"
    else:
        sandboxes = root / ".olympus" / "sandboxes"
        baselines = root / ".olympus" / "sandbox_baselines"
        tmp_root = root / ".olympus" / "sandbox_tmp"
    if sandboxes.is_dir():
        for entry in sorted(sandboxes.iterdir()):
            if entry.is_dir() and entry.name.startswith(prefix):
                if git_host:
                    _run_git(root, "worktree", "remove", "--force", str(entry))
                shutil.rmtree(entry, ignore_errors=True)
    if baselines.is_dir():
        for entry in sorted(baselines.iterdir()):
            if entry.name.startswith(prefix):
                entry.unlink(missing_ok=True)
    if tmp_root.is_dir():
        for entry in sorted(tmp_root.iterdir()):
            if entry.is_dir() and entry.name.startswith(prefix):
                shutil.rmtree(entry, ignore_errors=True)
