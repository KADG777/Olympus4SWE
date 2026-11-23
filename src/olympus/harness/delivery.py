"""Run approved criterion checks against isolated copies of the composed delivery."""

from __future__ import annotations

import shutil
import stat
import subprocess
from hashlib import file_digest, sha256
from pathlib import Path
from typing import Any

from olympus.core.contracts import TaskPlan, new_id
from olympus.core.fingerprint import content_hash
from olympus.harness.worktree import build_sandbox_env, create_sandbox, remove_sandbox

DELIVERY_EVENT = "delivery.checks_completed"
_INTERNAL_DIRS = {".git", ".olympus"}


def _delivery_version(workspace: Path, base_commit: str | None) -> str:
    entries: dict[str, tuple[int, str]] = {}

    def raise_walk_error(error: OSError) -> None:
        raise error

    for root, dirs, files in workspace.walk(on_error=raise_walk_error):
        if root == workspace:
            dirs[:] = [name for name in dirs if name not in _INTERNAL_DIRS]
            files = [name for name in files if name not in _INTERNAL_DIRS]
        for name in dirs + files:
            path = root / name
            mode = path.lstat().st_mode
            value = ""
            if stat.S_ISLNK(mode):
                target = path.readlink()
                resolved = path.resolve().relative_to(workspace)
                if target.is_absolute() or (resolved.parts and resolved.parts[0] in _INTERNAL_DIRS):
                    raise ValueError(f"Cannot bind an external link: {path.relative_to(workspace)}")
                value = target.as_posix()
            elif stat.S_ISREG(mode):
                with path.open("rb") as file:
                    value = file_digest(file, "sha256").hexdigest()
            elif not stat.S_ISDIR(mode):
                raise ValueError(f"Unsupported delivery entry: {path.relative_to(workspace)}")
            entries[path.relative_to(workspace).as_posix()] = (mode, value)
    return content_hash({"base_commit": base_commit, "files": entries})


def run_delivery_checks(
    plan: TaskPlan,
    *,
    workspace_root: Path,
    overlay_dir: Path,
    run_dir: Path,
    sandbox_root: Path | str | None = None,
) -> dict[str, Any]:
    """Return a receipt tied to the checked inputs and saved per-check logs."""
    receipt_id = new_id("delivery")
    receipt_dir = run_dir / "artifacts" / receipt_id
    delivery_version = None
    results = []
    for position, check in enumerate(plan.delivery_checks, start=1):
        result: dict[str, Any] = {
            "requirement_id": check.requirement_id,
            "criterion_index": check.criterion_index,
        }
        sandbox = None
        failure_reason = "无法建立隔离验收环境"
        try:
            sandbox = create_sandbox(
                workspace_root,
                label=f"{receipt_id}-{position}",
                sandbox_root=sandbox_root,
            )
            workspace = sandbox.workspace_path
            failure_reason = "无法核验交付快照"
            if overlay_dir.is_dir():
                shutil.copytree(overlay_dir, workspace, dirs_exist_ok=True)
            version = _delivery_version(workspace, sandbox.base_commit)
            if delivery_version is None:
                delivery_version = version
            if version != delivery_version:
                result["reason"] = "本次检查的输入与交付版本不一致"
            else:
                failure_reason = "无法运行验收命令"
                proc = subprocess.run(
                    ["bash", "-lc", check.command],
                    cwd=workspace,
                    env=build_sandbox_env(workspace, host_root=workspace_root),
                    capture_output=True,
                    text=True,
                    errors="replace",
                    timeout=120,
                    check=False,
                )
                if proc.returncode < 0 or proc.returncode >= 126:
                    result["reason"] = f"验收命令未正常完成（退出码 {proc.returncode}）"
                else:
                    failure_reason = "无法保存可核验的执行日志"
                    log = (
                        f"command: {check.command}\nexit_code: {proc.returncode}\n"
                        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}\n"
                    ).encode()
                    receipt_dir.mkdir(parents=True, exist_ok=True)
                    with (receipt_dir / f"check-{position}.log").open("xb") as log_file:
                        log_file.write(log)
                    result.update(
                        delivery_version=version,
                        exit_code=proc.returncode,
                        log_hash=sha256(log).hexdigest(),
                    )
        except subprocess.TimeoutExpired:
            result["reason"] = "验收命令超时，未取得可用结果"
        except (OSError, ValueError, RuntimeError):
            result["reason"] = failure_reason
        finally:
            if sandbox is not None:
                remove_sandbox(sandbox)
        results.append(result)
    return {
        "receipt_id": receipt_id,
        "plan_hash": plan.content_hash,
        "spec_hash": plan.spec_hash,
        "delivery_version": delivery_version,
        "results": results,
    }
