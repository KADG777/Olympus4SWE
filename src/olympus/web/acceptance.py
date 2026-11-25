"""Project approved GoalSpec criteria from the current delivery's recorded checks."""

from __future__ import annotations

import os
import re
import stat
from collections import Counter, defaultdict
from hashlib import file_digest, sha256
from pathlib import Path
from typing import Any

from olympus.core.contracts import GoalSpec, RunEvent, TaskPlan
from olympus.harness.delivery import DELIVERY_EVENT

_VERSION = re.compile(r"sha256:[a-f0-9]{64}")
_RECEIPT_ID = re.compile(r"delivery-[a-f0-9]{12}")

_PRODUCT_ARTIFACTS: dict[str, tuple[str, str]] = {
    "report": ("report.json", "application/json; charset=utf-8"),
    "boot_log": ("boot.log", "text/plain; charset=utf-8"),
    "screenshot": ("home.png", "image/png"),
}


def _valid_segment(part: str) -> bool:
    return bool(part) and part not in (".", "..") and "/" not in part and "\\" not in part


def _open_safe_regular_fd(trusted_root: Path, *parts: str) -> int | None:
    """Walk ``trusted_root / *parts`` via dirfd + ``O_NOFOLLOW | O_NONBLOCK``."""
    if not parts or not all(_valid_segment(part) for part in parts):
        return None
    o_directory = getattr(os, "O_DIRECTORY", 0)
    o_nofollow = getattr(os, "O_NOFOLLOW", 0)
    if os.open not in os.supports_dir_fd or not o_directory or not o_nofollow:
        return None
    dir_flags = os.O_RDONLY | o_directory | o_nofollow
    file_flags = os.O_RDONLY | o_nofollow | getattr(os, "O_NONBLOCK", 0)
    try:
        root_resolved = trusted_root.resolve()
        if not root_resolved.is_dir():
            return None
        cur_fd = os.open(str(root_resolved), os.O_RDONLY | o_directory)
        try:
            for directory in parts[:-1]:
                nxt_fd = os.open(directory, dir_flags, dir_fd=cur_fd)
                os.close(cur_fd)
                cur_fd = nxt_fd
            fd = os.open(parts[-1], file_flags, dir_fd=cur_fd)
        finally:
            os.close(cur_fd)
    except OSError:
        return None
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            os.close(fd)
            return None
        return fd
    except OSError:
        os.close(fd)
        return None


def open_safe_artifact_bytes(trusted_root: Path, *parts: str) -> bytes | None:
    """Read one regular file strictly under ``trusted_root`` without following symlinks."""
    fd = _open_safe_regular_fd(trusted_root, *parts)
    if fd is None:
        return None
    try:
        with os.fdopen(fd, "rb") as handle:
            fd = -1
            return handle.read()
    except OSError:
        return None
    finally:
        if fd >= 0:
            os.close(fd)


def _safe_artifact_exists(trusted_root: Path, *parts: str) -> bool:
    fd = _open_safe_regular_fd(trusted_root, *parts)
    if fd is None:
        return False
    os.close(fd)
    return True


def read_product_artifact(
    run_dir: Path, kind: str, *, artifacts_root: Path | None = None
) -> tuple[bytes, str] | None:
    """Read one allowlisted product verification artifact under ``artifacts_root``."""
    entry = _PRODUCT_ARTIFACTS.get(kind)
    if entry is None:
        return None
    filename, media_type = entry
    root = artifacts_root if artifacts_root is not None else run_dir.parent
    content = open_safe_artifact_bytes(root, run_dir.name, "artifacts", "verify_product", filename)
    if content is None:
        return None
    return content, media_type


def latest_product_verification(
    events: list[RunEvent], run_dir: Path, *, artifacts_root: Path | None = None
) -> dict[str, Any] | None:
    """Whitelist the latest ``product.verified`` payload into a safe view."""
    latest: RunEvent | None = None
    for event in events:
        if event.event_type == "product.verified":
            latest = event
    if latest is None:
        return None
    payload = latest.payload
    passed = payload.get("passed")
    if not isinstance(passed, bool):
        return None
    raw_summary = payload.get("summary")
    summary = raw_summary if isinstance(raw_summary, str) else ""
    failures: list[str] = []
    raw_failures = payload.get("failures")
    if isinstance(raw_failures, list):
        failures = [item for item in raw_failures if isinstance(item, str)]
    checks: list[dict[str, Any]] = []
    raw_checks = payload.get("checks")
    if isinstance(raw_checks, list):
        for item in raw_checks:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            status = item.get("status")
            if not isinstance(name, str) or not isinstance(status, str):
                continue
            raw_detail = item.get("detail")
            raw_port = item.get("port")
            checks.append(
                {
                    "name": name,
                    "status": status,
                    "detail": raw_detail if isinstance(raw_detail, str) else "",
                    "port": (
                        raw_port
                        if isinstance(raw_port, int) and not isinstance(raw_port, bool)
                        else None
                    ),
                }
            )
    root = artifacts_root if artifacts_root is not None else run_dir.parent
    run_name = run_dir.name
    screenshot_declared_absent = (
        "screenshot_path" in payload
        and not (
            isinstance(payload.get("screenshot_path"), str)
            and bool(payload["screenshot_path"].strip())
        )
    ) or any(
        check["name"] == "screenshot" and check["status"] != "passed" for check in checks
    )
    return {
        "passed": passed,
        "summary": summary,
        "failures": failures,
        "event_seq": latest.event_seq,
        "verified_at": latest.created_at.isoformat(),
        "checks": checks,
        "has_report": _safe_artifact_exists(
            root, run_name, "artifacts", "verify_product", "report.json"
        ),
        "has_boot_log": _safe_artifact_exists(
            root, run_name, "artifacts", "verify_product", "boot.log"
        ),
        "has_screenshot": (
            False
            if screenshot_declared_absent
            else _safe_artifact_exists(
                root, run_name, "artifacts", "verify_product", "home.png"
            )
        ),
    }


def _current_receipt(events: list[RunEvent], plan: TaskPlan) -> RunEvent | None:
    for event in reversed(events):
        if event.event_type in {"plan.saved", "task.leased", "task.retried_by_human"}:
            return None
        if event.event_type == DELIVERY_EVENT and event.actor == "chronos":
            receipt = event.payload
            if (
                receipt.get("plan_hash") == plan.content_hash
                and receipt.get("spec_hash") == plan.spec_hash
            ):
                return event
            return None
    return None


def _evidence_log_path(run_dir: Path, receipt_id: str, position: int) -> Path:
    path = run_dir.resolve() / "artifacts" / receipt_id / f"check-{position}.log"
    if path.resolve() != path:
        raise ValueError("Delivery evidence log cannot be a symlink")
    return path


def read_delivery_log(run_dir: Path, evidence: dict[str, Any]) -> bytes | None:
    """Read the complete log for an evidence reference from the current projection."""
    try:
        path = _evidence_log_path(run_dir, evidence["receipt_id"], evidence["check_position"])
        log = path.read_bytes()
        return log if sha256(log).hexdigest() == evidence["log_hash"] else None
    except (OSError, ValueError, RuntimeError):
        return None


def _check_evidence(
    run_dir: Path, event: RunEvent, position: int, key: tuple[str, int]
) -> tuple[dict[str, Any] | None, str]:
    receipt = event.payload
    version = receipt.get("delivery_version")
    receipt_id = receipt.get("receipt_id")
    results = receipt.get("results")
    if not isinstance(results, list) or len(results) < position:
        return None, "缺少此条件的实际执行记录"
    result = results[position - 1]
    if (
        not isinstance(result, dict)
        or (result.get("requirement_id"), result.get("criterion_index")) != key
    ):
        return None, "执行记录无法定位到此条件"
    raw_reason = result.get("reason")
    if isinstance(raw_reason, str) and raw_reason:
        return None, raw_reason
    if (
        not isinstance(version, str)
        or not _VERSION.fullmatch(version)
        or result.get("delivery_version") != version
    ):
        return None, "无法确认执行记录对应当前交付版本"
    exit_code = result.get("exit_code")
    if (
        not isinstance(receipt_id, str)
        or not _RECEIPT_ID.fullmatch(receipt_id)
        or type(exit_code) is not int
        or not 0 <= exit_code < 126
    ):
        return None, "缺少可核验的验收日志或退出码"

    try:
        log_path = _evidence_log_path(run_dir, receipt_id, position)
        with log_path.open("rb") as log_file:
            if file_digest(log_file, "sha256").hexdigest() != result.get("log_hash"):
                return None, "验收日志与记录的指纹不一致"
            log_file.seek(0)
            log = log_file.read(8001)
    except (OSError, ValueError, RuntimeError):
        return None, "验收日志缺失或不可读"
    return {
        "delivery_version": version,
        "event_seq": event.event_seq,
        "receipt_id": receipt_id,
        "check_position": position,
        "log_hash": result["log_hash"],
        "exit_code": exit_code,
        "log": log[:8000].decode("utf-8", errors="replace") + ("\n…" if len(log) > 8000 else ""),
    }, "" if exit_code == 0 else f"验收命令退出码 {exit_code}，详见执行日志"


def build_delivery_acceptance(
    spec: GoalSpec | None,
    plan: TaskPlan | None,
    events: list[RunEvent],
    run_dir: Path,
    *,
    artifacts_root: Path | None = None,
) -> dict[str, Any] | None:
    if spec is None:
        return None

    counts = {"passed": 0, "failed": 0, "unverified": 0}
    rows = []
    mappings: dict[tuple[str, int], list[int]] = defaultdict(list)
    risks: dict[str, list[str]] = defaultdict(list)
    receipt = None
    if plan is not None and (plan.spec_id, plan.spec_hash) == (spec.spec_id, spec.content_hash):
        receipt = _current_receipt(events, plan)
        for position, check in enumerate(plan.delivery_checks, start=1):
            mappings[check.requirement_id, check.criterion_index].append(position)
        for task in plan.tasks:
            if task.accepted_risk_reason:
                for requirement_id in task.requirement_refs:
                    risks[requirement_id].append(f"{task.task_id}：{task.accepted_risk_reason}")

    requirement_counts = Counter(req.id for req in spec.requirements)
    for req in spec.requirements:
        for index, text in enumerate(req.acceptance_criteria, start=1):
            key = (req.id, index)
            status = "unverified"
            evidence = None
            positions = mappings[key]
            risk_reasons = list(dict.fromkeys(risks[req.id]))
            if risk_reasons:
                reason = "人工接受风险：" + "；".join(risk_reasons)
            elif requirement_counts[req.id] != 1 or len(positions) != 1:
                reason = "缺少此条件与交付验收检查的可靠映射"
            elif receipt is None:
                reason = "缺少与当前交付版本对应的实际执行记录"
            else:
                evidence, reason = _check_evidence(run_dir, receipt, positions[0], key)
                if evidence:
                    status = "passed" if evidence["exit_code"] == 0 else "failed"

            counts[status] += 1
            rows.append(
                {
                    "requirement_id": req.id,
                    "criterion_index": index,
                    "text": text,
                    "status": status,
                    "reason": reason,
                    "risk_accepted": bool(risk_reasons),
                    "evidence": evidence,
                }
            )

    return {
        "spec_id": spec.spec_id,
        "spec_version": spec.version,
        "spec_hash": spec.content_hash,
        "counts": counts,
        "criteria": rows,
        "product_verification": latest_product_verification(
            events, run_dir, artifacts_root=artifacts_root
        ),
    }
