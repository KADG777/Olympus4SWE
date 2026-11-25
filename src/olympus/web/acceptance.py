"""Project approved GoalSpec criteria from the current delivery's recorded checks."""

from __future__ import annotations

import re
from collections import Counter, defaultdict, deque
from hashlib import file_digest, sha256
from pathlib import Path
from typing import Any

from olympus.core.contracts import GoalSpec, RunEvent, TaskPlan
from olympus.harness.delivery import DELIVERY_EVENT
from olympus.stages.verify_product import PRODUCT_VERIFIED_EVENT

_VERSION = re.compile(r"sha256:[a-f0-9]{64}")
_RECEIPT_ID = re.compile(r"delivery-[a-f0-9]{12}")


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
    }


def build_post_dag_verification(events: list[RunEvent]) -> dict[str, Any] | None:
    """Keep the current plan's recent checks without reading mutable sandbox files."""
    product_passed = None
    last_recheck = None
    attempts: deque[dict[str, Any]] = deque(maxlen=100)
    for event in events:
        payload = event.payload
        if event.event_type == "plan.saved":
            product_passed = None
            last_recheck = None
            attempts.clear()
            continue
        if event.event_type == PRODUCT_VERIFIED_EVENT:
            product_passed = payload.get("passed")
        if event.event_type == "post_dag.repro_rechecked":
            last_recheck = {
                "checked_count": payload["checked_count"],
                "still_failing_count": payload["still_failing_count"],
            }
            round_number = payload["repair_round"] + 1
        elif event.event_type == "review.verdict_recorded" and payload.get("mode") == "post_dag":
            round_number = payload["review_cycle"]
        else:
            continue
        for check in payload.get("checks", []):
            attempts.append({
                **check,
                "event_seq": event.event_seq,
                "round": round_number,
                "candidate": payload.get("candidate"),
            })
    if product_passed is None and last_recheck is None and not attempts:
        return None
    return {
        "product_passed": product_passed,
        "last_recheck": last_recheck,
        "attempts": list(attempts),
    }
