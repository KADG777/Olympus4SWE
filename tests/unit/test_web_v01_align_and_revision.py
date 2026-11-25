"""v0.1 Stage 1 resynthesis fidelity plus explicit revision/reopen commands."""

from __future__ import annotations

import json
import subprocess
import threading
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from olympus.core.contracts import (
    AcceptanceCriterion,
    AmbiguityItem,
    DeliverableKind,
    EarsPattern,
    GoalSpec,
    RequirementItem,
    RunStatus,
    SpecStatus,
)
from olympus.core.store import RunStore
from olympus.providers.base import ChatMessage, LLMResponse, MockLLMProvider, ToolSpec
from olympus.stages.align import clarification_note_states, has_pending_resynthesis
from olympus.web.server import AriadneApp, create_fastapi_app
from olympus.web.supervisor import RunSupervisor

Responder = Callable[[Sequence[ChatMessage], Sequence[ToolSpec]], LLMResponse]


def _init_git_repo(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "README.md").write_text("# orders\n", encoding="utf-8")
    subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "dev@example.com"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "Dev"], cwd=root, check=True)
    subprocess.run(["git", "add", "README.md"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=root, check=True, capture_output=True)


def _draft_spec() -> GoalSpec:
    return GoalSpec(
        spec_id="spec-orders",
        version=1,
        goal="构建运营可自助使用的订单导出服务",
        deliverable_kind=DeliverableKind.WEB_APP,
        observable_surfaces=["HTTP API", "Web 控制台"],
        actors=["运营"],
        test_seams=["POST /orders/export"],
        requirements=[
            RequirementItem(
                id="R1",
                actor="运营",
                description="运营可按日期范围导出订单明细。",
                ears_criteria=[
                    AcceptanceCriterion(
                        id="AC-1",
                        pattern=EarsPattern.EVENT_DRIVEN,
                        condition="运营提交导出请求",
                        outcome="在 5 秒内生成文件并返回下载链接",
                    )
                ],
            )
        ],
        constraints=["使用服务端数据库持久化"],
        out_of_scope=["不导出用户隐私字段"],
        open_ambiguities=[
            AmbiguityItem(
                id="A1-format",
                question="导出文件格式？",
                impact="决定导出实现",
                options=["CSV", "Excel"],
                default_assumption="CSV",
                blocking=True,
            ),
            AmbiguityItem(
                id="A2-scope",
                question="导出范围如何界定？",
                impact="决定查询边界",
                options=["当前筛选结果", "全量订单"],
                default_assumption="当前筛选结果",
                blocking=True,
            ),
        ],
        status=SpecStatus.AWAITING_HUMAN,
    )


def _fused_spec(draft: GoalSpec, answers: dict[str, str], notes: list[str]) -> GoalSpec:
    """Deterministically fuse every human answer and note into the spec body."""
    human_lines = [value for value in answers.values() if value.strip()]
    human_lines.extend(note for note in notes if note.strip())
    return draft.model_copy(
        update={
            "requirements": [
                draft.requirements[0].model_copy(
                    update={
                        "description": " ".join(
                            [draft.requirements[0].description, *human_lines]
                        ),
                        "ears_criteria": [
                            AcceptanceCriterion(
                                id="AC-1",
                                pattern=EarsPattern.UBIQUITOUS,
                                outcome="导出文件严格遵循运营确认的格式与范围",
                            )
                        ],
                    }
                )
            ],
            "constraints": [*draft.constraints, *human_lines],
        }
    )


def _design_json(spec: GoalSpec) -> str:
    return json.dumps(
        {
            "architecture_summary": "FastAPI 导出服务与 Web 控制台。",
            "spec_id": spec.spec_id,
            "spec_hash": spec.content_hash,
            "shared_invariants": [
                {
                    "id": "INV-1",
                    "category": "validation",
                    "rule": "导出必须校验日期范围并拒绝空范围。",
                    "enforcement_hint": "ExportService 参数守卫",
                    "applies_to": ["M1"],
                }
            ],
            "shared_contract_paths": [],
            "modules": [
                {
                    "id": "M1",
                    "name": "ExportService",
                    "responsibility": "按筛选条件生成导出文件。",
                    "requirement_refs": ["R1"],
                    "target_paths": ["app.py"],
                }
            ],
            "data_models": [],
            "interfaces": [],
            "execution_strategy": "serial_handoff",
            "strategy_rationale": "单模块保持交付原子性。",
        }
    )


def _plan_json(spec: GoalSpec, design: dict[str, object]) -> str:
    return json.dumps(
        {
            "plan_id": "plan-orders",
            "spec_id": spec.spec_id,
            "spec_hash": spec.content_hash,
            "design_id": design["design_id"],
            "design_hash": design["content_hash"],
            "version": 1,
            "tasks": [
                {
                    "task_id": "T1",
                    "title": "实现订单导出",
                    "objective": "按确认的格式与范围导出订单。",
                    "requirement_refs": ["R1"],
                    "design_refs": ["M1"],
                    "write_scope": ["app.py"],
                    "acceptance_checks": [
                        {
                            "id": "C1",
                            "kind": "command",
                            "description": "导出冒烟检查",
                            "command": "python3 -c 'print(1)'",
                        }
                    ],
                }
            ],
        }
    )


class _BlueprintResponder:
    """Mock provider that drafts, fuses, and optionally fails the first resynthesis."""

    def __init__(
        self,
        *,
        answers: dict[str, str],
        notes: list[str] | None = None,
        fail_first_resynthesis: bool = False,
        blocking: bool = True,
    ) -> None:
        self.store: RunStore | None = None
        self.draft = _draft_spec()
        if not blocking:
            self.draft = self.draft.model_copy(update={"open_ambiguities": []})
        self.answers = answers
        self.notes = list(notes or [])
        self.fail_first_resynthesis = fail_first_resynthesis
        self.resynthesis_calls = 0
        self.resynthesis_prompts: list[str] = []

    def _latest_spec(self) -> GoalSpec | None:
        assert self.store is not None
        runs = self.store.list_runs(limit=1)
        if not runs:
            return None
        return self.store.get_run(str(runs[0]["run_id"]))["spec"]

    def __call__(self, messages: Sequence[ChatMessage], tools: Sequence[ToolSpec]) -> LLMResponse:
        del tools
        system = messages[0].content
        user = messages[1].content
        if "Fuse EVERY human clarification" in system:
            self.resynthesis_calls += 1
            self.resynthesis_prompts.append(user)
            if self.fail_first_resynthesis and self.resynthesis_calls == 1:
                raise RuntimeError("transient fusion outage")
            return LLMResponse(
                content=_fused_spec(self.draft, self.answers, self.notes).model_dump_json()
            )
        if "Requirements Alignment Architect" in system:
            return LLMResponse(content=self.draft.model_dump_json())
        if "White-Box Solution Architect" in system:
            spec = self._latest_spec()
            assert spec is not None
            return LLMResponse(content=_design_json(spec))
        if "Stage 2 Task Planner" in system or "Task Planner" in system:
            assert self.store is not None
            spec = self._latest_spec()
            assert spec is not None
            design = self.store.get_run(str(self.store.list_runs(limit=1)[0]["run_id"]))["design"]
            assert design is not None
            return LLMResponse(
                content=_plan_json(spec, design.model_dump(mode="json"))
            )
        raise AssertionError(f"Unexpected system prompt: {system[:120]}")


def _make_app(
    tmp_path: Path,
    responder: Responder,
) -> tuple[AriadneApp, RunSupervisor, TestClient]:
    _init_git_repo(tmp_path)
    store = RunStore(tmp_path / ".olympus" / "olympus.db")
    supervisor = RunSupervisor(
        store=store,
        workspace_root=tmp_path,
        provider_builder=lambda **_kwargs: MockLLMProvider(responder=responder),
    )
    app = AriadneApp(store=store, workspace_root=tmp_path, supervisor=supervisor)
    return app, supervisor, TestClient(create_fastapi_app(app))


def _start_run(
    supervisor: RunSupervisor,
    *,
    workflow: str = "full_sdlc_tdd",
    auto_approve: bool = False,
    constraints: list[str] | None = None,
) -> str:
    started = supervisor.start_run(
        title="订单导出",
        raw_input="构建运营可自助使用的订单导出服务",
        workflow_name=workflow,
        provider_config={"provider": "mock"},
        auto_approve=auto_approve,
        constraints=constraints,
        sync=True,
    )
    return str(started["run_id"])


def _projection(client: TestClient, run_id: str) -> dict:
    response = client.get(f"/api/runs/{run_id}")
    assert response.status_code == 200, response.text
    return response.json()["projection"]


def test_clarify_note_resynthesizes_and_rejects_stale_approval(tmp_path: Path) -> None:
    answers = {"A1-format": "Excel 工作簿", "A2-scope": "全量订单"}
    responder = _BlueprintResponder(
        answers=answers, notes=["导出格式固定为 Excel 工作簿"]
    )
    app, supervisor, client = _make_app(tmp_path, responder)
    responder.store = app.store
    try:
        run_id = _start_run(supervisor)
        before = _projection(client, run_id)
        assert before["status"] == "awaiting_human"
        assert before["pending_resynthesis"] is False
        old_hash = before["spec"]["content_hash"]
        old_version = before["spec"]["version"]

        response = client.post(
            f"/api/runs/{run_id}/command",
            json={
                "action": "clarify",
                "answers": answers,
                "clarification_note": "导出格式固定为 Excel 工作簿",
            },
        )
        assert response.status_code == 200, response.text
        supervisor.wait(run_id, timeout=30)

        after = _projection(client, run_id)
        assert after["pending_resynthesis"] is False
        assert after["resynthesis_error"] is None
        assert after["spec"]["version"] > old_version
        assert after["spec"]["content_hash"] != old_hash
        assert after["clarification_notes"]
        assert all(note["consumed"] for note in after["clarification_notes"])
        assert any(
            "Excel 工作簿" in constraint for constraint in after["spec"]["constraints"]
        )

        stale = client.post(
            f"/api/runs/{run_id}/command",
            json={
                "action": "approve_spec",
                "target_hash": old_hash,
                "expected_state_revision": after["state_revision"],
            },
        )
        assert stale.status_code == 409, stale.text
    finally:
        supervisor.shutdown()


def test_multi_turn_clarification_retains_turn_one_note(tmp_path: Path) -> None:
    answers = {"A1-format": "Excel 工作簿", "A2-scope": "全量订单"}
    notes = ["导出格式固定为 Excel 工作簿"]
    responder = _BlueprintResponder(answers=answers, notes=notes)
    app, supervisor, client = _make_app(tmp_path, responder)
    responder.store = app.store
    try:
        run_id = _start_run(supervisor)
        first = client.post(
            f"/api/runs/{run_id}/command",
            json={
                "action": "clarify",
                "answers": {"A1-format": answers["A1-format"]},
                "clarification_note": notes[0],
            },
        )
        assert first.status_code == 200, first.text
        supervisor.wait(run_id, timeout=30)
        mid = _projection(client, run_id)
        assert mid["pending_resynthesis"] is False
        assert any(note["consumed"] is False for note in mid["clarification_notes"])

        second = client.post(
            f"/api/runs/{run_id}/command",
            json={"action": "clarify", "answers": {"A2-scope": answers["A2-scope"]}},
        )
        assert second.status_code == 200, second.text
        supervisor.wait(run_id, timeout=30)

        final = _projection(client, run_id)
        assert final["pending_resynthesis"] is False
        body = json.dumps(final["spec"], ensure_ascii=False)
        assert notes[0] in body
        assert answers["A1-format"] in body
        assert answers["A2-scope"] in body
        assert all(note["consumed"] for note in final["clarification_notes"])
    finally:
        supervisor.shutdown()


def test_resynthesis_failure_preserves_answers_and_retries_in_place(tmp_path: Path) -> None:
    answers = {"A1-format": "Excel 工作簿", "A2-scope": "全量订单"}
    note = "导出必须保留订单号列"
    responder = _BlueprintResponder(
        answers=answers,
        notes=[note],
        fail_first_resynthesis=True,
    )
    app, supervisor, client = _make_app(tmp_path, responder)
    responder.store = app.store
    try:
        run_id = _start_run(supervisor)
        response = client.post(
            f"/api/runs/{run_id}/command",
            json={
                "action": "clarify",
                "answers": answers,
                "clarification_note": note,
            },
        )
        assert response.status_code == 200, response.text
        supervisor.wait(run_id, timeout=30)

        failed = _projection(client, run_id)
        assert failed["pending_resynthesis"] is True
        assert "transient fusion outage" in (failed["resynthesis_error"] or "")
        assert failed["hitl_banner"]["kind"] == "resynthesis_failed"
        assert failed["hitl_banner"]["actions"] == ["retry_workflow"]
        assert any(note["consumed"] is False for note in failed["clarification_notes"])
        assert {
            amb["id"]: amb["answer"] for amb in failed["spec"]["open_ambiguities"]
        } == answers

        retried = client.post(
            f"/api/runs/{run_id}/command",
            json={"action": "retry_workflow", "sync": True},
        )
        assert retried.status_code == 200, retried.text
        supervisor.wait(run_id, timeout=30)

        recovered = _projection(client, run_id)
        assert recovered["pending_resynthesis"] is False
        assert recovered["resynthesis_error"] is None
        assert all(note["consumed"] for note in recovered["clarification_notes"])
        assert responder.resynthesis_calls == 2
        assert note in responder.resynthesis_prompts[1]
    finally:
        supervisor.shutdown()


def test_approve_spec_rejected_while_resynthesis_pending(tmp_path: Path) -> None:
    answers = {"A1-format": "CSV", "A2-scope": "当前筛选结果"}
    responder = _BlueprintResponder(answers=answers)
    app, supervisor, client = _make_app(tmp_path, responder)
    responder.store = app.store
    try:
        run_id = _start_run(supervisor)
        clarify = client.post(
            f"/api/runs/{run_id}/command",
            json={
                "action": "clarify",
                "answers": answers,
                "clarification_note": "导出文件名包含时间戳",
                "auto_advance": False,
            },
        )
        assert clarify.status_code == 200, clarify.text
        projection = _projection(client, run_id)
        assert projection["pending_resynthesis"] is True
        assert projection["hitl_banner"]["kind"] == "resynthesizing"

        blocked = client.post(
            f"/api/runs/{run_id}/command",
            json={
                "action": "approve_spec",
                "target_hash": projection["spec"]["content_hash"],
                "expected_state_revision": projection["state_revision"],
            },
        )
        assert blocked.status_code == 400, blocked.text
        assert "awaiting clarification resynthesis" in blocked.text
    finally:
        supervisor.shutdown()


def test_request_revision_restarts_from_align_and_is_idempotent(tmp_path: Path) -> None:
    responder = _BlueprintResponder(answers={})
    app, supervisor, client = _make_app(tmp_path, responder)
    responder.store = app.store
    try:
        run_id = _start_run(supervisor, constraints=["使用服务端数据库持久化"])
        payload = {
            "action": "request_revision",
            "feedback": "导出改为按周汇总，而不是逐单明细",
            "operation_id": "op-revision-1",
            "sync": True,
        }
        first = client.post(f"/api/runs/{run_id}/command", json=payload)
        assert first.status_code == 200, first.text
        supervisor.wait(run_id, timeout=30)

        events = app.store.list_events(run_id)
        assert sum(event.event_type == "run.revision_requested" for event in events) == 1
        recorded = supervisor._recorded_raw_input(run_id, "")
        assert "[User revision feedback]" in recorded
        assert "按周汇总" in recorded
        assert supervisor._recorded_constraints(run_id) == ["使用服务端数据库持久化"]

        again = client.post(f"/api/runs/{run_id}/command", json=payload)
        assert again.status_code == 200, again.text
        assert again.json()["command_result"]["deduplicated"] is True
        assert (
            sum(
                event.event_type == "run.revision_requested"
                for event in app.store.list_events(run_id)
            )
            == 1
        )

        empty = client.post(
            f"/api/runs/{run_id}/command",
            json={"action": "request_revision", "feedback": "   "},
        )
        assert empty.status_code == 400, empty.text
    finally:
        supervisor.shutdown()


def test_reopen_run_restarts_completed_run_and_is_idempotent(tmp_path: Path) -> None:
    responder = _BlueprintResponder(answers={}, blocking=False)
    app, supervisor, client = _make_app(tmp_path, responder)
    responder.store = app.store
    try:
        run_id = _start_run(
            supervisor, workflow="product_blueprint", auto_approve=True
        )
        completed = _projection(client, run_id)
        assert completed["status"] == "completed"

        payload = {
            "action": "reopen_run",
            "feedback": "追加导出历史记录的下载链接",
            "operation_id": "op-reopen-1",
            "sync": True,
        }
        opened = client.post(f"/api/runs/{run_id}/command", json=payload)
        assert opened.status_code == 200, opened.text
        supervisor.wait(run_id, timeout=30)

        events = app.store.list_events(run_id)
        assert sum(event.event_type == "run.reopened" for event in events) == 1
        assert "追加导出历史记录的下载链接" in supervisor._recorded_raw_input(run_id, "")

        again = client.post(f"/api/runs/{run_id}/command", json=payload)
        assert again.status_code == 200, again.text
        assert again.json()["command_result"]["deduplicated"] is True

        non_completed = client.post(
            f"/api/runs/{run_id}/command",
            json={"action": "reopen_run", "feedback": "再追加一次", "sync": True},
        )
        assert non_completed.status_code == 400, non_completed.text
    finally:
        supervisor.shutdown()


def test_request_revision_during_inflight_resynthesis_realigns_with_feedback(
    tmp_path: Path,
) -> None:
    """Revision coalesced while resynthesis is active must still re-run align with feedback."""
    resynth_entered = threading.Event()
    allow_resynth = threading.Event()
    align_prompts: list[str] = []
    answers = {"A1-format": "Excel 工作簿", "A2-scope": "全量订单"}
    draft = _draft_spec()

    def responder(messages: Sequence[ChatMessage], tools: Sequence[ToolSpec]) -> LLMResponse:
        del tools
        system = messages[0].content
        user = messages[1].content
        if "Fuse EVERY human clarification" in system:
            resynth_entered.set()
            assert allow_resynth.wait(timeout=10.0)
            return LLMResponse(content=_fused_spec(draft, answers, []).model_dump_json())
        if "Requirements Alignment Architect" in system:
            align_prompts.append(user)
            if "按周汇总" in user:
                revised = draft.model_copy(
                    update={
                        "spec_id": "spec-orders-weekly",
                        "goal": "构建按周汇总的订单导出服务",
                        "open_ambiguities": [],
                    }
                )
                return LLMResponse(content=revised.model_dump_json())
            return LLMResponse(content=draft.model_dump_json())
        raise AssertionError(f"Unexpected prompt: {system[:100]}")

    app, supervisor, client = _make_app(tmp_path, responder)
    try:
        run_id = _start_run(supervisor)
        clarify = client.post(
            f"/api/runs/{run_id}/command",
            json={"action": "clarify", "answers": answers},
        )
        assert clarify.status_code == 200, clarify.text
        assert resynth_entered.wait(timeout=10.0)

        revision = client.post(
            f"/api/runs/{run_id}/command",
            json={
                "action": "request_revision",
                "feedback": "改为按周汇总并保留下载历史",
                "operation_id": "op-rev-inflight",
            },
        )
        assert revision.status_code == 200, revision.text
        assert revision.json()["command_result"]["workflow"].get("coalesced") is True

        allow_resynth.set()
        supervisor.wait(run_id, timeout=30)

        after = _projection(client, run_id)
        events = app.store.list_events(run_id)
        rev_seq = next(ev.event_seq for ev in events if ev.event_type == "run.revision_requested")
        assert not any(
            ev.event_type == "spec.resynthesized" and ev.event_seq > rev_seq for ev in events
        )
        assert len(align_prompts) == 2
        assert "改为按周汇总并保留下载历史" in align_prompts[1]
        assert after["spec"]["spec_id"] == "spec-orders-weekly"
        assert after["status"] == "awaiting_human"
        assert after["pending_resynthesis"] is False
    finally:
        allow_resynth.set()
        supervisor.shutdown()


def test_queued_revision_respects_subsequent_human_pause(tmp_path: Path) -> None:
    """Pause arriving after a coalesced revision must keep the run paused."""
    resynth_entered = threading.Event()
    allow_resynth = threading.Event()
    align_calls = 0
    answers = {"A1-format": "Excel 工作簿", "A2-scope": "全量订单"}
    draft = _draft_spec()

    def responder(messages: Sequence[ChatMessage], tools: Sequence[ToolSpec]) -> LLMResponse:
        nonlocal align_calls
        del tools
        system = messages[0].content
        if "Fuse EVERY human clarification" in system:
            resynth_entered.set()
            assert allow_resynth.wait(timeout=10.0)
            return LLMResponse(content=_fused_spec(draft, answers, []).model_dump_json())
        if "Requirements Alignment Architect" in system:
            align_calls += 1
            return LLMResponse(content=draft.model_dump_json())
        raise AssertionError(f"Unexpected prompt: {system[:100]}")

    app, supervisor, client = _make_app(tmp_path, responder)
    try:
        run_id = _start_run(supervisor)
        assert align_calls == 1
        assert client.post(
            f"/api/runs/{run_id}/command",
            json={"action": "clarify", "answers": answers},
        ).status_code == 200
        assert resynth_entered.wait(timeout=10.0)

        assert client.post(
            f"/api/runs/{run_id}/command",
            json={"action": "request_revision", "feedback": "改为按月导出"},
        ).status_code == 200
        paused_resp = client.post(
            f"/api/runs/{run_id}/command",
            json={"action": "pause"},
        )
        assert paused_resp.status_code == 200, paused_resp.text
        assert paused_resp.json()["projection"]["status"] == "paused"

        allow_resynth.set()
        supervisor.wait(run_id, timeout=30)

        after = _projection(client, run_id)
        assert after["status"] == "paused"
        assert align_calls == 1
    finally:
        allow_resynth.set()
        supervisor.shutdown()


def test_persisted_revision_auto_advance_false_preserves_feedback_on_retry(
    tmp_path: Path,
) -> None:
    """Retry after request_revision(auto_advance=False) must pass saved feedback to align."""
    align_prompts: list[str] = []
    draft = _draft_spec()

    def responder(messages: Sequence[ChatMessage], tools: Sequence[ToolSpec]) -> LLMResponse:
        del tools
        system = messages[0].content
        user = messages[1].content
        if "Requirements Alignment Architect" in system:
            align_prompts.append(user)
            return LLMResponse(content=draft.model_dump_json())
        raise AssertionError(f"Unexpected prompt: {system[:100]}")

    app, supervisor, client = _make_app(tmp_path, responder)
    try:
        run_id = _start_run(supervisor)
        rev = client.post(
            f"/api/runs/{run_id}/command",
            json={
                "action": "request_revision",
                "feedback": "新增按门店维度分组导出",
                "auto_advance": False,
            },
        )
        assert rev.status_code == 200, rev.text
        retry = client.post(
            f"/api/runs/{run_id}/command",
            json={"action": "retry_workflow", "sync": True},
        )
        assert retry.status_code == 200, retry.text
        supervisor.wait(run_id, timeout=30)

        assert len(align_prompts) == 2
        assert "新增按门店维度分组导出" in align_prompts[1]
    finally:
        supervisor.shutdown()


def test_second_clarify_after_resynthesis_triggers_fresh_resynthesis(tmp_path: Path) -> None:
    """Changing an answer after a prior resynthesis must re-enter resynthesis before approval."""
    first_answers = {"A1-format": "Excel 工作簿", "A2-scope": "全量订单"}
    second_answers = {"A1-format": "PDF 报表", "A2-scope": "全量订单"}
    draft = _draft_spec()
    resynth_calls = 0

    def responder(messages: Sequence[ChatMessage], tools: Sequence[ToolSpec]) -> LLMResponse:
        nonlocal resynth_calls
        del tools
        system = messages[0].content
        user = messages[1].content
        if "Fuse EVERY human clarification" in system:
            resynth_calls += 1
            active_answers = second_answers if "PDF 报表" in user else first_answers
            return LLMResponse(content=_fused_spec(draft, active_answers, []).model_dump_json())
        if "Requirements Alignment Architect" in system:
            return LLMResponse(content=draft.model_dump_json())
        raise AssertionError(f"Unexpected prompt: {system[:100]}")

    app, supervisor, client = _make_app(tmp_path, responder)
    try:
        run_id = _start_run(supervisor)
        r1 = client.post(
            f"/api/runs/{run_id}/command",
            json={"action": "clarify", "answers": first_answers},
        )
        assert r1.status_code == 200, r1.text
        supervisor.wait(run_id, timeout=30)
        assert resynth_calls == 1

        # Second clarify changes Excel -> PDF without advancing yet: must mark pending & block.
        r2 = client.post(
            f"/api/runs/{run_id}/command",
            json={
                "action": "clarify",
                "answers": {"A1-format": "PDF 报表"},
                "auto_advance": False,
            },
        )
        assert r2.status_code == 200, r2.text
        pending = _projection(client, run_id)
        assert pending["pending_resynthesis"] is True

        blocked = client.post(
            f"/api/runs/{run_id}/command",
            json={
                "action": "approve_spec",
                "target_hash": pending["spec"]["content_hash"],
                "expected_state_revision": pending["state_revision"],
            },
        )
        assert blocked.status_code == 400, blocked.text

        resumed = client.post(
            f"/api/runs/{run_id}/command",
            json={"action": "retry_workflow", "sync": True},
        )
        assert resumed.status_code == 200, resumed.text
        supervisor.wait(run_id, timeout=30)
        assert resynth_calls == 2
        final = _projection(client, run_id)
        assert final["pending_resynthesis"] is False
        assert any("PDF 报表" in c for c in final["spec"]["constraints"])
    finally:
        supervisor.shutdown()


def test_default_answers_still_require_strict_canonical_resynthesis(tmp_path: Path) -> None:
    """Strict-canonical spec missing test_seams must still require repair on default answers."""
    incomplete_draft = _draft_spec().model_copy(update={"test_seams": []})
    repaired = _draft_spec().model_copy(update={"open_ambiguities": []})

    def responder(messages: Sequence[ChatMessage], tools: Sequence[ToolSpec]) -> LLMResponse:
        del tools
        system = messages[0].content
        if "Fuse EVERY human clarification" in system:
            return LLMResponse(content=repaired.model_dump_json())
        if "Requirements Alignment Architect" in system:
            return LLMResponse(content=incomplete_draft.model_dump_json())
        raise AssertionError(f"Unexpected prompt: {system[:100]}")

    app, supervisor, client = _make_app(tmp_path, responder)
    try:
        run_id = _start_run(supervisor)
        clarify = client.post(
            f"/api/runs/{run_id}/command",
            json={
                "action": "clarify",
                "answers": {"A1-format": "CSV", "A2-scope": "当前筛选结果"},
                "auto_advance": False,
            },
        )
        assert clarify.status_code == 200, clarify.text
        proj = _projection(client, run_id)
        assert proj["pending_resynthesis"] is True

        blocked = client.post(
            f"/api/runs/{run_id}/command",
            json={
                "action": "approve_spec",
                "target_hash": proj["spec"]["content_hash"],
                "expected_state_revision": proj["state_revision"],
            },
        )
        assert blocked.status_code == 400, blocked.text
    finally:
        supervisor.shutdown()


def test_approve_spec_conflict_and_replay_precedence_during_pending_resynthesis(
    tmp_path: Path,
) -> None:
    """Stale revision/hash return 409 and replay returns 200 before the 400 resynthesis guard."""
    answers = {"A1-format": "CSV", "A2-scope": "当前筛选结果"}
    responder = _BlueprintResponder(answers=answers)
    app, supervisor, client = _make_app(tmp_path, responder)
    responder.store = app.store
    try:
        run_id = _start_run(supervisor)
        before = _projection(client, run_id)
        old_hash = before["spec"]["content_hash"]
        old_rev = before["state_revision"]

        clarify = client.post(
            f"/api/runs/{run_id}/command",
            json={
                "action": "clarify",
                "answers": answers,
                "clarification_note": "补充审计日志字段",
                "auto_advance": False,
            },
        )
        assert clarify.status_code == 200, clarify.text
        pending = _projection(client, run_id)
        assert pending["pending_resynthesis"] is True

        stale_rev = client.post(
            f"/api/runs/{run_id}/command",
            json={
                "action": "approve_spec",
                "target_hash": pending["spec"]["content_hash"],
                "expected_state_revision": old_rev,
            },
        )
        assert stale_rev.status_code == 409, stale_rev.text

        stale_hash = client.post(
            f"/api/runs/{run_id}/command",
            json={
                "action": "approve_spec",
                "target_hash": old_hash,
                "expected_state_revision": pending["state_revision"],
            },
        )
        assert stale_hash.status_code == 409, stale_hash.text

        current_blocked = client.post(
            f"/api/runs/{run_id}/command",
            json={
                "action": "approve_spec",
                "target_hash": pending["spec"]["content_hash"],
                "expected_state_revision": pending["state_revision"],
            },
        )
        assert current_blocked.status_code == 400, current_blocked.text
    finally:
        supervisor.shutdown()


def test_unsupervised_run_does_not_report_pending_resynthesis(tmp_path: Path) -> None:
    """Unsupervised runs must not show pending_resynthesis or block approval in projection."""
    responder = _BlueprintResponder(answers={})
    app, supervisor, client = _make_app(tmp_path, responder)
    try:
        run_id = app.store.create_run(title="Unsupervised Clarify")
        app.store.save_spec(run_id, _draft_spec())
        app.store.set_run_status(run_id, RunStatus.AWAITING_HUMAN)

        clarify = client.post(
            f"/api/runs/{run_id}/command",
            json={
                "action": "clarify",
                "answers": {"A1-format": "Excel", "A2-scope": "全量订单"},
                "clarification_note": "人工自行确认",
            },
        )
        assert clarify.status_code == 200, clarify.text
        proj = _projection(client, run_id)
        assert proj["pending_resynthesis"] is False
        assert (proj.get("hitl_banner") or {}).get("kind") != "resynthesizing"

        approved = client.post(
            f"/api/runs/{run_id}/command",
            json={
                "action": "approve_spec",
                "target_hash": proj["spec"]["content_hash"],
                "expected_state_revision": proj["state_revision"],
            },
        )
        assert approved.status_code == 200, approved.text
    finally:
        supervisor.shutdown()


def test_malformed_consumed_note_ids_does_not_mark_notes_consumed(tmp_path: Path) -> None:
    """Non-list consumed_note_ids must not trigger the legacy cutoff fallback."""
    store = RunStore(tmp_path / "malformed.db")
    run_id = store.create_run(title="Malformed Resynth")
    spec = _draft_spec().model_copy(update={"open_ambiguities": []})
    store.save_spec(run_id, spec)
    store.append_event(
        run_id=run_id,
        stage="align",
        event_type="spec.clarification_note",
        actor="zeus",
        payload={"spec_id": spec.spec_id, "feedback": "固定分类不可改"},
    )
    store.append_event(
        run_id=run_id,
        stage="align",
        event_type="spec.resynthesized",
        actor="align",
        payload={"spec_id": spec.spec_id, "consumed_note_ids": {"bad": True}},
    )
    events = store.list_events(run_id)
    states = clarification_note_states(events, spec_id=spec.spec_id)
    assert len(states) == 1
    assert states[0][1] is False
    assert has_pending_resynthesis(events, spec, spec_approved=False) is True


@pytest.mark.parametrize("mode", ["anchor", "append"])
def test_revision_before_resynth_anchor_or_validation_append_prevents_stale_writes(
    tmp_path: Path, mode: str
) -> None:
    """Revision landing before resynthesis anchor or validation append prevents stale writes."""
    from unittest.mock import patch

    from olympus.stages.align import AlignStage

    answers = {"A1-format": "Excel 工作簿", "A2-scope": "全量订单"}
    notes = ["新增按财务月份分组"]
    responder = _BlueprintResponder(
        answers=answers, notes=notes if mode == "anchor" else []
    )
    app, supervisor, client = _make_app(tmp_path / mode, responder)
    responder.store = app.store
    try:
        run_id = _start_run(supervisor)
        gate = threading.local()
        orig_resynth = AlignStage.resynthesize_after_clarification
        orig_list = app.store.list_events
        orig_append = app.store.append_event
        inserted = False

        def revise() -> None:
            nonlocal inserted
            inserted = True
            res = client.post(
                f"/api/runs/{run_id}/command",
                json={
                    "action": "request_revision",
                    "feedback": "改为按周导出",
                    "auto_advance": False,
                },
            )
            assert res.status_code == 200, res.text

        def wrapped(stage: AlignStage, **kwargs: object) -> GoalSpec:
            gate.in_resynthesize = True
            try:
                return orig_resynth(stage, **kwargs)  # type: ignore[arg-type]
            finally:
                gate.in_resynthesize = False

        def before_anchor(*args: object, **kwargs: object):
            if mode == "anchor" and getattr(gate, "in_resynthesize", False) and not inserted:
                revise()
            return orig_list(*args, **kwargs)  # type: ignore[arg-type]

        def before_validation_append(*args: object, **kwargs: object):
            if (
                mode == "append"
                and kwargs.get("event_type") == "spec.validation_failed"
                and not inserted
            ):
                revise()
            return orig_append(*args, **kwargs)  # type: ignore[arg-type]

        with (
            patch.object(AlignStage, "resynthesize_after_clarification", wrapped),
            patch.object(app.store, "list_events", before_anchor),
            patch.object(app.store, "append_event", before_validation_append),
        ):
            clarify = client.post(
                f"/api/runs/{run_id}/command",
                json={
                    "action": "clarify",
                    "answers": answers,
                    "clarification_note": notes[0],
                },
            )
            assert clarify.status_code == 200, clarify.text
            supervisor.wait(run_id, timeout=30)

        events = app.store.list_events(run_id)
        rev_seq = next(
            ev.event_seq for ev in events if ev.event_type == "run.revision_requested"
        )
        stale_types = {
            "spec.resynthesized",
            "spec.validation_failed",
            "spec.resynthesis_failed",
        }
        assert not any(
            ev.event_seq > rev_seq and ev.event_type in stale_types for ev in events
        )
    finally:
        supervisor.shutdown()


def test_consecutive_revisions_never_shadow_newer_feedback(tmp_path: Path) -> None:
    """Monotonic event-log folding preserves both earlier and newer revision feedback."""
    align_prompts: list[str] = []
    draft = _draft_spec()

    def responder(messages: Sequence[ChatMessage], tools: Sequence[ToolSpec]) -> LLMResponse:
        del tools
        if "Requirements Alignment Architect" in messages[0].content:
            align_prompts.append(messages[1].content)
            return LLMResponse(content=draft.model_dump_json())
        raise AssertionError(f"Unexpected prompt: {messages[0].content[:100]}")

    app, supervisor, client = _make_app(tmp_path, responder)
    try:
        run_id = _start_run(supervisor)
        assert client.post(
            f"/api/runs/{run_id}/command",
            json={
                "action": "request_revision",
                "feedback": "First revision",
                "auto_advance": False,
            },
        ).status_code == 200
        assert client.post(
            f"/api/runs/{run_id}/command",
            json={"action": "pause"},
        ).status_code == 200
        assert client.post(
            f"/api/runs/{run_id}/command",
            json={
                "action": "request_revision",
                "feedback": "Second revision",
                "auto_advance": False,
            },
        ).status_code == 200
        assert client.post(
            f"/api/runs/{run_id}/command",
            json={"action": "retry_workflow", "sync": True},
        ).status_code == 200
        supervisor.wait(run_id, timeout=30)

        assert len(align_prompts) == 2
        assert "First revision" in align_prompts[1]
        assert "Second revision" in align_prompts[1]
    finally:
        supervisor.shutdown()


def test_approve_spec_rejected_while_revision_unfulfilled(tmp_path: Path) -> None:
    """Approving the pre-revision GoalSpec before align runs must return 400."""
    responder = _BlueprintResponder(answers={}, blocking=False)
    app, supervisor, client = _make_app(tmp_path, responder)
    responder.store = app.store
    try:
        run_id = _start_run(supervisor)
        rev = client.post(
            f"/api/runs/{run_id}/command",
            json={
                "action": "request_revision",
                "feedback": "改成按周汇总",
                "auto_advance": False,
            },
        )
        assert rev.status_code == 200, rev.text
        proj = _projection(client, run_id)
        blocked = client.post(
            f"/api/runs/{run_id}/command",
            json={
                "action": "approve_spec",
                "target_hash": proj["spec"]["content_hash"],
                "expected_state_revision": proj["state_revision"],
                "auto_advance": False,
            },
        )
        assert blocked.status_code == 400, blocked.text
        assert "awaiting revision realignment" in blocked.text

        # Spoofing actor="align" via clarify must not forge realignment completion.
        spoof = client.post(
            f"/api/runs/{run_id}/command",
            json={
                "action": "clarify",
                "answers": {},
                "actor": "align",
                "auto_advance": False,
            },
        )
        assert spoof.status_code == 200, spoof.text
        spoof_proj = _projection(client, run_id)
        still_blocked = client.post(
            f"/api/runs/{run_id}/command",
            json={
                "action": "approve_spec",
                "target_hash": spoof_proj["spec"]["content_hash"],
                "expected_state_revision": spoof_proj["state_revision"],
                "auto_advance": False,
            },
        )
        assert still_blocked.status_code == 400, still_blocked.text
        assert "awaiting revision realignment" in still_blocked.text
    finally:
        supervisor.shutdown()


def test_revision_during_resynthesis_failure_abort_does_not_clobber_running(
    tmp_path: Path,
) -> None:
    """Revision arriving right before spec.resynthesis_failed must not revert status to awaiting."""
    from unittest.mock import patch

    answers = {"A1-format": "Excel 工作簿", "A2-scope": "全量订单"}
    responder = _BlueprintResponder(answers=answers, fail_first_resynthesis=True)
    app, supervisor, client = _make_app(tmp_path, responder)
    responder.store = app.store
    try:
        run_id = _start_run(supervisor)
        orig_append = app.store.append_event
        inserted = False

        def inject_before_failure_event(*args: object, **kwargs: object):
            nonlocal inserted
            if kwargs.get("event_type") == "spec.resynthesis_failed" and not inserted:
                inserted = True
                res = client.post(
                    f"/api/runs/{run_id}/command",
                    json={
                        "action": "request_revision",
                        "feedback": "改成按月导出",
                        "auto_advance": False,
                    },
                )
                assert res.status_code == 200, res.text
            return orig_append(*args, **kwargs)  # type: ignore[arg-type]

        with patch.object(app.store, "append_event", inject_before_failure_event):
            clarify = client.post(
                f"/api/runs/{run_id}/command",
                json={"action": "clarify", "answers": answers},
            )
            assert clarify.status_code == 200, clarify.text
            supervisor.wait(run_id, timeout=30)

        events = app.store.list_events(run_id)
        rev_seq = next(
            ev.event_seq for ev in events if ev.event_type == "run.revision_requested"
        )
        assert not any(
            ev.event_seq > rev_seq and ev.event_type == "spec.resynthesis_failed"
            for ev in events
        )
        assert app.store.get_run(run_id)["status"] == RunStatus.RUNNING
    finally:
        supervisor.shutdown()


def test_late_revision_after_graph_return_not_overwritten_by_status_cas_retry(
    tmp_path: Path,
) -> None:
    """Revision landing right before post-graph set_run_status must stay RUNNING."""
    from unittest.mock import patch

    answers = {"A1-format": "CSV", "A2-scope": "当前筛选结果"}
    responder = _BlueprintResponder(answers=answers)
    app, supervisor, client = _make_app(tmp_path, responder)
    responder.store = app.store
    try:
        run_id = _start_run(supervisor)
        orig_set_status = app.store.set_run_status
        injected = False

        def inject_before_awaiting(rid: str, status: RunStatus, **kwargs: object) -> int:
            nonlocal injected
            if (
                rid == run_id
                and status == RunStatus.AWAITING_HUMAN
                and kwargs.get("stage") == "graph"
                and not injected
            ):
                injected = True
                rev = client.post(
                    f"/api/runs/{run_id}/command",
                    json={
                        "action": "request_revision",
                        "feedback": "改成支持分批下载",
                        "auto_advance": False,
                    },
                )
                assert rev.status_code == 200, rev.text
            return orig_set_status(rid, status, **kwargs)  # type: ignore[arg-type]

        with patch.object(app.store, "set_run_status", inject_before_awaiting):
            clarify = client.post(
                f"/api/runs/{run_id}/command",
                json={"action": "clarify", "answers": answers},
            )
            assert clarify.status_code == 200, clarify.text
            supervisor.wait(run_id, timeout=30)

        assert injected is True
        assert app.store.get_run(run_id)["status"] == RunStatus.RUNNING
    finally:
        supervisor.shutdown()


def test_legacy_realigned_run_without_spec_aligned_event_remains_approvable(
    tmp_path: Path,
) -> None:
    """Runs realigned prior to spec.aligned (standalone spec.saved by align) remain approvable."""
    responder = _BlueprintResponder(answers={}, blocking=False)
    app, supervisor, client = _make_app(tmp_path, responder)
    try:
        run_id = _start_run(supervisor)
        rev = client.post(
            f"/api/runs/{run_id}/command",
            json={
                "action": "request_revision",
                "feedback": "改成按周汇总",
                "auto_advance": False,
            },
        )
        assert rev.status_code == 200, rev.text

        # Simulate a legacy database where AlignStage.propose_spec wrote spec.saved(actor="align")
        # without the newer spec.aligned audit event.
        legacy_realigned = _draft_spec().model_copy(
            update={"version": 2, "open_ambiguities": [], "status": SpecStatus.AWAITING_HUMAN}
        )
        app.store.save_spec(run_id, legacy_realigned, actor="align")
        app.store.set_run_status(run_id, RunStatus.AWAITING_HUMAN, actor="align", stage="align")

        proj = _projection(client, run_id)
        approved = client.post(
            f"/api/runs/{run_id}/command",
            json={
                "action": "approve_spec",
                "target_hash": proj["spec"]["content_hash"],
                "expected_state_revision": proj["state_revision"],
                "auto_advance": False,
            },
        )
        assert approved.status_code == 200, approved.text
        assert approved.json()["projection"]["spec_approved"] is True
    finally:
        supervisor.shutdown()


def test_second_revision_during_align_aborts_stale_graph_events_and_retries_align(
    tmp_path: Path,
) -> None:
    """Second revision arriving right after align saves spec must not emit stale graph events."""
    from unittest.mock import patch

    align_prompts: list[str] = []
    draft = _draft_spec().model_copy(update={"open_ambiguities": []})

    def responder(messages: Sequence[ChatMessage], tools: Sequence[ToolSpec]) -> LLMResponse:
        del tools
        if "Requirements Alignment Architect" in messages[0].content:
            align_prompts.append(messages[1].content)
            return LLMResponse(content=draft.model_dump_json())
        raise AssertionError(f"Unexpected prompt: {messages[0].content[:100]}")

    app, supervisor, client = _make_app(tmp_path, responder)
    try:
        run_id = _start_run(supervisor)
        orig_append = app.store.append_event
        injected = False

        def inject_rev2_before_align_finished(*args: object, **kwargs: object):
            nonlocal injected
            if (
                kwargs.get("event_type") == "graph.node_finished"
                and (kwargs.get("payload") or {}).get("node") == "align"
                and len(align_prompts) == 2
                and not injected
            ):
                injected = True
                res = client.post(
                    f"/api/runs/{run_id}/command",
                    json={
                        "action": "request_revision",
                        "feedback": "第二轮修订反馈",
                        "auto_advance": False,
                    },
                )
                assert res.status_code == 200, res.text
            return orig_append(*args, **kwargs)  # type: ignore[arg-type]

        with patch.object(app.store, "append_event", inject_rev2_before_align_finished):
            rev1 = client.post(
                f"/api/runs/{run_id}/command",
                json={
                    "action": "request_revision",
                    "feedback": "第一轮修订反馈",
                    "sync": True,
                },
            )
            assert rev1.status_code == 200, rev1.text
            supervisor.wait(run_id, timeout=30)

        assert injected is True
        events = app.store.list_events(run_id)
        rev2_seq = [ev.event_seq for ev in events if ev.event_type == "run.revision_requested"][-1]
        assert not any(
            ev.event_seq > rev2_seq
            and ev.event_type in ("graph.node_finished", "graph.paused")
            for ev in events
        )

        retry = client.post(
            f"/api/runs/{run_id}/command",
            json={"action": "retry_workflow", "sync": True},
        )
        assert retry.status_code == 200, retry.text
        supervisor.wait(run_id, timeout=30)
        assert len(align_prompts) == 3
        assert "第二轮修订反馈" in align_prompts[-1]
        assert _projection(client, run_id)["status"] == "awaiting_human"
    finally:
        supervisor.shutdown()
