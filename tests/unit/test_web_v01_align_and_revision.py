"""v0.1 Stage 1 resynthesis fidelity plus explicit revision/reopen commands."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path

from fastapi.testclient import TestClient

from olympus.core.contracts import (
    AcceptanceCriterion,
    AmbiguityItem,
    DeliverableKind,
    EarsPattern,
    GoalSpec,
    RequirementItem,
    SpecStatus,
)
from olympus.core.store import RunStore
from olympus.providers.base import ChatMessage, LLMResponse, MockLLMProvider, ToolSpec
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
        recorded = [
            event for event in events if event.event_type == "graph.input_recorded"
        ][-1]
        assert "[User revision feedback]" in recorded.payload["raw_input"]
        assert "按周汇总" in recorded.payload["raw_input"]
        assert recorded.payload["constraints"] == ["使用服务端数据库持久化"]

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
        recorded = [
            event for event in events if event.event_type == "graph.input_recorded"
        ][-1]
        assert "追加导出历史记录的下载链接" in recorded.payload["raw_input"]

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
