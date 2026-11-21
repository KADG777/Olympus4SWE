from collections.abc import Sequence
from pathlib import Path
from threading import Thread

from olympus.harness.driver import ActiveTurnHandle, DriverTurnRequest
from olympus.harness.native import NativeDriver
from olympus.providers.types import ChatMessage, LLMResponse, MockLLMProvider, ToolCall, ToolSpec


def test_native_steer_preserves_exploration_and_emits_milestone(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("repository fact", encoding="utf-8")
    provider = MockLLMProvider(
        responses=[
            LLMResponse(
                content="Exploring",
                tool_calls=(
                    ToolCall(
                        id="read", name="read_file", arguments={"path": str(tmp_path / "notes.txt")}
                    ),
                ),
            ),
            LLMResponse(content="Done"),
        ]
    )
    handles: list[ActiveTurnHandle] = []
    steered: list[dict[str, object]] = []

    def on_event(kind: str, payload: dict[str, object]) -> None:
        if kind == "tool_end":

            def steer() -> None:
                assert handles[0].steer("Use the fact you just read", source="human")

            thread = Thread(target=steer)
            thread.start()
            thread.join()
        if kind == "agent.steered":
            steered.append(payload)

    outcome = NativeDriver(provider=provider).run_turn(
        DriverTurnRequest(
            workspace_path=tmp_path,
            env_root=tmp_path,
            actor="worker",
            system_prompt="System",
            user_prompt="Explore",
        ),
        on_event=on_event,
        on_handle=handles.append,
    )

    assert outcome.completed and outcome.driver_name == "native"
    messages, _ = provider.recorded_requests[1]
    assert any(msg.role == "tool" and "repository fact" in msg.content for msg in messages)
    assert messages[-1] == ChatMessage(role="user", content="Use the fact you just read")
    assert len(steered) == 1 and steered[0]["source"] == "human"
    assert "Use the fact you just read" not in str(steered[0])
    assert not handles[0].steer("Too late", source="human")


def test_native_cancel_during_provider_skips_tools_and_further_calls(tmp_path: Path) -> None:
    handles: list[ActiveTurnHandle] = []
    target = tmp_path / "cancelled.txt"

    def respond(messages: Sequence[ChatMessage], tools: Sequence[ToolSpec]) -> LLMResponse:
        thread = Thread(target=handles[0].cancel, kwargs={"reason": "human_paused"})
        thread.start()
        thread.join()
        return LLMResponse(
            content="",
            tool_calls=(
                ToolCall(
                    id="write",
                    name="write_file",
                    arguments={
                        "path": str(target),
                        "content": "should never be written",
                    },
                ),
            ),
        )

    provider = MockLLMProvider(responder=respond)
    outcome = NativeDriver(provider=provider).run_turn(
        DriverTurnRequest(
            workspace_path=tmp_path,
            env_root=tmp_path,
            actor="worker",
            system_prompt="System",
            user_prompt="Explore",
        ),
        on_handle=handles.append,
    )

    assert not outcome.completed and outcome.stop_reason == "cancelled"
    assert not target.exists()
    assert len(provider.recorded_requests) == 1


def test_native_cancel_during_reservation_releases_unused_call(tmp_path: Path) -> None:
    from olympus.core.contracts import BudgetLedger
    from olympus.core.store import RunStore

    handles: list[ActiveTurnHandle] = []

    class CancellingStore(RunStore):
        def reserve_llm_call(self, run_id: str, *, actor: str, purpose: str) -> BudgetLedger:
            ledger = super().reserve_llm_call(run_id, actor=actor, purpose=purpose)
            handles[0].cancel(reason="human_paused")
            return ledger

    store = CancellingStore(tmp_path / "run.db")
    run_id = store.create_run(title="Cancellation at reservation")
    provider = MockLLMProvider(responses=[LLMResponse(content="Should not be called")])
    outcome = NativeDriver(provider=provider, store=store, run_id=run_id).run_turn(
        DriverTurnRequest(
            workspace_path=tmp_path,
            env_root=tmp_path,
            actor="worker",
            system_prompt="System",
            user_prompt="Explore",
        ),
        on_handle=handles.append,
    )

    assert outcome.stop_reason == "cancelled"
    assert not provider.recorded_requests
    assert store.get_run(run_id)["budget_ledger"] == BudgetLedger()
