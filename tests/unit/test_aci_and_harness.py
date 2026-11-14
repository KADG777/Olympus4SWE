"""Unit tests for Poka-Yoke ACI tools, git worktree sandbox, and AgentLoop."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from olympus.core.contracts import BudgetPolicy
from olympus.core.store import RunStore
from olympus.harness.loop import AgentLoop
from olympus.harness.worktree import capture_sandbox_diff, create_sandbox, remove_sandbox
from olympus.providers.base import LLMResponse, MockLLMProvider, ToolCall, UsageStats
from olympus.tools.aci import WorkspaceToolkit


def test_aci_rejects_relative_and_escape_paths_and_enforces_write_scope(tmp_path: Path) -> None:
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    allowed_file = src_dir / "allowed.py"
    allowed_file.write_text("x = 1\n", encoding="utf-8")
    blocked_file = src_dir / "other.py"
    blocked_file.write_text("y = 2\n", encoding="utf-8")

    toolkit = WorkspaceToolkit(tmp_path, allowed_write_scope=["src/allowed.py"])

    # Relative path rejected with actionable hint containing workspace root
    rel_err = toolkit.execute("read_file", {"path": "src/allowed.py"})
    assert "Relative path" in rel_err
    assert str(tmp_path) in rel_err

    # Escape outside workspace rejected
    esc_err = toolkit.execute("read_file", {"path": "/etc/passwd"})
    assert "escapes workspace root" in esc_err

    # Out-of-scope write rejected
    scope_err = toolkit.execute(
        "write_file",
        {"path": str(blocked_file), "content": "y = 99\n"},
    )
    assert "outside allowed write_scope" in scope_err
    assert blocked_file.read_text(encoding="utf-8") == "y = 2\n"

    # In-scope edit succeeds
    ok = toolkit.execute(
        "edit_file",
        {"path": str(allowed_file), "target": "x = 1", "replacement": "x = 42"},
    )
    assert "1 replacement" in ok
    assert allowed_file.read_text(encoding="utf-8") == "x = 42\n"


def test_aci_read_only_hides_write_tools_and_truncates_with_hint(tmp_path: Path) -> None:
    big = tmp_path / "big.txt"
    big.write_text("\n".join(f"line-{i}" for i in range(500)), encoding="utf-8")

    ro = WorkspaceToolkit(tmp_path, read_only=True, max_output_chars=300)
    names = [t.name for t in ro.tool_specs()]
    assert names == ["glob_files", "grep_files", "read_file"]

    out = ro.execute("read_file", {"path": str(big), "start_line": 1, "end_line": 200})
    assert "[TRUNCATED:" in out
    assert "start_line=" in out


def test_git_worktree_sandbox_isolates_changes_and_captures_diff(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.py").write_text("def f() -> int:\n    return 1\n", encoding="utf-8")
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Tester"], cwd=repo, check=True)
    subprocess.run(["git", "add", "app.py"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo, check=True, capture_output=True)

    handle = create_sandbox(repo, label="task-1-a1")
    try:
        assert handle.is_git_worktree is True
        assert handle.base_commit is not None

        # Modify tracked + add untracked file inside sandbox only
        (handle.workspace_path / "app.py").write_text(
            "def f() -> int:\n    return 2\n", encoding="utf-8"
        )
        (handle.workspace_path / "new_mod.py").write_text("VAL = 9\n", encoding="utf-8")

        diff = capture_sandbox_diff(handle)
        assert "return 2" in diff
        assert "new_mod.py" in diff
        assert "VAL = 9" in diff
        # Main repo untouched
        assert "return 1" in (repo / "app.py").read_text(encoding="utf-8")
        assert not (repo / "new_mod.py").exists()
    finally:
        remove_sandbox(handle)
    assert not handle.workspace_path.exists()


def test_agent_loop_preserves_prefix_and_errors_and_respects_budget(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    target = ws / "hello.txt"
    target.write_text("hello world\n", encoding="utf-8")

    toolkit = WorkspaceToolkit(ws, allowed_write_scope=["hello.txt"])
    store = RunStore(tmp_path / "state.db")
    run_id = store.create_run(title="Loop test", budget_policy=BudgetPolicy(max_llm_calls=5))
    trace_path = tmp_path / "trace.jsonl"

    # Step 1: intentional bad relative path -> error stays in context
    # Step 2: corrected absolute path -> succeeds
    # Step 3: final answer
    provider = MockLLMProvider(
        responses=[
            LLMResponse(
                content="Try reading file",
                tool_calls=(
                    ToolCall(id="tc-1", name="read_file", arguments={"path": "hello.txt"}),
                ),
                usage=UsageStats(input_tokens=100, output_tokens=20),
            ),
            LLMResponse(
                content="Use absolute path instead",
                tool_calls=(
                    ToolCall(id="tc-2", name="read_file", arguments={"path": str(target)}),
                ),
                usage=UsageStats(input_tokens=140, output_tokens=25),
            ),
            LLMResponse(
                content="Done reading hello world.",
                tool_calls=(),
                usage=UsageStats(input_tokens=180, output_tokens=15),
            ),
        ]
    )

    loop = AgentLoop(
        provider=provider,
        toolkit=toolkit,
        store=store,
        run_id=run_id,
        actor="hephaestus",
        max_steps=5,
        trace_path=trace_path,
    )
    res = loop.run(system_prompt="SYS", user_prompt="Read hello.txt")

    assert res.completed is True
    assert res.steps_taken == 3
    assert res.final_content == "Done reading hello world."
    assert res.total_input_tokens == 420
    assert res.total_output_tokens == 60

    # Verify exact prefix invariance across every LLM call (KV-cache property)
    req0_msgs, req0_tools = provider.recorded_requests[0]
    req1_msgs, req1_tools = provider.recorded_requests[1]
    req2_msgs, req2_tools = provider.recorded_requests[2]
    assert req0_tools == req1_tools == req2_tools
    assert req1_msgs[: len(req0_msgs)] == req0_msgs
    assert req2_msgs[: len(req1_msgs)] == req1_msgs
    # Failed tool output preserved in messages
    assert "Relative path" in req1_msgs[-1].content

    # Verify trace.jsonl and monotonic budget ledger
    traces = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    assert len(traces) == 3
    assert len({t["prefix_hash"] for t in traces}) == 1
    snap = store.get_run(run_id)
    assert snap["budget_ledger"].llm_calls_used == 3


def test_agent_loop_stops_on_budget_exhaustion_and_max_steps(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "state.db")
    run_id = store.create_run(title="Budget stop", budget_policy=BudgetPolicy(max_llm_calls=1))

    provider = MockLLMProvider(
        responses=[
            LLMResponse(
                content="Call tool",
                tool_calls=(ToolCall(id="tc-1", name="read_file", arguments={"path": "/missing"}),),
            ),
            LLMResponse(content="Should not reach"),
        ]
    )
    loop = AgentLoop(provider=provider, store=store, run_id=run_id, max_steps=5)
    res = loop.run(system_prompt="SYS", user_prompt="Go")
    assert res.completed is False
    assert res.stop_reason.startswith("budget_exhausted")
    assert res.steps_taken == 1

    # Also check max_steps stopping condition independent of budget
    loop2 = AgentLoop(
        provider=MockLLMProvider(
            responses=[
                LLMResponse(
                    content="step1",
                    tool_calls=(ToolCall(id="tc-1", name="read_file", arguments={"path": "/a"}),),
                ),
                LLMResponse(
                    content="step2",
                    tool_calls=(ToolCall(id="tc-2", name="read_file", arguments={"path": "/b"}),),
                ),
            ]
        ),
        max_steps=2,
    )
    res2 = loop2.run(system_prompt="SYS", user_prompt="Go")
    assert res2.completed is False
    assert res2.stop_reason == "max_steps_exceeded"
    assert res2.steps_taken == 2
