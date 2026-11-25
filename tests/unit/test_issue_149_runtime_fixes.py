"""Unit tests for Issue #149: NODE_OPTIONS isolation, sandbox-relative repros, and heading lines."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

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
from olympus.providers.base import MockLLMProvider
from olympus.stages.align import _resynthesis_fidelity_violations
from olympus.stages.orchestrate import OrchestrateStage

_KEEPALIVE_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "olympus"
    / "harness"
    / "pi_net_keepalive.cjs"
)


@pytest.mark.skipif(shutil.which("node") is None, reason="node binary not available")
@pytest.mark.parametrize(
    ("node_options_template", "expected_preserved"),
    [
        ("--require={preload}", None),
        ('"--require={preload}"', None),
        ('--require="{preload}"', None),
        ("--max-old-space-size=2048 --require={preload}", "--max-old-space-size=2048"),
        ("--require {preload} --max-old-space-size=1024", "--max-old-space-size=1024"),
        ("-r {preload}", None),
    ],
)
def test_pi_net_keepalive_strips_itself_from_node_options_for_children(
    node_options_template: str,
    expected_preserved: str | None,
) -> None:
    node = shutil.which("node")
    assert node is not None
    preload = str(_KEEPALIVE_SCRIPT.resolve())
    env = dict(os.environ)
    env["NODE_OPTIONS"] = node_options_template.format(preload=preload)

    probe = (
        "const net = require('net');"
        "const { execFileSync } = require('child_process');"
        "const childOut = execFileSync(process.execPath, ['-e', "
        "\"const net = require('net'); console.log(JSON.stringify({"
        "patched: Boolean(net.Socket.prototype.connect.__olympusKeepAlivePatched), "
        "nodeOptions: process.env.NODE_OPTIONS ?? null}))\"], { encoding: 'utf8' }).trim();"
        "console.log(JSON.stringify({"
        "parentPatched: Boolean(net.Socket.prototype.connect.__olympusKeepAlivePatched), "
        "parentNodeOptions: process.env.NODE_OPTIONS ?? null, "
        "child: JSON.parse(childOut)}));"
    )
    proc = subprocess.run(
        [node, "-e", probe],
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
    )
    data = json.loads(proc.stdout.strip())
    assert data["parentPatched"] is True
    assert data["parentNodeOptions"] == expected_preserved
    assert data["child"]["patched"] is False
    assert data["child"]["nodeOptions"] == expected_preserved


@pytest.mark.skipif(shutil.which("node") is None, reason="node binary not available")
def test_pi_net_keepalive_preserves_distinct_preload_and_strips_renamed_symlink(
    tmp_path: Path,
) -> None:
    node = shutil.which("node")
    assert node is not None
    other_dir = tmp_path / "other preload"
    other_dir.mkdir()
    other_preload = other_dir / "pi_net_keepalive.cjs"
    other_preload.write_text("globalThis.__otherLoaded = true;\n", encoding="utf-8")
    apostrophe_preload = tmp_path / "other's.cjs"
    apostrophe_preload.write_text("globalThis.__apostropheLoaded = true;\n", encoding="utf-8")

    renamed_link = tmp_path / "keepalive_alias.cjs"
    renamed_link.symlink_to(_KEEPALIVE_SCRIPT.resolve())

    env = dict(os.environ)
    env["NODE_OPTIONS"] = (
        f"--require={apostrophe_preload} "
        f'--require="{other_preload}" '
        f'--require="{renamed_link}"'
    )
    probe = (
        "const { execFileSync } = require('child_process');"
        "const childOut = execFileSync(process.execPath, ['-e', "
        "\"const net = require('net'); console.log(JSON.stringify({"
        "other: Boolean(globalThis.__otherLoaded), "
        "apostrophe: Boolean(globalThis.__apostropheLoaded), "
        "patched: Boolean(net.Socket.prototype.connect.__olympusKeepAlivePatched)}))\"], "
        "{ encoding: 'utf8' }).trim();"
        "console.log(JSON.stringify({"
        "parentNodeOptions: process.env.NODE_OPTIONS ?? null, "
        "child: JSON.parse(childOut)}));"
    )
    proc = subprocess.run(
        [node, "-e", probe],
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
    )
    data = json.loads(proc.stdout.strip())
    assert str(other_preload) in (data["parentNodeOptions"] or "")
    assert str(apostrophe_preload) in (data["parentNodeOptions"] or "")
    assert str(renamed_link) not in (data["parentNodeOptions"] or "")
    assert data["child"]["other"] is True
    assert data["child"]["apostrophe"] is True
    assert data["child"]["patched"] is False


def test_normalize_repro_command_strips_sandbox_cd_and_rewrites_sibling_paths(
    tmp_path: Path,
) -> None:
    sandbox_root = tmp_path / "sandboxes"
    review_sandbox = sandbox_root / "run-a2cdf582011d-global-review-2"
    repro_sandbox = sandbox_root / "run-a2cdf582011d-postdag-repro-3"

    raw = f"cd {review_sandbox} && node --input-type=module -e \"console.log('ok')\""
    normalized_at_review = OrchestrateStage._normalize_repro_command(
        raw, sandbox_path=review_sandbox, sandbox_root=sandbox_root
    )
    assert normalized_at_review == "node --input-type=module -e \"console.log('ok')\""

    normalized_at_recheck = OrchestrateStage._normalize_repro_command(
        raw, sandbox_path=repro_sandbox, sandbox_root=sandbox_root
    )
    assert normalized_at_recheck == "node --input-type=module -e \"console.log('ok')\""

    quoted = f"cd '{review_sandbox}' ; pytest {review_sandbox}/tests/test_app.py"
    assert (
        OrchestrateStage._normalize_repro_command(
            quoted, sandbox_path=repro_sandbox, sandbox_root=sandbox_root
        )
        == "pytest __OLYMPUS_SANDBOX_ROOT__/tests/test_app.py"
    )

    subshell = f"(cd {review_sandbox})"
    normalized_subshell = OrchestrateStage._normalize_repro_command(
        subshell, sandbox_path=repro_sandbox, sandbox_root=sandbox_root
    )
    assert normalized_subshell == "(cd __OLYMPUS_SANDBOX_ROOT__)"
    syntax = subprocess.run(
        ["bash", "-n", "-c", normalized_subshell], capture_output=True, check=False
    )
    assert syntax.returncode == 0

    mid_cd = f'cd web && (cd "{review_sandbox}" && pwd)'
    assert (
        OrchestrateStage._normalize_repro_command(
            mid_cd, sandbox_path=repro_sandbox, sandbox_root=sandbox_root
        )
        == 'cd web && (cd "__OLYMPUS_SANDBOX_ROOT__" && pwd)'
    )

    custom_label = f"cd '{sandbox_root}/custom-global-review-2/web' && ls"
    assert (
        OrchestrateStage._normalize_repro_command(
            custom_label, sandbox_path=repro_sandbox, sandbox_root=None
        )
        == "cd '__OLYMPUS_SANDBOX_ROOT__/web' && ls"
    )

    relative = "cd web && npm test"
    assert (
        OrchestrateStage._normalize_repro_command(
            relative, sandbox_path=repro_sandbox, sandbox_root=sandbox_root
        )
        == "cd web && npm test"
    )


def test_recheck_post_dag_repros_ignores_hardcoded_old_review_sandbox(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    (workspace / "app.txt").write_text("INITIAL\n", encoding="utf-8")
    subprocess.run(["git", "init"], cwd=workspace, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "dev@example.com"], cwd=workspace, check=True)
    subprocess.run(["git", "config", "user.name", "Dev"], cwd=workspace, check=True)
    subprocess.run(["git", "add", "app.txt"], cwd=workspace, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=workspace, check=True, capture_output=True)

    sandbox_root = tmp_path / "sandboxes"
    stale_review_sandbox = sandbox_root / "run-test123-global-review-2"
    stale_review_sandbox.mkdir(parents=True)
    (stale_review_sandbox / "app.txt").write_text("BROKEN\n", encoding="utf-8")

    artifacts_root = tmp_path / "runs"
    store = RunStore(tmp_path / "olympus.db")
    run_id = store.create_run(title="repro-test", workflow_name="full_sdlc_tdd")
    run_dir = artifacts_root / run_id
    overlay = run_dir / "accepted_overlay"
    overlay.mkdir(parents=True)
    (overlay / "app.txt").write_text("FIXED\n", encoding="utf-8")

    stage = OrchestrateStage(
        store=store,
        provider=MockLLMProvider(),
        workspace_root=workspace,
        artifacts_root=artifacts_root,
        sandbox_root=sandbox_root,
    )
    stale_cmd = f"cd {stale_review_sandbox} && grep -q FIXED app.txt"
    inline_single_quote_cmd = (
        f"python3 -c 'import pathlib, sys; "
        f'sys.exit(0 if "FIXED" in pathlib.Path("{stale_review_sandbox}/app.txt").read_text() '
        f"else 1)'"
    )
    failures, still_failing = stage._recheck_post_dag_repros(
        run_id=run_id,
        run_dir=run_dir,
        repro_commands=[stale_cmd, inline_single_quote_cmd],
        repair_round=2,
    )
    assert failures == []
    assert still_failing == []


def test_resynthesis_fidelity_skips_pure_heading_lines_ending_in_colon() -> None:
    current = GoalSpec(
        spec_id="spec-bn",
        goal="构建 AI 工程师图文社区",
        deliverable_kind=DeliverableKind.WEB_APP,
        observable_surfaces=["Web 信息流"],
        actors=["AI 工程师"],
        test_seams=["HTTP API"],
        requirements=[
            RequirementItem(
                id="R1",
                actor="AI 工程师",
                description="浏览与发布笔记。",
                ears_criteria=[
                    AcceptanceCriterion(
                        id="AC-1",
                        pattern=EarsPattern.UBIQUITOUS,
                        outcome="展示笔记列表",
                    )
                ],
            )
        ],
        open_ambiguities=[
            AmbiguityItem(
                id="A1",
                question="形态？",
                impact="交付形态",
                default_assumption="Web 应用",
                answer="Web 应用",
                blocking=True,
            )
        ],
        status=SpecStatus.AWAITING_HUMAN,
    )
    candidate = current.model_copy(
        update={
            "requirements": [
                RequirementItem(
                    id="R1",
                    actor="AI 工程师",
                    description=(
                        "平台固定包含 AI 新闻 / AI 工具 / AI 产品 三类话题并支持自定义标签。"
                    ),
                    ears_criteria=[
                        AcceptanceCriterion(
                            id="AC-1",
                            pattern=EarsPattern.UBIQUITOUS,
                            outcome="提供 AI 新闻 / AI 工具 / AI 产品 三类固定话题筛选",
                        )
                    ],
                )
            ]
        }
    )
    extra_feedback = (
        "以下是已确认的核心边界（来自 BlackNote Dev 基准假设）：\n"
        "1. 话题固定包含 AI 新闻 / AI 工具 / AI 产品 三类并支持自定义标签\n"
        "针对各阻塞项的明确答复："
    )
    assert _resynthesis_fidelity_violations(current, candidate, extra_feedback=extra_feedback) == []

    # Substantive items ending in a colon MUST still be enforced even with intro-like prefixes.
    for extra_directive in ("2. 必须禁用 Redis：", "说明：必须启用 Redis："):
        violations = _resynthesis_fidelity_violations(
            current, candidate, extra_feedback=f"{extra_feedback}\n{extra_directive}"
        )
        assert any("Redis" in v for v in violations)
