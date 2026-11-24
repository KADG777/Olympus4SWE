"""Mutation tests for the 0-token diff escape-hatch pre-gate."""

from __future__ import annotations

from pathlib import Path

from olympus.harness.escape_hatches import (
    EscapeHatchResult,
    EscapeHatchViolation,
    scan_diff_escape_hatches,
)


def _write(root: Path, rel_path: str, content: str) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _new_file_diff(rel_path: str, content: str) -> str:
    """Raw new-file block exactly as capture_sandbox_diff emits it."""
    return f"--- /dev/null\n+++ b/{rel_path}\n{content}"


def _added_line_diff(rel_path: str, *, context: list[str], added: list[str]) -> str:
    lines = [f" {line}" for line in context] + [f"+{line}" for line in added]
    return (
        f"--- a/{rel_path}\n+++ b/{rel_path}\n"
        f"@@ -1,{len(context)} +1,{len(context) + len(added)} @@\n" + "\n".join(lines) + "\n"
    )


def test_python_escape_hatches_and_silent_excepts_are_rejected(tmp_path: Path) -> None:
    content = (
        "def load():\n"
        "    try:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        pass\n"
        "\n"
        "def explode():\n"
        "    try:\n"
        "        return 2\n"
        "    except:\n"
        "        return 0\n"
        "\n"
        "VALUE = 1  # type: ignore\n"
        "OTHER = 2  # noqa\n"
    )
    _write(tmp_path, "pkg/mod.py", content)

    result = scan_diff_escape_hatches(_new_file_diff("pkg/mod.py", content), tmp_path)

    assert result.rejected is True
    assert {violation.rule for violation in result.violations} >= {
        "silent-except",
        "bare-except",
        "bare-type-ignore",
        "bare-noqa",
    }
    message = result.rejection_message()
    assert "pkg/mod.py:4" in message
    assert "[silent-except]" in message
    assert "pkg/mod.py:13" in message
    assert "[bare-type-ignore]" in message


def test_annotated_suppressions_and_specific_excepts_pass(tmp_path: Path) -> None:
    content = (
        "import logging\n"
        "\n"
        "LOG = logging.getLogger(__name__)\n"
        "\n"
        "def load():\n"
        "    try:\n"
        "        return 1\n"
        "    except (KeyError, ValueError) as exc:\n"
        "        LOG.warning('fallback', exc_info=exc)\n"
        "        return 2\n"
        "\n"
        "VALUE = 1  # type: ignore[attr-defined] # vendor stub is untyped\n"
        "OTHER = 2  # noqa: E501 -- generated table row\n"
    )
    _write(tmp_path, "ok.py", content)

    result = scan_diff_escape_hatches(_new_file_diff("ok.py", content), tmp_path)

    assert result == EscapeHatchResult(violations=())


def test_js_and_ts_escape_hatches_are_rejected_only_without_reasons(tmp_path: Path) -> None:
    content = (
        "const a = 1; // @ts-ignore\n"
        "const b = 2; // @ts-expect-error\n"
        "const c = 3; // eslint-disable-next-line\n"
        "const d = 4; // @ts-expect-error third-party types are wrong\n"
        "// eslint-disable-next-line no-console -- CLI output is the product surface\n"
    )
    _write(tmp_path, "web/ui.ts", content)

    result = scan_diff_escape_hatches(_new_file_diff("web/ui.ts", content), tmp_path)

    rules = [(violation.line, violation.rule) for violation in result.violations]
    assert rules == [
        (1, "bare-ts-ignore"),
        (2, "bare-ts-expect-error"),
        (3, "bare-eslint-disable"),
    ]


def test_preexisting_escape_hatches_outside_added_lines_are_not_flagged(
    tmp_path: Path,
) -> None:
    content = "LEGACY = 1  # type: ignore\nOLD = 2  # noqa\nNEW = 3\n"
    _write(tmp_path, "legacy.py", content)

    diff = _added_line_diff(
        "legacy.py",
        context=["LEGACY = 1  # type: ignore", "OLD = 2  # noqa"],
        added=["NEW = 3"],
    )
    result = scan_diff_escape_hatches(diff, tmp_path)

    assert result.rejected is False


def test_deleted_missing_and_binary_files_are_skipped(tmp_path: Path) -> None:
    binary = tmp_path / "blob.py"
    binary.write_bytes(b"\x00\x01VALUE = 1  # noqa\n")
    deleted_diff = (
        "diff --git a/gone.py b/gone.py\n"
        "deleted file mode 100644\n"
        "--- a/gone.py\n+++ /dev/null\n"
        "@@ -1,1 +0,0 @@\n-VALUE = 1  # type: ignore\n"
    )
    missing_diff = "--- /dev/null\n+++ b/missing.py\nVALUE = 1  # type: ignore\n"
    binary_diff = "--- /dev/null\n+++ b/blob.py\nVALUE = 1  # noqa\n"

    assert scan_diff_escape_hatches(deleted_diff, tmp_path).rejected is False
    assert scan_diff_escape_hatches(missing_diff, tmp_path).rejected is False
    assert scan_diff_escape_hatches(binary_diff, tmp_path).rejected is False


def test_git_hunks_use_post_image_line_numbers(tmp_path: Path) -> None:
    content = "".join(f"ROW_{index} = {index}\n" for index in range(1, 5))
    content += "TAIL = 1  # type: ignore\n"
    content += "ROW_5 = 5\n"
    _write(tmp_path, "pkg/hunked.py", content)
    diff = (
        "diff --git a/pkg/hunked.py b/pkg/hunked.py\n"
        "--- a/pkg/hunked.py\n+++ b/pkg/hunked.py\n"
        "@@ -4,2 +4,3 @@\n"
        " ROW_4 = 4\n"
        "+TAIL = 1  # type: ignore\n"
        " ROW_5 = 5\n"
    )

    result = scan_diff_escape_hatches(diff, tmp_path)

    assert [(violation.line, violation.rule) for violation in result.violations] == [
        (5, "bare-type-ignore")
    ]

    _write(tmp_path, "web/inc.ts", "let i = 0;\n++ i; // @ts-ignore\n")
    plus_diff = (
        "diff --git a/web/inc.ts b/web/inc.ts\n"
        "--- a/web/inc.ts\n+++ b/web/inc.ts\n"
        "@@ -1,1 +1,2 @@\n"
        " let i = 0;\n"
        "+++ i; // @ts-ignore\n"
    )
    assert [(v.line, v.rule) for v in scan_diff_escape_hatches(plus_diff, tmp_path).violations] == [
        (2, "bare-ts-ignore")
    ]


def test_result_formats_each_violation_for_the_repair_prompt() -> None:
    violation = EscapeHatchViolation(
        file_path="a.py", line=3, rule="bare-noqa", message="Annotate it."
    )
    result = EscapeHatchResult(violations=(violation,))

    assert violation.format() == "a.py:3: [bare-noqa] Annotate it."
    assert "1 newly added line(s)" in result.rejection_message()
    assert "- a.py:3: [bare-noqa] Annotate it." in result.rejection_message()
    assert EscapeHatchResult(violations=()).rejection_message() == ""


def test_git_quoted_paths_and_hunk_content_headers_are_parsed_accurately(
    tmp_path: Path,
) -> None:
    rel = 'dir with space/bad "file".py'
    content = 'HEADER = """\n--- a/fake.py\n+++ b/fake.py\n"""\nX = 1  # noqa\n'
    _write(tmp_path, rel, content)
    diff = (
        f'diff --git "a/{rel}" "b/{rel}"\n'
        "--- /dev/null\n"
        '+++ "b/dir with space/bad \\"file\\".py"\n'
        "@@ -0,0 +1,5 @@\n"
        '+HEADER = """\n'
        "+--- a/fake.py\n"
        "++++ b/fake.py\n"
        '+"""\n'
        "+X = 1  # noqa\n"
    )

    result = scan_diff_escape_hatches(diff, tmp_path)

    assert [(v.file_path, v.line, v.rule) for v in result.violations] == [
        (rel, 5, "bare-noqa")
    ]


def test_changing_existing_except_body_to_pass_is_caught(tmp_path: Path) -> None:
    content = (
        "def run():\n"
        "    try:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        pass\n"
    )
    _write(tmp_path, "pkg/body_only.py", content)
    diff = (
        "diff --git a/pkg/body_only.py b/pkg/body_only.py\n"
        "--- a/pkg/body_only.py\n+++ b/pkg/body_only.py\n"
        "@@ -4,2 +4,2 @@\n"
        "     except Exception:\n"
        "-        raise\n"
        "+        pass\n"
    )

    result = scan_diff_escape_hatches(diff, tmp_path)

    assert [(v.line, v.rule) for v in result.violations] == [(5, "silent-except")]


def test_js_strings_containing_directive_text_are_not_flagged(tmp_path: Path) -> None:
    content = (
        'const a = "// @ts-ignore";\n'
        "const b = '/* eslint-disable */';\n"
        "const c = `// @ts-expect-error`;\n"
        "const d = 1; // @ts-ignore\n"
    )
    _write(tmp_path, "web/strings.ts", content)

    result = scan_diff_escape_hatches(_new_file_diff("web/strings.ts", content), tmp_path)

    assert [(v.line, v.rule) for v in result.violations] == [(4, "bare-ts-ignore")]


def test_non_git_workspace_untouched_legacy_noqa_is_not_flagged(tmp_path: Path) -> None:
    from olympus.harness.worktree import (
        capture_sandbox_diff,
        create_sandbox,
        remove_sandbox,
        snapshot_sandbox_state,
    )

    repo = tmp_path / "nongit"
    repo.mkdir()
    _write(repo, "legacy.py", "OLD = 1  # noqa\nVALUE = 2\n")
    sandbox = create_sandbox(repo, label="nongit-check", sandbox_root=tmp_path / "sandboxes")
    try:
        baseline = snapshot_sandbox_state(sandbox)
        _write(sandbox.workspace_path, "legacy.py", "OLD = 1  # noqa\nVALUE = 3\n")
        diff_text = capture_sandbox_diff(sandbox, baseline)
        result = scan_diff_escape_hatches(
            diff_text, sandbox.workspace_path, baseline_roots=(repo,)
        )
        assert result.rejected is False
        # Even if a prior overlay/attempt wrote a file differing from repo_root before snapshot:
        _write(sandbox.workspace_path, "legacy.py", "OVERLAY_OLD = 1  # noqa\nVALUE = 10\n")
        baseline2 = snapshot_sandbox_state(sandbox)
        _write(sandbox.workspace_path, "legacy.py", "OVERLAY_OLD = 1  # noqa\nVALUE = 11\n")
        diff_text2 = capture_sandbox_diff(sandbox, baseline2)
        assert scan_diff_escape_hatches(diff_text2, sandbox.workspace_path).rejected is False
        # Large (>256KB) non-git file with untouched legacy noqa must also stay clean.
        big_prefix = "OLD = 1  # noqa\n" + ("# padding line\n" * 20_000)
        _write(sandbox.workspace_path, "big.py", big_prefix + "TAIL = 1\n")
        baseline3 = snapshot_sandbox_state(sandbox)
        _write(sandbox.workspace_path, "big.py", big_prefix + "TAIL = 2\n")
        diff_text3 = capture_sandbox_diff(sandbox, baseline3)
        assert scan_diff_escape_hatches(diff_text3, sandbox.workspace_path).rejected is False
    finally:
        remove_sandbox(sandbox)


def test_js_regex_literal_with_quote_does_not_hide_ts_ignore(tmp_path: Path) -> None:
    content = (
        "const re = /'/;\n"
        "// @ts-ignore\n"
        'const re2 = /"/; // @ts-ignore\n'
        "if (true) /'/; // @ts-ignore\n"
        'if (true) /"/; // @ts-ignore\n'
    )
    _write(tmp_path, "web/regex.ts", content)

    result = scan_diff_escape_hatches(_new_file_diff("web/regex.ts", content), tmp_path)

    assert [(v.line, v.rule) for v in result.violations] == [
        (2, "bare-ts-ignore"),
        (3, "bare-ts-ignore"),
        (4, "bare-ts-ignore"),
        (5, "bare-ts-ignore"),
    ]
