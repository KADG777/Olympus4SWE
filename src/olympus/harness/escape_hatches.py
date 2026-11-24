"""0-token diff pre-gate rejecting newly added escape hatches and silent excepts.

The gate is a ratchet: only lines added by the current diff are scanned, so a
pre-existing escape hatch never fails a run that did not touch it. Files that are
binary, deleted, missing, or unparseable are skipped instead of crashing.
"""

from __future__ import annotations

import ast
import difflib
import io
import re
import tokenize
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

_PY_SUFFIXES = frozenset({".py"})
_JS_SUFFIXES = frozenset({".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"})

_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
_NOQA_RE = re.compile(r"^#\s*noqa\b(.*)$", re.IGNORECASE)
_NOQA_VALID_RE = re.compile(
    r"^#\s*noqa:\s*[A-Z0-9][A-Z0-9,\s_]*(?:,\s*[A-Z0-9_]+)*\s+(?:--|-|—|#)\s+\S+",
    re.IGNORECASE,
)
_TYPE_IGNORE_RE = re.compile(r"^#\s*type:\s*ignore\b(.*)$")
_TYPE_IGNORE_VALID_RE = re.compile(r"^#\s*type:\s*ignore\[[^\]]+\]\s*(?:--|#|—)\s+\S+")
_JS_TS_IGNORE_RE = re.compile(r"^\s*@(ts-ignore|ts-expect-error)\b(.*)$")
_JS_ESLINT_RE = re.compile(r"^\s*(eslint-disable(?:-next-line|-line)?)\b(.*)$")

_MAX_REPORTED_VIOLATIONS = 20

_GIT_SIMPLE_ESCAPES: dict[str, int] = {
    "a": 0x07,
    "b": 0x08,
    "f": 0x0C,
    "n": 0x0A,
    "r": 0x0D,
    "t": 0x09,
    "v": 0x0B,
    "\\": ord("\\"),
    '"': ord('"'),
}


@dataclass(frozen=True)
class EscapeHatchViolation:
    file_path: str
    line: int
    rule: str
    message: str

    def format(self) -> str:
        return f"{self.file_path}:{self.line}: [{self.rule}] {self.message}"


@dataclass(frozen=True)
class EscapeHatchResult:
    violations: tuple[EscapeHatchViolation, ...]

    @property
    def rejected(self) -> bool:
        return bool(self.violations)

    def rejection_message(self) -> str:
        if not self.violations:
            return ""
        lines = [
            f"Escape-hatch pre-gate rejected {len(self.violations)} newly added line(s):"
        ]
        shown = self.violations[:_MAX_REPORTED_VIOLATIONS]
        lines.extend(f"- {violation.format()}" for violation in shown)
        if len(self.violations) > len(shown):
            lines.append(f"- ... and {len(self.violations) - len(shown)} more")
        lines.append(
            "Fix the underlying defect, or annotate the suppression with an explicit reason."
        )
        return "\n".join(lines)


def _unescape_git_quoted(raw: str) -> str:
    """Decode a Git C-quoted path (without surrounding double quotes) into UTF-8."""
    buf = bytearray()
    index = 0
    length = len(raw)
    while index < length:
        ch = raw[index]
        if ch != "\\" or index + 1 >= length:
            buf.extend(ch.encode("utf-8"))
            index += 1
            continue
        nxt = raw[index + 1]
        if (
            nxt in "01234567"
            and index + 3 < length
            and all(c in "01234567" for c in raw[index + 1 : index + 4])
        ):
            buf.append(int(raw[index + 1 : index + 4], 8) & 0xFF)
            index += 4
        elif nxt in _GIT_SIMPLE_ESCAPES:
            buf.append(_GIT_SIMPLE_ESCAPES[nxt])
            index += 2
        else:
            buf.extend(nxt.encode("utf-8"))
            index += 2
    return buf.decode("utf-8", errors="replace")


def _parse_diff_header_path(header_line: str, marker: str) -> tuple[bool, str | None]:
    """Parse a ``--- `` or ``+++ `` diff header, handling Git C-quoted paths.

    Returns ``(is_valid_header, repo_relative_path_or_None_for_dev_null)``.
    """
    if not header_line.startswith(marker):
        return False, None
    rest = header_line[len(marker) :].split("\t", 1)[0].rstrip("\r\n")
    if not rest:
        return False, None
    if rest == "/dev/null":
        return True, None
    if len(rest) >= 2 and rest[0] == '"' and rest[-1] == '"':
        decoded = _unescape_git_quoted(rest[1:-1])
    else:
        decoded = rest
    if decoded == "/dev/null":
        return True, None
    expected_prefix = "a/" if marker == "--- " else "b/"
    if decoded.startswith(expected_prefix):
        return True, decoded[len(expected_prefix) :]
    return False, None


@dataclass
class _ParsedDiffFile:
    added_lines: set[int]
    synthetic_full_overlay: bool = False


def _collect_added_files(diff_text: str) -> dict[str, _ParsedDiffFile]:
    """Map repo-relative paths to post-image line numbers added by this diff.

    Tracks exact ``old_rem`` / ``new_rem`` counts per hunk so content lines that
    happen to start with ``--- ``, ``+++ ``, ``@@ ``, or ``diff --git `` never
    corrupt file header parsing. Also handles the raw-content new-file blocks
    emitted by ``_head_scoped_diff`` (``--- /dev/null`` / ``+++ b/<rel>`` with no
    hunk header).
    """
    files: dict[str, _ParsedDiffFile] = {}
    lines = diff_text.splitlines()
    index = 0
    current: str | None = None
    source_is_existing = False
    mode = "idle"
    next_lineno = 0
    old_rem = 0
    new_rem = 0
    raw_lineno = 0

    while index < len(lines):
        line = lines[index]
        if mode == "hunk" and current is not None:
            if line.startswith("\\"):
                index += 1
                continue
            if old_rem > 0 or new_rem > 0:
                if line.startswith("+") and new_rem > 0:
                    files[current].added_lines.add(next_lineno)
                    next_lineno += 1
                    new_rem -= 1
                    index += 1
                    if old_rem == 0 and new_rem == 0:
                        mode = "await_hunk"
                    continue
                if line.startswith("-") and old_rem > 0:
                    old_rem -= 1
                    index += 1
                    if old_rem == 0 and new_rem == 0:
                        mode = "await_hunk"
                    continue
                if (line.startswith(" ") or line == "") and old_rem > 0 and new_rem > 0:
                    old_rem -= 1
                    new_rem -= 1
                    next_lineno += 1
                    index += 1
                    if old_rem == 0 and new_rem == 0:
                        mode = "await_hunk"
                    continue
            mode = "await_hunk"

        if line.startswith("diff --git "):
            current = None
            source_is_existing = False
            mode = "idle"
            index += 1
            continue

        if index + 1 < len(lines):
            src_ok, src_path = _parse_diff_header_path(line, "--- ")
            dst_ok, dst_path = _parse_diff_header_path(lines[index + 1], "+++ ")
            if src_ok and dst_ok:
                current = dst_path
                source_is_existing = src_path is not None
                if current is not None:
                    files.setdefault(current, _ParsedDiffFile(added_lines=set()))
                    following = lines[index + 2] if index + 2 < len(lines) else ""
                    if following.startswith("@@ "):
                        mode = "await_hunk"
                    else:
                        mode = "raw"
                        raw_lineno = 0
                else:
                    mode = "idle"
                index += 2
                continue

        if line.startswith("@@ ") and current is not None and mode != "raw":
            match = _HUNK_RE.match(line)
            if match is not None:
                old_start = int(match.group(1))
                old_rem = int(match.group(2)) if match.group(2) is not None else 1
                next_lineno = int(match.group(3))
                new_rem = int(match.group(4)) if match.group(4) is not None else 1
                if source_is_existing and old_start == 0 and old_rem == 0 and next_lineno == 1:
                    files[current].synthetic_full_overlay = True
                mode = "hunk" if (old_rem > 0 or new_rem > 0) else "await_hunk"
            index += 1
            continue

        if mode == "raw" and current is not None:
            raw_lineno += 1
            files[current].added_lines.add(raw_lineno)
            index += 1
            continue

        index += 1
    return files


def _collect_added_lines(diff_text: str) -> dict[str, set[int]]:
    """Map repo-relative paths to the post-image line numbers this diff adds."""
    return {path: info.added_lines for path, info in _collect_added_files(diff_text).items()}


def _added_lines_from_baseline(before_text: str, after_text: str) -> set[int]:
    """Compute exact 1-based post-image added line numbers via SequenceMatcher."""
    before_lines = before_text.splitlines()
    after_lines = after_text.splitlines()
    matcher = difflib.SequenceMatcher(None, before_lines, after_lines, autojunk=False)
    added: set[int] = set()
    for tag, _i1, _i2, j1, j2 in matcher.get_opcodes():
        if tag in ("insert", "replace"):
            added.update(range(j1 + 1, j2 + 1))
    return added


def _read_baseline_text(rel_path: str, baseline_roots: Sequence[Path]) -> str | None:
    for base_root in baseline_roots:
        candidate = base_root / rel_path
        try:
            if not candidate.is_file() or candidate.is_symlink():
                continue
            raw = candidate.read_bytes()
        except OSError:
            continue
        if b"\x00" in raw:
            return None
        return raw.decode("utf-8", errors="replace")
    return None


def _is_silent_pass_body(body: list[ast.stmt]) -> bool:
    meaningful = [
        stmt
        for stmt in body
        if not (
            isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Constant)
            and isinstance(stmt.value.value, str)
        )
    ]
    return len(meaningful) == 1 and isinstance(meaningful[0], ast.Pass)


def _catches_broad_exception(exc_type: ast.expr | None) -> bool:
    if exc_type is None:
        return True
    if isinstance(exc_type, ast.Name):
        return exc_type.id in {"Exception", "BaseException"}
    if isinstance(exc_type, ast.Tuple):
        return any(_catches_broad_exception(element) for element in exc_type.elts)
    return False


def _handler_body_lines(body: list[ast.stmt]) -> set[int]:
    lines: set[int] = set()
    for stmt in body:
        start = getattr(stmt, "lineno", None)
        end = getattr(stmt, "end_lineno", start)
        if isinstance(start, int):
            last = end if isinstance(end, int) and end >= start else start
            lines.update(range(start, last + 1))
    return lines


def _python_comment_violations(
    text: str, rel_path: str, added_lines: set[int]
) -> list[EscapeHatchViolation]:
    violations: list[EscapeHatchViolation] = []
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return violations
    for token in tokens:
        if token.type != tokenize.COMMENT:
            continue
        lineno = token.start[0]
        if lineno not in added_lines:
            continue
        comment = token.string.strip()
        if _NOQA_RE.match(comment) and not _NOQA_VALID_RE.match(comment):
            violations.append(
                EscapeHatchViolation(
                    rel_path,
                    lineno,
                    "bare-noqa",
                    "Bare '# noqa' without a rule code and reason; fix the lint or write "
                    "'# noqa: <CODE> -- <reason>'.",
                )
            )
        if _TYPE_IGNORE_RE.match(comment) and not _TYPE_IGNORE_VALID_RE.match(comment):
            violations.append(
                EscapeHatchViolation(
                    rel_path,
                    lineno,
                    "bare-type-ignore",
                    "Bare '# type: ignore' without '[code]' and a reason; fix the typing or write "
                    "'# type: ignore[code] # <reason>'.",
                )
            )
    return violations


def _python_ast_violations(
    text: str, rel_path: str, added_lines: set[int]
) -> list[EscapeHatchViolation]:
    violations: list[EscapeHatchViolation] = []
    try:
        tree = ast.parse(text, filename=rel_path)
    except (SyntaxError, ValueError):
        return violations
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        header_added = node.lineno in added_lines
        body_added = bool(_handler_body_lines(node.body) & added_lines)
        if node.type is None:
            if header_added:
                violations.append(
                    EscapeHatchViolation(
                        rel_path,
                        node.lineno,
                        "bare-except",
                        "Bare 'except:' swallows BaseException/KeyboardInterrupt; catch a specific "
                        "exception.",
                    )
                )
        elif (
            _catches_broad_exception(node.type)
            and _is_silent_pass_body(node.body)
            and (header_added or body_added)
        ):
            report_line = (
                node.lineno
                if header_added
                else min(_handler_body_lines(node.body) & added_lines)
            )
            violations.append(
                EscapeHatchViolation(
                    rel_path,
                    report_line,
                    "silent-except",
                    "Silent 'except Exception: pass' hides failures; handle, re-raise, or log "
                    "with context.",
                )
            )
    return violations


_REGEX_PREFIX_TOKENS = frozenset(
    {
        "=",
        "(",
        "[",
        "{",
        "}",
        ",",
        ";",
        ":",
        "!",
        "&",
        "|",
        "?",
        "+",
        "-",
        "*",
        "%",
        "^",
        "~",
        "<",
        ">",
        "\n",
    }
)
_REGEX_PREFIX_KEYWORDS = frozenset(
    {"return", "case", "throw", "in", "of", "typeof", "void", "delete", "new", "yield", "await"}
)
_CONTROL_PAREN_KEYWORDS = frozenset({"if", "while", "for", "with"})


def _iter_js_comments(text: str) -> list[tuple[int, str, bool]]:
    """Extract ``(start_line, comment_body, is_block)`` skipping strings, templates, and regexes."""
    comments: list[tuple[int, str, bool]] = []
    index = 0
    length = len(text)
    line = 1
    can_start_regex = True
    prev_ident = ""
    paren_allows_regex: list[bool] = []
    while index < length:
        ch = text[index]
        if ch == "\n":
            line += 1
            index += 1
            continue
        if ch in (" ", "\t", "\r"):
            index += 1
            continue
        if ch in ("'", '"'):
            quote = ch
            index += 1
            while index < length:
                c = text[index]
                if c == "\\":
                    if index + 1 < length and text[index + 1] == "\n":
                        line += 1
                    index += 2
                    continue
                if c == "\n":
                    line += 1
                    index += 1
                    break
                index += 1
                if c == quote:
                    break
            can_start_regex = False
            prev_ident = ""
            continue
        if ch == "`":
            index += 1
            while index < length:
                c = text[index]
                if c == "\\":
                    if index + 1 < length and text[index + 1] == "\n":
                        line += 1
                    index += 2
                    continue
                if c == "\n":
                    line += 1
                index += 1
                if c == "`":
                    break
            can_start_regex = False
            prev_ident = ""
            continue
        if ch == "/" and index + 1 < length:
            nxt = text[index + 1]
            if nxt == "/":
                start_line = line
                index += 2
                start = index
                while index < length and text[index] != "\n":
                    index += 1
                comments.append((start_line, text[start:index], False))
                continue
            if nxt == "*":
                start_line = line
                index += 2
                start = index
                while index < length:
                    if text[index] == "*" and index + 1 < length and text[index + 1] == "/":
                        comments.append((start_line, text[start:index], True))
                        index += 2
                        break
                    if text[index] == "\n":
                        line += 1
                    index += 1
                else:
                    comments.append((start_line, text[start:], True))
                continue
            if can_start_regex:
                index += 1
                in_class = False
                while index < length:
                    c = text[index]
                    if c == "\\":
                        index += 2
                        continue
                    if c == "\n":
                        line += 1
                        index += 1
                        break
                    if c == "[":
                        in_class = True
                    elif c == "]":
                        in_class = False
                    elif c == "/" and not in_class:
                        index += 1
                        while index < length and text[index].isalpha():
                            index += 1
                        break
                    index += 1
                can_start_regex = False
                prev_ident = ""
                continue
        if ch.isalpha() or ch in ("_", "$"):
            start = index
            while index < length and (text[index].isalnum() or text[index] in ("_", "$")):
                index += 1
            ident = text[start:index]
            can_start_regex = ident in _REGEX_PREFIX_KEYWORDS
            prev_ident = ident
            continue
        if ch.isdigit():
            while index < length and (text[index].isalnum() or text[index] in (".", "_")):
                index += 1
            can_start_regex = False
            prev_ident = ""
            continue
        if ch == "(":
            paren_allows_regex.append(prev_ident in _CONTROL_PAREN_KEYWORDS)
            can_start_regex = True
            prev_ident = ""
            index += 1
            continue
        if ch == ")":
            can_start_regex = paren_allows_regex.pop() if paren_allows_regex else False
            prev_ident = ""
            index += 1
            continue
        can_start_regex = ch in _REGEX_PREFIX_TOKENS
        prev_ident = ""
        index += 1
    return comments


def _js_violations(
    text: str, rel_path: str, added_lines: set[int]
) -> list[EscapeHatchViolation]:
    violations: list[EscapeHatchViolation] = []
    for lineno, comment_body, is_block in _iter_js_comments(text):
        if lineno not in added_lines:
            continue
        body = comment_body.strip()
        if not is_block and (ts_match := _JS_TS_IGNORE_RE.match(body)) is not None:
            directive = ts_match.group(1)
            tail = ts_match.group(2).strip()
            if directive == "ts-ignore" or not tail or tail in {"--", ":"}:
                violations.append(
                    EscapeHatchViolation(
                        rel_path,
                        lineno,
                        f"bare-{directive}",
                        f"Avoid '@{directive}' without an explanation; fix the types or use "
                        "'@ts-expect-error <reason>'.",
                    )
                )
        eslint_match = _JS_ESLINT_RE.match(body)
        if eslint_match is not None:
            directive = eslint_match.group(1)
            tail = eslint_match.group(2).strip()
            if not re.search(r"--\s+\S+", tail):
                violations.append(
                    EscapeHatchViolation(
                        rel_path,
                        lineno,
                        "bare-eslint-disable",
                        f"Bare '{directive}' without '-- <reason>'; fix the lint or document "
                        "'-- <reason>'.",
                    )
                )
    return violations


def scan_diff_escape_hatches(
    diff_text: str,
    workspace_root: Path,
    *,
    baseline_roots: Sequence[Path] = (),
) -> EscapeHatchResult:
    """Return violations for escape hatches introduced on lines this diff adds."""
    root = Path(workspace_root)
    parsed_files = _collect_added_files(diff_text)
    violations: list[EscapeHatchViolation] = []
    for rel_path, parsed in sorted(parsed_files.items()):
        added_lines = parsed.added_lines
        if not added_lines:
            continue
        relative = PurePosixPath(rel_path)
        suffix = relative.suffix.lower()
        if suffix not in _PY_SUFFIXES and suffix not in _JS_SUFFIXES:
            continue
        if relative.is_absolute() or ".." in relative.parts:
            continue
        file_path = root / rel_path
        try:
            if not file_path.is_file():
                continue
            text = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if "\x00" in text:
            continue
        if parsed.synthetic_full_overlay and baseline_roots:
            before_text = _read_baseline_text(rel_path, baseline_roots)
            if before_text is not None:
                added_lines = _added_lines_from_baseline(before_text, text)
                if not added_lines:
                    continue
        if suffix in _PY_SUFFIXES:
            violations.extend(_python_comment_violations(text, rel_path, added_lines))
            violations.extend(_python_ast_violations(text, rel_path, added_lines))
        else:
            violations.extend(_js_violations(text, rel_path, added_lines))
    violations.sort(key=lambda item: (item.file_path, item.line, item.rule))
    return EscapeHatchResult(violations=tuple(violations))
