"""Static architecture fitness tests enforcing module dependency boundaries.

These tests mechanically protect the layering documented in ``AGENTS.md`` and
``docs/ARCHITECTURE.md`` so unintended cross-layer coupling fails loudly in CI.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OLYMPUS_SRC = REPO_ROOT / "src" / "olympus"
WORKFLOWS_DIR = OLYMPUS_SRC / "graph" / "workflows"

VENDOR_LLM_SDKS = frozenset({"openai", "anthropic"})
WEB_AND_CLI_FRAMEWORKS = frozenset({"fastapi", "uvicorn", "typer"})
OLYMPUS_SUBPACKAGES = frozenset(
    {
        "core",
        "providers",
        "tools",
        "harness",
        "stages",
        "graph",
        "session",
        "web",
        "cli",
    }
)


@dataclass(frozen=True)
class ImportEdge:
    """One resolved import target discovered via AST inspection."""

    file: Path
    lineno: int
    target: str

    @property
    def top_level(self) -> str:
        return self.target.split(".", 1)[0]

    @property
    def olympus_subpackage(self) -> str | None:
        parts = self.target.split(".")
        if len(parts) >= 2 and parts[0] == "olympus":
            return parts[1]
        return None


def _module_name_for_file(file_path: Path, src_root: Path) -> str:
    rel = file_path.relative_to(src_root).with_suffix("")
    parts = list(rel.parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _resolve_relative_module(
    current_module: str,
    *,
    is_package_init: bool,
    level: int,
    module: str | None,
) -> str:
    """Resolve an AST ``ImportFrom`` relative import to a dotted module path."""
    parts = current_module.split(".") if current_module else []
    base = parts if is_package_init else parts[:-1]
    ascend = level - 1
    if ascend > len(base):
        resolved_base: list[str] = []
    elif ascend > 0:
        resolved_base = base[:-ascend]
    else:
        resolved_base = list(base)
    if module:
        resolved_base.extend(module.split("."))
    return ".".join(resolved_base)


def collect_imports_from_source(
    source: str,
    *,
    file_path: Path,
    current_module: str,
    is_package_init: bool = False,
) -> list[ImportEdge]:
    """Parse Python source and return all resolved import targets."""
    tree = ast.parse(source, filename=str(file_path))
    edges: list[ImportEdge] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                edges.append(ImportEdge(file=file_path, lineno=node.lineno, target=alias.name))
        elif isinstance(node, ast.ImportFrom):
            if node.level > 0:
                base_target = _resolve_relative_module(
                    current_module,
                    is_package_init=is_package_init,
                    level=node.level,
                    module=node.module,
                )
                if node.module:
                    edges.append(
                        ImportEdge(file=file_path, lineno=node.lineno, target=base_target)
                    )
                else:
                    for alias in node.names:
                        if alias.name == "*":
                            if base_target:
                                edges.append(
                                    ImportEdge(
                                        file=file_path, lineno=node.lineno, target=base_target
                                    )
                                )
                        else:
                            combined = (
                                f"{base_target}.{alias.name}" if base_target else alias.name
                            )
                            edges.append(
                                ImportEdge(file=file_path, lineno=node.lineno, target=combined)
                            )
            elif node.module:
                if node.module == "olympus":
                    for alias in node.names:
                        target = (
                            "olympus" if alias.name == "*" else f"olympus.{alias.name}"
                        )
                        edges.append(
                            ImportEdge(file=file_path, lineno=node.lineno, target=target)
                        )
                else:
                    edges.append(
                        ImportEdge(file=file_path, lineno=node.lineno, target=node.module)
                    )
    return edges


def _iter_package_imports(package_dir: Path) -> list[ImportEdge]:
    src_root = REPO_ROOT / "src"
    edges: list[ImportEdge] = []
    for py_file in sorted(package_dir.rglob("*.py")):
        if "__pycache__" in py_file.parts:
            continue
        source = py_file.read_text(encoding="utf-8")
        edges.extend(
            collect_imports_from_source(
                source,
                file_path=py_file,
                current_module=_module_name_for_file(py_file, src_root),
                is_package_init=py_file.name == "__init__.py",
            )
        )
    return edges


def _format_violations(violations: list[ImportEdge], hint: str) -> str:
    lines = [
        f"  - {edge.file.relative_to(REPO_ROOT)}:{edge.lineno} imports '{edge.target}'"
        for edge in violations
    ]
    return "Architecture boundary violation(s):\n" + "\n".join(lines) + f"\nHint: {hint}"


def test_core_has_no_outer_layer_or_vendor_dependencies() -> None:
    """``olympus.core`` must stay a pure foundation with zero sibling or SDK imports."""
    forbidden_subpackages = OLYMPUS_SUBPACKAGES - {"core"}
    forbidden_roots = VENDOR_LLM_SDKS | WEB_AND_CLI_FRAMEWORKS | {"tests"}
    violations = [
        edge
        for edge in _iter_package_imports(OLYMPUS_SRC / "core")
        if edge.top_level in forbidden_roots
        or edge.olympus_subpackage in forbidden_subpackages
    ]
    assert not violations, _format_violations(
        violations,
        "Keep olympus.core independent of outer packages, vendor SDKs, and web/CLI frameworks.",
    )


def test_vendor_llm_sdks_confined_to_providers() -> None:
    """Only ``olympus.providers`` may import vendor LLM SDKs directly."""
    violations = [
        edge
        for edge in _iter_package_imports(OLYMPUS_SRC)
        if edge.top_level in VENDOR_LLM_SDKS
        and "providers" not in edge.file.relative_to(OLYMPUS_SRC).parts[:1]
    ]
    assert not violations, _format_violations(
        violations,
        "Route all LLM calls through olympus.providers abstractions instead of vendor SDKs.",
    )


def test_domain_and_runtime_layers_do_not_import_web_or_cli() -> None:
    """Core and runtime packages must remain decoupled from Web/CLI presentation layers."""
    isolated_packages = ("core", "providers", "tools", "harness", "stages", "graph", "session")
    violations: list[ImportEdge] = []
    for pkg in isolated_packages:
        for edge in _iter_package_imports(OLYMPUS_SRC / pkg):
            if edge.top_level in WEB_AND_CLI_FRAMEWORKS or edge.olympus_subpackage in {
                "web",
                "cli",
            }:
                violations.append(edge)
    assert not violations, _format_violations(
        violations,
        "Web and CLI layers should project runtime state, not be imported by core/runtime modules.",
    )


def test_web_does_not_import_cli() -> None:
    """``olympus.web`` must stay decoupled from CLI commands and Typer."""
    violations = [
        edge
        for edge in _iter_package_imports(OLYMPUS_SRC / "web")
        if edge.top_level == "typer" or edge.olympus_subpackage == "cli"
    ]
    assert not violations, _format_violations(
        violations,
        "Share domain logic through core/session/graph rather than importing olympus.cli in web.",
    )


def test_providers_and_tools_respect_bottom_layer_boundaries() -> None:
    """Bottom-layer adapters must not reach upward into orchestration or UI layers."""
    provider_forbidden = OLYMPUS_SUBPACKAGES - {"providers"}
    provider_violations = [
        edge
        for edge in _iter_package_imports(OLYMPUS_SRC / "providers")
        if edge.olympus_subpackage in provider_forbidden
    ]
    assert not provider_violations, _format_violations(
        provider_violations,
        "Keep olympus.providers self-contained as a pluggable LLM compatibility layer.",
    )

    tools_forbidden = {"harness", "stages", "graph", "session", "web", "cli"}
    tools_violations = [
        edge
        for edge in _iter_package_imports(OLYMPUS_SRC / "tools")
        if edge.olympus_subpackage in tools_forbidden
    ]
    assert not tools_violations, _format_violations(
        tools_violations,
        "Keep olympus.tools below harness/stages orchestration layers.",
    )


def test_production_code_never_imports_tests_or_evals() -> None:
    """Enforce one-way dependency: ``tests/evals/ -> src/olympus/``, never the reverse."""
    violations = [
        edge
        for edge in _iter_package_imports(OLYMPUS_SRC)
        if edge.top_level == "tests"
    ]
    assert not violations, _format_violations(
        violations,
        "Production code must stay independent of tests/evals; expose read-only contracts instead.",
    )


def test_declarative_workflows_contain_only_yaml() -> None:
    """Declarative workflow templates under ``graph/workflows/`` must stay data-only YAML."""
    non_yaml = sorted(
        path
        for path in WORKFLOWS_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() not in {".yaml", ".yml"}
    )
    assert not non_yaml, (
        "Unexpected non-YAML file(s) in src/olympus/graph/workflows/: "
        + ", ".join(str(p.relative_to(REPO_ROOT)) for p in non_yaml)
    )


def test_relative_import_resolution_catches_escape() -> None:
    """Verify the AST helper catches both relative and nested cross-layer imports."""
    sample = """
from ..web import server
from .. import cli
import openai

def helper() -> None:
    from ...providers.dialects import openai_chat
"""
    edges = collect_imports_from_source(
        sample,
        file_path=OLYMPUS_SRC / "core" / "sub" / "demo.py",
        current_module="olympus.core.sub.demo",
    )
    targets = {edge.target for edge in edges}
    assert "olympus.core.web" in targets
    assert "olympus.core.cli" in targets
    assert "openai" in targets
    assert "olympus.providers.dialects" in targets

    pkg_edges = collect_imports_from_source(
        "from ..web import server\nfrom olympus import cli, providers\n",
        file_path=OLYMPUS_SRC / "core" / "contracts.py",
        current_module="olympus.core.contracts",
    )
    assert {edge.target for edge in pkg_edges} == {
        "olympus.web",
        "olympus.cli",
        "olympus.providers",
    }
