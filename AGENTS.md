# AGENTS.md — Olympus4SWE

Olympus4SWE is a first-principles, Agent-Native full-lifecycle software engineering system (Python 3.13 + `uv` backend, React + TypeScript + `pnpm` frontend).

This file is the canonical technical guide for both human contributors and AI coding agents working in this repository. For team roles, onboarding, and PR workflow, read [CONTRIBUTING.md](CONTRIBUTING.md).

## Repository Map

- Architecture & principles: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- Canonical domain glossary: [GLOSSARY.md](GLOSSARY.md)
- Team workflow & track ownership: [CONTRIBUTING.md](CONTRIBUTING.md)
- Backend source (`src/olympus/`):
  - `core/` — Strict Pydantic contracts, SHA-256 content fingerprints, DAG & `write_scope` validation, SQLite event/state store, deterministic control primitives
  - `providers/` — Pluggable LLM provider layer (`dialects/`, `registry.py`, `factory.py`, `base.py`, `types.py`) isolating official vendor SDKs (`openai`, `anthropic`)
  - `tools/` — Safe ACI file/shell primitives with absolute-path enforcement and bounded output
  - `harness/` — Transparent append-only loop, `AgentDriver` abstraction (`native`, `pi`), `DiffGate`, `escape_hatches`, `git worktree` sandbox, `TurnReceipt`
  - `stages/` — Stage 1 `Align` (`align.py`), Stage 2a `Design` (`design.py`), Stage 2 `Orchestrate & Verify` (`orchestrate.py`)
  - `graph/` — Declarative workflow loader/engine (`engine.py`, `loader.py`, `progress.py`) and YAML workflow templates (`workflows/*.yaml`)
  - `session/` — Headless session routing, policy, and runner (`contracts.py`, `policy.py`, `router.py`, `runner.py`)
  - `web/` — FastAPI server, OpenAPI schema export (`openapi.json`), read-only event projection, supervisor, and built static assets
  - `cli/` — Typer CLI entrypoints (`olympus`)
- Frontend source (`web/src/`): React + TypeScript Ariadne & Oracle Board workbench, consuming typed OpenAPI contracts (`web/src/api/schema.d.ts`)
- Tests & benchmarks: `tests/unit/` (unit, contract, architecture, and browser tests), `tests/evals/` (isolated evaluation suites, datasets, scorers, and reports)

## Architecture Invariants (Enforced by CI)

These boundaries are mechanically checked by `tests/unit/test_architecture_invariants.py` and `.github/workflows/ci.yml`:

1. **Deterministic control plane vs. probabilistic LLM plane**: State transitions, DAG validation, `write_scope` / `DiffGate` enforcement, monotonic budget ledgers, and hash-bound approvals belong exclusively to deterministic code. Never let LLM output mutate them directly.
2. **Pure foundation (`core/`)**: `src/olympus/core/` must not import any sibling `olympus.*` package, vendor LLM SDKs, web/CLI frameworks, or `tests`.
3. **Vendor SDK isolation**: Only `src/olympus/providers/` may import `openai` or `anthropic`. All upper layers (`stages/`, `graph/`, `session/`, `web/`, `cli/`) must depend on `olympus.providers` abstractions.
4. **Presentation isolation**: Core and runtime layers (`core/`, `providers/`, `tools/`, `harness/`, `stages/`, `graph/`, `session/`) must not import `olympus.web`, `olympus.cli`, `fastapi`, `uvicorn`, or `typer`.
5. **Evaluation independence**: Dependency is strictly one-way (`tests/evals/ → src/olympus/`). Production code under `src/olympus/` must never import `tests`, and benchmark schemas stay separate in `tests/evals/schemas.py`.
6. **Frontend contract discipline**: No untyped `.js`/`.jsx` files under `web/src/`. Any backend API schema change must keep `web/src/api/schema.d.ts` and built assets under `src/olympus/web/static` in sync.

## Platform Core Change Protocol

Before modifying Platform Core (`core/`, `harness/`, `stages/`, `graph/engine.py|loader.py|progress.py`, `session/`, `web/server.py|schemas.py|projection.py|supervisor.py|openapi.py`, `pyproject.toml`, or `.github/`):

- Open or reference a GitHub Issue that aligns the interface contract and `write_scope` first.
- Prefer exposing clean, read-only fields or typed protocols over adding feature-specific conditionals inside core modules.

## Verification & Git Delivery

- Run all Python commands via `uv run`:
  - `uv run ruff check .`
  - `uv run mypy src`
  - `uv run pytest -q`
- When touching `web/` or backend API schemas, also run:
  - `pnpm --dir web generate:api`
  - `pnpm --dir web typecheck`
  - `pnpm --dir web build`
- Follow `GitHub Issue → Branch → PR (Closes #<id>)` for all non-trivial changes. Never push directly to `main`.
