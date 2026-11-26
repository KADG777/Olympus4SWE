# Changelog

All notable changes to Olympus4SWE are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-11-25

First stable release covering the evolution from `v0.0` (`e87b7e3`) to `main` (`1b4fc18`).

### Added

- **Post-DAG verification and clean-directory product E2E.** Added a 0-token escape-hatch pre-gate (`src/olympus/harness/escape_hatches.py`), a post-DAG unified `Argus` review and repair loop, and clean-directory product E2E verification (`src/olympus/stages/verify_product.py`) with screenshot evidence (`#144`).
- **Stage 1 clarification fidelity and resynthesis.** Preserved free-form human clarification notes, added asynchronous `GoalSpec` resynthesis before approval, and exposed explicit `request_revision` and `reopen_run` controls across the backend and Ariadne (`#148`, `#158`, `#159`).
- **Oracle Board and Ariadne v0.1 workbench.** Cut the backend over to FastAPI with typed OpenAPI contracts (`#76`), unified the frontend into a React 19 + TypeScript + Vite SPA with generated API types (`#78`), and added the two-layer Oracle Board + 5-God Pantheon Dock (`#108`), workspace picker and Git URL clone (`#110`), 4-phase staged workflow board (`#117`), and Product E2E evidence and Post-DAG timeline views (`#158`, `#160`).
- **Headless session CLI and benchmark adapters.** Added `olympus session start` and `olympus session reply`, Harbor interactive benchmark adapters, and local Docker judging (`#107`), plus the archived `BlackNote Dev` `v0.0` longitudinal benchmark snapshot (`#82`).
- **Standalone API key rotation proxy.** Added `tools/api-key-ledger/` for multi-key ledger tracking and rotation (`#134`), aligned with upstream keep-alive and read timeouts (`#146`).
- **Repository governance and bilingual docs.** Added `CONTRIBUTING.md`, `.github/CODEOWNERS`, CI-enforced architecture invariant tests (`#120`), and bilingual `README.md` and `README.zh-CN.md` (`#128`).

### Changed

- **Vendor SDK isolation.** Routed OpenAI and Anthropic wire dialects through the official `openai` and `anthropic` SDKs inside `src/olympus/providers/` (`#77`).
- **Pi driver defaults and token accounting.** Defaulted `stage_driver`, `worker_driver`, and `reviewer_driver` (`Argus`) to `auto`, selecting `pi` on eligible remote providers (`#118`, `#140`), with T3-aligned native Pi provider overrides and full input, cache, and reasoning token accounting (`#88`, `#131`).
- **Parallel vertical DAG planning.** Aligned `tdd_first` and `vertical_slices` planning with capability-scaled vertical slicing and runtime extension seams (`#93`, `#166`).
- **Core DAG slimming.** Removed speculative prose and bash-AST validators from `src/olympus/core/dag.py` and extracted pure path-overlap checks into `src/olympus/core/scope_paths.py` (`#142`).
- **Stateful repair and blocked recovery.** Replaced full sandbox wipes on recoverable rejections with stateful in-place repair, run-proven two-axis `Argus` review, and in-place recovery for blocked tasks (`#88`, `#129`).
- **Network keep-alive and fast paths.** Added TCP keep-alive (`src/olympus/harness/pi_net_keepalive.cjs`), `>=300s` read timeouts, non-thinking router fast paths, and incremental worker edits (`#138`).

### Fixed

- **Session workflow status projection.** Reconciled `build_workflow_view` node statuses from `SessionRunner` domain events so headless session runs show accurate stage progress in Ariadne (`#164`).
- **Post-approval clarification UI.** Hid the editable Stage 1 `ClarifyPanel` and non-blocking ambiguities after `GoalSpec` approval (`#162`).
- **Pi keep-alive isolation and Argus repro paths.** Isolated `NODE_OPTIONS` for `pi_net_keepalive.cjs` so it does not leak into child processes and normalized `Argus` reproduction paths (`#151`).
- **Workflow convergence and reply routing.** Fixed workflow convergence, stranded in-flight task requeuing on resume, deterministic reply routing fast paths, and an `httpx` exception cause cycle (`#136`).
- **Harbor resilience and planner allowlists.** Added an explicit `design_refs` allowlist and hardened Harbor bridge process cleanup and timeout handling (`#133`).
- **Ephemeral runtime artifact cleanup.** Cleaned newly created out-of-scope test artifacts (SQLite journals, logs, temporary uploads) before `DiffGate` evaluation (`#140`).

### Removed

- **LLM Watchdog.** Removed the LLM Watchdog runtime execution chain (`#140`). The `--monitor-model` CLI option stays as a deprecated no-op, and historical `watchdog.*` events remain readable for backwards compatibility.

### Verified

- **BlackNote Dev live E2E (`run-4e19c8d93539`).** Completed a 4-wave, 7-slice parallel full-lifecycle run with `27/27` delivery acceptance checks passing and `9/9` Playwright restart E2E checks passing.

## [0.0] - 2026-11-21

Baseline release (`e87b7e3`, tag `v0.0`).

### Added

- **Deterministic SDLC control plane.** Implemented Pydantic contracts, SHA-256 content fingerprints, DAG and disjoint `write_scope` validation, SQLite WAL event/state store, monotonic budget ledgers, and hash-bound approvals (`c281ad7`, `#53`, `#59`), along with GitHub Actions CI and Issue/PR templates (`#2`).
- **Two-stage pipeline.** Implemented Stage 1 `Align` (`GoalSpec`), Stage 2a `Design` (`DesignSpec` v2), and Stage 2 `Orchestrate & Verify` with isolated `git worktree` sandboxes and cold-start review (`#10`, `#18`, `#30`).
- **Meta-Harness execution layer.** Added the `AgentDriver` protocol (`NativeDriver` and `PiDriver`), post-flight `DiffGate`, live milestone observability, and unlimited default `worker_max_steps` (`#66`, `#68`, `#72`).
- **Ariadne web console.** Built the event-sourced web console with SSE updates, approval review views, and React Flow + ELK workflow/DAG graphs (`#12`, `#14`, `#48`, `#60`).
- **Decoupled evaluation suite.** Added the Stage 1 four-class benchmark runner and `ClarifyCodeBench` / `Ambig-SWE` / `SWE-Interact` evaluation adapters (`#4`, `#6`, `#10`).
