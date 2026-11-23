# Glossary

Canonical domain terms for Olympus4SWE. Keep definitions behavioral and implementation-free.

- **Oracle**: Tracked end-to-end engineering request issued by Zeus with an explicit `Scope` (`brownfield`, `greenfield`, or `spec_only`) and `Task Type` (`feature`, `fix`, `refactor`, or `blueprint`), progressing across the Pantheon pipeline to verified delivery or a frozen blueprint.
- **Workspace**: Registered local repository or target directory managed by the console to scope and filter Oracles by project without taking ownership of or deleting files on disk.
- **Oracle Board**: L1 human-agent collaboration surface organized by who holds the ball (`Awaiting Your Ruling`, `In Flight · Pantheon Active`, `Delivered & Verified`), letting Zeus resolve pending rulings in place and open L2 Ariadne Cockpit tabs on demand.
- **Pantheon Dock**: Pinned five-seat pipeline footer across the Oracle Board (`Prometheus`, `Athena`, `Hephaestus`, `Momus`, `Argus`) surfacing each god's live state (`idle`, `working`, `waiting_on_human`), active and waiting Oracle counts, and cross-column card highlighting.
- **GoalSpec**: Frozen, human-approved behavioral contract specifying requirements, constraints, assumptions, out-of-scope boundaries, and acceptance criteria.
- **DesignSpec**: Frozen white-box architecture contract specifying global invariants, module boundaries, interface contracts, and shared contract paths.
- **TaskPlan**: Validated dependency graph of `TaskNode` items bound by content hashes to upstream approved specs.
- **TaskNode**: Single bounded execution unit owning an objective, declared `write_scope`, optional `forbidden_globs`, and acceptance checks.
- **AgentDriver**: Pluggable execution contract through which the orchestrator runs an agent turn in an isolated sandbox, observes live milestone events, delivers mid-turn `steer` guidance, and cancels execution. Covers both the built-in native loop and external coding harnesses (such as Pi or ACP-backed agents).
- **Prometheus (Align)**: Stage 1 intent-alignment role that gathers read-only repository evidence, surfaces blocking ambiguities for Zeus, and freezes the human-approved `GoalSpec`.
- **Athena (Design & Plan)**: Stage 2a architecture and planning role that authors the white-box `DesignSpec` and decomposes work into a validated dependency `TaskPlan` before code execution begins.
- **Hephaestus (Coding Worker)**: Stage 2 execution role that implements a `TaskNode` inside an isolated sandbox via its configured `AgentDriver`.
- **Momus (Review & Audit)**: Stage 2c cold-start code review and rework-audit role that inspects worker diffs for defects (`Important` vs. `Nit`) and enforces bounded convergence before final verification.
- **Argus (Independent Verifier)**: Cold-start, read-only Stage 2 review role that audits a task's diff and verification outputs against frozen contracts with zero shared conversation history with the implementer. Distinct from `Watchdog`.
- **Watchdog**: Runtime progress health monitor operating alongside an active turn. Combines deterministic stall/loop detection (L1) with a lightweight semantic evaluator (L2) to publish human-readable progress summaries, send bounded `steer` nudges, or pause an unhealthy task into `blocked` for human decision. Never replaces `Argus` acceptance or silently discards work.
- **Diff Gate**: Deterministic post-execution boundary check that inspects all sandbox changes (tracked and untracked) against a task's authorized `write_scope` and `forbidden_globs`, rejecting out-of-scope mutations before acceptance or overlay promotion.
- **Steer**: Mid-turn, non-destructive guidance injected into an active agent session (by `Watchdog` or a human operator) to redirect a stuck or drifting turn without discarding its accumulated context.
- **Turn Receipt**: Privacy-safe summary of a completed execution turn recording aggregate call/token counts, sanitized tool invocations, timing, and artifact metrics—never raw prompt or tool body text.
- **Delivery Checks**: Isolated, content-fingerprinted verification commands executed against the composed accepted delivery to produce auditable pass/fail evidence.
