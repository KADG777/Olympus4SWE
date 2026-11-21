# Glossary

Canonical domain terms for Olympus4SWE. Keep definitions behavioral and implementation-free.

- **GoalSpec**: Frozen, human-approved behavioral contract specifying requirements, constraints, assumptions, out-of-scope boundaries, and acceptance criteria.
- **DesignSpec**: Frozen white-box architecture contract specifying global invariants, module boundaries, interface contracts, and shared contract paths.
- **TaskPlan**: Validated dependency graph of `TaskNode` items bound by content hashes to upstream approved specs.
- **TaskNode**: Single bounded execution unit owning an objective, declared `write_scope`, optional `forbidden_globs`, and acceptance checks.
- **AgentDriver**: Pluggable execution contract through which the orchestrator runs an agent turn in an isolated sandbox, observes live milestone events, delivers mid-turn `steer` guidance, and cancels execution. Covers both the built-in native loop and external coding harnesses (such as Pi or ACP-backed agents).
- **Hephaestus (Coding Worker)**: Stage 2 execution role that implements a `TaskNode` inside an isolated sandbox via its configured `AgentDriver`.
- **Argus (Independent Verifier)**: Cold-start, read-only Stage 2 review role that audits a task's diff and verification outputs against frozen contracts with zero shared conversation history with the implementer. Distinct from `Watchdog`.
- **Watchdog**: Runtime progress health monitor operating alongside an active turn. Combines deterministic stall/loop detection (L1) with a lightweight semantic evaluator (L2) to publish human-readable progress summaries, send bounded `steer` nudges, or pause an unhealthy task into `blocked` for human decision. Never replaces `Argus` acceptance or silently discards work.
- **Diff Gate**: Deterministic post-execution boundary check that inspects all sandbox changes (tracked and untracked) against a task's authorized `write_scope` and `forbidden_globs`, rejecting out-of-scope mutations before acceptance or overlay promotion.
- **Steer**: Mid-turn, non-destructive guidance injected into an active agent session (by `Watchdog` or a human operator) to redirect a stuck or drifting turn without discarding its accumulated context.
- **Turn Receipt**: Privacy-safe summary of a completed execution turn recording aggregate call/token counts, sanitized tool invocations, timing, and artifact metrics—never raw prompt or tool body text.
- **Delivery Checks**: Isolated, content-fingerprinted verification commands executed against the composed accepted delivery to produce auditable pass/fail evidence.
