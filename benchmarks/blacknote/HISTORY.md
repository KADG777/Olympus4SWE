# BlackNote Dev — Longitudinal Harness Benchmark History

This directory tracks the evolution of **Olympus4SWE** across major versions using a fixed, real-world full-lifecycle benchmark: **`BlackNote Dev`** (`full_sdlc_tdd` workflow; see [case.json](case.json)).

Each major release (`v0.0`, `v0.1`, ...) runs the same ambiguous user prompt, archives its frozen planning artifacts, structured telemetry (`metrics.json`), and clean runnable `accepted_overlay` snapshot (`app/`), allowing direct side-by-side comparison of both **control-plane metrics** and **generated software quality**.

---

## Cross-Version Scorecard

| Version | Git Tag / Commit | Model & Worker Driver | Tasks Accepted | Overlay Tests | Wall Clock (since execution) | LLM Calls (Ledger) | Booked Cost (RMB) | Watchdog (`HEALTHY` / False Pause / Error) | Diff Gate Rejections |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`v0.0`** | [`v0.0`](../../) (`e87b7e3`, PR #72) | `pi@bailian/qwen3.8-flash` + `native` Argus | **5 / 8** (Wave 0 + Wave 1; stopped mid-`T2-web` Attempt 2) | **540 passed** (`0` failed, `3.33s`, 49 files) | `8h 53m` (`10h 04m` total) | `2,349` (`11.57M` in / `2.37M` out) | **¥15.65** *(est. ¥24–32 incl. unbooked cache & Watchdog)* | `1,244` / **`0`** / `475` | `2` (`T1-interaction`, `T2-web`) |
| `v0.1` | *in progress* | `pi@bailian/qwen3.8-flash` *(baseline)* | — | — | — | — | — | — | — |

---

## Version Notes

### `v0.0` — Meta-Harness Baseline (`e87b7e3`, 2026-11-21)

- **Artifacts**: [metrics.json](v0.0/metrics.json) · [goal_spec.json](v0.0/artifacts/goal_spec.json) · [design_spec.json](v0.0/artifacts/design_spec.json) · [task_plan.json](v0.0/artifacts/task_plan.json) · [pytest_overlay_report.txt](v0.0/artifacts/pytest_overlay_report.txt) · [app/](v0.0/app/)
- **What Worked (validated under 10h live stress)**:
  - **Phase 1–3 (`5 min`)**: Disambiguation, automated design cycle detection & repair (`v1 -> v2`, 5 layered modules), and 8-task 4-wave TDD DAG generation completed cleanly.
  - **Wave 0 + Wave 1 (`5/8 accepted`, `540/540` tests green)**: Huge leap over pre-Meta-Harness baseline (`run-771faf110778`), which failed completely in Wave 0 at step 20.
  - **Deterministic Control Plane**:
    - `pi_scope_guard.ts` blocked a live path-escape attempt (`/tmp/...`) in 4 ms; model self-corrected immediately.
    - `Diff Gate` deterministically caught stray out-of-scope SQLite test artifacts twice (`data/*.db` in `T1-interaction`, `file:/.../*.db` in `T2-web`), keeping `accepted_overlay` 100% clean.
    - Cold-start `Argus` caught 1 genuine `important` defect on `T1-interaction` Attempt 2 and cleanly approved 5 modules (`important=0`).
    - Two-tier `Watchdog` ran `1,244` `HEALTHY` checks with **`0` false pauses** and degraded fail-open across transient `qwen3.8-flash` parse/HTTP errors.
- **Top 5 Real-World Bottlenecks Exposed for `v0.1`**:
  1. **P0 — Full sandbox wipe on Diff Gate / Argus rejection (~5h wasted)**: Both `T1-interaction` (240 steps) and `T2-web` (650 steps, 607/608 passing) were wiped and rebuilt from scratch solely because of 2–3 stray out-of-scope `.db` files. `v0.1` needs **in-place retry / patch-forward** on recoverable gate/review rejections.
  2. **P0 — Long-session step & context bloat (340–650 steps)**: Removing the static step cap (PR #72) unblocked legitimate 91–111 step modules, but `T1-interaction` (340 steps, 85 KB tests) and `T2-web` (650 steps) suffered over-testing and 3–4 min/step tail latency. `v0.1` needs task-sizing / compaction / convergence guidance.
  3. **P1 — Watchdog JSON parse failure rate (`413 / 1,719 ≈ 24%`)**: Needs strict structured JSON mode (`response_format: json_object`), stronger extraction, and steady-state backoff.
  4. **P1 — Cost accounting gaps**: `PiDriver` settled `usage.input` without `usage.cacheRead`, and `Watchdog` calls were excluded from `BudgetLedger`.
  5. **P2 — External driver credential failover**: `PooledProvider` failover did not propagate dynamically to `PiDriver`'s static `models.json` when the primary key hit `403 AccessDenied.Unpurchased`.
