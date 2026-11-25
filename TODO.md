# TODO

Deferred product, UX, and evaluation improvements identified during Stage 1/2 alignment and BlackNote E2E audits.

## Frontend (Ariadne / Oracle Board)

- [ ] **Clarification panel free-form supplement**: Add an optional free-form supplementary input box alongside the 1–3 blocking ambiguity inputs so humans can state extra product boundaries, fixed taxonomies, or constraints during clarification without needing a separate revision round.
- [ ] **Web `/clarify` async resynthesis**: Wire Web API `clarify` submissions (including optional `feedback`) to trigger `AlignStage.resynthesize_after_clarification` in the background supervisor when human input goes beyond recorded defaults.
- [ ] **Explicit revision entrypoints on artifact approval gates**: Expose a dedicated revision / edit feedback form directly on `GoalSpec`, `DesignSpec`, and `TaskPlan` approval panels (wired to `request_revision`) instead of only Approve / Reject controls.

## Evaluation & Benchmarks

- [ ] **BlackNote benchmark driver parity**: Pass baseline assumptions as structured `constraints` on `olympus session start` (or drive a real `revise:` round against the drafted `GoalSpec`) instead of stuffing assumptions into a clarification reply preamble.
- [ ] **Headless Linux UI typography / icon polish**: Prefer SVG icons or bundled webfonts over system emoji glyphs in generated web apps so headless Linux browser screenshots render cleanly without missing-glyph boxes.
