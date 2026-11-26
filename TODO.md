# TODO

Deferred product, UX, and evaluation items remaining after `v0.1.0`.

Completed Stage 1 clarification supplement, async `/clarify` resynthesis, and explicit approval-gate revision controls shipped in `v0.1.0` (`#158`, `#159`, `#160`, `#162`). Active frontend follow-ups are tracked in GitHub Issues (`#150`, `#152`, `#154`, `#155`).

## Evaluation & Benchmarks

- [ ] **BlackNote benchmark driver parity**: Pass baseline assumptions as structured `--constraint` values on `olympus session start` (or drive a real `revise:` round against the drafted `GoalSpec`) instead of stuffing assumptions into a clarification reply preamble, and archive the `v0.1.0` scorecard in `benchmarks/blacknote/HISTORY.md`.
- [ ] **Headless Linux UI typography / icon polish**: Prefer SVG icons or bundled webfonts over system emoji glyphs in generated web apps so headless Linux browser screenshots render cleanly without missing-glyph boxes.
