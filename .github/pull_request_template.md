Closes #<issue-id>

## Summary

<!-- What changed and why. Keep it short and factual. -->

## Track & Scope

- [ ] Evaluation & Benchmarking (`tests/evals/`)
- [ ] Ariadne Web UX (`web/src/`)
- [ ] Workflows, Providers & CLI DX (`graph/workflows/`, `providers/dialects/`, `cli/`, `docs/`)
- [ ] Platform Core (`core/`, `harness/`, `stages/`, `session/`, `src/olympus/web/`) — *contract aligned in linked Issue*

## Verification

- [ ] `uv run ruff check .`
- [ ] `uv run mypy src`
- [ ] `uv run pytest -q`
- [ ] Frontend checks (if touching `web/` or backend API schemas): `pnpm --dir web generate:api && pnpm --dir web typecheck && pnpm --dir web build`
