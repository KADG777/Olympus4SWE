# web-graph — Ariadne Graph Island

React Flow (`@xyflow/react`) + ELK (`elkjs/lib/elk.bundled.js`) renderer for Ariadne's
Topology graphs. See [ADR-0001](../docs/adr/0001-ariadne-graph-island-reactflow-elk.md).

The build writes an offline IIFE bundle into `src/olympus/web/static/graph/`
(`ariadne-graph.js`, `ariadne-graph.css`). Those files are committed, so `uv run`,
`pytest` and wheel builds never need Node.

```bash
pnpm --dir web-graph install
pnpm --dir web-graph build   # tsc --noEmit && vite build
```

Every bundle starts with `/*! ariadne-graph | source-sha256:<hex> */`, a hash over
`package.json`, `pnpm-lock.yaml`, `tsconfig.json`, `vite.config.ts` and `src/`.
`tests/unit/test_web_ariadne.py` recomputes it, so editing sources without rebuilding fails CI.

## Host contract

```js
const handle = window.AriadneGraph.mountDag(container, {
  runId, nodes, edges, selectedId, onSelect, emptyMessage, ariaLabel,
});
handle.update(nextProps); // patches in place; ELK reruns only when topology changes
handle.fitView();
handle.destroy();
```

`mountWorkflow` renders the Orchestration Workflow: gate / validator / stage cards, the
`skipped` status and the traversed trajectory (solid) versus untaken routes and loops (dashed).

Both graphs share one canvas (`src/canvas/GraphCanvas.tsx`: scroll-safe React Flow, fixed-size
markers, 1:1 auto-fit) and one physical card size plus ELK / fallback helpers
(`src/layout/shared.ts`); `src/dag/` and `src/workflow/` only add their own cards and layout options.
Types live in [`src/types.ts`](src/types.ts). All styling reads `tokens.css` variables.
