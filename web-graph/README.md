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

`mountWorkflow` is a reserved slot for the Orchestration Workflow cut-over (#57).
Types live in [`src/types.ts`](src/types.ts). All styling reads `tokens.css` variables.
