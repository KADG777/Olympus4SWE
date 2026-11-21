import "@xyflow/react/dist/base.css";
import "./graph.css";

import { DagGraph } from "./dag/DagGraph";
import { createIsland } from "./island";
import type { AriadneGraphAPI, DagGraphProps, WorkflowGraphProps } from "./types";
import { WorkflowGraph } from "./workflow/WorkflowGraph";

export type * from "./types";

declare global {
  interface Window {
    AriadneGraph?: AriadneGraphAPI;
  }
}

/** The only global the island exposes; the vanilla console talks to nothing else. */
export const AriadneGraph: AriadneGraphAPI = Object.freeze({
  version: "1",
  mountDag: (container: HTMLElement, initialProps: DagGraphProps) =>
    createIsland(container, DagGraph, initialProps),
  mountWorkflow: (container: HTMLElement, initialProps: WorkflowGraphProps) =>
    createIsland(container, WorkflowGraph, initialProps),
});

window.AriadneGraph = AriadneGraph;
