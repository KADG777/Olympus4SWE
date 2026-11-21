import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Position, type EdgeTypes, type NodeHandle, type NodeTypes } from "@xyflow/react";

import { GraphCanvas, frameHeight, useMarkerPrefix } from "../canvas/GraphCanvas";
import { OrthogonalEdge, type OrthogonalEdgeType } from "../edges/OrthogonalEdge";
import type { Point } from "../geometry";
import { dispatchToHost, type IslandController } from "../island";
import type { EdgeTone, WorkflowGraphProps } from "../types";
import {
  WF_NODE_H,
  WF_NODE_W,
  classifyNode,
  computeWorkflowLayout,
  normalizeWorkflow,
  workflowFallback,
  workflowSignature,
  type WorkflowLayout,
  type WorkflowLayoutEdge,
} from "./layout";
import { WorkflowCard, type WorkflowCardNode } from "./WorkflowCard";

const NODE_TYPES: NodeTypes = { workflow: WorkflowCard };
const EDGE_TYPES: EdgeTypes = { orthogonal: OrthogonalEdge };

/** Compact frame: one card row plus routed skip/back-edge channels. */
const MIN_HEIGHT = 220;
const MAX_HEIGHT = 360;

const NODE_HANDLES: NodeHandle[] = [
  { type: "target", position: Position.Left, x: 0, y: WF_NODE_H / 2, width: 1, height: 1 },
  { type: "source", position: Position.Right, x: WF_NODE_W - 1, y: WF_NODE_H / 2, width: 1, height: 1 },
];

/** Taken path in full contrast; untaken router fan-out and repair loops recede. */
function workflowEdgeTone(edge: WorkflowLayoutEdge): EdgeTone {
  if (edge.traversed) return "traversed";
  if (edge.conditional || edge.back) return "conditional";
  return "default";
}

export function WorkflowGraph(props: WorkflowGraphProps & { controller: IslandController }) {
  const { runId, selectedId, focusId, onSelect, controller } = props;
  const markerPrefix = useMarkerPrefix();

  const graph = useMemo(() => normalizeWorkflow(props.nodes, props.edges), [props.nodes, props.edges]);
  const signature = useMemo(
    () => workflowSignature(runId, graph.nodes, graph.edges),
    [runId, graph],
  );

  const latestGraph = useRef(graph);
  latestGraph.current = graph;
  const onSelectRef = useRef(onSelect);
  onSelectRef.current = onSelect;

  const [layout, setLayout] = useState<WorkflowLayout | null>(null);
  const version = useRef(0);

  useEffect(() => {
    const ticket = ++version.current;
    const { nodes, edges } = latestGraph.current;
    if (!nodes.length) return undefined;
    let cancelled = false;
    void computeWorkflowLayout(nodes, edges, signature).then((result) => {
      if (!cancelled && ticket === version.current) setLayout(result);
    });
    return () => {
      cancelled = true;
    };
  }, [signature]);

  const select = useCallback((id: string) => {
    const handler = onSelectRef.current;
    if (handler) dispatchToHost(() => handler(id));
  }, []);

  const current = layout && layout.signature === signature ? layout : null;

  const nodes = useMemo<WorkflowCardNode[]>(() => {
    if (!current) return [];
    return graph.nodes.map((view) => ({
      id: view.id,
      type: "workflow" as const,
      position: current.positions.get(view.id) || { x: 0, y: 0 },
      width: WF_NODE_W,
      height: WF_NODE_H,
      handles: NODE_HANDLES,
      draggable: false,
      connectable: false,
      selectable: false,
      focusable: false,
      data: { view, kind: classifyNode(view.id, view.kind), selected: view.id === selectedId, select },
    }));
  }, [current, graph, selectedId, select]);

  const edges = useMemo<OrthogonalEdgeType[]>(() => {
    if (!current) return [];
    return graph.edges.map((edge) => {
      const tone = workflowEdgeTone(edge);
      return {
        id: edge.id,
        source: edge.source,
        target: edge.target,
        type: "orthogonal" as const,
        selectable: false,
        focusable: false,
        zIndex: tone === "traversed" ? 1 : 0,
        data: {
          points: current.routes.get(edge.id) || [],
          tone,
          markerId: `${markerPrefix}-${tone}`,
          attrs: {
            traversed: edge.traversed ? "true" : "false",
            conditional: edge.conditional ? "true" : "false",
            backedge: edge.back ? "true" : "false",
          },
        },
      };
    });
  }, [current, graph, markerPrefix]);

  const estimate = useMemo(
    () => (graph.nodes.length ? workflowFallback(graph.nodes, graph.edges, signature).height : 0),
    [graph, signature],
  );

  const focus = useMemo<Point | null>(() => {
    const pos = current && focusId ? current.positions.get(focusId) : undefined;
    return pos ? { x: pos.x + WF_NODE_W / 2, y: pos.y + WF_NODE_H / 2 } : null;
  }, [current, focusId]);

  if (!graph.nodes.length) {
    return (
      <div className="ag-island" data-graph="workflow" data-empty="true">
        <p className="ag-notice" role="status">
          {props.emptyMessage || "当前工作流未提供拓扑描述"}
        </p>
      </div>
    );
  }

  return (
    <GraphCanvas
      graph="workflow"
      layout={current}
      height={frameHeight(current ? current.height : estimate, MIN_HEIGHT, MAX_HEIGHT)}
      ariaLabel={props.ariaLabel || "工作流拓扑有向图"}
      nodes={nodes}
      edges={edges}
      nodeTypes={NODE_TYPES}
      edgeTypes={EDGE_TYPES}
      markerPrefix={markerPrefix}
      controller={controller}
      focus={focus}
    />
  );
}
