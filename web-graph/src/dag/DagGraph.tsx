import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Position, ViewportPortal, type EdgeTypes, type NodeHandle, type NodeTypes } from "@xyflow/react";

import { GraphCanvas, frameHeight, useMarkerPrefix } from "../canvas/GraphCanvas";
import { OrthogonalEdge, edgeTone, type OrthogonalEdgeType } from "../edges/OrthogonalEdge";
import { dispatchToHost, type IslandController } from "../island";
import { computeLineage, edgeLineage, nodeLineage } from "../lineage";
import type { DagGraphProps } from "../types";
import {
  NODE_H,
  NODE_W,
  computeDagLayout,
  layeredFallback,
  normalizeGraph,
  topologySignature,
  type DagLayout,
} from "./layout";
import { TaskCard, type TaskCardNode } from "./TaskCard";

const NODE_TYPES: NodeTypes = { task: TaskCard };
const EDGE_TYPES: EdgeTypes = { orthogonal: OrthogonalEdge };

/** Adaptive frame bounds (px). */
const MIN_HEIGHT = 300;
const MAX_HEIGHT = 540;

const NODE_HANDLES: NodeHandle[] = [
  { type: "target", position: Position.Left, x: 0, y: NODE_H / 2, width: 1, height: 1 },
  { type: "source", position: Position.Right, x: NODE_W - 1, y: NODE_H / 2, width: 1, height: 1 },
];

export function DagGraph(props: DagGraphProps & { controller: IslandController }) {
  const { runId, selectedId, onSelect, controller } = props;
  const markerPrefix = useMarkerPrefix();

  const graph = useMemo(() => normalizeGraph(props.nodes, props.edges), [props.nodes, props.edges]);
  const signature = useMemo(
    () => topologySignature(runId, graph.nodes, graph.edges),
    [runId, graph],
  );

  const latestGraph = useRef(graph);
  latestGraph.current = graph;
  const onSelectRef = useRef(onSelect);
  onSelectRef.current = onSelect;

  const [layout, setLayout] = useState<DagLayout | null>(null);
  const [hoveredId, setHoveredId] = useState<string | null>(null);
  const version = useRef(0);

  useEffect(() => {
    setHoveredId(null);
    const ticket = ++version.current;
    const { nodes, edges } = latestGraph.current;
    if (!nodes.length) return undefined;
    let cancelled = false;
    void computeDagLayout(nodes, edges, signature).then((result) => {
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

  const hover = useCallback((id: string | null) => {
    setHoveredId(id);
  }, []);

  const current = layout && layout.signature === signature ? layout : null;
  const lineage = useMemo(() => computeLineage(graph.edges, hoveredId), [graph.edges, hoveredId]);

  const nodes = useMemo<TaskCardNode[]>(() => {
    if (!current) return [];
    return graph.nodes.map((view) => ({
      id: view.id,
      type: "task" as const,
      position: current.positions.get(view.id) || { x: 0, y: 0 },
      width: NODE_W,
      height: NODE_H,
      handles: NODE_HANDLES,
      draggable: false,
      connectable: false,
      selectable: false,
      focusable: false,
      data: {
        view,
        selected: view.id === selectedId,
        lineage: nodeLineage(lineage, view.id),
        select,
        hover,
      },
    }));
  }, [current, graph, lineage, selectedId, select, hover]);

  const edges = useMemo<OrthogonalEdgeType[]>(() => {
    if (!current) return [];
    return graph.edges.map((edge) => {
      const tone = edgeTone(edge);
      const relation = edgeLineage(lineage, edge.source, edge.target);
      return {
        id: edge.id,
        source: edge.source,
        target: edge.target,
        type: "orthogonal" as const,
        selectable: false,
        focusable: false,
        zIndex: relation === "related" ? 2 : tone === "default" ? 0 : 1,
        data: {
          points: current.routes.get(edge.id) || [],
          tone,
          markerId: `${markerPrefix}-${tone}`,
          lineage: relation,
        },
      };
    });
  }, [current, graph, lineage, markerPrefix]);

  const estimate = useMemo(
    () => (graph.nodes.length ? layeredFallback(graph.nodes, graph.edges, signature).height : 0),
    [graph, signature],
  );

  if (!graph.nodes.length) {
    return (
      <div className="ag-island" data-graph="dag" data-empty="true">
        <p className="ag-notice" role="status">
          {props.emptyMessage || "当前运行尚未生成任务 DAG"}
        </p>
      </div>
    );
  }

  return (
    <GraphCanvas
      graph="dag"
      layout={current}
      height={frameHeight(current ? current.height : estimate, MIN_HEIGHT, MAX_HEIGHT)}
      ariaLabel={props.ariaLabel || "任务依赖有向无环图"}
      nodes={nodes}
      edges={edges}
      nodeTypes={NODE_TYPES}
      edgeTypes={EDGE_TYPES}
      markerPrefix={markerPrefix}
      controller={controller}
    >
      {current ? (
        <ViewportPortal>
          <div className="ag-waves" aria-hidden="true">
            {current.waves.map((wave) => (
              <div
                key={wave.layer}
                className="ag-wave"
                data-wave={wave.layer}
                style={{ width: NODE_W, transform: `translate(${wave.x}px, 0px)` }}
              >
                Wave {wave.layer}
              </div>
            ))}
          </div>
        </ViewportPortal>
      ) : null}
    </GraphCanvas>
  );
}
