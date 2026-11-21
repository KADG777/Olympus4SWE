import { useCallback, useEffect, useId, useLayoutEffect, useMemo, useRef, useState } from "react";
import {
  Position,
  ReactFlow,
  ViewportPortal,
  useReactFlow,
  type EdgeTypes,
  type NodeHandle,
  type NodeTypes,
} from "@xyflow/react";

import { dispatchToHost, type IslandController } from "../island";
import type { DagGraphProps } from "../types";
import { ArrowMarkers, OrthogonalEdge, edgeTone, type OrthogonalEdgeType } from "../edges/OrthogonalEdge";
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

/** Viewport padding and adaptive frame bounds (px). */
const PAD = 24;
const MIN_HEIGHT = 300;
const MAX_HEIGHT = 540;
/** Initial auto-fit never magnifies (<= 1) and never crushes below this. */
const MIN_FIT_ZOOM = 0.8;

const frameHeight = (graphHeight: number): number =>
  Math.round(Math.min(MAX_HEIGHT, Math.max(MIN_HEIGHT, graphHeight + PAD * 2)));

const NODE_HANDLES: NodeHandle[] = [
  { type: "target", position: Position.Left, x: 0, y: NODE_H / 2, width: 1, height: 1 },
  { type: "source", position: Position.Right, x: NODE_W - 1, y: NODE_H / 2, width: 1, height: 1 },
];

/**
 * Centers the laid-out graph at 1:1 when it fits, scales down (>= MIN_FIT_ZOOM) when it does not,
 * and re-fits only on topology change or container resize before the user has moved the viewport.
 */
function AutoFit({
  layout,
  rootRef,
  controller,
  userMoved,
}: {
  layout: DagLayout | null;
  rootRef: React.RefObject<HTMLDivElement | null>;
  controller: IslandController;
  userMoved: React.RefObject<boolean>;
}) {
  const flow = useReactFlow();
  const fitted = useRef<string | null>(null);

  const fit = useCallback((): boolean => {
    const root = rootRef.current;
    if (!layout || !root || !flow.viewportInitialized) return false;
    const width = root.clientWidth;
    const height = root.clientHeight;
    if (!width || !height) return false;
    const zoom = Math.max(
      MIN_FIT_ZOOM,
      Math.min(1, (width - PAD * 2) / layout.width, (height - PAD * 2) / layout.height),
    );
    const x = layout.width * zoom + PAD * 2 <= width ? (width - layout.width * zoom) / 2 : PAD;
    const y = layout.height * zoom + PAD * 2 <= height ? (height - layout.height * zoom) / 2 : PAD;
    void flow.setViewport({ x: Math.round(x), y: Math.round(y), zoom });
    fitted.current = layout.signature;
    userMoved.current = false;
    return true;
  }, [flow, layout, rootRef, userMoved]);

  controller.fit = () => {
    fit();
  };

  useLayoutEffect(() => {
    if (layout && fitted.current !== layout.signature) fit();
  });

  useEffect(() => {
    const root = rootRef.current;
    if (!root || typeof ResizeObserver === "undefined") return undefined;
    const observer = new ResizeObserver(() => {
      if (!layout) return;
      if (fitted.current !== layout.signature || !userMoved.current) fit();
    });
    observer.observe(root);
    return () => observer.disconnect();
  }, [fit, layout, rootRef, userMoved]);

  return null;
}

export function DagGraph(props: DagGraphProps & { controller: IslandController }) {
  const { runId, selectedId, onSelect, controller } = props;
  const markerPrefix = `ag-arrow-${useId().replace(/[^a-zA-Z0-9_-]/g, "")}`;
  const rootRef = useRef<HTMLDivElement | null>(null);
  const userMoved = useRef(false);

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
  const version = useRef(0);

  useEffect(() => {
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

  const current = layout && layout.signature === signature ? layout : null;

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
      data: { view, selected: view.id === selectedId, select },
    }));
  }, [current, graph, selectedId, select]);

  const edges = useMemo<OrthogonalEdgeType[]>(() => {
    if (!current) return [];
    return graph.edges.map((edge) => {
      const tone = edgeTone(edge);
      return {
        id: edge.id,
        source: edge.source,
        target: edge.target,
        type: "orthogonal" as const,
        selectable: false,
        focusable: false,
        zIndex: tone === "default" ? 0 : 1,
        data: { points: current.routes.get(edge.id) || [], tone, markerId: `${markerPrefix}-${tone}` },
      };
    });
  }, [current, graph, markerPrefix]);

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
    <div
      ref={rootRef}
      className="ag-island"
      data-graph="dag"
      data-layout={current ? current.mode : "pending"}
      role="group"
      aria-label={props.ariaLabel || "任务依赖有向无环图"}
      style={{ height: frameHeight(current ? current.height : estimate) }}
    >
      <ArrowMarkers prefix={markerPrefix} />
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={NODE_TYPES}
        edgeTypes={EDGE_TYPES}
        nodesDraggable={false}
        nodesConnectable={false}
        nodesFocusable={false}
        edgesFocusable={false}
        elementsSelectable={false}
        disableKeyboardA11y
        deleteKeyCode={null}
        selectionKeyCode={null}
        multiSelectionKeyCode={null}
        zoomOnScroll={false}
        panOnScroll={false}
        zoomOnPinch
        zoomOnDoubleClick={false}
        panOnDrag
        preventScrolling={false}
        minZoom={0.4}
        maxZoom={2}
        proOptions={{ hideAttribution: true }}
        onMoveStart={(event) => {
          if (event) userMoved.current = true;
        }}
      >
        <AutoFit layout={current} rootRef={rootRef} controller={controller} userMoved={userMoved} />
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
      </ReactFlow>
    </div>
  );
}
