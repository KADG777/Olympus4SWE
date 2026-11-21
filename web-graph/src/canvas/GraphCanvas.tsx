import {
  useCallback,
  useEffect,
  useId,
  useLayoutEffect,
  useRef,
  type ReactNode,
  type RefObject,
} from "react";
import {
  ReactFlow,
  useReactFlow,
  type Edge,
  type EdgeTypes,
  type Node,
  type NodeTypes,
} from "@xyflow/react";

import { ArrowMarkers } from "../edges/OrthogonalEdge";
import type { Point } from "../geometry";
import type { IslandController } from "../island";

/** Viewport padding (px) shared by every graph frame. */
export const PAD = 24;
/**
 * Initial auto-fit never magnifies and, by default, never shrinks either: both Topology graphs
 * open at the same crisp 1:1 card scale, and a graph larger than its frame pans instead.
 */
export const DEFAULT_MIN_FIT_ZOOM = 1;

/** What the canvas needs to know about a finished layout to frame it. */
export interface CanvasLayout {
  signature: string;
  mode: string;
  width: number;
  height: number;
}

/** Stable, CSS-safe id prefix for one canvas's arrowhead markers. */
export function useMarkerPrefix(): string {
  return `ag-arrow-${useId().replace(/[^a-zA-Z0-9_-]/g, "")}`;
}

/** Frame height that follows the laid-out graph within [min, max]. */
export function frameHeight(graphHeight: number, min: number, max: number): number {
  return Math.round(Math.min(max, Math.max(min, graphHeight + PAD * 2)));
}

/** Center when the graph fits; otherwise keep `focus` in view, clamped to the graph's edges. */
function axisOffset(size: number, extent: number, focus: number | null, zoom: number): number {
  if (extent + PAD * 2 <= size) return (size - extent) / 2;
  if (focus === null) return PAD;
  return Math.min(PAD, Math.max(size - PAD - extent, size / 2 - focus * zoom));
}

/**
 * Centers the laid-out graph at 1:1 when it fits; otherwise scales down no further than minZoom
 * and keeps the focus point in view. Re-fits only on topology change, or on container resize
 * before the user has moved the viewport.
 */
function AutoFit({
  layout,
  rootRef,
  controller,
  userMoved,
  minZoom,
  focusRef,
}: {
  layout: CanvasLayout | null;
  rootRef: RefObject<HTMLDivElement | null>;
  controller: IslandController;
  userMoved: RefObject<boolean>;
  minZoom: number;
  focusRef: RefObject<Point | null>;
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
      minZoom,
      Math.min(1, (width - PAD * 2) / layout.width, (height - PAD * 2) / layout.height),
    );
    const focus = focusRef.current;
    const x = axisOffset(width, layout.width * zoom, focus ? focus.x : null, zoom);
    const y = axisOffset(height, layout.height * zoom, focus ? focus.y : null, zoom);
    void flow.setViewport({ x: Math.round(x), y: Math.round(y), zoom });
    fitted.current = layout.signature;
    userMoved.current = false;
    return true;
  }, [flow, focusRef, layout, minZoom, rootRef, userMoved]);

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

export interface GraphCanvasProps<N extends Node, E extends Edge> {
  /** `data-graph` hook (`dag` | `workflow`). */
  graph: string;
  layout: CanvasLayout | null;
  height: number;
  ariaLabel: string;
  nodes: N[];
  edges: E[];
  nodeTypes: NodeTypes;
  edgeTypes: EdgeTypes;
  markerPrefix: string;
  controller: IslandController;
  /** Lower bound of the initial auto-fit zoom (default 1); the upper bound is always 1. */
  minFitZoom?: number;
  /** Graph-space point kept in view when the graph overflows the frame on initial fit. */
  focus?: Point | null;
  /** Extra viewport content (e.g. Wave headers in a ViewportPortal). */
  children?: ReactNode;
}

/**
 * The one read-only, scroll-safe React Flow canvas both Topology graphs share: fixed-size
 * tone-matched markers, non-draggable cards, page-scrolling wheel, drag-to-pan and auto-fit.
 */
export function GraphCanvas<N extends Node, E extends Edge>(props: GraphCanvasProps<N, E>) {
  const { layout, controller } = props;
  const rootRef = useRef<HTMLDivElement | null>(null);
  const userMoved = useRef(false);
  const focusRef = useRef<Point | null>(props.focus ?? null);
  focusRef.current = props.focus ?? null;

  return (
    <div
      ref={rootRef}
      className="ag-island"
      data-graph={props.graph}
      data-layout={layout ? layout.mode : "pending"}
      role="group"
      aria-label={props.ariaLabel}
      style={{ height: props.height }}
    >
      <ArrowMarkers prefix={props.markerPrefix} />
      <ReactFlow<N, E>
        nodes={props.nodes}
        edges={props.edges}
        nodeTypes={props.nodeTypes}
        edgeTypes={props.edgeTypes}
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
        <AutoFit
          layout={layout}
          rootRef={rootRef}
          controller={controller}
          userMoved={userMoved}
          minZoom={props.minFitZoom ?? DEFAULT_MIN_FIT_ZOOM}
          focusRef={focusRef}
        />
        {props.children}
      </ReactFlow>
    </div>
  );
}
