import ELK from "elkjs/lib/elk.bundled.js";
import type { ElkExtendedEdge, ElkNode, ElkPort } from "elkjs/lib/elk-api";

import type { Point } from "../geometry";
import type { DagEdgeView, DagNodeView } from "../types";

/** Fixed 1:1 physical card size; CSS reads the same numbers via inline style. */
export const NODE_W = 184;
export const NODE_H = 76;
/** Vertical band above the first card row that hosts the Wave column headers. */
export const WAVE_BAND = 36;
const LAYER_GAP = 96;
const ROW_GAP = 24;

export type LayoutMode = "elk" | "fallback";

export interface LayoutEdge extends DagEdgeView {
  id: string;
}

export interface DagLayout {
  signature: string;
  mode: LayoutMode;
  positions: Map<string, Point>;
  routes: Map<string, Point[]>;
  waves: { layer: number; x: number }[];
  width: number;
  height: number;
}

export const edgeKey = (source: string, target: string): string => `${source}\u2192${target}`;

/** Cycle-safe longest-path wave for nodes whose host layer is missing or invalid. */
function fillLayers(nodes: DagNodeView[], edges: LayoutEdge[]): Map<string, number> {
  const parents = new Map<string, string[]>();
  edges.forEach((edge) => parents.set(edge.target, [...(parents.get(edge.target) || []), edge.source]));
  const memo = new Map<string, number>();
  const given = new Map(nodes.map((node) => [node.id, node.layer]));
  const visit = (id: string, visiting: Set<string>): number => {
    const host = given.get(id);
    if (typeof host === "number" && Number.isInteger(host) && host >= 0) return host;
    if (memo.has(id)) return memo.get(id) as number;
    if (visiting.has(id)) return 0;
    visiting.add(id);
    const deps = (parents.get(id) || []).map((parent) => visit(parent, visiting));
    visiting.delete(id);
    const layer = deps.length ? Math.max(...deps) + 1 : 0;
    memo.set(id, layer);
    return layer;
  };
  return new Map(nodes.map((node) => [node.id, visit(node.id, new Set())]));
}

/** Dedupe nodes/edges, drop dangling edges, and give every node an integer wave. */
export function normalizeGraph(
  rawNodes: DagNodeView[],
  rawEdges: DagEdgeView[],
): { nodes: DagNodeView[]; edges: LayoutEdge[] } {
  const seen = new Set<string>();
  const unique = (rawNodes || []).filter((node) => {
    if (!node || typeof node.id !== "string" || seen.has(node.id)) return false;
    seen.add(node.id);
    return true;
  });
  const edgeIds = new Set<string>();
  const edges: LayoutEdge[] = [];
  (rawEdges || []).forEach((edge) => {
    if (!edge || !seen.has(edge.source) || !seen.has(edge.target)) return;
    const id = edgeKey(edge.source, edge.target);
    if (edgeIds.has(id)) return;
    edgeIds.add(id);
    edges.push({ ...edge, id });
  });
  const layers = fillLayers(unique, edges);
  const nodes = unique.map((node) => ({ ...node, layer: layers.get(node.id) ?? 0 }));
  return { nodes, edges };
}

/** Layout reruns only when this changes; status/selection patches keep the viewport. */
export function topologySignature(runId: string, nodes: DagNodeView[], edges: LayoutEdge[]): string {
  const nodePart = nodes.map((node) => `${node.id}@${node.layer}`).sort();
  const edgePart = edges.map((edge) => edge.id).sort();
  return JSON.stringify([runId || "", nodePart, edgePart]);
}

export function hasCycle(nodes: DagNodeView[], edges: LayoutEdge[]): boolean {
  const indegree = new Map(nodes.map((node) => [node.id, 0]));
  const children = new Map<string, string[]>();
  edges.forEach((edge) => {
    indegree.set(edge.target, (indegree.get(edge.target) || 0) + 1);
    children.set(edge.source, [...(children.get(edge.source) || []), edge.target]);
  });
  const queue = nodes.filter((node) => indegree.get(node.id) === 0).map((node) => node.id);
  let visited = 0;
  while (queue.length) {
    const id = queue.shift() as string;
    visited += 1;
    (children.get(id) || []).forEach((child) => {
      const left = (indegree.get(child) || 0) - 1;
      indegree.set(child, left);
      if (left === 0) queue.push(child);
    });
  }
  return visited !== nodes.length;
}

function columns(nodes: DagNodeView[]): Map<number, number> {
  const layers = Array.from(new Set(nodes.map((node) => node.layer))).sort((a, b) => a - b);
  return new Map(layers.map((layer, index) => [layer, index]));
}

function wavesFrom(nodes: DagNodeView[], positions: Map<string, Point>): DagLayout["waves"] {
  const byLayer = new Map<number, number>();
  nodes.forEach((node) => {
    const pos = positions.get(node.id);
    if (pos && !byLayer.has(node.layer)) byLayer.set(node.layer, pos.x);
  });
  return Array.from(byLayer, ([layer, x]) => ({ layer, x })).sort((a, b) => a.layer - b.layer);
}

/** Deterministic layered coordinates + simple orthogonal routes; never throws, tolerates cycles. */
export function layeredFallback(
  nodes: DagNodeView[],
  edges: LayoutEdge[],
  signature: string,
): DagLayout {
  const column = columns(nodes);
  const rows = new Map<number, number>();
  const positions = new Map<string, Point>();
  nodes.forEach((node) => {
    const col = column.get(node.layer) ?? 0;
    const row = rows.get(col) || 0;
    rows.set(col, row + 1);
    positions.set(node.id, { x: col * (NODE_W + LAYER_GAP), y: WAVE_BAND + row * (NODE_H + ROW_GAP) });
  });
  const portY = (ids: string[], id: string, top: number): number =>
    top + (NODE_H * (ids.indexOf(id) + 1)) / (ids.length + 1);
  const incoming = new Map<string, string[]>();
  const outgoing = new Map<string, string[]>();
  const byY = (key: "source" | "target") => (a: LayoutEdge, b: LayoutEdge) =>
    (positions.get(a[key]) as Point).y - (positions.get(b[key]) as Point).y || (a.id < b.id ? -1 : 1);
  nodes.forEach((node) => {
    incoming.set(node.id, edges.filter((e) => e.target === node.id).sort(byY("source")).map((e) => e.id));
    outgoing.set(node.id, edges.filter((e) => e.source === node.id).sort(byY("target")).map((e) => e.id));
  });
  let bottom = Math.max(...Array.from(positions.values(), (pos) => pos.y + NODE_H));
  const routes = new Map<string, Point[]>();
  let loop = 0;
  edges.forEach((edge, index) => {
    const s = positions.get(edge.source) as Point;
    const t = positions.get(edge.target) as Point;
    const sy = portY(outgoing.get(edge.source) || [], edge.id, s.y);
    const ty = portY(incoming.get(edge.target) || [], edge.id, t.y);
    const start = { x: s.x + NODE_W, y: sy };
    const end = { x: t.x, y: ty };
    if (t.x > s.x) {
      const channel = t.x - LAYER_GAP / 2 + ((index % 5) - 2) * 6;
      routes.set(edge.id, [start, { x: channel, y: sy }, { x: channel, y: ty }, end]);
    } else {
      loop += 1;
      const under = bottom + 16 + loop * 10;
      routes.set(edge.id, [
        start,
        { x: start.x + 16, y: sy },
        { x: start.x + 16, y: under },
        { x: end.x - 16, y: under },
        { x: end.x - 16, y: ty },
        end,
      ]);
    }
  });
  if (loop) bottom += 16 + loop * 10 + 8;
  const width = Math.max(...Array.from(positions.values(), (pos) => pos.x + NODE_W));
  return {
    signature,
    mode: "fallback",
    positions,
    routes,
    waves: wavesFrom(nodes, positions),
    width,
    height: bottom,
  };
}

let engine: InstanceType<typeof ELK> | null = null;

const ROOT_OPTIONS: Record<string, string> = {
  "elk.algorithm": "layered",
  "elk.direction": "RIGHT",
  "elk.edgeRouting": "ORTHOGONAL",
  "elk.layered.layering.strategy": "INTERACTIVE",
  "elk.layered.crossingMinimization.strategy": "LAYER_SWEEP",
  "elk.layered.nodePlacement.strategy": "BRANDES_KOEPF",
  "elk.layered.nodePlacement.bk.fixedAlignment": "BALANCED",
  "elk.layered.considerModelOrder.strategy": "NODES_AND_EDGES",
  "elk.layered.mergeEdges": "false",
  "elk.layered.spacing.nodeNodeBetweenLayers": String(LAYER_GAP),
  "elk.layered.spacing.edgeNodeBetweenLayers": "20",
  "elk.layered.spacing.edgeEdgeBetweenLayers": "12",
  "elk.spacing.nodeNode": String(ROW_GAP),
  "elk.spacing.edgeNode": "18",
  "elk.spacing.edgeEdge": "12",
  "elk.spacing.portPort": "14",
  "elk.padding": "[top=0,left=0,bottom=0,right=0]",
  "elk.randomSeed": "1",
};

/** ELK Layered with waves pinned to `node.layer` and distributed west-in / east-out ports. */
export async function elkLayout(
  nodes: DagNodeView[],
  edges: LayoutEdge[],
  signature: string,
): Promise<DagLayout> {
  engine = engine || new ELK();
  const column = columns(nodes);
  const rows = new Map<number, number>();
  const ports = new Map<string, ElkPort[]>(nodes.map((node) => [node.id, []]));
  edges.forEach((edge) => {
    ports.get(edge.source)?.push({
      id: `${edge.id}#out`, width: 0, height: 0, layoutOptions: { "elk.port.side": "EAST" },
    });
    ports.get(edge.target)?.push({
      id: `${edge.id}#in`, width: 0, height: 0, layoutOptions: { "elk.port.side": "WEST" },
    });
  });
  const graph: ElkNode = {
    id: "ariadne-dag",
    layoutOptions: ROOT_OPTIONS,
    children: nodes.map((node) => {
      const col = column.get(node.layer) ?? 0;
      const row = rows.get(col) || 0;
      rows.set(col, row + 1);
      return {
        id: node.id,
        width: NODE_W,
        height: NODE_H,
        x: col * (NODE_W + LAYER_GAP),
        y: row * (NODE_H + ROW_GAP),
        ports: ports.get(node.id),
        layoutOptions: {
          "elk.portConstraints": "FIXED_SIDE",
          "elk.spacing.portPort": "14",
        },
      };
    }),
    edges: edges.map(
      (edge): ElkExtendedEdge => ({ id: edge.id, sources: [`${edge.id}#out`], targets: [`${edge.id}#in`] }),
    ),
  };
  const result = await engine.layout(graph);
  const positions = new Map<string, Point>();
  (result.children || []).forEach((child) => {
    positions.set(child.id, { x: child.x ?? 0, y: child.y ?? 0 });
  });
  const routes = new Map<string, Point[]>();
  (result.edges || []).forEach((edge) => {
    const sections = edge.sections || [];
    if (!sections.length) return;
    const points: Point[] = [];
    sections.forEach((section) => {
      points.push(section.startPoint, ...(section.bendPoints || []), section.endPoint);
    });
    routes.set(edge.id, points.map((point) => ({ x: point.x, y: point.y })));
  });
  if (positions.size !== nodes.length || routes.size !== edges.length) {
    throw new Error("ELK returned an incomplete layout");
  }
  const all = [
    ...Array.from(positions.values()).flatMap((p) => [p, { x: p.x + NODE_W, y: p.y + NODE_H }]),
    ...Array.from(routes.values()).flat(),
  ];
  const minX = Math.min(...all.map((p) => p.x));
  const minY = Math.min(...all.map((p) => p.y));
  const shift = (p: Point): Point => ({ x: p.x - minX, y: p.y - minY + WAVE_BAND });
  positions.forEach((pos, id) => positions.set(id, shift(pos)));
  routes.forEach((points, id) => routes.set(id, points.map(shift)));
  const layout: DagLayout = {
    signature,
    mode: "elk",
    positions,
    routes,
    waves: wavesFrom(nodes, positions),
    width: Math.max(...all.map((p) => p.x)) - minX,
    height: Math.max(...all.map((p) => p.y)) - minY + WAVE_BAND,
  };
  assertWavesAligned(nodes, layout);
  return layout;
}

/** ELK must honour the host's waves: one shared x per wave, columns ordered by wave. */
function assertWavesAligned(nodes: DagNodeView[], layout: DagLayout): void {
  const xs = new Map<number, number>();
  nodes.forEach((node) => {
    const x = (layout.positions.get(node.id) as Point).x;
    const first = xs.get(node.layer);
    if (first === undefined) xs.set(node.layer, x);
    else if (Math.abs(first - x) > 0.5) throw new Error(`wave ${node.layer} is not column-aligned`);
  });
  const ordered = Array.from(xs).sort((a, b) => a[0] - b[0]);
  ordered.forEach(([layer, x], index) => {
    if (index && x <= ordered[index - 1][1]) throw new Error(`wave ${layer} is out of order`);
  });
}

/** Never rejects: cyclic input or any ELK failure degrades to the deterministic fallback. */
export async function computeDagLayout(
  nodes: DagNodeView[],
  edges: LayoutEdge[],
  signature: string,
): Promise<DagLayout> {
  if (hasCycle(nodes, edges)) return layeredFallback(nodes, edges, signature);
  try {
    return await elkLayout(nodes, edges, signature);
  } catch (error) {
    console.warn("[AriadneGraph] ELK layout failed; using layered fallback", error);
    return layeredFallback(nodes, edges, signature);
  }
}
