import type { RunProjection } from "../api/types";
import { NODE_LABELS, statusMeta, WORKFLOW_STATUS_TEXT } from "./labels";

type WorkflowGraph = NonNullable<RunProjection["workflow"]["graph"]>;
type WorkflowGraphNode = WorkflowGraph["nodes"][number];
type WorkflowGraphEdge = WorkflowGraph["edges"][number];

export type WorkflowNodeKind = "gate" | "validator" | "stage";
export type WorkflowStepTone = "ok" | "live" | "warn" | "bad" | "skipped" | "muted";
export type WorkflowPhaseState = "done" | "active" | "waiting" | "failed" | "pending";
export type WorkflowPhaseTone = "ok" | "live" | "warn" | "bad" | "muted";
export type ConnectorState = "traversed" | "skipped" | "pending";

export interface RouteBadge {
  text: string;
  tone: "muted" | "warn" | "bad";
}

export interface WorkflowStageStep {
  id: string;
  label: string;
  kind: WorkflowNodeKind;
  kindLabel: string;
  status: string;
  tone: WorkflowStepTone;
  glyph: string;
  statusText: string;
  routeBadge: RouteBadge | null;
  connectorBefore: ConnectorState | null;
  ariaLabel: string;
  title: string;
}

export interface WorkflowStageColumn {
  id: string;
  indexLabel: string;
  title: string;
  god: string;
  state: WorkflowPhaseState;
  statusText: string;
  statusTone: WorkflowPhaseTone;
  steps: WorkflowStageStep[];
}

export interface WorkflowStagesModel {
  columns: WorkflowStageColumn[];
  totalNodes: number;
}

interface PhaseDefinition {
  id: string;
  title: string;
  god: string;
}

const PHASE_DEFINITIONS: readonly PhaseDefinition[] = [
  { id: "align", title: "需求对齐", god: "Prometheus" },
  { id: "design", title: "白盒架构", god: "Athena" },
  { id: "plan", title: "任务编排", god: "Athena" },
  { id: "execute", title: "执行验收", god: "Hephaestus" },
  { id: "other", title: "扩展步骤", god: "Olympus" },
];

const NODE_PHASE_BY_ID: Record<string, string> = {
  intake: "align",
  align: "align",
  critic: "align",
  gate_spec: "align",
  design: "design",
  validate_design: "design",
  gate_design: "design",
  plan: "plan",
  validate_plan: "plan",
  gate_plan: "plan",
  execute: "execute",
  verify: "execute",
};

const KIND_LABELS: Record<WorkflowNodeKind, string> = {
  gate: "关口",
  validator: "校验",
  stage: "",
};

const STEP_GLYPHS: Record<string, string> = {
  completed: "✓",
  running: "▶",
  paused: "◷",
  failed: "✕",
  error: "✕",
  skipped: "—",
  pending: "○",
};

export function classifyWorkflowNode(name: string, kind?: string | null): WorkflowNodeKind {
  const raw = String(kind || "").toLowerCase();
  const id = String(name || "").toLowerCase();
  if (raw === "human_gate" || raw.startsWith("gate") || id.startsWith("gate_")) return "gate";
  if (raw.startsWith("validate") || id.startsWith("validate_")) return "validator";
  return "stage";
}

export function workflowStepTone(status: string): WorkflowStepTone {
  switch (status) {
    case "completed":
      return "ok";
    case "running":
      return "live";
    case "paused":
      return "warn";
    case "failed":
    case "error":
      return "bad";
    case "skipped":
      return "skipped";
    default:
      return "muted";
  }
}

function resolvePhaseId(node: WorkflowGraphNode): string {
  const direct = NODE_PHASE_BY_ID[node.name];
  if (direct) return direct;
  const id = node.name.toLowerCase();
  const kind = String(node.kind || "").toLowerCase();
  if (id.includes("spec") || id.includes("align") || id.includes("intake") || id.includes("critic") || kind.includes("align") || kind.includes("intake")) {
    return "align";
  }
  if (id.includes("design") || kind.includes("design")) return "design";
  if (id.includes("plan") || kind.includes("plan") || kind.includes("decompose")) return "plan";
  if (id.includes("exec") || id.includes("verify") || kind.includes("exec") || kind.includes("verify")) {
    return "execute";
  }
  return "other";
}

function buildRouteBadge(
  nodeId: string,
  status: string,
  nodePhase: string,
  order: Map<string, number>,
  nodePhaseMap: Map<string, string>,
  kindMap: Map<string, WorkflowNodeKind>,
  edges: WorkflowGraphEdge[],
  traversedSet: Set<string>,
): RouteBadge | null {
  const srcIndex = order.get(nodeId);
  if (srcIndex === undefined) return null;

  const outgoing = edges.filter((edge) => edge.source === nodeId && order.has(edge.target));

  for (const edge of outgoing) {
    const key = `${edge.source}\u2192${edge.target}`;
    if (!traversedSet.has(key)) continue;
    const dstIndex = order.get(edge.target)!;
    if (dstIndex <= srcIndex) {
      return { text: `回退${NODE_LABELS[edge.target] || edge.target}`, tone: "warn" };
    }
    if (dstIndex > srcIndex + 1) {
      const targetKind = kindMap.get(edge.target);
      const samePhase = nodePhaseMap.get(edge.target) === nodePhase;
      if (samePhase && targetKind === "gate") {
        return { text: "直达关口", tone: "muted" };
      }
      return { text: `直达${NODE_LABELS[edge.target] || edge.target}`, tone: "muted" };
    }
  }

  if (status === "failed" || status === "error") {
    for (const edge of outgoing) {
      const dstIndex = order.get(edge.target)!;
      if (dstIndex <= srcIndex) {
        return { text: `回退 ${NODE_LABELS[edge.target] || edge.target}`, tone: "bad" };
      }
    }
  }

  return null;
}

function summarizePhase(steps: WorkflowStageStep[]): {
  state: WorkflowPhaseState;
  statusText: string;
  statusTone: WorkflowPhaseTone;
} {
  const total = steps.length;
  const completed = steps.filter((step) => step.status === "completed").length;
  const skipped = steps.filter((step) => step.status === "skipped").length;

  if (steps.some((step) => step.status === "failed" || step.status === "error")) {
    return { state: "failed", statusText: "✕ 中断", statusTone: "bad" };
  }
  if (steps.some((step) => step.status === "paused")) {
    return { state: "waiting", statusText: "◷ 待确认", statusTone: "warn" };
  }
  if (steps.some((step) => step.status === "running")) {
    return { state: "active", statusText: "▶ 运行中", statusTone: "live" };
  }
  if (completed + skipped === total && total > 0) {
    if (completed === 0) {
      return { state: "done", statusText: "— 已跳过", statusTone: "muted" };
    }
    return { state: "done", statusText: `✓ ${completed}/${total}`, statusTone: "ok" };
  }
  if (completed > 0) {
    return { state: "active", statusText: `▶ ${completed}/${total}`, statusTone: "live" };
  }
  return { state: "pending", statusText: "◷ 待开始", statusTone: "muted" };
}

function connectorBetween(
  prev: WorkflowStageStep,
  cur: WorkflowStageStep,
  traversedSet: Set<string>,
): ConnectorState {
  if (prev.status === "skipped" || cur.status === "skipped") return "skipped";
  if (traversedSet.has(`${prev.id}\u2192${cur.id}`)) return "traversed";
  if (prev.status === "completed" && cur.status !== "pending") return "traversed";
  return "pending";
}

export function buildWorkflowStages(projection: RunProjection): WorkflowStagesModel {
  const workflow = projection.workflow;
  const rawNodes = workflow.graph?.nodes || [];
  const rawEdges = workflow.graph?.edges || [];
  const statuses = workflow.node_statuses || {};

  const order = new Map<string, number>();
  const uniqueNodes: WorkflowGraphNode[] = [];
  for (const node of rawNodes) {
    if (!node?.name || order.has(node.name)) continue;
    order.set(node.name, uniqueNodes.length);
    uniqueNodes.push(node);
  }

  if (!uniqueNodes.length) {
    return { columns: [], totalNodes: 0 };
  }

  const traversedSet = new Set(
    (workflow.traversed_edges || []).map((edge) => `${edge.source}\u2192${edge.target}`),
  );

  const nodePhaseMap = new Map<string, string>();
  const kindMap = new Map<string, WorkflowNodeKind>();
  const grouped = new Map<string, WorkflowGraphNode[]>();

  for (const node of uniqueNodes) {
    const phaseId = resolvePhaseId(node);
    nodePhaseMap.set(node.name, phaseId);
    kindMap.set(node.name, classifyWorkflowNode(node.name, node.kind));
    const list = grouped.get(phaseId);
    if (list) list.push(node);
    else grouped.set(phaseId, [node]);
  }

  const columns: WorkflowStageColumn[] = [];
  for (const def of PHASE_DEFINITIONS) {
    const nodesInPhase = grouped.get(def.id);
    if (!nodesInPhase?.length) continue;

    const steps: WorkflowStageStep[] = nodesInPhase.map((node) => {
      const status = statuses[node.name] || "pending";
      const label = NODE_LABELS[node.name] || node.name;
      const kind = kindMap.get(node.name) || "stage";
      const kindLabel = KIND_LABELS[kind];
      const meta = statusMeta(status);
      const statusText = WORKFLOW_STATUS_TEXT[status] || `${meta.glyph} ${meta.label}`;
      const routeBadge = buildRouteBadge(
        node.name,
        status,
        def.id,
        order,
        nodePhaseMap,
        kindMap,
        rawEdges,
        traversedSet,
      );
      const baseAria = `${label} (${node.name}) ${statusText}`;
      return {
        id: node.name,
        label,
        kind,
        kindLabel,
        status,
        tone: workflowStepTone(status),
        glyph: STEP_GLYPHS[status] || "○",
        statusText,
        routeBadge,
        connectorBefore: null,
        ariaLabel: kindLabel ? `${baseAria} · ${kindLabel}` : baseAria,
        title: `${label} · ${node.kind || node.name}`,
      };
    });

    for (let i = 1; i < steps.length; i += 1) {
      steps[i] = {
        ...steps[i],
        connectorBefore: connectorBetween(steps[i - 1], steps[i], traversedSet),
      };
    }

    const summary = summarizePhase(steps);
    const indexNumber = String(columns.length + 1).padStart(2, "0");
    columns.push({
      id: def.id,
      indexLabel: `STAGE ${indexNumber}`,
      title: def.title,
      god: def.god,
      ...summary,
      steps,
    });
  }

  return { columns, totalNodes: uniqueNodes.length };
}
