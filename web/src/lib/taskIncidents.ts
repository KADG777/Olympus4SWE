import type { RunEvent } from "../api/types";

type Note = { seq: number; actor: string; label: string };
export type AttemptRef = { number: number; seq: number };
export interface TaskIncident {
  id: string;
  failure: Note;
  reason?: string;
  attempt?: number;
  previousFailure?: AttemptRef;
  nextAttempt?: AttemptRef;
  intervention?: Note;
  restart?: Note;
  result?: Note;
}

export function positiveAttempt(value: unknown): number | undefined {
  return typeof value === "number" && Number.isSafeInteger(value) && value > 0 ? value : undefined;
}

function failureLabel(event: RunEvent) {
  const p = event.payload ?? {};
  if (event.event_type === "task.diff_gate_rejected") return "文件变更未通过范围检查";
  if (event.event_type === "task.escape_hatch_rejected") return "代码变更未通过门禁检查";
  if (event.event_type === "review.verdict_recorded" && p.passed === false) return "独立审查未通过";
  if (event.event_type === "task.failed" || event.event_type === "task.blocked" ||
      (event.event_type === "agent.swimlane" && (p.status === "blocked" || p.status === "failed"))) {
    return "任务执行受阻";
  }
}

function summaryText(value: unknown) {
  if (typeof value !== "string" || !value.trim()) return;
  // Only show short recorded summaries. Omit credential-shaped or multiline content.
  if (/[\r\n]|bearer\s|api[_ -]?key|token\s*[:=]|password|secret|sk-[\w-]{8,}|gh[pousr]_/i.test(value)) return;
  return value.trim().slice(0, 180);
}

/** One card per task, rooted at its most recent recorded failure. */
export function taskIncidents(events: RunEvent[]): TaskIncident[] {
  const byTask = new Map<string, TaskIncident>();
  const attempts = new Map<string, number>();
  for (const event of events) {
    const p = event.payload ?? {};
    const id = p.task_id;
    if (typeof id !== "string" || !id) continue;
    const kind = event.event_type;
    const seq = event.event_seq;
    const actor = event.actor;
    const raw = kind === "agent.swimlane" ? p.attempts : p.attempt;
    const count = positiveAttempt(raw);
    if (kind === "task.leased") {
      if (count === undefined) attempts.delete(id);
      else attempts.set(id, count);
    }
    const attempt = raw == null ? attempts.get(id) : count;
    const incident = byTask.get(id);
    const failure = failureLabel(event);
    if (failure) {
      if (attempt && incident?.attempt && attempt < incident.attempt) continue;
      // A terminal blocked event can follow the specific rejection in the same attempt.
      if (kind === "task.blocked" && incident && attempt !== undefined && attempt === incident.attempt &&
          !incident.result && !incident.restart && !incident.intervention) continue;
      const previousFailure = attempt && incident?.attempt && incident.attempt < attempt
        ? { number: incident.attempt, seq: incident.failure.seq } : incident?.previousFailure;
      byTask.set(id, { id, attempt, previousFailure, failure: { seq, actor, label: failure },
        reason: kind === "agent.swimlane" ? summaryText(p.summary) : undefined });
      continue;
    }
    if (!incident || incident.result || (incident.attempt !== undefined && attempt !== undefined && attempt < incident.attempt)) continue;
    if (!incident.nextAttempt && incident.attempt && count && count > incident.attempt &&
        (kind === "task.leased" || kind === "review.verdict_recorded" || kind === "task.accepted")) {
      incident.nextAttempt = { number: count, seq };
    }
    const note = (label: string): Note => ({ label, seq, actor });
    if (kind === "task.retried_by_human") incident.intervention = note("人工已请求重试");
    if (kind === "task.requeued") incident.intervention = note("已重新排队");
    if (kind === "task.reassigned_human") incident.intervention = note("已改派人工");
    if (kind === "task.leased") incident.restart = note("已获得新的执行租约");
    if (kind === "task.accepted") incident.result = note("任务已验收");
    if (kind === "task.accepted_with_risk") incident.result = note("人工接受风险放行");
  }
  return [...byTask.values()].sort((a, b) =>
    Number(Boolean(a.result)) - Number(Boolean(b.result)) || b.failure.seq - a.failure.seq);
}
