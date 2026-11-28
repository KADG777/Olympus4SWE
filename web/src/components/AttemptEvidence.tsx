import { useEffect, useId, useState } from "react";

import { ApiError, fetchAttemptArtifacts } from "../api/client";
import { useT } from "../i18n";
import { summarizeAttempt } from "../lib/attemptEvidence";
import type { AttemptRef, TaskIncident } from "../lib/taskIncidents";

type LoadState = "loading" | "missing" | "error" | ReturnType<typeof summarizeAttempt>;

function AttemptColumn({ runId, taskId, attempt, onEvent }: {
  runId: string; taskId: string; attempt: AttemptRef; onEvent: (seq: number) => void;
}) {
  const t = useT();
  const [state, setState] = useState<LoadState>("loading");
  const [request, setRequest] = useState(0);
  useEffect(() => {
    const abort = new AbortController();
    setState("loading");
    fetchAttemptArtifacts(runId, taskId, attempt.number, abort.signal).then(art => {
      if (!abort.signal.aborted) setState(summarizeAttempt(art, taskId, attempt.number));
    }).catch(error => {
      if (!abort.signal.aborted) setState(error instanceof ApiError && error.status === 404 ? "missing" : "error");
    });
    return () => abort.abort();
  }, [runId, taskId, attempt.number, request]);

  const title = t("第 {n} 次尝试", { n: attempt.number });
  return <section className="attempt-column" aria-label={title}>
    <header className="replay-row"><strong>{title}</strong>
      <button className="btn" aria-label={t("查看事件 {n}", { n: attempt.seq })}
        onClick={() => onEvent(attempt.seq)}>#{attempt.seq}</button>
    </header>
    {state === "loading" ? <p role="status">{t("正在读取尝试产物…")}</p>
      : state === "missing" ? <p>{t("本次尝试没有归档产物。")}</p>
      : state === "error" ? <p role="alert">{t("尝试产物读取失败。")} <button className="btn"
          onClick={() => setRequest(n => n + 1)}>{t("重试")}</button></p>
      : <>
        <h4>{t("验收检查")}</h4>
        {state.checks.length ? <ul>{state.checks.map(([id, outcome]) => <li key={id}>
          <code>{id}</code>{" "}<span className="badge"
            data-variant={outcome === "通过" ? "ok" : outcome === "未通过" ? "bad" : "muted"}>{t(outcome)}</span>
        </li>)}</ul> : <p>{t("没有已记录的检查结果。")}</p>}
        {state.reviewCount > 0 && <p>{t("独立复核：{pass}/{total} 份通过", {
          pass: state.reviewsPassed, total: state.reviewCount,
        })}</p>}
        <h4>{t("已记录的补丁片段")}</h4>
        {state.hasPatch ? <>
          <code>+{state.added} / −{state.removed}</code>
          {state.files.length > 0 && <ul>{state.files.map(path => <li key={path}><code>{path}</code></li>)}</ul>}
        </> : <p>{t("没有已记录的补丁。")}</p>}
      </>}
  </section>;
}

export function AttemptEvidence({ runId, incident, onEvent }: {
  runId: string; incident: TaskIncident; onEvent: (seq: number) => void;
}) {
  const t = useT();
  const [open, setOpen] = useState(false);
  const panelId = useId();
  if (!incident.attempt) return <p>{t("故障记录缺少有效的尝试编号。")}</p>;
  const current = { number: incident.attempt, seq: incident.failure.seq };
  const pair = incident.nextAttempt ? [current, incident.nextAttempt] : [incident.previousFailure, current];
  return <div className="attempt-evidence">
    <button className="btn" type="button" aria-expanded={open} aria-controls={panelId}
      onClick={() => setOpen(value => !value)}>{t(open ? "收起尝试证据" : "查看尝试证据")}</button>
    {open && <div id={panelId}>
      <div className="attempt-columns">
        {pair.map(ref => ref && <AttemptColumn key={ref.number} runId={runId} taskId={incident.id}
          attempt={ref} onEvent={onEvent} />)}
      </div>
      {!incident.nextAttempt && <p>{t("截至回放终点，未记录后续尝试。")}</p>}
    </div>}
  </div>;
}
