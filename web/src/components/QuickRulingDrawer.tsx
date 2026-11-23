import { useCallback, useEffect, useRef, useState } from "react";

import type { ApprovalAction, CommandBody, GoalSpec, RunProjection } from "../api/types";
import { useRunStream, type StreamPhase } from "../api/useRunStream";
import {
  APPROVAL_LABELS,
  boardBadgeVariant,
  GOD_META,
  localizeError,
  pendingAmbiguities,
  shortOracleCode,
  STATUS_BADGE_LABELS,
  TASK_TYPE_LABELS,
  taskBlocker,
  unresolvedAmbiguities,
  WORKSPACE_MODE_LABELS,
} from "../lib/labels";
import {
  buildApprovalSnapshot,
  executeRunCommand,
  resolvePointer,
  type ApprovalOrigin,
  type ApprovalSnapshot,
  type OriginVersion,
  type Workbench,
  type WorkbenchState,
} from "../state/workbench";
import { ApprovalReviewCard } from "./HitlBanner";
import { ClarifyPanel } from "./IntentClarify";
import { Badge } from "./ui";

const APPROVAL_ACTIONS = Object.keys(APPROVAL_LABELS) as ApprovalAction[];

export function QuickRulingDrawer({
  workbench,
  state,
}: {
  workbench: Workbench;
  state: WorkbenchState;
}) {
  const runId = state.activeSurface === "board" ? state.quickDrawerRunId : null;
  const [projection, setProjection] = useState<RunProjection | null>(null);
  const [phase, setPhase] = useState<StreamPhase | null>(null);
  const [approvalSnapshot, setApprovalSnapshot] = useState<ApprovalSnapshot | null>(null);
  const [approvalOrigins, setApprovalOrigins] = useState<
    Partial<Record<ApprovalAction, ApprovalOrigin>>
  >({});
  const [clarifyGen, setClarifyGen] = useState(0);
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState<{ tone: "warn" | "bad"; text: string } | null>(null);

  const drawerRef = useRef<HTMLElement>(null);
  const activeRunIdRef = useRef<string | null>(runId);
  const projectionRef = useRef<RunProjection | null>(null);
  const snapshotRef = useRef<ApprovalSnapshot | null>(null);
  const originsRef = useRef<Partial<Record<ApprovalAction, ApprovalOrigin>>>({});

  activeRunIdRef.current = runId;
  projectionRef.current = projection;
  snapshotRef.current = approvalSnapshot;
  originsRef.current = approvalOrigins;

  useEffect(() => {
    setProjection(null);
    setPhase(runId ? "loading" : null);
    setApprovalSnapshot(null);
    setApprovalOrigins({});
    setNotice(null);
    setBusy(false);
    setClarifyGen(0);
  }, [runId]);

  const applyDrawerProjection = useCallback((incoming: RunProjection) => {
    const previous = projectionRef.current;
    const sameRun = previous?.run_id === incoming.run_id;
    if (
      sameRun &&
      previous &&
      (incoming.state_revision < previous.state_revision ||
        incoming.latest_event_seq < previous.latest_event_seq ||
        (incoming.state_revision === previous.state_revision &&
          incoming.latest_event_seq === previous.latest_event_seq))
    ) {
      return;
    }
    projectionRef.current = incoming;
    setProjection(incoming);
    const nextApprovals = buildApprovalSnapshot(incoming, snapshotRef.current, originsRef.current);
    if (nextApprovals) {
      snapshotRef.current = nextApprovals.snapshot;
      originsRef.current = nextApprovals.origins;
      setApprovalSnapshot(nextApprovals.snapshot);
      setApprovalOrigins(nextApprovals.origins);
    }
  }, []);

  useRunStream(
    runId,
    0,
    {
      current: () => projectionRef.current,
      apply: applyDrawerProjection,
      phase: (nextPhase) => setPhase(nextPhase),
      activity: () => undefined,
      missing: (message) => setNotice({ tone: "bad", text: message }),
    },
    { sse: workbench.sse },
  );

  useEffect(() => {
    if (!runId) return undefined;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      const hitl = document.getElementById("hitl-dialog") as HTMLDialogElement | null;
      if (hitl?.open) return;
      workbench.closeQuickDrawer();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [runId, workbench]);

  if (!runId) return null;

  const summary =
    state.allRuns.find((item) => item.run_id === runId) ||
    state.runs.find((item) => item.run_id === runId) ||
    null;

  const god = summary ? GOD_META[summary.active_god] || GOD_META.prometheus : GOD_META.prometheus;
  const badgeVariant = summary
    ? boardBadgeVariant(summary.ball_holder, summary.status_badge)
    : "warn";
  const badgeLabel = summary
    ? STATUS_BADGE_LABELS[summary.status_badge] || summary.status_badge
    : "Ruling";

  const sendDrawerCommand = async (body: CommandBody, successMessage: string): Promise<void> => {
    const targetRunId = runId;
    const current = projectionRef.current;
    if (!current || current.run_id !== targetRunId || busy) return;
    setBusy(true);
    setNotice(null);
    const result = await executeRunCommand(targetRunId, current.state_revision, body);
    const stillActive =
      activeRunIdRef.current === targetRunId &&
      workbench.state.activeSurface === "board" &&
      workbench.state.quickDrawerRunId === targetRunId;

    if (result.kind === "network_error" || result.kind === "rejected") {
      if (stillActive) {
        setNotice({ tone: "bad", text: result.message });
        setBusy(false);
      }
      workbench.toast(result.message);
      return;
    }

    if (result.kind === "conflict") {
      if (result.projection) {
        if (stillActive) applyDrawerProjection(result.projection);
        workbench.refreshRuns().catch(() => undefined);
      }
      if (stillActive) {
        setNotice({ tone: "warn", text: result.message });
        setBusy(false);
      }
      workbench.toast(result.message);
      return;
    }

    workbench.toast(successMessage);
    if (stillActive) {
      applyDrawerProjection(result.projection);
      setBusy(false);
    }
    await workbench.refreshRuns().catch(() => undefined);
    if (
      activeRunIdRef.current === targetRunId &&
      workbench.state.quickDrawerRunId === targetRunId
    ) {
      workbench.closeQuickDrawer();
    }
  };

  const handleClarify = async (answers: Record<string, string>) => {
    const current = projectionRef.current;
    if (!current) return;
    const missing = pendingAmbiguities(current).filter((amb) => amb.blocking && !answers[amb.id]);
    if (missing.length) {
      const msg = `还有 ${missing.length} 个阻断歧义未回答`;
      setNotice({ tone: "warn", text: msg });
      workbench.toast(msg);
      return;
    }
    if (!Object.keys(answers).length) {
      const msg = "请先回答至少一个澄清问题";
      setNotice({ tone: "warn", text: msg });
      workbench.toast(msg);
      return;
    }
    setClarifyGen((value) => value + 1);
    await sendDrawerCommand({ action: "clarify", answers }, "已提交澄清并生成新的 GoalSpec 版本");
  };

  const handleApprove = (action: ApprovalAction) => {
    const binding = snapshotRef.current?.reviews.get(action);
    const current = projectionRef.current;
    if (!binding || !current) return;
    const { artifact, command } = binding;
    if (command.action === "approve_spec") {
      const spec = artifact as GoalSpec;
      const blockers = unresolvedAmbiguities(spec);
      const blockingCritique = (spec.minority_notes || []).some(
        (note) => note.severity === "blocking" && !note.resolved,
      );
      if (blockers.length || blockingCritique) {
        workbench.openDialog({
          title: `GoalSpec v${spec.version} 仍有未决阻断项`,
          description: "请先处理阻断项。若要批准此版本，请填写接受风险的理由。",
          label: "接受风险理由（必填）",
          defaultValue: "首期范围已与业务确认，接受未决歧义风险先行冻结规格。",
          requireNonEmpty: true,
          onConfirm: (reason) =>
            sendDrawerCommand(
              { ...command, action: "approve_spec", accept_risk: true, reason },
              "已带风险批准 GoalSpec，进入架构设计",
            ),
        });
        return;
      }
      void sendDrawerCommand(command, "已批准 GoalSpec，进入架构设计");
      return;
    }
    if (command.action === "approve_design") {
      void sendDrawerCommand(command, "已批准 DesignSpec，启动任务分解");
      return;
    }
    const message =
      current.workflow?.stop_after === "plan"
        ? "已批准任务计划，蓝图编排已完成"
        : "已批准任务计划，开始执行";
    void sendDrawerCommand(command, message);
  };

  const handleLocateOrigin = (action: ApprovalAction, ref: string, baseline: boolean) => {
    const binding = snapshotRef.current?.reviews.get(action);
    const artifact = baseline ? binding?.baseline : binding?.artifact;
    if (!binding || !artifact) return;
    const { parts, value } = resolvePointer(artifact, ref);
    if (value === undefined) return;
    const requirements = (artifact as { requirements?: { source_ref?: string | null }[] })
      .requirements;
    const source =
      parts[0] === "requirements" ? (requirements?.[Number(parts[1])]?.source_ref ?? null) : null;
    const version: OriginVersion = baseline ? "baseline" : "candidate";
    setApprovalOrigins((prev) => ({
      ...prev,
      [action]: {
        version,
        ref,
        text: typeof value === "string" ? value : JSON.stringify(value),
        source,
      },
    }));
    requestAnimationFrame(() => {
      const card = drawerRef.current?.querySelector(`[data-review-action="${action}"]`);
      const details = card?.querySelector<HTMLDetailsElement>(
        `[data-review-disclosure="${version}"]`,
      );
      if (!details) return;
      details.open = true;
      details.querySelector<HTMLElement>(".approval-origin code")?.focus();
    });
  };

  const openCockpit = () => {
    workbench.openOracleTab(runId);
  };

  const openTaskGraph = () => {
    workbench.openOracleTab(runId);
    workbench.setView("topology");
  };

  const pendingClarifications = projection ? pendingAmbiguities(projection) : [];
  const reviews =
    approvalSnapshot && projection && approvalSnapshot.runId === projection.run_id
      ? approvalSnapshot.reviews
      : new Map();
  const blocker = projection ? taskBlocker(projection, null) : null;
  const blockerTaskId = blocker?.task_id || null;
  const gateBanner = projection?.hitl_banner || null;
  const isPaused = projection?.status === "paused";
  const promoteGate = gateBanner?.kind === "blueprint_completed";
  const canRetryWorkflow = Boolean(
    gateBanner && Array.isArray(gateBanner.actions) && gateBanner.actions.includes("retry_workflow"),
  );
  const runFailed = gateBanner?.kind === "run_failed" || projection?.status === "failed";
  const hasAnyRuling =
    pendingClarifications.length > 0 ||
    reviews.size > 0 ||
    Boolean(blocker) ||
    isPaused ||
    promoteGate ||
    canRetryWorkflow ||
    runFailed;

  const openBlockerHint = () => {
    const current = projectionRef.current;
    if (!blockerTaskId || !current) return;
    const expectedRevision = current.state_revision;
    workbench.openDialog({
      title: `向任务 ${blockerTaskId} 注入人工提示并重试`,
      description:
        "提示原文保留在审计事件中，随下次隔离沙盒执行一并注入 Worker 上下文，不直接篡改需求契约。",
      label: "注入提示内容",
      defaultValue: "",
      placeholder: "输入要注入的人工提示，例如需修复的边界或重试思路。",
      requireNonEmpty: true,
      onConfirm: (hint) =>
        sendDrawerCommand(
          {
            action: "retry_task",
            task_id: blockerTaskId,
            hint,
            expected_state_revision: expectedRevision,
          },
          `已向 ${blockerTaskId} 注入提示并重新排入就绪队列`,
        ),
    });
  };

  const openBlockerReassign = () => {
    const current = projectionRef.current;
    if (!blockerTaskId || !current) return;
    const expectedRevision = current.state_revision;
    workbench.openDialog({
      title: `将任务 ${blockerTaskId} 改派给人工工程师`,
      description:
        "改派后人工提交的产物仍须经过相同的程序门禁与 Argus 独立复核才能成为已验收状态。",
      label: "改派备注",
      defaultValue: "",
      placeholder: "输入改派备注，例如接手人与需关注的验收要点。",
      requireNonEmpty: true,
      onConfirm: (note) =>
        sendDrawerCommand(
          {
            action: "reassign_human",
            task_id: blockerTaskId,
            note,
            expected_state_revision: expectedRevision,
          },
          `已记录 ${blockerTaskId} 改派人工并回到就绪态`,
        ),
    });
  };

  const openBlockerAcceptRisk = () => {
    const current = projectionRef.current;
    if (!blockerTaskId || !current) return;
    const expectedRevision = current.state_revision;
    workbench.openDialog({
      title: `接受风险完成任务 ${blockerTaskId}`,
      description:
        "必须填写明确理由。该决定与 Agent 建议分开存储，追踪矩阵标记为“未验证 · 人工接受”，绝不显示为“已验证”。",
      label: "接受风险理由（必填）",
      defaultValue: "",
      placeholder: "输入接受风险的明确理由，例如影响范围与缓解措施。",
      requireNonEmpty: true,
      onConfirm: (reason) =>
        sendDrawerCommand(
          {
            action: "accept_risk",
            task_id: blockerTaskId,
            reason,
            expected_state_revision: expectedRevision,
          },
          `已记录人工接受风险，${blockerTaskId} 标记为未验证 · 人工接受`,
        ),
    });
  };

  return (
    <div className="ruling-drawer-layer" id="quick-ruling-drawer-layer">
      <div
        className="ruling-drawer-scrim"
        aria-hidden="true"
        onClick={() => workbench.closeQuickDrawer()}
      />
      <aside
        ref={drawerRef}
        id="quick-ruling-drawer"
        className="ruling-drawer"
        role="dialog"
        aria-labelledby="quick-drawer-title"
        data-run-id={runId}
      >
        <header className="ruling-drawer-head">
          <div className="ruling-drawer-head-top">
            <div className="ruling-drawer-tags">
              <span className="oracle-card-code">{shortOracleCode(runId)}</span>
              {summary ? (
                <>
                  <span>{summary.workspace_name}</span>
                  <span aria-hidden="true">·</span>
                  <span>{TASK_TYPE_LABELS[summary.task_type] || summary.task_type}</span>
                  <span aria-hidden="true">·</span>
                  <span>{WORKSPACE_MODE_LABELS[summary.workspace_mode] || summary.workspace_mode}</span>
                </>
              ) : null}
              <Badge variant={badgeVariant}>{badgeLabel}</Badge>
            </div>
            <button
              type="button"
              className="btn btn-quiet"
              id="btn-drawer-close"
              aria-label="Close Quick Ruling Drawer"
              onClick={() => workbench.closeQuickDrawer()}
            >
              ×
            </button>
          </div>

          <h2 className="ruling-drawer-title" id="quick-drawer-title">
            {projection?.title || summary?.title || runId}
          </h2>

          <div className="ruling-drawer-bar">
            <span className="oracle-card-god">
              <span className="oracle-god-monogram" aria-hidden="true">
                {god.monogram}
              </span>
              <span>{`${god.name} → You`}</span>
            </span>
            <button
              type="button"
              className="btn btn-quiet"
              id="btn-drawer-open-cockpit"
              onClick={openCockpit}
            >
              Open Full Ariadne Cockpit ↗
            </button>
          </div>
        </header>

        <div className="ruling-drawer-body scroller">
          {notice ? (
            <p
              className="ruling-drawer-notice"
              id="quick-drawer-notice"
              data-tone={notice.tone}
              role="alert"
            >
              {notice.text}
            </p>
          ) : null}

          {!projection ? (
            <p className="ruling-drawer-empty" id="quick-drawer-loading">
              {phase === "missing" ? "找不到该 Oracle 的运行记录。" : "正在加载 Oracle 裁决上下文…"}
            </p>
          ) : (
            <>
              {projection.raw_input ? (
                <section className="ruling-drawer-intent" aria-label="Oracle Intent">
                  <p className="dialog-label">Intent</p>
                  <p className="ruling-drawer-intent-text">{projection.raw_input}</p>
                </section>
              ) : null}

              {pendingClarifications.length > 0 ? (
                <ClarifyPanel
                  projection={projection}
                  generation={clarifyGen}
                  onSubmit={handleClarify}
                  submitting={busy}
                />
              ) : null}

              {reviews.size > 0 ? (
                <section
                  className="approval-reviews"
                  id="quick-drawer-approvals"
                  aria-label="审批候选版本"
                >
                  {APPROVAL_ACTIONS.map((action) => {
                    const binding = reviews.get(action);
                    const review = projection.approval_reviews?.[action];
                    if (!binding || !review) return null;
                    return (
                      <ApprovalReviewCard
                        key={action}
                        action={action}
                        review={review}
                        artifact={binding.artifact}
                        origin={approvalOrigins[action]}
                        onLocate={(ref, old) => handleLocateOrigin(action, ref, old)}
                        onOpenTaskGraph={openTaskGraph}
                        onApprove={handleApprove}
                        disabled={busy}
                      />
                    );
                  })}
                </section>
              ) : null}

              {blocker ? (
                <section className="hitl-banner" id="quick-drawer-blocker" data-tone="danger">
                  <div className="hitl-copy">
                    <p className="hitl-headline">
                      <span aria-hidden="true">✕</span>
                      <span>{blocker.headline || "任务受阻，需人工介入"}</span>
                    </p>
                    <p className="hitl-detail">
                      {`受影响任务：${(blocker.affected_task_ids || []).join("、")}。原因：${blocker.detail}`}
                    </p>
                  </div>
                  <div className="hitl-actions">
                    <button
                      type="button"
                      className="btn btn-primary"
                      id="btn-drawer-blocker-hint"
                      disabled={busy}
                      onClick={openBlockerHint}
                    >
                      注入提示
                    </button>
                    <button
                      type="button"
                      className="btn"
                      id="btn-drawer-blocker-retry"
                      disabled={busy || !blockerTaskId}
                      onClick={() =>
                        blockerTaskId &&
                        void sendDrawerCommand(
                          { action: "retry_task", task_id: blockerTaskId, hint: "" },
                          `已将任务 ${blockerTaskId} 重置为就绪 (READY)`,
                        )
                      }
                    >
                      重试任务
                    </button>
                    <button
                      type="button"
                      className="btn"
                      id="btn-drawer-blocker-reassign"
                      disabled={busy}
                      onClick={openBlockerReassign}
                    >
                      改派给人
                    </button>
                    <button
                      type="button"
                      className="btn btn-danger"
                      id="btn-drawer-blocker-accept-risk"
                      disabled={busy}
                      onClick={openBlockerAcceptRisk}
                    >
                      接受风险
                    </button>
                  </div>
                </section>
              ) : null}

              {isPaused ? (
                <section className="hitl-banner" id="quick-drawer-paused" data-tone="warn">
                  <div className="hitl-copy">
                    <p className="hitl-headline">
                      <span aria-hidden="true">◷</span>
                      <span>编排当前处于暂停状态</span>
                    </p>
                    <p className="hitl-detail">恢复调度后，Pantheon 将继续推进剩余任务。</p>
                  </div>
                  <div className="hitl-actions">
                    <button
                      type="button"
                      className="btn btn-primary"
                      id="btn-drawer-resume"
                      disabled={busy}
                      onClick={() =>
                        void sendDrawerCommand({ action: "resume" }, "已恢复任务编排调度")
                      }
                    >
                      恢复编排
                    </button>
                  </div>
                </section>
              ) : null}

              {canRetryWorkflow || runFailed ? (
                <section className="hitl-banner" id="quick-drawer-failed" data-tone="danger">
                  <div className="hitl-copy">
                    <p className="hitl-headline">
                      <span aria-hidden="true">✕</span>
                      <span>{gateBanner?.headline || "工作流执行中断"}</span>
                    </p>
                    {gateBanner?.detail ? (
                      <p className="hitl-detail">{localizeError(gateBanner.detail)}</p>
                    ) : null}
                  </div>
                  <div className="hitl-actions">
                    {canRetryWorkflow ? (
                      <button
                        type="button"
                        className="btn btn-primary"
                        id="btn-drawer-retry-workflow"
                        disabled={busy}
                        onClick={() =>
                          void sendDrawerCommand(
                            { action: "retry_workflow" },
                            "已从断点重新启动工作流",
                          )
                        }
                      >
                        从断点重试
                      </button>
                    ) : null}
                    <button type="button" className="btn" onClick={openCockpit}>
                      查看节点日志
                    </button>
                  </div>
                </section>
              ) : null}

              {promoteGate ? (
                <section className="hitl-banner" id="quick-drawer-promote" data-tone="ok">
                  <div className="hitl-copy">
                    <p className="hitl-headline">
                      <span aria-hidden="true">✓</span>
                      <span>{gateBanner?.headline || "产品蓝图已定稿"}</span>
                    </p>
                    {gateBanner?.detail ? (
                      <p className="hitl-detail">{gateBanner.detail}</p>
                    ) : null}
                  </div>
                  <div className="hitl-actions">
                    <button
                      type="button"
                      className="btn btn-primary"
                      id="btn-drawer-promote"
                      disabled={busy}
                      onClick={() =>
                        void sendDrawerCommand(
                          { action: "promote_to_execute", workflow_name: "full_sdlc_tdd" },
                          "已升级为 full_sdlc_tdd，启动沙盒代码执行",
                        )
                      }
                    >
                      转入沙盒执行 (full_sdlc_tdd)
                    </button>
                  </div>
                </section>
              ) : null}

              {!hasAnyRuling ? (
                <div className="ruling-drawer-empty" id="quick-drawer-resolved">
                  <p>该 Oracle 当前没有待裁决关口，可关闭抽屉或进入 Ariadne 控制台查看完整进度。</p>
                </div>
              ) : null}
            </>
          )}
        </div>
      </aside>
    </div>
  );
}
