import type { ReactNode } from "react";

import type { RunSummary } from "../api/types";
import {
  boardBadgeVariant,
  GOD_META,
  PENDING_ACTION_CTA,
  shortOracleCode,
  STATUS_BADGE_LABELS,
  TASK_TYPE_LABELS,
  WORKSPACE_MODE_LABELS,
} from "../lib/labels";

export interface OracleCardProps {
  run: RunSummary;
  isOpenTab: boolean;
  dimmed?: boolean;
  highlighted?: boolean;
  onOpenTab: (runId: string) => void;
  /** Modular slot for T4 Quick Ruling Drawer trigger on human-held cards. */
  onQuickRule?: (runId: string) => void;
  /** Optional custom footer action slot for downstream feature extensions. */
  actionSlot?: ReactNode;
}

export function OracleCard({
  run,
  isOpenTab,
  dimmed = false,
  highlighted = false,
  onOpenTab,
  onQuickRule,
  actionSlot,
}: OracleCardProps) {
  const god = GOD_META[run.active_god] || GOD_META.prometheus;
  const badgeVariant = boardBadgeVariant(run.ball_holder, run.status_badge);
  const badgeLabel = STATUS_BADGE_LABELS[run.status_badge] || run.status_badge;
  const typeLabel = TASK_TYPE_LABELS[run.task_type] || run.task_type;
  const modeLabel = WORKSPACE_MODE_LABELS[run.workspace_mode] || run.workspace_mode;
  const code = shortOracleCode(run.run_id);

  const handoffText =
    run.ball_holder === "human"
      ? `${god.name} → You`
      : run.worktree_branch
        ? `${god.name} · ${run.worktree_branch}`
        : `${god.name} · ${god.role}`;

  const quickLabel = run.pending_action ? PENDING_ACTION_CTA[run.pending_action] : "Review";

  return (
    <article
      className="oracle-card"
      data-run-id={run.run_id}
      data-ball={run.ball_holder}
      data-god={run.active_god}
      data-mode={run.workspace_mode}
      data-dimmed={dimmed ? "true" : undefined}
      data-highlighted={highlighted ? "true" : undefined}
    >
      <button
        type="button"
        className="oracle-card-trigger"
        onClick={() => onOpenTab(run.run_id)}
      >
        <div className="oracle-card-meta">
          <span className="oracle-card-tags">
            <span>{run.workspace_name}</span>
            <span aria-hidden="true">·</span>
            <span>{typeLabel}</span>
            <span aria-hidden="true">·</span>
            <span>{modeLabel}</span>
          </span>
          <span className="oracle-status-pill" data-variant={badgeVariant}>
            {badgeLabel}
          </span>
        </div>

        <h3 className="oracle-card-title">
          <span className="oracle-card-code">{code}</span>
          <span>{run.title}</span>
        </h3>
      </button>

      <div className="oracle-card-foot">
        <span className="oracle-card-god">
          <span className="oracle-god-monogram" data-god={run.active_god} aria-hidden="true">
            {god.monogram}
          </span>
          <span className="oracle-card-handoff">{handoffText}</span>
        </span>

        <div className="oracle-card-actions">
          {isOpenTab ? (
            <span className="oracle-open-indicator" title="Opened in Ariadne Tabs">
              Tab Open
            </span>
          ) : null}
          {actionSlot ? (
            actionSlot
          ) : onQuickRule && run.ball_holder === "human" ? (
            <button
              type="button"
              className="oracle-quick-cta"
              data-quick-rule={run.run_id}
              onClick={() => onQuickRule(run.run_id)}
            >
              {quickLabel} →
            </button>
          ) : (
            <button
              type="button"
              className="oracle-open-cta"
              onClick={() => onOpenTab(run.run_id)}
            >
              Open Cockpit →
            </button>
          )}
        </div>
      </div>
    </article>
  );
}
