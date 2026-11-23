import type { components } from "./schema";

type Schemas = components["schemas"];

export type RunProjection = Schemas["RunProjection"];
export type RunSummary = Schemas["RunSummary"];
export type RunListResponse = Schemas["RunListResponse"];
export type RunProjectionResponse = Schemas["RunProjectionResponse"];
export type PantheonSeatSummary = Schemas["PantheonSeatSummary"];
export type WorkspaceSummary = Schemas["WorkspaceSummary"];
export type WorkspaceListResponse = Schemas["WorkspaceListResponse"];
export type WorkspaceResponse = Schemas["WorkspaceResponse"];
export type DeleteWorkspaceResponse = Schemas["DeleteWorkspaceResponse"];
export type FilesystemBrowseResponse = Schemas["FilesystemBrowseResponse"];
export type FilesystemBrowseEntry = Schemas["FilesystemBrowseEntry"];
export type QuickRoot = Schemas["QuickRoot"];
export type CloneWorkspaceBody = Schemas["CloneWorkspaceRequest"];
export type WorkspaceMode = RunSummary["workspace_mode"];
export type TaskType = RunSummary["task_type"];
export type BallHolder = RunSummary["ball_holder"];
export type PantheonGod = RunSummary["active_god"];
export type PendingAction = RunSummary["pending_action"];
export type SeatState = PantheonSeatSummary["state"];
export type RunEvent = Schemas["RunEvent"];
export type RunProgressItem = Schemas["RunProgressItem"];
export type DagNode = Schemas["DagNode"];
export type DagEdge = Schemas["DagEdge"];
export type GoalSpec = Schemas["GoalSpec"];
export type DesignSpec = Schemas["DesignSpec"];
export type TaskPlan = Schemas["TaskPlan"];
export type AmbiguityItem = Schemas["AmbiguityItem"];
export type ApprovalReview = Schemas["ApprovalReview"];
export type ReviewSection = Schemas["ReviewSection"];
export type ReviewEntry = Schemas["ReviewEntry"];
export type KeyOutputItem = Schemas["KeyOutputItem"];
export type DeliveryAcceptance = Schemas["DeliveryAcceptance"];
export type DeliveryCriterion = Schemas["DeliveryCriterion"];
export type TaskArtifacts = Schemas["TaskArtifacts"];
export type RecordedAttemptArtifacts = Schemas["RecordedAttemptArtifacts"];
export type AttemptArtifactsResponse = Schemas["AttemptArtifactsResponse"];
export type SubagentStep = Schemas["SubagentStep"];
export type TurnReceipt = Schemas["TurnReceipt"];
export type WorkflowView = Schemas["WorkflowView"];
export type WorkflowSummary = Schemas["WorkflowSummary"];
export type WorkflowsResponse = Schemas["WorkflowsResponse"];
export type ProviderStatus = Schemas["ProviderStatus"];
export type ProvidersResponse = Schemas["ProvidersResponse"];
export type DemoPreset = Schemas["DemoPreset"];
export type DemosResponse = Schemas["DemosResponse"];
export type HitlBanner = Schemas["HitlBanner"];
export type TaskBlocker = Schemas["TaskBlocker"];
export type CreateRunResponse = Schemas["CreateRunResponse"];
export type CommandResponse = Schemas["CommandResponse"];
export type ConflictResponse = Schemas["ConflictResponse"];
export type ErrorResponse = Schemas["ErrorResponse"];

export type ReviewArtifact = GoalSpec | DesignSpec | TaskPlan;

/** Generated request types mark server-side defaults as required; on the wire they are optional. */
type Loose<T, K extends PropertyKey> = Omit<T, K & keyof T> & Partial<Pick<T, K & keyof T>>;
type Defaults = "auto_advance" | "sync";

export type CreateRunBody = Loose<Schemas["CreateRunRequest"], Defaults | "auto_approve" | "init_git">;
export type CreateWorkspaceBody = Loose<Schemas["CreateWorkspaceRequest"], "create_if_missing" | "init_git">;

export type CommandBody =
  | Loose<Schemas["ApproveSpecCommand"], Defaults | "accept_risk" | "reason">
  | Loose<Schemas["ApproveDesignCommand"], Defaults | "reason">
  | Loose<Schemas["ApprovePlanCommand"], Defaults | "reason">
  | Loose<Schemas["ClarifyCommand"], Defaults>
  | Loose<Schemas["RetryTaskCommand"], Defaults | "hint">
  | Loose<Schemas["ReassignHumanCommand"], Defaults | "note">
  | Loose<Schemas["AcceptRiskCommand"], Defaults>
  | Loose<Schemas["PauseCommand"], Defaults>
  | Loose<Schemas["ResumeCommand"], Defaults>
  | Loose<Schemas["RetryWorkflowCommand"], Defaults>
  | Loose<Schemas["PromoteToExecuteCommand"], Defaults>;

export type ApprovalAction = "approve_spec" | "approve_design" | "approve_plan";

/** `event: state` frame of /api/runs/{run_id}/stream (the SSE route is outside the OpenAPI body). */
export interface StreamStateFrame {
  run_id: string;
  state_revision: number;
  events: { event_seq: number }[];
}

export interface ActivityPreview {
  attempt: number;
  preview: string;
}

export type ActivityPreviews = Record<string, ActivityPreview>;

/** `event: activity` frame: live worker previews keyed by task id. */
export interface StreamActivityFrame {
  run_id: string;
  previews: ActivityPreviews;
}
