import type { IslandController } from "../island";
import type { WorkflowGraphProps } from "../types";

/**
 * Reserved mount slot for the Orchestration Workflow graph (Ticket #57).
 * The console keeps its existing SVG workflow renderer until that cut-over lands.
 */
export function WorkflowGraph(props: WorkflowGraphProps & { controller: IslandController }) {
  return (
    <div className="ag-island" data-graph="workflow" data-empty="true">
      <p className="ag-notice" role="status">
        {props.emptyMessage || "工作流图岛尚未启用"}
      </p>
    </div>
  );
}
