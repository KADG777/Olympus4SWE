import { useState, type FormEvent } from "react";

import type { Workbench, WorkbenchState } from "../state/workbench";
import { NativeDialog } from "./Dialogs";

export function WorkspaceDialog({
  workbench,
  state,
}: {
  workbench: Workbench;
  state: WorkbenchState;
}) {
  const [rootPath, setRootPath] = useState("");
  const [name, setName] = useState("");
  const [createIfMissing, setCreateIfMissing] = useState(false);
  const [initGit, setInitGit] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const closeDialog = () => {
    const dialog = document.getElementById("workspace-manager-dialog") as HTMLDialogElement | null;
    if (dialog?.open) dialog.close();
  };

  const handleRegister = async (event: FormEvent) => {
    event.preventDefault();
    const trimmedPath = rootPath.trim();
    if (!trimmedPath) {
      setError("请填写本地仓库或工作目录路径");
      return;
    }
    setBusy(true);
    setError(null);
    const result = await workbench.registerWorkspace({
      root_path: trimmedPath,
      name: name.trim() || null,
      create_if_missing: createIfMissing,
      init_git: initGit,
    });
    setBusy(false);
    if (!result.ok) {
      setError(result.error);
      return;
    }
    setRootPath("");
    setName("");
    setCreateIfMissing(false);
    setInitGit(false);
  };

  const handleUnregister = async (workspaceId: string) => {
    setBusy(true);
    setError(null);
    await workbench.unregisterWorkspace(workspaceId);
    setBusy(false);
  };

  return (
    <NativeDialog
      id="workspace-manager-dialog"
      labelledBy="workspace-dialog-title"
      lightDismissFallback
      onClose={() => setError(null)}
    >
      <div className="workspace-dialog-body">
        <div className="workspace-dialog-head">
          <h2 className="dialog-title" id="workspace-dialog-title">
            Workspaces
          </h2>
          <button
            type="button"
            className="btn btn-quiet"
            aria-label="关闭工作区管理"
            onClick={closeDialog}
          >
            ×
          </button>
        </div>
        <p className="dialog-desc">
          注册本地仓库作为项目工作区以筛选 Oracle Board。移除注册仅清理索引条目，绝不会删除磁盘文件或历史编排记录。
        </p>

        <section className="workspace-section" aria-label="已注册工作区">
          <p className="dialog-label">Registered Workspaces ({state.workspaces.length})</p>
          <ul className="workspace-list scroller" id="workspace-list">
            {state.workspaces.length ? (
              state.workspaces.map((ws) => {
                const isSelected = state.selectedWorkspaceId === ws.workspace_id;
                const total = ws.active_oracle_count + ws.pending_ruling_count;
                return (
                  <li
                    className="workspace-row"
                    key={ws.workspace_id}
                    data-workspace-id={ws.workspace_id}
                    data-selected={isSelected ? "true" : "false"}
                  >
                    <div className="workspace-row-main">
                      <div className="workspace-row-title">
                        <span className="workspace-name">{ws.name}</span>
                        <span className="badge" data-variant="muted">
                          {ws.is_git_repo
                            ? ws.default_branch
                              ? `git · ${ws.default_branch}`
                              : "git"
                            : "local"}
                        </span>
                        {ws.pending_ruling_count > 0 ? (
                          <span className="badge" data-variant="warn">
                            {ws.pending_ruling_count} ruling
                          </span>
                        ) : null}
                        {total > 0 ? (
                          <span className="badge" data-variant="live">
                            {total} active
                          </span>
                        ) : null}
                      </div>
                      <p className="workspace-path mono">{ws.root_path}</p>
                    </div>
                    <div className="workspace-row-actions">
                      <button
                        type="button"
                        className="btn btn-quiet"
                        data-workspace-select={ws.workspace_id}
                        disabled={isSelected}
                        onClick={() => {
                          workbench.selectWorkspace(ws.workspace_id);
                          closeDialog();
                        }}
                      >
                        {isSelected ? "Current" : "Switch"}
                      </button>
                      <button
                        type="button"
                        className="btn btn-danger"
                        data-workspace-remove={ws.workspace_id}
                        disabled={busy}
                        onClick={() => void handleUnregister(ws.workspace_id)}
                      >
                        Unregister
                      </button>
                    </div>
                  </li>
                );
              })
            ) : (
              <li className="workspace-empty">尚未注册工作区</li>
            )}
          </ul>
        </section>

        <form className="workspace-register-form" id="workspace-register-form" onSubmit={(e) => void handleRegister(e)}>
          <p className="dialog-label">Register Local Workspace</p>
          <div className="form-grid">
            <div className="form-field">
              <label className="dialog-label" htmlFor="workspace-root-input">
                Directory Path
              </label>
              <input
                id="workspace-root-input"
                className="field"
                type="text"
                placeholder="/path/to/repository"
                value={rootPath}
                onChange={(event) => setRootPath(event.target.value)}
              />
            </div>
            <div className="form-field">
              <label className="dialog-label" htmlFor="workspace-name-input">
                Display Name (optional)
              </label>
              <input
                id="workspace-name-input"
                className="field"
                type="text"
                placeholder="Defaults to folder name"
                value={name}
                onChange={(event) => setName(event.target.value)}
              />
            </div>
          </div>
          <div className="workspace-checkbox-row">
            <label className="workspace-check">
              <input
                id="workspace-create-missing"
                type="checkbox"
                checked={createIfMissing}
                onChange={(event) => setCreateIfMissing(event.target.checked)}
              />
              <span>Create directory if missing</span>
            </label>
            <label className="workspace-check">
              <input
                id="workspace-init-git"
                type="checkbox"
                checked={initGit}
                onChange={(event) => setInitGit(event.target.checked)}
              />
              <span>Initialize Git repo</span>
            </label>
          </div>
          {error ? (
            <p className="notice" data-level="bad" id="workspace-dialog-error" role="alert">
              {error}
            </p>
          ) : null}
          <div className="dialog-actions">
            <button type="button" className="btn btn-quiet" onClick={closeDialog}>
              Done
            </button>
            <button
              type="submit"
              className="btn btn-primary"
              id="btn-register-workspace"
              disabled={busy}
            >
              {busy ? "Registering…" : "Register Workspace"}
            </button>
          </div>
        </form>
      </div>
    </NativeDialog>
  );
}
