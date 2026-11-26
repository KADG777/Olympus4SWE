# 作业 3 任务边界与 PR 合入标准（临时协作规范）

> [!NOTE]
> **临时协作文档**：本文档用于对齐「课堂作业 3：多智能体软件工程工作流（Multi-Agent SE workflow）的可观测性（可视化）框架」的任务边界、PR 拆分与合入标准。**作业 3 全部功能合入主线后将直接删除本文件**，请勿在生产代码或其他长期文档中硬引用本路径。

> **适用基线**：`origin/main @ 1b4fc18`（已包含 Oracle Board + Ariadne v0.1 双层工作台及 PR #158 / #159 / #160 / #162 / #164 / #166）  
> **阅读前必读**：仓库根目录 [AGENTS.md](../AGENTS.md) 与 [CONTRIBUTING.md](../CONTRIBUTING.md)

---

## 核心协作原则

1. **后端严格禁改（按需提单）**：团队成员**禁止修改任何后端 Python 代码（`src/olympus/**/*.py`）及 `src/olympus/web/openapi.json`**；如前端交互确实缺少某项只读字段或接口，统一开 GitHub Issue 描述契约诉求并 `@kyrie21z`，由项目负责人评估并统一合入。
2. **前端交互与美观优先**：在现有 Oracle Board + Ariadne 工作台（`web/src/`）内，以**出色的交互流畅度、信息层次与视觉美观性**为第一目标；保持第三方依赖克制，不引入臃肿杂乱的组件全家桶即可。
3. **聚焦真缺口，不重复造轮子**：主线已具备 ~90% 作业 1/2 可视化与 13 类执行期 HITL 交互，作业 3 聚焦补齐**执行期可观测性仪表盘、全量事件探针、事后时间线回放/因果溯源、白盒模块依赖图与跨 Run 对比**。

---

## 一、主线现状：已具备（直接复用）vs 真实缺口（聚焦攻坚）

| 作业 3 维度 | 主线已具备（直接复用，禁止重造） | 真实缺口（本次作业 3 核心增量） |
| --- | --- | --- |
| **1. 作业 1 & 2 可视化**<br>（需求对齐 / 架构设计 / 编排与验收） | • **L1 Oracle Board**：球权三列看板、多 Workspace/模式筛选、底部 5 神席位实时联动、快捷裁决抽屉（`OracleBoard.tsx`、`PantheonDock.tsx`、`QuickRulingDrawer.tsx`）<br>• **作业 1 Align**：需求分诊卡（`IntentCard`）、结构化澄清与补充说明融合面板（`ClarifyPanel`）、`GoalSpecCard` 与关口 1 对比审批卡（`ApprovalReviewCard`）<br>• **作业 2 Design / Plan / Verify**：`DesignSpecCard`（策略/契约域/不变量/模块表/接口表）、编排阶段卡（`WorkflowStages`）+ 任务依赖 DAG 图（`DagGraphHost`）+ 四级追踪矩阵（`MatrixView`）+ 实机 `Product E2E` & `Post-DAG` 验收卡片 | 1. **白盒架构交互依赖图**：`DesignStructure` 目前只有静态表格，缺「模块 × 接口契约（Provider → Consumer）× 数据实体 × `write_scope`」交互式关系图及与下游 DAG 任务的联动高亮。<br>2. **规格演化回看**：批准完成后，无法回看 `GoalSpec` / `DesignSpec` / `TaskPlan` 历次版本变化记录。 |
| **2. 执行期观测数据与交互** | • **双路实时 SSE**：`/api/runs/{run_id}/stream`（`state` + `activity`）与 `/api/board/stream`<br>• **单任务实时观测**：`LiveProgressCard`（驱动/当前工具/最近工具/流式预览）、`SubPane` 迷你终端、`provider_retry`、`escape_hatch` 拦截、`TurnReceipt`<br>• **执行期交互**：13 个幂等 + 乐观锁 HITL 命令（`pause`/`resume`/`clarify`/`approve_*`/`request_revision`/`reopen_run`/`retry_task(hint)`/`accept_risk`/`reassign_human`/`retry_workflow`/`promote_to_execute`） | 1. **Run 级可观测性仪表盘**：后端 `BudgetSummary` 与 `TurnReceipt` 已提供 `input`/`output`/`reasoning`/`cache_read`/`cache_write` Token 与 `cache_hit_rate`，但前端仅在折叠区展示了总次数和总 Token，缺少直观的 Token/缓存/耗时/工具错误分布可视化。<br>2. **全量事件流浏览器**：`RunProjection.events` 仅带最后 80 条；检查器事件列表仅单行展示，缺接入 `GET /api/runs/{run_id}/events?after_seq=0` 的多维过滤、搜索、JSON Payload 折叠展开与点击反查节点。<br>3. **真实并发时间轴**：当前泳道与瀑布视图未按事件真实时间戳展示并行任务的重叠并发，视觉与信息密度均可大幅升级。 |
| **3. 事后回溯**<br>（最高区分度与增量价值） | • **完备的只读数据底座**：`GET /api/runs/{run_id}/events?after_seq=0`（Append-Only 全量事件流）、`GET /api/runs/{run_id}/task-artifacts/{task_id}/{attempt}`（历史 attempt 的 diff/manifest/check logs/verdicts）、归档历史列表 | 1. **事件时间线回放（Time-Travel Replay）**：支持拖动进度条、播放/暂停、步进、按关键里程碑跳转，确定性重放 Run 在任意 `event_seq` 时的工作流与 DAG 状态（明确标注只读回放态）。<br>2. **失败与返工因果链 + Attempt 对比**：在单任务下支持切换对比 `Attempt 1..N`，把「失败/门禁拒绝/逃生舱拦截/Argus Findings → 人工 Hint 或重试 → 新 Attempt Diff」串成清晰因果链路。<br>3. **人工决策审计时间线 & 复盘导出**：聚合全流程 HITL 裁决记录，支持一键导出单次 Run 的结构化复盘报告（Markdown / JSON），以及多 Run 指标横向对比。 |

---

## 二、该做什么（In Scope）与互不冲突的 PR 拆分

所有成员工作均在 **`web/src/` + `tests/unit/test_web_*_browser.py`** 内完成，直接消费现有 OpenAPI 只读接口。按路径正交拆为 **3 个可并行推进的 PR**（每人认领 1 个）：

### PR-1 · 执行期可观测性仪表盘 + 真实并发时间轴 + 全量事件探针
- **核心目标**：把运行期已采集的丰富遥测数据变成**高可读、强交互、高颜值**的观测视图。
- **交付功能**：
  1. **Run 级可观测性仪表盘**：可视化 Token 结构（Input / Output / Reasoning / Cache Read & Write）与 Cache 命中率、按阶段/角色/任务的耗时与调用分布、工具调用成功率与耗时排行、重试与逃生舱拦截统计（支持悬停联动与点击钻取任务）；
  2. **真实时间轴并发甘特视图**：基于 `events` 的真实 `created_at` 时间戳绘制并行 Worker 波次、验收与 `Post-DAG` 轮次的并发重叠时间轴，支持缩放与点击高亮；
  3. **全量事件流探针（Event Explorer）**：接入 `GET /api/runs/{run_id}/events?after_seq=0`，支持按 `stage` / `actor` / `task_id` / `event_type` 筛选、关键词检索、JSON Payload 交互式展开与一键复制、点击事件反向定位工作流节点或 DAG 任务。
- **建议 `write_scope`**：
  - 新增：`web/src/components/ObservabilityView.tsx`、`web/src/components/EventExplorer.tsx`、`web/src/lib/observability.ts`、`web/src/styles/observability.css`、`tests/unit/test_web_observability_browser.py`
  - 最小接线：`web/src/App.tsx`、`web/src/components/StageHeader.tsx`、`web/src/components/InspectorDock.tsx`、`web/src/api/client.ts`、`web/src/lib/labels.ts`、`web/src/styles/tokens.css`（如需新增语义化图表色板 Token）

### PR-2 · 事后回溯引擎：时间线回放 + 失败返工因果链 + 审计导出
- **核心目标**：打造作业 3 最具亮点的事后回溯体验，做到顺滑可玩、零副作用。
- **交付功能**：
  1. **交互式时间线回放（Time-Travel Replay）**：编写纯前端确定性事件折叠器（`replayReducer.ts`），支持时间轴拖拽（Scrubbing）、播放/倍速/暂停、键盘左右步进、按里程碑胶囊快进（需求冻结 / 架构批准 / 计划批准 / 任务阻塞 / 门禁拦截 / 实机验收），联动驱动工作流阶段与任务 DAG 状态变化；**回放模式下顶部显著提示「只读回放」，强制禁用一切 HITL 写命令按钮，支持一键回到最新态（LIVE）**；
  2. **失败与返工因果链（Multi-Attempt Inspector）**：在任务检查器增加 `Attempt 1 / 2 / ...` 切换器（复用 `/task-artifacts/{task_id}/{attempt}`），清晰串联「失败报错 + 逃生舱拦截 / Argus Findings → 人工 Hint 或重试动作 → 新 Attempt Diff / 检查日志变化」；
  3. **人工决策审计线 & 一键复盘导出**：汇总全流程 HITL 裁决（澄清问答、审批 `target_hash`、风险接受、改派、修订/追加工单），支持一键导出离线复盘报告（Markdown / JSON）。
- **建议 `write_scope`**：
  - 新增：`web/src/lib/replayReducer.ts`、`web/src/components/ReplayView.tsx`、`web/src/components/AuditTimeline.tsx`、`web/src/styles/replay.css`、`tests/unit/test_web_replay_browser.py`
  - 最小接线：`web/src/App.tsx`、`web/src/components/StageHeader.tsx`、`web/src/components/InspectorDock.tsx`、`web/src/api/client.ts`、`web/src/lib/labels.ts`

### PR-3 · 白盒模块依赖图 + 跨 Run 横向对比（作业 1/2 补强 + 演示利器）
- **核心目标**：补齐架构设计白盒可视化与多 Run 对比能力，提升作业整体视觉完成度。
- **交付功能**：
  1. **DesignSpec 白盒模块依赖图**：将 `design.modules`、`design.interfaces`（Provider → Consumer）、`data_models` 归属与 `write_scope` 渲染为可拖拽/缩放/悬停高亮的模块依赖图，点击模块可联动查看关联需求与下游 `TaskNode`；
  2. **跨 Run 指标横向对比**：在 Oracle Board 支持勾选 2–4 个 Run 进入对比视图，可视化对比总耗时、LLM 调用次数、Token 与 Cache 命中率、重试与逃生舱拦截次数及验收通过率。
- **建议 `write_scope`**：
  - 新增：`web/src/components/DesignGraph.tsx`、`web/src/components/RunCompareView.tsx`、`web/src/styles/compare.css`、`tests/unit/test_web_design_compare_browser.py`
  - 最小接线：`web/src/components/Views.tsx`、`web/src/components/OracleBoard.tsx`、`web/src/styles/tokens.css`（如需新增色板 Token）

---

## 三、不该做什么（Out of Scope / 红线）

触犯以下任一红线的 PR 将直接退回：

1. **后端绝对红线（严禁触碰 `src/olympus/`）**：
   - 成员**不得修改任何 Python 后端代码**（包括 `core/`、`providers/`、`tools/`、`harness/`、`stages/`、`graph/`、`session/`、`cli/`，以及 `src/olympus/web/*.py` 与 `openapi.json`）；
   - 如确需新增后端只读字段（例如历史 attempt 的 `worker.jsonl` 工具轨迹 `tool_calls`、历史版本规格列表等），**必须先提 GitHub Issue（见文末模板）交由 `@kyrie21z` 修改合入**，前端基于更新后的 `schema.d.ts` 继续开发。
2. **严禁推翻重造或破坏现有主链路**：
   - 不得新起独立前端项目、第二套 Web 服务或破坏现有 Oracle Board / 澄清 / 三关口审批 / DAG / 检查器的既有交互与选择器（现有浏览器测试须 100% 继续通过）。
3. **前端依赖克制与工程卫生**：
   - 以交互性与美观性为导向，允许按需引入**少量轻量、维护良好**的可视化辅助库，但**严禁堆砌杂乱依赖**（如同时引入多套 UI 框架、重型 ECharts/AntD/MUI 全家桶、或无关运行时库）；
   - 严禁在 `web/src/` 下出现 `.js`/`.jsx` 文件；严禁引入外部监控基础设施（Prometheus / Grafana / Langfuse / Docker 等）；
   - 严禁捏造/伪造假数据：所有可视化与回放必须 100% 源自真实 API（`projection` / `events` / `task-artifacts`），缺失字段做优雅空态提示，绝不编造虚假耗时或指标。
4. **遵循设计系统 CI 门禁（兼顾美观与规范）**：
   - 仓库的 `tests/unit/test_design_system.py` 会自动检查样式规范：**所有颜色定义（包括新增的图表色板，如 `--chart-1..6`）必须集中写在 `web/src/styles/tokens.css` 中（同时兼顾亮色/暗色主题）**，其他 CSS/TSX 文件统一通过 `var(--token)` 引用，不写裸 `#hex`/`rgb`/`hsl`；遵守字重、字体、圆角、阴影与动效 Token 规范；
   - 如因极少数交互细节确需调整设计门禁规则，先在 Issue 中说明理由并与负责人对齐，严禁私自删除或绕过测试。

---

## 四、PR 合并硬标准（Merge Checklist）

每个 PR 提交 Review 前，须逐项满足以下全部条件：

- [ ] **1. Issue 先行与 `write_scope` 严格收敛**
  - 已创建对应 GitHub Issue（写明 Goal、Non-goals、`write_scope`、验收标准）；PR 描述含 `Closes #<issue-id>`；
  - `git diff --stat` 确认**零改动 `src/olympus/**/*.py` 及 `openapi.json`**（静态构建产物 `src/olympus/web/static/` 除外），不顺手修改无关文件。
- [ ] **2. 静态产物与类型严格同步**
  - 已运行 `pnpm --dir web typecheck && pnpm --dir web build`，确保 `src/olympus/web/static/` 与 `web/src/` 同步更新（通过 `ariadne-source-sha256` 校验，且不提交 `.map` 文件）；
  - 如新增了前端依赖，必须在 PR 描述中说明**引入理由与体积影响**，且 `web/pnpm-lock.yaml` 同步更新。
- [ ] **3. 全量验证命令全绿**
  ```bash
  uv run ruff check .
  uv run mypy src
  uv run pytest -q
  pnpm --dir web typecheck
  pnpm --dir web build
  ```
  - 特别确认 `test_architecture_invariants.py`、`test_design_system.py`、`test_web_openapi_contract.py` 及全部现有 `test_web_*_browser.py` 无回归。
- [ ] **4. 交互测试 + 真机视觉验收证据**
  - 核心前端纯逻辑（如 `replayReducer.ts`、指标统计、事件过滤）须覆盖空运行、阻塞/失败、多 Attempt 等边界用例；
  - 须补充对应的 Playwright 浏览器测试（参考 `tests/unit/test_web_run_progress_browser.py`）；
  - PR 描述必须附上**关键交互态实机截图（建议同时包含亮色 + 暗色主题）**，直观证明排版美观、无溢出、无控制台报错、回放态禁发写请求。
- [ ] **5. 独立 Review 门禁与合入规范**
  - PR 处于 `Ready for review` 状态（非 Draft），CI 全绿；
  - 通过独立 Code Review（交互体验、视觉美观度、边界条件、契约与设计门禁）并清零全部 `[Blocking]` 反馈；
  - 采用 **Squash and merge** 合入 `main`，严禁直推 `main`。

---

## 五、需要后端新增只读字段/接口时的提单模板

如在前端开发过程中发现现有接口确实缺少某项只读数据（例如：`task-artifacts` 需要返回历史 attempt 的 `tool_calls`、或需要独立历史 `GoalSpec/DesignSpec` 版本接口），请**不要自行修改 Python 代码**，直接按以下模板开 GitHub Issue 并 `@kyrie21z`：

```markdown
### [Contract Request] 为 <功能名> 补充只读字段/端点
- **关联前端 Issue**：#<id>
- **当前缺口**：现有 `<GET /api/...>` 缺少 `<具体字段>`，导致 `<交互场景>` 无法实现。
- **期望只读契约**：
  - 路径与参数：`GET /api/runs/{run_id}/...`
  - 期望返回字段（只读，不改变状态机）：
    - `field_name`: `type` — 含义与空态约定
- **数据来源确认**：该数据已存在于 SQLite `events` / 磁盘 `artifacts/...`（无需改控制面）。
```
