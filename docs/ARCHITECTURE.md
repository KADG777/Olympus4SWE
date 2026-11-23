# Olympus4SWE 架构设计（第一性原理版）

> 本文档基于 `Olympus4SWE` 独立产品愿景与 `docs/references/oiloil-curated/`（13 篇业界实战文献），以**第一性原理**剔除冗余仪式后确立 `Olympus4SWE` 的技术架构。  
> 先前 `SWE_Homework/03 Agents Meeting` 与 `04 Agent Orchestrate` 仅作为问题域与边界参考，**不继承其机械多轮仪式、人为打分公式或 LangGraph 重脚手架**。

---

## 1. 第一性原理：不可简化的 4 个根本问题

Agent-Native 软件工程系统本质上只为解决 4 个真实痛点：

1. **意图有歧义 → 做错方向**：模型未经取证与澄清即按隐含猜测写代码。
2. **复杂任务长程失焦 & 并发互相踩踏**：上下文衰减（Context Rot）+ 多路写入无隔离造成隐性决策冲突。
3. **自写自审确认偏误**：同一段上下文中的模型天然偏向自证清白、用桩代码或表面通过敷衍验收。
4. **过程黑盒、失控难干预**：无法追踪中间决策、无法在关键节点暂停/拍板/恢复、无法回放调试。

---

## 2. 核心架构原则（来自 13 篇实战文献的承重准则）

1. **简单可组合原语，拒绝重型黑盒框架**（`13`, `06`, `11`）：核心使用**原生 Python 透明状态机 + SQLite 持久化**，Prompt 与工具调用完全透明可控，不引入重型第三方编排框架。
2. **按需引入复杂度（不够再加）**（`13`, `04`, `07`）：小任务单写入者直接推进；仅当多文件并发或跨视角防盲区有真实收益时，才拆分 Task DAG 或启用并行只读批评。
3. **单写入者 / 不相交写域隔离 + 只读卸载**（`12`, `09`, `01`）：代码写入要么单线程连续推进，要么通过 `git worktree` + 严格不相交的 `write_scope` 锁隔离；大上下文调研与独立审查交给冷启动只读子 Agent。
4. **冷启动独立验收与收敛护栏**（`01`, `04`, `03`, `08`）：程序检查（测试/Lint/类型）+ 零历史独立审查（输入 = 冻结 Spec + Diff + 实测输出）；强制 `Important`/`Nit` 分级与 `max_cycles=3` 收敛上限，次轮只审增量 Diff，防死循环与过度工程化。
5. **确定性控制面 vs. 概率模型面**（`08`, `05`, `01`）：状态迁移、DAG 拓扑、`write_scope` 锁、单调预算账本、Commit/Hash 绑定的人类审批全部由确定性代码掌控；LLM 仅输出提案、代码、测试与审查意见。
6. **文件系统优先 + 防呆 ACI + KV 缓存纪律**（`07`, `10`, `06`, `11`）：强制绝对路径、结构化精炼返回值与可操作报错、固定前缀 + Append-Only 上下文与失败证据保留。
7. **单一事件源驱动 CLI 与未来 Web 投影**（`05`, `06`）：所有状态迁移、工具调用、Diff、实测证据与人工裁决全部写入只追加 `SQLite` 事件表；**先跑通 CLI 闭环**，后续 Web 控制台直接只读投影同一份事件流。

---

## 3. 两阶段统一内核架构

以极简统一内核串起「意图对齐（Stage 1）」与「编排执行及验证（Stage 2）」：

### Stage 1: `Align`（意图对齐与契约冻结）
- **事实取证**：先只读探测仓库与既有约束，识别真正阻塞实现的歧义、硬约束冲突与权衡选项。
- **可选并行只读挑刺**：复杂需求可并发拉起 `Product` / `Architecture` / `QA` 三个冷启动只读视角审同一份草案，**只暴露遗漏、风险与反对意见（保留少数意见给人类，不做机械加权打分或强行凑两轮讨论）**。
- **一次性澄清与冻结**：合并去重问题给人类集中澄清一次 → 生成带 `out_of_scope` 与可机读验收项的 `GoalSpec`，人类确认后计算规范化 **SHA-256 指纹**并绑定 `ApprovalRecord`。

### Stage 2: `Orchestrate & Verify`（自适应分解、隔离执行与独立验收）
- **自适应规划**：单步可完成的任务直接生成单节点 `TaskPlan`；复杂任务拆分为 **Task DAG**（边表示“前驱产物已 `accepted`”），由确定性校验器强制拦截图环、缺失依赖与并发节点的 `write_scope` 重叠。
- **确定性调度循环**：
  - `waiting → ready → running → verifying → accepted`（及 `failed` / `blocked` / `cancelled` / `stale`）；
  - 取得唯一租约代际（Lease）与 `write_scope` 锁后在 `git worktree` 中运行透明 Coding Loop；
  - 产出经 **0-Token 程序门禁（pytest/ruff/script）+ 冷启动独立审查**双重通过后方能 `accepted`；
  - `Important` 问题带完整上下文打回修复（`max_cycles=3`），超限转 `blocked` 交由人类裁决；人类可随时 `pause` / `resume` / `retry` / `accept_risk`（标明未验证人工接受，不伪造通过）。

---

## 4. 代码库模块化结构

```text
Olympus4SWE/
├── AGENTS.md                  # <=100 行仓库入口导航
├── GLOSSARY.md                # 核心领域术语表
├── pyproject.toml             # uv + Python 3.13
├── docs/
│   ├── ARCHITECTURE.md        # 本文档
│   ├── adr/                   # 架构决策记录（ADR）
│   └── references/            # 本地参考资料（已 .gitignore）
├── src/olympus/
│   ├── core/                  # 强类型契约、SHA-256 指纹、DAG/Scope 校验、SQLite 事件与状态存储
│   ├── providers/             # 轻量统一 LLM 接口（OpenAI-Compatible / Anthropic / Mock）
│   ├── tools/                 # 防呆 ACI 原语（read/edit/glob/grep/bash）与截断/报错引导
│   ├── harness/               # 透明 Append-Only Loop、KV 前缀稳定、git worktree 沙盒、Trace 落盘
│   ├── stages/
│   │   ├── align.py           # Stage 1: 意图取证、可选并行只读挑刺、澄清与 Spec 冻结
│   │   └── orchestrate.py     # Stage 2: 自适应规划、确定性 DAG 调度、冷启动验收与收敛循环
│   ├── skills/                # 渐进式披露 Skill 包
│   ├── web/                   # FastAPI 后端、OpenAPI 契约、Board/Pantheon 投影与 SSE 流
│   └── cli/                   # CLI 入口
├── web/                       # 双层 Oracle Board + Ariadne React/TS 前端工作台源码
└── tests/
    ├── unit/                  # 内核不变量、DAG 冲突、幂等与指纹失效、设计系统与浏览器 E2E 测试
    └── evals/                 # 真实 SWE 端到端评测任务
```

---

## 5. 分期实施路线（CLI 闭环优先，不够再加）

1. **M1 — 确定性内核与持久化底座**（当前首步）：`pyproject.toml`、`AGENTS.md`、`core/fingerprint.py`、`core/contracts.py`、`core/dag.py`、`core/store.py`（SQLite WAL + 事件流 + 单调预算 + 乐观锁/幂等命令 + Hash 绑定审批）及完整单测。
2. **M2 — 透明 Harness、防呆 ACI 与 Provider 层**：统一 LLM 客户端（含离线 Mock/Replay）、绝对路径 ACI 工具集、Append-Only Loop、`git worktree` 隔离沙盒。
3. **M3 — Stage 1 (`Align`) + Stage 2 (`Orchestrate & Verify`) + CLI 闭环**：跑通 `olympus align` → `olympus run` → `olympus status/events/approve` 完整端到端流程。
4. **M4 — 真实 Eval 基线与双层 `Oracle Board + Ariadne` Web 工作台**：基于既有 SQLite 事件流构建 FastAPI + SSE + React/TypeScript 双层工作台——L1 **Oracle Board**（Workspace 注册与切换、`Scope × Task Type` 下发、三列 Ball-in-Court 裁决队列、Quick Ruling Drawer 与底部 5-God **Pantheon Dock**）与 L2 **Ariadne Cockpit**（多标签深潜拓扑、契约审阅、矩阵与瀑布视图）。
