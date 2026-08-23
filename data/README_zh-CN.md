> 📖 View this in [English](./README.md)

# 数据说明

本目录数据可分为两层：

- `standard_*`：标准化全量数据层
- `eval_*`：评测子集 + 评测真值结构

> **公开版策略**：为公开仓库考虑，已从被跟踪的数据文件中移除原始新闻/文章全文（`text`、`description`、`summary` 等字段）。文件保留标题、元数据、结构与 RiskNet 标注。含原文全文的合作版可由 RiskNet 作者在数据使用协议下提供。

## 核心文件

### 1) `standard_cases.jsonl`
标准化后的案例级语料（JSONL，每行一条）。

- 粒度：一条 case/news
- 常见关键字段：`id`、`title`、来源元信息与标注（原文全文 `text` 按公开版策略已移除）
- 用途：作为上游标准数据来源，用于构建评测数据

### 2) `standard_incidents.jsonl`
标准化后的事件级结构。

- 粒度：一条 incident/event cluster
- 结构：`incident_id`、`ids`
- 含义：`ids` 表示属于同一事件的 case ID 列表

### 3) `eval_cases.jsonl`
事件对齐实验使用的案例级评测子集。

- 粒度：一条评测 case/news
- 含有更丰富的结构化字段（例如风险/事件标注）
- 用途：embedding、召回、pairwise 训练与评测输入

### 4) `eval_structure.json`
评测使用的事件-案例真值结构（gold）。

- 顶层结构：`{ "events": [...] }`
- 每个事件项：`incident_id`、`ids`
- 用途：将预测聚类结果与真值结构进行对比评估

### 5) `chinese_eval_cases.jsonl`
中文 AI 风险事件评测集（case 级），由中文新闻源人工标注。

- 粒度：一条评测 case/news
- 与 `eval_cases.jsonl` 采用相同的嵌套结构（`event_annotation` / `ai_risk` / `ai_tech`）
- 用途：对英文训练模型做零样本跨语言评测

### 6) `chinese_eval_structure.json`
中文评测集的事件-案例真值结构（gold）。

- 顶层结构：`{ "events": [...], "metadata": {...} }`
- 每个事件项：`incident_id`、`ids`，可选 `is_ai_risk`
- 200 事件 / 1428 篇标注 report，覆盖 130+ 种事件类型
- 用途：跨语言聚类/对齐指标评测

## 文件关系

- `standard_cases.jsonl` 与 `standard_incidents.jsonl` 是标准化全量层。
- `eval_cases.jsonl` 是评测子集的案例内容。
- `eval_structure.json` 是对应评测范围的真值分组结构。

简要理解：
- `eval_cases.jsonl` 提供样本标识、元数据与 RiskNet 标注（原文全文见合作版）；
- `eval_structure.json` 提供”这些样本应如何按事件分组”。

## 注意事项

- 文件中 ID 可能是数值类型，代码中通常会统一为字符串处理。
- 若重新生成评测数据，请保证 `eval_cases.jsonl` 与 `eval_structure.json` 同步更新。
