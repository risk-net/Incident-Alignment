> 📖 View this in [English](./README.md)

# 数据说明

本目录数据可分为两层：

- `standard_*`：标准化全量数据层
- `eval_*`：评测子集 + 评测真值结构

## 核心文件

### 1) `standard_cases.jsonl`
标准化后的案例级语料（JSONL，每行一条）。

- 粒度：一条 case/news
- 常见关键字段：`id`、`title`、`text`（以及来源元信息）
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

## 文件关系

- `standard_cases.jsonl` 与 `standard_incidents.jsonl` 是标准化全量层。
- `eval_cases.jsonl` 是评测子集的案例内容。
- `eval_structure.json` 是对应评测范围的真值分组结构。

简要理解：
- `eval_cases.jsonl` 提供“样本内容”；
- `eval_structure.json` 提供“这些样本应如何按事件分组”。

## 注意事项

- 文件中 ID 可能是数值类型，代码中通常会统一为字符串处理。
- 若重新生成评测数据，请保证 `eval_cases.jsonl` 与 `eval_structure.json` 同步更新。
