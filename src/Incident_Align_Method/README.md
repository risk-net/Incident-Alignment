# Incident_Align_Method

`Incident_Align_Method` 是论文中事件对齐方法的核心实现目录。这里关注的问题不是普通新闻相似度检索，而是把多篇报道同一 AI 风险事件的新闻对齐到同一个事件簇中。

这个目录的定位是：

- 公开方法流程与核心实现
- 支持论文实验链路复现
- 提供从评测数据构造到全量应用的完整代码参考

它不是一个通用安装包，也不承诺开箱即用的产品化体验。公开版更强调“方法可理解、流程可跟踪、实验可复现”。

## 目录作用

本目录覆盖以下主链路：

1. 构造事件对齐评测数据
2. 生成双视角 embeddings
3. 建立 FAISS 索引
4. 执行双路召回
5. 构造 pairwise 训练/验证/测试数据
6. 训练 pairwise 分类器
7. 搜索图参数并解码为事件簇
8. 将方法迁移到全量数据库场景

可以把整个方法理解为两阶段流程：

- 第一阶段：双路召回
  - `E_text` 表示新闻主题语义视角
  - `E_event` 表示事件要素与事件过程视角
- 第二阶段：pairwise + 图聚类
  - 先判断候选新闻对是否属于同一事件
  - 再把局部 pair 关系解码成全局事件簇

## 适合谁看

这个目录主要面向三类读者：

- 想理解论文方法细节的读者
- 想复现实验主链路的研究者
- 想把方法迁移到自有数据库或新闻库上的开发者

如果你只想做 baseline 对比，或者只想对已有聚类结果重新打分，请使用 `src/Incident_Align_Evaluation`，而不是本目录。

## 目录结构

### `data_preparation/`

负责把仓库原始数据整理成事件对齐方法可直接消费的评测数据。

- `generate_dataset.py`
  - 从 `standard_cases.jsonl`、`standard_incidents.jsonl` 以及 AIID/AIAAIC 标注生成评测数据
  - 主要输出 `data/eval_cases.jsonl` 和 `data/eval_structure.json`
- `convert_json_to_jsonl.py`
  - 进行格式转换，便于流式处理

### `candidate_generation/`

负责 embedding 构建、索引建立和双路召回。

- `download_model.py`
  - 下载本方法依赖的编码模型
- `build_embeddings.py`
  - 生成 `E_text`、`E_event` 和要素级 embeddings
- `build_faiss_index.py`
  - 基于 embeddings 构建 FAISS 索引
- `run_recall.py`
  - 执行评测集上的双路召回与融合排序

典型产物：

- `embeddings/`
- `faiss_index/`
- `outputs/recall.jsonl`
- `outputs/metrics/recall_metrics.json`

### `pairwise_and_clustering/`

负责 pairwise 数据构造、模型训练和图聚类解码。

- `pairwise_data_io.py`
  - 数据读写、字段定义、split 与 pair 构造工具
- `prepare_pairwise_data.py`
  - 基于召回结果生成多次 repeat 的 train/dev/test pairwise 数据
- `pairwise_model.py`
  - Deep+Wide 模型与特征构造
- `train_deepwide_pairwise.py`
  - 唯一训练入口（current 架构）
- `graph_clustering.py`
  - 图聚类与 cluster-level 指标计算
- `decode_graph_from_pairwise_checkpoints.py`
  - 搜索 checkpoint 与图配置，并输出最终 test 聚类结果

典型产物：

- `prepared_pairwise/`
- `outputs/.../repeat_*/epoch_*.pt`
- `outputs/.../best_model/`

### `full_application/`

负责把实验链路迁移到数据库与真实应用场景。

- `build_db_embeddings_and_faiss.py`
  - 从 PostgreSQL `ai_risk_events_news` 表构建全量 embeddings 与 FAISS
- `run_full_dual_recall.py`
  - 在全量库上执行双路召回
- `run_full_inference.py`
  - 执行统一的 pairwise 推理与图聚类
- `build_faiss_index_full.py`
  - 针对已有全量 embeddings 重建索引的辅助脚本
- `README.md`
  - 全量应用链路的单独说明

## 方法设计概览

### 1. 双视角表示

`candidate_generation/build_embeddings.py` 中定义了两类核心视图：

- `E_text`
  - 由 `title + body/text` 组成
  - 更偏向主题语义和全文叙事
- `E_event`
  - 由时间、主体、事件类型、AI 技术、AI 风险、原因、过程、结果等字段拼接
  - 更偏向事件结构与要素对齐

脚本默认使用 `BGE-M3` 编码两种视图，并可进一步生成要素级 embeddings，例如：

- `actor_main`
- `ai_system`
- `domain`
- `event_type`
- `event_cause`
- `event_process`
- `event_result`
- `ai_risk_description`

### 2. 双路召回

`candidate_generation/run_recall.py` 会分别在 `E_text` 和 `E_event` 的 FAISS 索引上检索 topK 候选，并做融合排序。

当前支持的融合方式包括：

- `max`
- `mean`
- `wavg`
- `maxmin`

召回阶段的目标是尽量保留高召回候选，而不是直接完成最终同事件判断。

### 3. Pairwise 分类

`pairwise_and_clustering/train_deepwide_pairwise.py` 训练的是 Deep+Wide 风格的二分类模型，用来判断两个候选新闻是否属于同一事件。

模型输入由两部分构成：

- Deep 部分
  - 读取多个文本/事件字段的 embedding
  - 构造 `q`、`c`、`|q-c|`、`q*c` 等交互特征
- Wide / Tabular 部分
  - 类别字段本身
  - 类别字段是否相等
  - 召回分数归一化特征
  - 字段相似度特征
  - 缺失标记特征

### 4. 图聚类解码

`pairwise_and_clustering/graph_clustering.py` 和 `decode_graph_from_pairwise_checkpoints.py` 负责把 pairwise 预测转成事件簇。

当前实现支持：

- 边规则
  - `either`
  - `mutual`
- 合并策略
  - `closure`
  - `k_support`
  - `complete_link`

`decode_graph_from_pairwise_checkpoints.py` 会在 dev split 上搜索最优阈值和图参数，在 test split 上输出最终 pair 指标和 cluster 指标，并额外生成 `best_model/` 目录供统一评测或固定发布使用。

## 两条主要使用主线

### 1. 实验复现主线

从仓库根目录执行，典型顺序如下：

```bash
python src/Incident_Align_Method/data_preparation/generate_dataset.py
python src/Incident_Align_Method/candidate_generation/build_embeddings.py
python src/Incident_Align_Method/candidate_generation/build_faiss_index.py
python src/Incident_Align_Method/candidate_generation/run_recall.py
python src/Incident_Align_Method/pairwise_and_clustering/prepare_pairwise_data.py
python src/Incident_Align_Method/pairwise_and_clustering/train_deepwide_pairwise.py
python src/Incident_Align_Method/pairwise_and_clustering/decode_graph_from_pairwise_checkpoints.py
```

这条链路主要服务于：

- 方法验证
- 多次 repeat 实验
- checkpoint 选择
- dev/test 指标输出
- `best_model/` 结果固化

### 2. 全量应用主线

当需要对数据库中的全部事件级新闻执行事件对齐时，主要使用：

```bash
export DB_PASSWORD='your-password'

python src/Incident_Align_Method/full_application/build_db_embeddings_and_faiss.py \
  --config config/Incident_Align_Method-full_application-config.ini

python src/Incident_Align_Method/full_application/run_full_dual_recall.py \
  --config config/Incident_Align_Method-full_application-config.ini

python src/Incident_Align_Method/full_application/run_full_inference.py \
  --config config/Incident_Align_Method-full_application-config.ini
```

这条链路主要服务于：

- 全库候选生成
- 全库 pairwise 判别
- 全库图聚类
- 事件簇落地与分析

## 复现前提

公开版默认假设你从仓库根目录运行脚本。大多数脚本会围绕这些工作目录组织：

- `data/`
- `embeddings/`
- `faiss_index/`
- `prepared_pairwise/`
- `outputs/`

实验主线常见输入包括：

- `data/standard_cases.jsonl`
- `data/standard_incidents.jsonl`
- `data/AIID-AIAAIC/...`

实验主线常见中间产物包括：

- `data/eval_cases.jsonl`
- `data/eval_structure.json`
- `embeddings/case_ids.txt`
- `embeddings/emb_text.npy`
- `embeddings/emb_event.npy`
- `faiss_index/*.index`
- `outputs/recall.jsonl`
- `prepared_pairwise/manifest.json`

最终实验产物通常包括：

- pairwise checkpoints
- recall 指标
- pair 指标
- cluster 指标
- test split 上的聚类结果

## 配置文件

本目录当前使用的公开版配置主要包括：

- `config/Incident_Align_Method-build_embeddings-config.ini`
- `config/Incident_Align_Method-build_faiss_index-config.ini`
- `config/Incident_Align_Method-run_recall-config.ini`
- `config/Incident_Align_Method-decode_graph_from_pairwise_checkpoints-config.ini`
- `config/Incident_Align_Method-full_application-config.ini`

其中：

- `build_embeddings` 控制数据文件、模型路径与 embeddings 输出目录
- `build_faiss_index` 控制索引输入输出目录
- `run_recall` 控制召回 topK、融合方式与输出路径
- `decode_graph_from_pairwise_checkpoints` 控制 checkpoint 搜索和图参数搜索
- `full_application` 控制数据库连接、全量 embeddings、全量召回与推理输出

公开发布时建议：

- 所有配置只保留相对路径或可替换示例路径
- 不在配置文件中保留作者本地绝对路径
- 不把私有数据库密码、私有 checkpoint 或私有目录写入仓库

## 与评估目录的关系

如果你想按统一公开口径重新评测最终最佳模型结果，请使用 `src/Incident_Align_Evaluation/accuracy.py`，而不是直接复用方法目录中的历史评估文件。

例如：

```bash
python src/Incident_Align_Evaluation/accuracy.py \
  --pred_file outputs/.../best_model/model_selection.json \
  --true_file data/eval_structure.json
```

或：

```bash
python src/Incident_Align_Evaluation/accuracy.py \
  --pred_file outputs/.../best_model/clusters_test.json \
  --true_file data/eval_structure.json
```
