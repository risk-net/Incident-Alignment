# Incident-Alignment
[English](./README.md)

AI 风险事件对齐项目：将多篇新闻对齐到同一真实事件簇（incident cluster），并输出可评估、可落地的聚类结果。

## 项目目标

本项目提供两条主线：

- 方法主线（`src/Incident_Align_Method`）：
  双路召回（`E_text` + `E_event`）+ pairwise 判别 + 图聚类解码。
- 评估主线（`src/Incident_Align_Evaluation`）：
  对已有聚类结果统一评估，或运行基线方法进行对比。

## 目录结构

```text
Incident-Alignment/
├── config/
│   ├── Incident_Align_Method-build_embeddings-config.ini
│   ├── Incident_Align_Method-build_faiss_index-config.ini
│   ├── Incident_Align_Method-run_recall-config.ini
│   ├── Incident_Align_Method-decode_graph_from_pairwise_checkpoints-config.ini
│   └── Incident_Align_Method-full_application-config.ini
├── data/
│   ├── standard_cases.jsonl
│   ├── standard_incidents.jsonl
│   ├── eval_cases.jsonl
│   ├── eval_structure.json
│   ├── README.md
│   └── README_zh-CN.md
├── src/
│   ├── Incident_Align_Method/
│   │   ├── candidate_generation/
│   │   ├── data_preparation/
│   │   ├── pairwise_and_clustering/
│   │   └── full_application/
│   └── Incident_Align_Evaluation/
├── outputs/
└── pyproject.toml
```

## 环境要求

- Python `>= 3.11`
- 建议使用虚拟环境（`conda` 或 `venv`）

安装依赖：

```bash
cd /home/nlper/zlh/Incident-Alignment
pip install -e .
```

`pyproject.toml` 中核心依赖包括：`torch`、`sentence-transformers`、`faiss-cpu`、`numpy`、`pandas`、`scikit-learn`、`scipy` 等。

## 数据说明

方法主线默认依赖这四个文件：

- `data/standard_cases.jsonl`
- `data/standard_incidents.jsonl`
- `data/eval_cases.jsonl`
- `data/eval_structure.json`

详细字段与关系说明见：

- 英文：[data/README.md](data/README.md)
- 中文：[data/README_zh-CN.md](data/README_zh-CN.md)

## 快速开始（实验复现主线）

从仓库根目录执行：

```bash
cd /home/nlper/zlh/Incident-Alignment

python src/Incident_Align_Method/data_preparation/generate_dataset.py
python src/Incident_Align_Method/candidate_generation/build_embeddings.py
python src/Incident_Align_Method/candidate_generation/build_faiss_index.py
python src/Incident_Align_Method/candidate_generation/run_recall.py
python src/Incident_Align_Method/pairwise_and_clustering/prepare_pairwise_data.py
python src/Incident_Align_Method/pairwise_and_clustering/train_deepwide_pairwise.py
python src/Incident_Align_Method/pairwise_and_clustering/decode_graph_from_pairwise_checkpoints.py
```

默认关键产物：

- `embeddings/`
- `faiss_index/`
- `prepared_pairwise/`
- `outputs/recall.jsonl`
- `outputs/metrics/recall_metrics.json`
- `outputs/pairwise_train/`
- `outputs/graph_decode_from_all_checkpoints/`

## 统一评估与基线

### 1) 评估已有预测簇

```bash
python src/Incident_Align_Evaluation/accuracy.py \
  --pred_file outputs/graph_decode_from_all_checkpoints/best_model/model_selection.json \
  --true_file data/eval_structure.json
```

### 2) 单次向量基线

```bash
python src/Incident_Align_Evaluation/vector_baseline.py \
  --data_dir data \
  --output_dir outputs/Incident_Align_Evaluation/vector_baseline
```

### 3) Repeated text-sim 稳定性基线

```bash
python src/Incident_Align_Evaluation/baseline_textsim_threshold_repeat.py \
  --structure_file data/eval_structure.json \
  --embeddings_dir embeddings \
  --output_dir outputs/Incident_Align_Evaluation/textsim_threshold_repeat
```

说明：基线脚本内部仍兼容历史默认输出目录；建议始终显式传 `--output_dir`，保证结果统一落在 `outputs/Incident_Align_Evaluation/`。

## 全量应用流程（数据库场景）

配置文件：`config/Incident_Align_Method-full_application-config.ini`

运行前设置数据库密码环境变量：

```bash
export DB_PASSWORD='your-password'
```

执行全量流程：

```bash
python src/Incident_Align_Method/full_application/build_db_embeddings_and_faiss.py \
  --config config/Incident_Align_Method-full_application-config.ini

python src/Incident_Align_Method/full_application/run_full_dual_recall.py \
  --config config/Incident_Align_Method-full_application-config.ini

python src/Incident_Align_Method/full_application/run_full_inference.py \
  --config config/Incident_Align_Method-full_application-config.ini
```

默认输出根目录：`outputs/full_application_artifacts/`。

## 配置文件说明

- `Incident_Align_Method-build_embeddings-config.ini`
  控制输入 case 文件、编码模型路径、embedding 输出目录。
- `Incident_Align_Method-build_faiss_index-config.ini`
  控制 embedding 输入目录与 FAISS 输出目录。
- `Incident_Align_Method-run_recall-config.ini`
  控制召回参数（`TOPK_PER_ROUTE`、`FUSE_MODE` 等）与输出文件路径。
- `Incident_Align_Method-decode_graph_from_pairwise_checkpoints-config.ini`
  控制训练输出目录、图解码参数搜索网格与最终输出目录。
- `Incident_Align_Method-full_application-config.ini`
  控制数据库连接、全量 embedding/召回/推理全过程参数。

## 当前实现约定

- Pairwise 训练入口统一为：
  `src/Incident_Align_Method/pairwise_and_clustering/train_deepwide_pairwise.py`
- Pairwise 架构统一为单一 `current` 版本。
- 训练输出目录统一使用：`outputs/pairwise_train`。

## 补充文档

- 方法总览：`src/Incident_Align_Method/README.md`
- 候选生成：`src/Incident_Align_Method/candidate_generation/README.md`
- Pairwise 与聚类：`src/Incident_Align_Method/pairwise_and_clustering/README.md`
- 全量应用：`src/Incident_Align_Method/full_application/README.md`
- 评估模块：`src/Incident_Align_Evaluation/README.md`
