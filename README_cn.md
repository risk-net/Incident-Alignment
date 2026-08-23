# Incident-Alignment
[English](./README.md)

AI 风险事件对齐项目：将多篇新闻对齐到同一真实事件簇（incident cluster），并输出可评估、可落地的聚类结果。

## 项目目标

本项目提供两条主线：

- 方法主线（`src/Incident_Align_Method`）：
  双路召回（`E_text` + `E_event`）+ pairwise 判别 + 图聚类解码。
- 评估主线（`src/Incident_Align_Evaluation`）：
  对已有聚类结果统一评估，或运行基线方法进行对比。

## 项目前提

- 注意，如果使用这个对齐项目，需要确保本地有AI风险新闻的数据库，需要先运行本组织的Risk-Identification项目，识别得到AI风险新闻后，再进行这一部分。

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
│   ├── chinese_eval_cases.jsonl      # 中文评测集 case
│   ├── chinese_eval_structure.json   # 中文评测集 gold 事件
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

- Python `>= 3.9, < 3.13`
- 建议使用 GPU（PyTorch CUDA）进行 embedding 生成与模型训练；CPU 可用但较慢
- 全量应用流程需要 PostgreSQL 数据库（实验复现可省略）

### 1) 创建虚拟环境

使用 `conda`：

```bash
conda create -n incident-alignment python=3.9 -y
conda activate incident-alignment
```

或使用 `venv`：

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2) 安装依赖

```bash
cd <仓库根目录>
pip install -e .
```

`pyproject.toml` 中核心依赖包括：`torch`、`sentence-transformers`、`faiss-cpu`、`numpy`、`pandas`、`scikit-learn`、`scipy`、`psycopg2-binary`、`tqdm` 等。

### 3) 下载 embedding 模型

`build_embeddings.py` 默认使用本地模型目录 `models/bge-m3`。运行前需下载 BGE-M3 并放到该目录，或修改配置中的 `MODEL_PATH` 为 HuggingFace 模型名（`BAAI/bge-m3`）：

```bash
# 方式 A：下载到本地
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3').save('models/bge-m3')"

# 方式 B：直接使用 HuggingFace（编辑 config/Incident_Align_Method-build_embeddings-config.ini）
#   MODEL_PATH = BAAI/bge-m3
```

### 4)（可选）全量应用流程的数据库

全量应用流程从 PostgreSQL 读取数据。在 `config/Incident_Align_Method-full_application-config.ini` 中配置连接，并导出密码：

```bash
export DB_PASSWORD='<你的密码>'
```

## 数据说明

方法主线默认依赖这四个文件：

- `data/standard_cases.jsonl`
- `data/standard_incidents.jsonl`
- `data/eval_cases.jsonl`
- `data/eval_structure.json`

评测相关文件：

- `data/chinese_eval_cases.jsonl` — 中文 AI 风险事件评测集（case 级）
- `data/chinese_eval_structure.json` — 中文评测集 gold 事件簇

> **注意**：按公开版策略，已从被跟踪的数据文件中移除原文全文。文件保留标题、元数据、结构与 RiskNet 标注。含原文全文的合作版可由 RiskNet 作者在数据使用协议下提供。

详细字段与关系说明见：

- 英文：[data/README.md](data/README.md)
- 中文：[data/README_zh-CN.md](data/README_zh-CN.md)

### 数据获取

英文评测文件（`standard_cases.jsonl`、`eval_cases.jsonl`）源自公开的 AI 事件数据库（AIID / AIAAIC）。如需复现或重新生成，请参考本组织的 **Risk-Identification** 上游流程（AI 风险新闻识别），该流程产出本项目用于对齐的事件新闻库。

> 若原始数据文件体积过大不便入库，可通过运行上游识别流程获取，或联系作者获取整理后的数据集。

## 快速开始（实验复现主线）

从仓库根目录（`<仓库根目录>`）执行：

```bash
cd <仓库根目录>

python src/Incident_Align_Method/data_preparation/generate_dataset.py
python src/Incident_Align_Method/candidate_generation/build_embeddings.py
python src/Incident_Align_Method/candidate_generation/build_faiss_index.py
python src/Incident_Align_Method/candidate_generation/run_recall.py
python src/Incident_Align_Method/pairwise_and_clustering/prepare_pairwise_data.py
python src/Incident_Align_Method/pairwise_and_clustering/train_deepwide_pairwise.py
python src/Incident_Align_Method/pairwise_and_clustering/decode_graph_from_pairwise_checkpoints.py
```

### 运行前必改的配置参数

运行前请在 `config/` 中配置以下参数：

| 文件 | 参数 | 需要修改的内容 |
|------|------|----------------|
| `Incident_Align_Method-build_embeddings-config.ini` | `MODEL_PATH` | 默认 `models/bge-m3`。若未下载本地模型，改为 HuggingFace 名 `BAAI/bge-m3`，或先下载到该目录 |
| `Incident_Align_Method-build_embeddings-config.ini` | `CASES_FILE`, `OUTPUT_DIR` | 输入 case 文件与 embedding 输出目录（按需） |
| `Incident_Align_Method-build_faiss_index-config.ini` | `EMBEDDINGS_DIR`, `OUTPUT_DIR` | 需与上一步 embedding 输出目录一致 |
| `Incident_Align_Method-run_recall-config.ini` | `STRUCTURE_FILE`, `FAISS_DIR`, `EMBEDDINGS_DIR`, `OUTPUT_FILE` | 数据文件与上两步产物路径 |
| `Incident_Align_Method-full_application-config.ini` | `checkpoint_path` | **必须**替换为你训练好的 pairwise 模型路径（由 `train_deepwide_pairwise.py` 产生） |
| `Incident_Align_Method-full_application-config.ini` | `[Database]` | 设置 `password_env` 指向的环境变量，运行前 `export DB_PASSWORD='<密码>'` |

> 中文评测集使用独立的 `-chinese-config.ini` 变体，路径与英文版同理。

默认关键产物：

- `outputs/embeddings/`
- `outputs/faiss_index/`
- `outputs/prepared_pairwise/`
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

### 2) 逻辑回归基线（5 折 CV）

将 Deep+Wide 分类器替换为逻辑回归（对 17 维手工特征做线性分类），验证深度交互特征建模的必要性。

先准备 5 折划分：
```bash
python src/Incident_Align_Method/data_preparation/prepare_5fold_cv_splits.py
```

运行基线：
```bash
python src/Incident_Align_Evaluation/baseline_logistic_regression_cv.py
```

输出：`outputs/Incident_Align_Evaluation/baseline_logistic_regression_cv/`。

## 中文评测集（跨语言测试）

人工标注的中文 AI 风险事件评测集（`data/chinese_eval_cases.jsonl` + `data/chinese_eval_structure.json`），用于英文训练模型的零样本跨语言评估。

### 1) 构建中文 embedding 与召回

```bash
python src/Incident_Align_Method/candidate_generation/build_embeddings.py \
  --config config/Incident_Align_Method-build_embeddings-chinese-config.ini

python src/Incident_Align_Method/candidate_generation/build_faiss_index.py \
  --config config/Incident_Align_Method-build_faiss_index-chinese-config.ini

python src/Incident_Align_Method/candidate_generation/run_recall.py \
  --config config/Incident_Align_Method-run_recall-chinese-config.ini
```

### 2) 推理 + 聚类 + 评估

```bash
python src/Incident_Align_Evaluation/run_chinese_eval_inference.py \
  --checkpoint outputs/your_pairwise_checkpoint.pt
```

> **`--checkpoint` 为必填参数**：指向训练好的 Deep+Wide pairwise 模型 checkpoint（由 `train_deepwide_pairwise.py` 训练产生）。详见 `run_chinese_eval_inference.py` 文件头部说明。

输出：`outputs/chinese_eval_alignment/`。

## 全量应用流程（数据库场景）

配置文件：`config/Incident_Align_Method-full_application-config.ini`

运行前设置数据库密码环境变量：

```bash
export DB_PASSWORD='your-password'
```

执行全量流程（embedding → 双路召回 → 推理 → 聚类）：

```bash
python src/Incident_Align_Method/full_application/run_full_pipeline.py \
  --config config/Incident_Align_Method-full_application-config.ini
```

或分步执行：

```bash
python src/Incident_Align_Method/full_application/build_db_embeddings_and_faiss.py \
  --config config/Incident_Align_Method-full_application-config.ini

python src/Incident_Align_Method/full_application/run_full_dual_recall.py \
  --config config/Incident_Align_Method-full_application-config.ini

python src/Incident_Align_Method/full_application/run_full_inference.py \
  --config config/Incident_Align_Method-full_application-config.ini

python src/Incident_Align_Method/full_application/run_full_clustering.py \
  --config config/Incident_Align_Method-full_application-config.ini
```

默认输出根目录：`outputs/full_application_artifacts/`。

## 配置文件说明

- `Incident_Align_Method-build_embeddings-config.ini`
  控制输入 case 文件、编码模型路径、embedding 输出目录。中文评测变体：`Incident_Align_Method-build_embeddings-chinese-config.ini`。
- `Incident_Align_Method-build_faiss_index-config.ini`
  控制 embedding 输入目录与 FAISS 输出目录。中文评测变体：`Incident_Align_Method-build_faiss_index-chinese-config.ini`。
- `Incident_Align_Method-run_recall-config.ini`
  控制召回参数（`TOPK_PER_ROUTE`、`FUSE_MODE` 等）与输出文件路径。中文评测变体：`Incident_Align_Method-run_recall-chinese-config.ini`。
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
