# Incident-Alignment
[中文文档](./README_cn.md)
AI Risk Incident Alignment Project: Align multiple news articles into the same real-world incident cluster, and output evaluable, production-ready clustering results.

## Project Objectives
This project follows two main pipelines:

- Method Pipeline (`src/Incident_Align_Method`):
  Dual-path retrieval (`E_text` + `E_event`) + pairwise discrimination + graph clustering decoding.
- Evaluation Pipeline (`src/Incident_Align_Evaluation`):
  Unified evaluation of existing clustering results, or comparison against baseline methods.

## Project Prerequisites

- Note that to use this alignment project, you must have a local database of AI risk news.
You must first run the **Risk-Identification** project from this organization to identify AI risk news before proceeding with this part.

## Directory Structure
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

## Environment Requirements

- Python `>= 3.9, < 3.13`
- GPU recommended (PyTorch CUDA) for embedding generation and model training; CPU works but is slower
- PostgreSQL database for the full-application pipeline (optional for experiment reproduction)

### 1) Create a virtual environment

With `conda`:
```bash
conda create -n incident-alignment python=3.9 -y
conda activate incident-alignment
```

Or with `venv`:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
cd <repo-root>   # 仓库根目录
pip install -e .
```

Core dependencies (in `pyproject.toml`): `torch`, `sentence-transformers`, `faiss-cpu`, `numpy`, `pandas`, `scikit-learn`, `scipy`, `psycopg2-binary`, `tqdm`, etc.

### 3) Download the embedding model

`build_embeddings.py` defaults to a local model at `models/bge-m3`. Either download BGE-M3 to that directory, or point `MODEL_PATH` to a HuggingFace model name (`BAAI/bge-m3`):

```bash
# Option A: local download
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3').save('models/bge-m3')"

# Option B: use HuggingFace directly (edit config/Incident_Align_Method-build_embeddings-config.ini)
#   MODEL_PATH = BAAI/bge-m3
```

### 4) (Optional) Database for full-application pipeline

The full-application pipeline reads from PostgreSQL. Set the connection in `config/Incident_Align_Method-full_application-config.ini` and export the password:

```bash
export DB_PASSWORD='<your-password>'
```

## Data Description
The method pipeline relies on four default files:
- `data/standard_cases.jsonl`
- `data/standard_incidents.jsonl`
- `data/eval_cases.jsonl`
- `data/eval_structure.json`

Additional evaluation files:
- `data/chinese_eval_cases.jsonl` — Chinese AI risk event benchmark (case-level)
- `data/chinese_eval_structure.json` — Chinese benchmark gold event clusters

> **Note**: Per the public-release policy, original article full-text has been
> removed from the tracked data files. Files retain titles, metadata,
> structures, and RiskNet annotations. A restricted version with the original
> full-text is available from the RiskNet authors under a data-use agreement.

For detailed field and relationship descriptions:
- English: [data/README.md](data/README.md)
- 中文：[data/README_zh-CN.md](data/README_zh-CN.md)

### Data acquisition

The English eval files (`standard_cases.jsonl`, `eval_cases.jsonl`) are derived from public AI incident databases (AIID / AIAAIC). If you need to reproduce or re-generate them, refer to the upstream **Risk-Identification** pipeline (AI risk news identification) in this organization, which produces the incident-aligned news database that this project aligns.

> If the raw data files are too large to ship in the repository, you may obtain them by running the upstream identification pipeline, or contact the authors for the assembled dataset.

## Quick Start (Experiment Reproduction Pipeline)
Run from the repository root (`<repo-root>`):
```bash
cd <repo-root>

python src/Incident_Align_Method/data_preparation/generate_dataset.py
python src/Incident_Align_Method/candidate_generation/build_embeddings.py
python src/Incident_Align_Method/candidate_generation/build_faiss_index.py
python src/Incident_Align_Method/candidate_generation/run_recall.py
python src/Incident_Align_Method/pairwise_and_clustering/prepare_pairwise_data.py
python src/Incident_Align_Method/pairwise_and_clustering/train_deepwide_pairwise.py
python src/Incident_Align_Method/pairwise_and_clustering/decode_graph_from_pairwise_checkpoints.py
```

### Before running — required config changes

Before running the pipeline, configure these parameters (all in `config/`):

| File | Parameter | What to change |
|------|-----------|----------------|
| `Incident_Align_Method-build_embeddings-config.ini` | `MODEL_PATH` | 默认 `models/bge-m3`。若未下载本地模型，改为 HuggingFace 名 `BAAI/bge-m3`，或先下载到该目录 |
| `Incident_Align_Method-build_embeddings-config.ini` | `CASES_FILE`, `OUTPUT_DIR` | 输入 case 文件与 embedding 输出目录（按需） |
| `Incident_Align_Method-build_faiss_index-config.ini` | `EMBEDDINGS_DIR`, `OUTPUT_DIR` | 需与上一步 embedding 输出目录一致 |
| `Incident_Align_Method-run_recall-config.ini` | `STRUCTURE_FILE`, `FAISS_DIR`, `EMBEDDINGS_DIR`, `OUTPUT_FILE` | 数据文件与上两步产物路径 |
| `Incident_Align_Method-full_application-config.ini` | `checkpoint_path` | **必须**替换为你训练好的 pairwise 模型路径（由 `train_deepwide_pairwise.py` 产生） |
| `Incident_Align_Method-full_application-config.ini` | `[Database]` | 设置 `password_env` 指向的环境变量，运行前 `export DB_PASSWORD='<密码>'` |

> 中文评测集使用独立的 `-chinese-config.ini` 变体，路径与英文版同理。

Default key outputs:
- `outputs/embeddings/`
- `outputs/faiss_index/`
- `outputs/prepared_pairwise/`
- `outputs/recall.jsonl`
- `outputs/metrics/recall_metrics.json`
- `outputs/pairwise_train/`
- `outputs/graph_decode_from_all_checkpoints/`

## Unified Evaluation & Baselines
### 1) Evaluate Predicted Clusters
```bash
python src/Incident_Align_Evaluation/accuracy.py \
  --pred_file outputs/graph_decode_from_all_checkpoints/best_model/model_selection.json \
  --true_file data/eval_structure.json
```

### 2) Logistic Regression Baseline (5-Fold CV)
Replaces the Deep+Wide classifier with logistic regression (linear layer on the 17 hand-crafted features) to validate the necessity of deep interaction modeling.

```bash
python src/Incident_Align_Evaluation/baseline_logistic_regression_cv.py
```

Requires 5-fold CV splits first:
```bash
python src/Incident_Align_Method/data_preparation/prepare_5fold_cv_splits.py
```

Output: `outputs/Incident_Align_Evaluation/baseline_logistic_regression_cv/`.

## Chinese Evaluation Set (Cross-Lingual Test)
Annotated Chinese AI risk event benchmark (`data/chinese_eval_cases.jsonl` + `data/chinese_eval_structure.json`), used for zero-shot cross-lingual evaluation of the English-trained model.

### 1) Build Chinese Embeddings & Recall
```bash
python src/Incident_Align_Method/candidate_generation/build_embeddings.py \
  --config config/Incident_Align_Method-build_embeddings-chinese-config.ini

python src/Incident_Align_Method/candidate_generation/build_faiss_index.py \
  --config config/Incident_Align_Method-build_faiss_index-chinese-config.ini

python src/Incident_Align_Method/candidate_generation/run_recall.py \
  --config config/Incident_Align_Method-run_recall-chinese-config.ini
```

### 2) Inference + Clustering + Evaluation
```bash
python src/Incident_Align_Evaluation/run_chinese_eval_inference.py \
  --checkpoint outputs/your_pairwise_checkpoint.pt
```

> **`--checkpoint` is required**: point it to the trained Deep+Wide pairwise checkpoint (produced by `train_deepwide_pairwise.py`). See `run_chinese_eval_inference.py` header for details.

Output: `outputs/chinese_eval_alignment/`.

## Full Application Pipeline (Database Scenario)
Configuration file: `config/Incident_Align_Method-full_application-config.ini`

Set the database password environment variable before running:
```bash
export DB_PASSWORD='your-password'
```

Execute the full pipeline (embedding → dual recall → inference → clustering):
```bash
python src/Incident_Align_Method/full_application/run_full_pipeline.py \
  --config config/Incident_Align_Method-full_application-config.ini
```

Or run each step individually:
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

Default output root directory: `outputs/full_application_artifacts/`.

## Configuration File Reference
- `Incident_Align_Method-build_embeddings-config.ini`
  Controls input case files, embedding model paths, and embedding output directories. Chinese variant: `Incident_Align_Method-build_embeddings-chinese-config.ini`.
- `Incident_Align_Method-build_faiss_index-config.ini`
  Controls embedding input directories and FAISS index output directories. Chinese variant: `Incident_Align_Method-build_faiss_index-chinese-config.ini`.
- `Incident_Align_Method-run_recall-config.ini`
  Controls retrieval parameters (`TOPK_PER_ROUTE`, `FUSE_MODE`, etc.) and output file paths. Chinese variant: `Incident_Align_Method-run_recall-chinese-config.ini`.
- `Incident_Align_Method-decode_graph_from_pairwise_checkpoints-config.ini`
  Controls training output directories, graph decoding parameter search grids, and final output directories.
- `Incident_Align_Method-full_application-config.ini`
  Controls database connections and full pipeline parameters for embedding, retrieval, and inference.

## Implementation Conventions
- Unified pairwise training entry point:
  `src/Incident_Align_Method/pairwise_and_clustering/train_deepwide_pairwise.py`
- Unified pairwise architecture under a single `current` version.
- Unified training output directory: `outputs/pairwise_train`.

## Supplementary Documentation
- Method Overview: `src/Incident_Align_Method/README.md`
- Candidate Generation: `src/Incident_Align_Method/candidate_generation/README.md`
- Pairwise & Clustering: `src/Incident_Align_Method/pairwise_and_clustering/README.md`
- Full Application: `src/Incident_Align_Method/full_application/README.md`
- Evaluation Module: `src/Incident_Align_Evaluation/README.md`