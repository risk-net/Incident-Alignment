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
- Python `>= 3.11`
- Virtual environment recommended (`conda` or `venv`)

Install dependencies:
```bash
cd /home/nlper/zlh/Incident-Alignment
pip install -e .
```

Core dependencies in `pyproject.toml` include: `torch`, `sentence-transformers`, `faiss-cpu`, `numpy`, `pandas`, `scikit-learn`, `scipy`, etc.

## Data Description
The method pipeline relies on four default files:
- `data/standard_cases.jsonl`
- `data/standard_incidents.jsonl`
- `data/eval_cases.jsonl`
- `data/eval_structure.json`

For detailed field and relationship descriptions:
- English: [data/README.md](data/README.md)
- 中文：[data/README_zh-CN.md](data/README_zh-CN.md)

## Quick Start (Experiment Reproduction Pipeline)
Run from the repository root:
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

Default key outputs:
- `embeddings/`
- `faiss_index/`
- `prepared_pairwise/`
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

### 2) Single-Pass Vector Baseline
```bash
python src/Incident_Align_Evaluation/vector_baseline.py \
  --data_dir data \
  --output_dir outputs/Incident_Align_Evaluation/vector_baseline
```

### 3) Repeated Text-Similarity Stability Baseline
```bash
python src/Incident_Align_Evaluation/baseline_textsim_threshold_repeat.py \
  --structure_file data/eval_structure.json \
  --embeddings_dir embeddings \
  --output_dir outputs/Incident_Align_Evaluation/textsim_threshold_repeat
```

Note: Baseline scripts maintain compatibility with historical default output directories; explicitly passing `--output_dir` is recommended to ensure results are uniformly saved under `outputs/Incident_Align_Evaluation/`.

## Full Application Pipeline (Database Scenario)
Configuration file: `config/Incident_Align_Method-full_application-config.ini`

Set the database password environment variable before running:
```bash
export DB_PASSWORD='your-password'
```

Execute the full pipeline:
```bash
python src/Incident_Align_Method/full_application/build_db_embeddings_and_faiss.py \
  --config config/Incident_Align_Method-full_application-config.ini

python src/Incident_Align_Method/full_application/run_full_dual_recall.py \
  --config config/Incident_Align_Method-full_application-config.ini

python src/Incident_Align_Method/full_application/run_full_inference.py \
  --config config/Incident_Align_Method-full_application-config.ini
```

Default output root directory: `outputs/full_application_artifacts/`.

## Configuration File Reference
- `Incident_Align_Method-build_embeddings-config.ini`
  Controls input case files, embedding model paths, and embedding output directories.
- `Incident_Align_Method-build_faiss_index-config.ini`
  Controls embedding input directories and FAISS index output directories.
- `Incident_Align_Method-run_recall-config.ini`
  Controls retrieval parameters (`TOPK_PER_ROUTE`, `FUSE_MODE`, etc.) and output file paths.
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