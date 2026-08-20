# Incident_Align_Evaluation

## 用途

该目录负责事件对齐评测，不负责方法训练与全量应用。

主要能力：
- 对已有聚类结果统一计算指标
- 运行逻辑回归基线（5 折 CV），验证深度建模必要性
- 在中文评测集上做跨语言 test-only 推理与评估

方法训练、pairwise 建模、全量推理请使用 `src/Incident_Align_Method/`。

## 脚本

- `accuracy.py`：统一评估入口，输入预测簇与 gold 结构，输出 `metrics.json`。
- `baseline_logistic_regression_cv.py`：逻辑回归基线（将 Deep+Wide 替换为线性模型），5 折 CV。
- `run_chinese_eval_inference.py`：中文评测集 test-only 推理 + complete-link 聚类 + 评估。

## 输入

- `data/eval_structure.json`：英文 gold 事件结构。
- `data/chinese_eval_structure.json`：中文 gold 事件结构。
- `data/chinese_eval_cases.jsonl`：中文评测集 case。
- `outputs/chinese_eval_recall.jsonl`：中文评测集召回候选（由中文配置的 recall 脚本生成）。
- `outputs/chinese_eval_embeddings/`：中文评测集 embedding（由中文配置的 embedding 脚本生成）。

## 运行方式

从仓库根目录执行。

### 1) 评估已有聚类结果

```bash
python src/Incident_Align_Evaluation/accuracy.py \
  --pred_file outputs/graph_decode_from_all_checkpoints/best_model/model_selection.json \
  --true_file data/eval_structure.json
```

### 2) 逻辑回归基线（5 折 CV）

先准备 5 折划分：
```bash
python src/Incident_Align_Method/data_preparation/prepare_5fold_cv_splits.py
```

运行基线：
```bash
python src/Incident_Align_Evaluation/baseline_logistic_regression_cv.py
```

输出：`outputs/Incident_Align_Evaluation/baseline_logistic_regression_cv/`（含 `summary.json`）。

### 3) 中文评测集跨语言评测

先构建中文 embedding 和召回（使用独立中文配置）：
```bash
python src/Incident_Align_Method/candidate_generation/build_embeddings.py \
  --config config/Incident_Align_Method-build_embeddings-chinese-config.ini

python src/Incident_Align_Method/candidate_generation/build_faiss_index.py \
  --config config/Incident_Align_Method-build_faiss_index-chinese-config.ini

python src/Incident_Align_Method/candidate_generation/run_recall.py \
  --config config/Incident_Align_Method-run_recall-chinese-config.ini
```

运行推理 + 聚类 + 评估：
```bash
python src/Incident_Align_Evaluation/run_chinese_eval_inference.py \
  --checkpoint outputs/your_pairwise_checkpoint.pt
```

> **`--checkpoint` 必填**：指向训练好的 Deep+Wide pairwise 模型 checkpoint（由 `train_deepwide_pairwise.py` 产生）。可用 `--threshold` 调整聚类阈值（默认 0.86）。

输出：`outputs/chinese_eval_alignment/`（含 `chinese_eval_metrics.json` + `chinese_pred_clusters.json`）。

**独立评估已有结果**（无需重跑推理）：若已有预测簇 `chinese_pred_clusters.json`，可直接对金标计算完整指标：
```bash
python src/Incident_Align_Evaluation/evaluate_chinese_eval.py \
  --pred_file outputs/chinese_eval_alignment/chinese_pred_clusters.json \
  --true_file data/chinese_eval_structure.json
```

输出完整指标（Hungarian F1 G→P / P→G / 对称、B-cubed、ARI、Induced-pair F1），与英文五折 CV 使用同一套评估函数，口径一致。

## 默认路径说明

脚本支持直接在任意工作目录运行；默认路径会基于脚本位置解析到仓库根目录，而不是依赖当前 shell 的工作目录。
