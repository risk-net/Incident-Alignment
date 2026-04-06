# Event_Align_Evaluation

## 用途

该目录只负责事件对齐评测，不负责方法训练与全量应用。

主要能力：
- 对已有聚类结果统一计算指标
- 运行可复现的 baseline（单次向量基线 + repeated text-sim 基线）

方法训练、pairwise 建模、全量推理请使用 `src/Incident_Align_Method/`。

## 脚本

- `accuracy.py`：统一评估入口，输入预测簇与 gold 结构，输出 `metrics.json`。
- `vector_baseline.py`：单次向量相似度基线，输出 `clusters.json` + `metrics.json`。
- `baseline_textsim_threshold_repeat.py`：repeated split + 阈值搜索稳定性评测。

## 输入

- `data/eval_structure.json`：gold 事件结构。
- `data/cases.jsonl`：案例文本（`vector_baseline.py` 使用）。
- `embeddings/`：文本向量（`baseline_textsim_threshold_repeat.py` 使用）。

## 输出

统一输出到：`outputs/Event_Align_Evaluation/`

- `vector_baseline/`
  - `clusters.json`
  - `metrics.json`
  - `faiss_index/`
- `textsim_threshold_repeat/`
  - `repeat_runs.json`
  - `repeat_summary.json`
  - `repeat_runs.csv`
  - similarity cache 文件

## 运行方式

从仓库根目录执行。

### 1) 评估已有聚类结果

```bash
python src/Event_Align_Evaluation/accuracy.py \
  --pred_file outputs/Event_Align_Evaluation/vector_baseline/clusters.json \
  --true_file data/eval_structure.json
```

### 2) 单次向量基线

```bash
python src/Event_Align_Evaluation/vector_baseline.py \
  --data_dir data \
  --output_dir outputs/Event_Align_Evaluation/vector_baseline
```

### 3) Repeated text-sim 基线

```bash
python src/Event_Align_Evaluation/baseline_textsim_threshold_repeat.py \
  --structure_file data/eval_structure.json \
  --embeddings_dir embeddings \
  --output_dir outputs/Event_Align_Evaluation/textsim_threshold_repeat
```

## 默认路径说明

3 个脚本都支持直接在任意工作目录运行；默认路径会基于脚本位置解析到仓库根目录，而不是依赖当前 shell 的工作目录。
