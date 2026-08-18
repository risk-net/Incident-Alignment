# full_application

## 用途

面向全量数据的事件对齐应用流程：从数据库构建向量与索引，执行全量召回，并完成全量推理与聚类。

## 脚本

- `build_db_embeddings_and_faiss.py`：从 PostgreSQL 生成 embeddings、FAISS 和索引库。
- `run_full_dual_recall.py`：执行全量双路召回与融合。
- `run_full_inference.py`：加载 pairwise checkpoint 做全量 pairwise 推理，产出 pair_predictions_full.jsonl。
- `run_full_clustering.py`：从 pair_predictions_full.jsonl 读回连续分数，做 complete-link 聚类（稀疏/稠密），产出 clusters.json 等。
- `build_faiss_index_full.py`：基于已有全量向量重建 FAISS 的辅助脚本。
- `config_utils.py`：统一读取与解析配置。

## 输入

- PostgreSQL 只读关系 `ai_risk_relevant_news`（也可在配置中通过 `source_relation` 改为其他 relation）。
- 配置文件：`config/Incident_Align_Method-full_application-config.ini`。
- pairwise checkpoint（推理阶段）。

## 输出

- `artifacts_root/embeddings/`
- `artifacts_root/faiss/`
- `artifacts_root/embedding_index/`
- `artifacts_root/recall/`
- `artifacts_root/full_inference/`

## 运行方式

```bash
# 一键跑全流程（四步）
python src/Incident_Align_Method/full_application/run_full_pipeline.py --config config/Incident_Align_Method-full_application-config.ini

# 或分步执行
python src/Incident_Align_Method/full_application/build_db_embeddings_and_faiss.py --config config/Incident_Align_Method-full_application-config.ini
python src/Incident_Align_Method/full_application/run_full_dual_recall.py --config config/Incident_Align_Method-full_application-config.ini
python src/Incident_Align_Method/full_application/run_full_inference.py --config config/Incident_Align_Method-full_application-config.ini
python src/Incident_Align_Method/full_application/run_full_clustering.py --config config/Incident_Align_Method-full_application-config.ini
```

> 推理与聚类已解耦：`run_full_inference.py` 只产出 `pair_predictions_full.jsonl`，
> 聚类由 `run_full_clustering.py` 独立执行，可用 `--threshold` / `--pair-file`
> 反复重跑，无需重新推理。
