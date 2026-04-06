# full_application

## 用途

面向全量数据的事件对齐应用流程：从数据库构建向量与索引，执行全量召回，并完成全量推理与聚类。

## 脚本

- `build_db_embeddings_and_faiss.py`：从 PostgreSQL 生成 embeddings、FAISS 和索引库。
- `run_full_dual_recall.py`：执行全量双路召回与融合。
- `run_full_inference.py`：加载 pairwise checkpoint 做全量推理与聚类输出。
- `build_faiss_index_full.py`：基于已有全量向量重建 FAISS 的辅助脚本。
- `config_utils.py`：统一读取与解析配置。

## 输入

- PostgreSQL 只读视图（推荐）`v_alignment_input_v1`，也可配置为其他 relation。
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
python src/Incident_Align_Method/full_application/build_db_embeddings_and_faiss.py --config config/Incident_Align_Method-full_application-config.ini
python src/Incident_Align_Method/full_application/run_full_dual_recall.py --config config/Incident_Align_Method-full_application-config.ini
python src/Incident_Align_Method/full_application/run_full_inference.py --config config/Incident_Align_Method-full_application-config.ini
```
