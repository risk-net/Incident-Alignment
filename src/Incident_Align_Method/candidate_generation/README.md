# candidate_generation

## 用途

负责候选生成阶段：构建向量、建立索引并执行双路召回。

## 脚本

- `download_model.py`：下载并缓存 BGE-M3 模型。
- `build_embeddings.py`：生成 `E_text`、`E_event`（可选要素向量）。
- `build_faiss_index.py`：为 embeddings 构建 FAISS 索引。
- `run_recall.py`：执行 text/event 双路召回并输出候选。

## 输入

- 评估数据与配置文件（见 `config/Incident_Align_Method-*.ini`）。
- `build_embeddings.py` 产物（供后续索引和召回使用）。

## 输出

- embeddings 文件（如 `emb_text.npy`、`emb_event.npy`）。
- FAISS 索引文件。
- 召回候选结果（如 `recall.jsonl`）。

## 运行方式

```bash
python src/Incident_Align_Method/candidate_generation/download_model.py
python src/Incident_Align_Method/candidate_generation/build_embeddings.py
python src/Incident_Align_Method/candidate_generation/build_faiss_index.py
python src/Incident_Align_Method/candidate_generation/run_recall.py
```
