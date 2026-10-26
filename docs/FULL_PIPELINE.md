# Optional full-corpus processing

> 📖 中文版: [FULL_PIPELINE_zh-CN.md](./FULL_PIPELINE_zh-CN.md)

This path is **not run** in the planned paper reassessment. Code is preserved under `research_pipeline/Incident_Align_Method/full_application/`:

1. `build_db_embeddings_and_faiss.py` — encode the source news corpus / index prerequisites.
2. `build_faiss_index_full.py` — prepare vector indices.
3. `run_full_dual_recall.py` — dual-channel retrieval.
4. `run_full_inference.py` — DeepWide pair scoring.
5. `run_full_clustering.py` — full graph sparse/dense clustering.
6. `run_full_pipeline.py` — original orchestration entry point.

Dependencies: original database and credentials, news corpus, BGE-M3 model weights, embedding store, index assets, trained DeepWide checkpoints and storage capacity. The project does not ship any of these. Review `config_utils.py` and existing original instructions before execution. Scripts are preserved as original functional adapters, **not** certified to run against the new core without validation.
