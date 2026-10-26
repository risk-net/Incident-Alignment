# 可选：全量语料处理

> 📖 English version: [FULL_PIPELINE.md](./FULL_PIPELINE.md)

这条路径在本次计划的论文返修中**不会运行**。相关代码保留在 `research_pipeline/Incident_Align_Method/full_application/` 下：

1. `build_db_embeddings_and_faiss.py` —— 对源新闻语料做编码 / 准备索引前置条件。
2. `build_faiss_index_full.py` —— 构建向量索引。
3. `run_full_dual_recall.py` —— 双通道召回。
4. `run_full_inference.py` —— DeepWide pair 打分。
5. `run_full_clustering.py` —— 全图稀疏 / 稠密聚类。
6. `run_full_pipeline.py` —— 原始编排入口。

依赖项：原始数据库与凭据、新闻语料、BGE-M3 模型权重、embedding 存储、索引资产、已训练的 DeepWide checkpoint，以及足够的存储空间。**本项目不分发其中任何一项。** 执行前请先阅读 `config_utils.py` 与原始说明文档。这些脚本是按原样保留的功能适配层，**未经过**与新核心的兼容性认证，不可直接视为可运行。

## 风险提示

- 该流水线会读写数据库、生成大规模 embedding 与索引，误跑可能覆盖历史资产；
- 请使用独立的产物根目录，不要指向已有的 `outputs/`；
- 配置中涉及数据库密码时，应从环境变量读取，**不得写入 Git**。
