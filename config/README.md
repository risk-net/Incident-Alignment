# config/ — 各阶段脚本的 ini 模板

本目录存放 `research_pipeline/` 下入口脚本读取的配置文件。所有路径**相对仓库根目录**，
且与 `docs/DATA_LAYOUT.md` 的目录树一致。

| 文件 | 段落名 | 读取者 |
|---|---|---|
| `Incident_Align_Method-build_embeddings-config.ini` | `[BuildEmbeddings]` | `candidate_generation/build_embeddings.py` |
| `Incident_Align_Method-build_faiss_index-config.ini` | `[BuildFaissIndex]` | `candidate_generation/build_faiss_index.py` |
| `Incident_Align_Method-run_recall-config.ini` | `[RunRecall]` | `candidate_generation/run_recall.py` |
| `Incident_Align_Method-decode_graph_from_pairwise_checkpoints-config.ini` | `[DecodeGraphFromPairwiseCheckpoints]` | `pairwise_and_clustering/decode_graph_from_pairwise_checkpoints.py` |
| `Incident_Align_Method-baseline-logistic-regression-config.ini` | `[BaselineLogisticRegression]` | `baselines/run_baselines.py`（LR 基线驱动层） |
| `chinese_eval_templates.ini` | 三段合并版 | 中文资产重建（需拆成上面三个同名文件后使用） |

## 什么时候需要它们

**核心评测链路（推荐路线）不需要任何 ini：**

```bash
python scripts/npz_to_score_jsonl.py --all --npz-dir outputs/dev/probs --out-dir outputs/dev
incident-align select-dev ... / evaluate-folds ...
python research_pipeline/Incident_Align_Method/pairwise_and_clustering/train_deepwide_pairwise_5fold.py
```

上面这条训练脚本自己按 `docs/DATA_LAYOUT.md` 解析路径，不读 ini。

**只有以下三种“重建”场景才需要 ini：**

1. 重建 BGE-M3 字段 embedding → `build_embeddings.py --config ...`
2. 重建 FAISS 索引 → `build_faiss_index.py --config ...`
3. 重建候选召回 → `run_recall.py --config ...`

以及**可选**的独立图解码 `decode_graph_from_pairwise_checkpoints.py --config ...`。

## 使用前必读

- **重建会改变下游结果**。embedding 或召回一旦重建，`dev_pairs`/`test_pairs` 集合就会变，
  历史阈值（0.52 / 0.45 / 0.59 / 0.63 / 0.61）与全部指标都不再可比，必须连训练一起重跑，
  并同步更新 `SHA256SUMS.txt` 与 `docs/ASSET_MANIFEST.json`。
- **`topk_per_route` 与训练脚本的 `TOPM_OUT` 是一对**：召回 150/路 → 合并约 223/query →
  训练时再截到 200。只改其中一个会静默改变 pair 集合。
- **`decode_graph_from_pairwise_checkpoints.py` 有 checkpoint 一致性校验**：它要求
  checkpoint 元数据里的 `prepared_dir` 等于配置里的 `prepared_dir`。随包 5 个 checkpoint
  是在历史路径下训练的，直接套用本模板会报 mismatch，需要先用当前训练脚本重训。
- 中文模板：`chinese_eval_templates.ini` 里三段需拆成三个独立文件（段落名不变）后传入
  `--config`；中文评测域是同一份 1,283 篇，5 折共用同一批 cases/recall/embeddings。
