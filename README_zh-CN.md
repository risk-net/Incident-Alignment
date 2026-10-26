# Incident Alignment｜事件对齐

> 📖 English version: [README.md](./README.md)

基于 DeepWide 的可复现事件聚类、阈值选择，以及英文 / 中文事件级评测。

## 本包包含什么（约 5.7 GB）

| 内容 | 位置 |
|---|---|
| 冻结的英文 / 中文评测集 case（**含新闻正文**） | `data/english/`、`data/chinese_eval/` |
| 候选召回（英文 5,998 query / 中文 1,283 query） | `data/*/recall_topk.jsonl` |
| BGE-M3 逐字段 embedding + 有效性掩码 | `data/*/embeddings/` |
| 两份 Gold 结构（英文 1,742 事件；中文 1,283 篇 / 185 事件） | `data/precomputed/`、`data/chinese_gold/` |
| 五折 DeepWide checkpoint | `checkpoints/deepwide/fold_0X.pt` |
| Dev 概率缓存 + 锁定阈值 | `outputs/dev/probs/`、`outputs/dev/locked_thetas.json` |
| 英文 / 中文指标产物 | `outputs/english/`、`outputs/chinese/` |

**不包含：** 全量新闻数据库、生产用 FAISS 索引、BGE-M3 权重（仅在**重建** embedding 时需要）、
以及任何数据库凭据。所有阶段都读取既有文件，不会自动重新生成数据。

详细清单见 [data/README_zh-CN.md](data/README_zh-CN.md)，目录契约见 [docs/DATA_LAYOUT_zh-CN.md](docs/DATA_LAYOUT_zh-CN.md)。

### 评委 / 从零复现路径

Git 检出只带金标、评测 case 与五折划分（约 52 MiB）。要从零重建 embedding、召回与 checkpoint：

```bash
python scripts/download_bge_m3.py          # 第三方权重（约 2.3 GiB）需你自行下载
```

然后按 [docs/REPRODUCIBILITY_zh-CN.md](docs/REPRODUCIBILITY_zh-CN.md) §1b 逐步执行。

## 快速开始

```bash
# 0) 基线自检 —— 不训练、不推理
bash scripts/run_full_validation.sh

# 1) Dev 概率（.npz 缓存 -> M6 JSONL，无需推理）
python scripts/npz_to_score_jsonl.py --all --npz-dir outputs/dev/probs --out-dir outputs/dev

# 2) 仅用 Dev 选阈并锁定
incident-align select-dev --gold data/english/folds/fold_01/dev/gold.csv \
  --scores outputs/dev/fold_01_scores.jsonl --out outputs/dev/fold_01_scan.json
#    ... 02–05 折重复，然后：
incident-align lock-thetas --dev-run 1=outputs/dev/fold_01_scan.json \
  --dev-run 2=outputs/dev/fold_02_scan.json --dev-run 3=outputs/dev/fold_03_scan.json \
  --dev-run 4=outputs/dev/fold_04_scan.json --dev-run 5=outputs/dev/fold_05_scan.json \
  --out outputs/dev/locked_thetas.json
```

完整步骤（含**英文必须使用每折 Gold** 的原因）见 [docs/REPRODUCIBILITY_zh-CN.md](docs/REPRODUCIBILITY_zh-CN.md)。

## 支持的任务

| 使用者目标 | 入口 |
|---|---|
| 在既有五折划分上训练 DeepWide | `research_pipeline/Incident_Align_Method/pairwise_and_clustering/train_deepwide_pairwise_5fold.py` |
| 从 checkpoint 导出冻结 pair 分数 | `scripts/export_pair_scores.py` |
| 把 Dev 概率缓存转成 M6 JSONL | `scripts/npz_to_score_jsonl.py` |
| 仅用 Dev 选阈 / 锁定阈值 | `incident-align select-dev` / `incident-align lock-thetas` |
| 评测英文测试集与中文 Gold v2.0.2 | `incident-align evaluate-folds` |
| 校验数据放置 | `scripts/check_assets.py --profile {chinese,english,train,folds}` |
| 资产变更后重生成校验和 | `scripts/regen_checksums.py` |
| 全量语料召回与聚类（可选，不属于本次复跑） | `research_pipeline/Incident_Align_Method/full_application/` |

## 文档

- [数据放置与校验](docs/DATA_LAYOUT_zh-CN.md)（英文：[DATA_LAYOUT.md](docs/DATA_LAYOUT.md)）
- [训练、阈值搜索与评测](docs/REPRODUCIBILITY_zh-CN.md)（英文：[REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)）
- [可选全量流水线](docs/FULL_PIPELINE_zh-CN.md)（英文：[FULL_PIPELINE.md](docs/FULL_PIPELINE.md)）
- [验证状态](docs/VALIDATION_STATUS_zh-CN.md)（英文：[VALIDATION_STATUS.md](docs/VALIDATION_STATUS.md)）
- [各阶段配置模板](config/README.md)

## 许可

代码：MIT（见 [LICENSE](LICENSE)）。新闻正文为第三方内容快照，版权归原出版方，**仅供研究复现**。
引用信息见 [CITATION.cff](CITATION.cff)。

## 开发期验证

`python -m pytest -q tests` 只覆盖**单元行为**；它**不能**替代与随包 checkpoint、冻结研究输入之间的端到端等价性测试。
