# 数据说明

> 📖 English version: [README.md](./README.md)

完整的数据放置规范、目录树与校验方式见 [docs/DATA_LAYOUT_zh-CN.md](../docs/DATA_LAYOUT_zh-CN.md)。本仓库**不包含**原始语料素材，也不包含 embedding 资产。

## 当前随包提供的数据文件

| 文件 | 大小 | 说明 |
|---|---|---|
| `data/chinese_gold/gold_v2.0.2_mapping.csv` | 27 KB | 中文评测集金标（唯一真值）：**1,283 篇 / 185 事件**，事件 ID `zh_gold_0001`…`zh_gold_0185` 连续无缺，SHA-256 `44d25a59…4caa` |
| `data/chinese_gold/gold_v2.0.2_structure.json` | 29 KB | 上述映射的 `{events:[{incident_id,ids}]}` 结构视图，供召回脚本读取 |
| `data/chinese_eval/chinese_eval_cases.jsonl` | 7.4 MB | 中文评测集 case 级数据，**1,283 行，含新闻正文**（`text` 字段 1,283/1,283 非空；`id` = `469377` 为采集占位符 `...`） |
| `data/precomputed/eval_structure.json` | 191 KB | 英文评测集 Gold 结构：**1,742 个事件 / 5,998 篇**，每个报告只属于一个事件；文件内 `metadata` 仅描述当前数据集规模与字段，不含历史对比字段 |

## 哪些进 Git、哪些单独分发

全包约 5.7 GiB，但**只有约 52 MiB 进入 Git**——这是评委从头跑通整条流水线所需的最小集合。
其余要么可重建，要么需要单独托管。

### 进入 Git（`data/` 下 43 个文件，51.8 MiB）

| 文件 | 体积 | 为什么必须进 Git |
|---|---|---|
| `chinese_gold/gold_v2.0.2_mapping.csv` | 27 KB | **中文金标本体**：1,283 篇 / 185 事件 |
| `chinese_gold/gold_v2.0.2_structure.json` | 29 KB | 召回脚本读取的结构视图 |
| `precomputed/eval_structure.json` | 191 KB | **英文金标本体**：1,742 事件 / 5,998 篇 |
| `chinese_eval/cases.jsonl` | 7.0 MiB | 1,283 篇中文 case（含正文），embedding/召回的上游输入 |
| `english/test/cases.jsonl` | 44.2 MiB | 5,998 篇英文 case（含正文），embedding/召回的上游输入 |
| `english/folds/fold_splits.json` | 1.5 KB | 五折划分定义（seed 42，1,742 / 5,998） |
| `english/folds/fold_0X/{train,dev,test}_incidents.json` | 各约 10 KB | 每折 incident 成员 |
| `english/folds/fold_0X/{train,test}_gold.csv`、`fold_0X/{dev,test}/gold.csv` | 各 12–36 KB | 每折金标（英文各折 Test 互不重叠，必须每折一份） |

### 单独分发 / 由评委重建

| 资产 | 体积 | 获取方式 |
|---|---|---|
| `data/english/test/embeddings/`、`data/chinese_eval/embeddings/` | 1.4 GiB + 361 MiB | 用 `scripts/download_bge_m3.py` + `build_embeddings.py` 重建 |
| `data/*/recall_topk.jsonl` 及每折召回 | 240 MiB + 48 MiB | 用 `build_faiss_index.py` + `run_recall.py` 重建 |
| `data/english/folds/fold_0X/train/`（已准备 pair） | 406 MiB | 训练脚本自动生成 |
| 每折 `dev`/`test` 的 cases+召回+embedding 子集 | 3.3 GiB | `python scripts/make_fold_subsets.py` |
| `checkpoints/deepwide/fold_0X.pt` | 5 × 81 MiB | 训练产生（或单独托管） |
| `models/bge-m3/` | ≈2.3 GiB | **需你自行下载**：`python scripts/download_bge_m3.py` |

`.gitattributes` 已为 `data/**/*.{jsonl,csv,json}` 声明 Git LFS；克隆后执行
`git lfs install && git lfs pull` 取回实体文件。

## 关于正文与版权

- **两个评测集正文均随包提供**：中文 **1,283 / 1,283** 篇。
- 正文为第三方新闻文章的冻结快照，**版权归原出版方所有**；本仓库的 MIT 许可**仅覆盖代码，不覆盖正文**。
- 正文**仅供研究复现**使用（embedding、召回、pairwise 模型输入），不得商用转载或作为新闻语料二次发布。
- 每条评测样本的原文链接保留在 `case_link` 字段，可回溯至原始出版方。

## 尚未随包提供的数据（需自行放置）

按 `docs/DATA_LAYOUT_zh-CN.md` 的约定位置放入：

- `data/english/**`：英文 cases、召回、embedding、五折划分与 test gold；
- `data/chinese_eval/recall_topk.jsonl` 与 `data/chinese_eval/embeddings/`：中文召回与字段 embedding；
- `checkpoints/deepwide/fold_0X.pt`：五折 DeepWide checkpoint。

`python scripts/check_assets.py --root . --profile chinese` 可检查上述文件是否就位（当前会报告 `cases.jsonl`、`recall_topk.jsonl`、`embeddings/case_ids.txt` 缺失，属预期状态）。
