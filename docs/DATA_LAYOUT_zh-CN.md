# 数据放置与资产清单

> 📖 English version: [DATA_LAYOUT.md](./DATA_LAYOUT.md)

本仓库**不再分发**底层语料、字段 embedding、训练权重或全量 FAISS 索引。请把**既有的**资产放到 `data/` 与 `checkpoints/` 下。这里的路径只是**发布约定位置**：适配器和 CLI 参数必须指向这些位置；仅把文件复制过去，并不会自动配置旧训练脚本（部分脚本使用自己的路径）。

## 0. Git 跟踪范围

只有**评委重建全部资产所需的最小集合**进入 Git（约 5.7 GiB 中的 52 MiB）：

- `data/chinese_gold/gold_v2.0.2_mapping.csv` + `gold_v2.0.2_structure.json` —— 中文金标
- `data/precomputed/eval_structure.json` —— 英文金标结构
- `data/chinese_eval/cases.jsonl`、`data/english/test/cases.jsonl` —— **含正文**的评测 case
- `data/english/folds/fold_splits.json` + `fold_0X/{train,dev,test}_incidents.json` —— 五折定义
- `data/english/folds/fold_0X/*_gold.csv` 与 `fold_0X/dev/gold.csv` —— 每折金标

不跟踪（需重建或单独托管）：字段 embedding、召回、已准备 pair、每折 `dev`/`test` 子集、
checkpoint、BGE-M3 权重，以及整个 `outputs/`。精确清单与体积见 `data/README_zh-CN.md`
的「哪些进 Git」一节，重建顺序见 `docs/REPRODUCIBILITY_zh-CN.md` §1b。

## 1. 目录树

```text
data/
├── english/
│   ├── folds/
│   │   ├── fold_splits.json                   # 既有划分元数据（只读）
│   │   ├── fold_01/                          # fold_02 ... fold_05 同样重复
│   │   │   ├── train/                        # 已准备好的训练输入（原始 schema）
│   │   │   └── dev/
│   │   │       ├── cases.jsonl               # case 输入，原始 loader 格式
│   │   │       ├── recall_topk.jsonl         # 候选召回，原始 loader 格式
│   │   │       ├── embeddings/               # case_ids.txt, emb_<field>.npy, valid_mask_<field>.npy
│   │   │       └── dev_gold.csv              # report_id,event_id（必要时需转换）
│   │   └── ...
│   └── test/
│       ├── cases.jsonl                       # 原始 case schema
│       ├── recall_topk.jsonl                 # 原始召回 schema
│       ├── gold.csv                          # report_id,event_id
│       └── embeddings/                       # case_ids.txt + 各字段数组
├── chinese_gold/
│   └── gold_v2.0.2_mapping.csv               # 1,283 行，字段 report_id,gold_v2_event_id
├── chinese_eval/
│   ├── cases.jsonl                           # 1,283 条 case，原始 loader 格式
│   ├── recall_topk.jsonl                     # 1,283 条 query
│   └── embeddings/                           # 1,283 个 ID + 各文本字段数组
├── precomputed/                              # 旧脚本引用的其他冻结输入
└── full_pipeline/                            # 仅可选；核心评测永不使用
checkpoints/
└── deepwide/
    ├── fold_01.pt                            # 实际扩展名 / 文件名可配置
    └── ... fold_05.pt
outputs/
├── dev/                                      # 导出的 pair 概率与阈值锁定结果
├── english/                                  # 预测与指标
└── chinese/                                  # 预测与指标
```

**重要说明：** 目录树中的文件名（`gold_v2.0.2_mapping.csv` 除外）都是**推荐的本地落盘位置，并不代表使用者现有文件的命名规则**。原始工程的读取器可能要求 JSON 而不是 JSONL、不同的字段键、不同的 checkpoint 名，或按字段拆分的文件。请用 `research_pipeline/` 里的 loader 确认，再把命令指向真实文件。**不要在未同步调整路径的情况下重命名二进制资产。**

## 2. 各工作流所需的输入

| 工作流 | 必需输入 | 写入位置 | 是否生成新素材 |
|---|---|---|---|
| 五折训练 | 固定的 train/dev 划分、原始已准备的 pair 数据、特征 embedding、训练配置 | `checkpoints/deepwide/` | 不新建数据集 |
| Dev 阈值选择 | 五折 Dev gold 映射 + 五折 DeepWide pair 概率 | `outputs/dev/` | 否 |
| 英文留出评测 | 英文 test gold、各折候选分数、锁定的阈值 | `outputs/english/` | 否 |
| 中文 Gold 评测 | **v1.1** 映射、1,283 条对齐 case、既有召回、对齐的 embedding、checkpoint、锁定阈值 | `outputs/chinese/` | 否 |
| 全量语料处理（可选） | 原始语料 / 数据库访问权限、预训练 BGE-M3、向量索引、推理资产、数据库凭据 | 外部全量流水线输出 | 重；不属于计划内的复跑 |

## 3. 精确的 Gold 映射契约

中文 Gold 文件是 CSV，表头为 **`report_id,gold_v2_event_id`**（允许多余列），包含 **1,283 个唯一 report ID** 和恰好 **185 个不同事件 ID** `zh_gold_0001`…`zh_gold_0185`。请从单独冻结的中文 Gold v1.1 产物中获取（文件名保留 `gold_v2.0.2_mapping.csv` 以兼容历史路径）；**不要用历史版本的 `zh_eval_*` 映射替代**。

`incident-align` 使用的英文 gold schema 同样是 `report_id,gold_v2_event_id`（该名称只是通用 loader 字段，**并不表示**英文集是 v2.0.2 版本）。如果你的英文 gold 表头不同，请显式转换成 `data/english/test/gold.csv`，不要修改源数据。

## 4. DeepWide 字段 embedding 规则

每个 embedding 目录都必须包含 `case_ids.txt`，以及 checkpoint 所声明 `text_fields` 对应的 `emb_<field>.npy` 与 `valid_mask_<field>.npy`。ID 的顺序**必须与数组行序一致**，请先核对 checkpoint 中的字段元数据。导出工具目前假定 `case_ids.txt` 等于 `sorted(cases)`；如果你的原始准备顺序不同，**不要强行对 embedding 排序**：先把 loader 与顺序契约对齐，再执行。

## 5. 分数文件契约

`incident-align` 每行读取一个 JSON 对象。`scripts/export_pair_scores.py` 写出 `query_id`、`candidate_id`、`probability`；两个 ID 必须属于同一评测域；数值必须有限且位于 0 到 1 之间。候选集合必须与配置的召回、对应折号一致。五折的分数文件必须分别保存，不能合并。

## 6. 校验数据放置

```bash
python scripts/check_assets.py --root . --profile chinese
python scripts/check_assets.py --root . --profile english
python scripts/check_assets.py --root . --profile train
python scripts/check_assets.py --root . --profile full  # 仅做清单清点，可选
```

该校验脚本执行的是**清单式检查**（文件是否存在、CSV Gold 唯一性、可选的 embedding ID 计数）；它**不能**证明 checkpoint 与特征兼容、召回的语义有效，或数值等价。发布结果之前，请使用真实 loader 与模型做集成测试。

## 7. 发布与安全

默认情况下，**不要**把语料、专有新闻正文、数据库密码、API Key、模型权重或大规模 embedding 提交到 Git。冻结的 Gold 产物应单独发布，附带版本号与 SHA-256；允许分发的大体积产物请托管在模型 / 数据托管服务上，在 Git 中只保留哈希与获取说明。
