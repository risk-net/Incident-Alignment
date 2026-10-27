# baselines/ — 基准对照

本目录只做**一个**基线：**逻辑回归（LR）**。

> 子项状态：B1（全单例）、B2（召回分数直接阈值）已按需求裁掉，不做。
> 需要"上游召回覆盖率"时另见候选生成阶段的 `run_k_sensitivity` 思路，
> 不属于本目录的对照表。

## 为什么是这个基线

主方法 DeepWide = **Dual Recall → pair（17 维手工特征 + Deep MLP）→ complete-link 聚类**。

LR 基线**只删掉 Deep MLP**：17 维手工特征直接进 `nn.Linear(17, 1)`。
因此它是与主方法构成**唯一变量**对照的基线，用来回答审稿人最常问的那句：

> "把深度交互建模拿掉，会差多少？"

其余一切保持相同，保证比较公平：

| 环节 | 主方法 DeepWide | 本目录 LR 基线 | 是否相同 |
|---|---|---|---|
| 候选召回 | `recall_topk.jsonl`, TOPM_OUT=200 | 同 | ✅ |
| prepared pairs | `data/english/folds/fold_0X/train/repeat_01/` | 同（直接复用） | ✅ |
| 负采样 | neg_ratio=4, seed=42+fold | 同 | ✅ |
| 训练超参 | 8 epoch / bs 512 / lr 2e-4 / wd 1e-4 | 同 | ✅ |
| 选 checkpoint | dev 上 pair-level positive F1 | 同 | ✅ |
| τ 网格 | 0.10–0.95 / step 0.01 | 同 | ✅ |
| 边规则 / 聚类 | mutual + complete-link | 同 | ✅ |
| 指标 | `graph_clustering` 那一套 | 同 | ✅ |
| **分类器** | **Deep MLP(512→256) + Wide** | **`nn.Linear(17,1)`** | ❌ **唯一变量** |

17 维特征 = 7 个类别一致位（`eq_<cat_field>`）+ 10 个文本字段余弦（`sim_<text_field>`），
见 `pairwise_model.get_wide_feature_names`。

## 文件

| 文件 | 作用 |
|---|---|
| `run_baselines.py` | **统一入口**。按序跑「英文五折 → 中文零样本」，汇总出 `comparison.csv`（含 DeepWide 参照列）与 `session.json` |
| `baseline_logistic_regression_5fold.py` | 实际训练/评测逻辑（英文五折训练 + dev 选 τ + test 评测 + 中文零样本打分） |

## 运行

```bash
# 前置：需要 torch 的环境（本机默认 python 没有 torch）
PY=/data2/conda_envs/AI_risk_event/bin/python

# 一步到位：英文五折训练+评测（约 1–2 小时 CPU）→ 中文零样本评测 → 对照表
$PY research_pipeline/Incident_Align_Method/baselines/run_baselines.py --all

# —— 收敛版（推荐用于论文对照）——
# LR 只有 17 个参数，8 轮下 3/5 折的 dev F1 仍在上升（未收敛）。
# 用 20 轮给足训练预算，回答"LR 到底能到多少"，约 2.5–3 小时。
$PY .../run_baselines.py --all --epochs 20

# 先只跑英文，确认没问题再补中文
$PY .../run_baselines.py --datasets english
$PY .../run_baselines.py --datasets chinese      # 秒级，复用英文权重

# 分折跑（中断后可续）
$PY .../run_baselines.py --datasets english --folds 1,2

# 只重出对照表，不重算
$PY .../run_baselines.py --reuse-only
```

### 关于 epoch 预算

| 预算 | 用途 | 说明 |
|---|---|---|
| `--epochs 8`（默认） | **同预算对照** | 与主方法 DeepWide 完全相同的 8 轮，回答"同预算下差多少" |
| `--epochs 20` | **收敛版对照** | 给 LR 足够轮数跑到平台期，回答"给足训练预算后 LR 仍不及吗" |

两套都跑最稳：审稿人无论从"预算公平"还是"能力上限"哪个角度问，都有对应数字。
`summary.json` 里会自动写入 `convergence` 字段（每折 dev F1 曲线、是否收敛），
终端也会打印收敛检查表。

也可以用配置文件（段落名固定 `[BaselineLogisticRegression]`）：

```bash
$PY research_pipeline/Incident_Align_Method/baselines/run_baselines.py \
    --config config/Incident_Align_Method-baseline-logistic-regression-config.ini
```

## 输出

```
outputs/benchmarks/
├── comparison.csv / .json          ★ 对照表：LR vs DeepWide × 英文/中文（不进 git）
├── session.json                    命令、输入文件 sha256、环境（不进 git）
└── logistic_regression/
    ├── fold_0X/result.json         每折：best_epoch、best_dev、test_metrics
    ├── fold_0X/lr_fold0X_epochN.pt  每 epoch 权重
    ├── summary.json                英文五折 mean ± std
    └── chinese/
        ├── fold_0X_scores.jsonl    中文 pair 分数
        ├── fold_0X_result.json     中文每折指标
        └── summary.json            中文 mean ± std
```

`outputs/benchmarks/**` 被 `.gitignore` 的 `outputs/*` 覆盖，**不会进 git**。

## 评测集

| 数据集 | 规模 | 评测方式 | 阈值来源 |
|---|---|---|---|
| English | 5,998 篇 / 1,742 事件，5 折 incident-level CV | 每折在**各自的** held-out test 折上评测（约 1,200 篇） | dev 折上按 Hungarian G→P 选 τ |
| Chinese | 1,283 篇 / 185 事件（`gold_v2.0.2`） | 单一评测域 | `outputs/dev/locked_thetas.json` 里锁定的每折 τ |

**中文是零样本**：用英文五折各自训练出的 LR 权重直接给中文打分，
5 折之间的差异只来自训练折不同，不是重训中文。

## 必须先知道的坑

1. **不要用单份全量 Gold 评英文。** 英文五折 test 集互不重叠，必须按折评测
   （详见 `docs/VALIDATION_STATUS.md` 第 3 条）。
2. **DeepWide 参照列是既有产物，不是本次重算的。** 来源：
   `outputs/english/graph_decode_5fold/metrics.json`（英文）与
   `outputs/chinese/metrics.json`（中文）。对照表里 `source` 字段会写明。
3. **历史数字 0.7602 已在新版下得到验证。** 新版 8 轮 LR 跑出 **0.7575 ± 0.0308**，
   与 0.7602 差 0.003，落在阈值网格粒度之内（旧版 τ step 0.02，新版 0.01）。
   说明旧稿的 0.7602 本身没有错。论文引用时用本目录 `comparison.csv` 的值。
4. **LR 在 8 轮下未收敛。** 5 折里有 3 折的 dev pair-F1 到第 8 轮仍在单调上升
   （fold_02/03/05，末轮增量 +0.014 / +0.019 / +0.019）。因此 **8 轮的结果低估了 LR 的能力**，
   差值 `DeepWide − LR` 会被高估。要下"深度建模是否必要"的结论，请用 `--epochs 20` 的收敛版。
5. **中文 τ 沿用 DeepWide 的锁定值。** 这样两边在同一阈值上比较，是唯一不作弊的选择；
   对照表里两边 τ 相同，比较仍成立。注意 LR 对 τ 极敏感（中文 τ 从 0.45 变到 0.63，
   预测簇数就从 189 变到 423），这是 LR 概率分布窄的固有特性，解读时需留意。
6. **CPU 上很慢。** 5 折 × 8 epoch，每 epoch 约 13 万 pair 的前向+特征构建，
   8 轮整体约 1–2 小时，20 轮约 2.5–3 小时。中断后可用 `--folds` 分折续跑。
   脚本已默认单线程（`--threads 1`）：模型只有 17 维，多线程同步开销反而拖慢
   （实测 121 线程只吃到 1.76 核）。
