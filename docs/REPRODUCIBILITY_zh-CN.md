# 复现训练与留出评测

> 📖 English version: [REPRODUCIBILITY.md](./REPRODUCIBILITY.md)

下面这些命令都是**由研究者分步执行的独立步骤**，不是一次性的全自动复跑。发布包内含可执行的核心工作流，但原始五折训练与旧版预处理有各自的输入契约，必须与你现有的文件对齐。本 ZIP **不附带**任何外部数据集或模型 checkpoint。

## 环境准备

```bash
python -m pip install -e '.[test,deepwide]'
python -m pytest -q tests
python scripts/check_assets.py --profile chinese
```

## 1. 训练：固定在既有五折产物上

原始已验证的训练入口（其参数与路径需对照 `--help` 或源码核对）：

```bash
python research_pipeline/Incident_Align_Method/pairwise_and_clustering/train_deepwide_pairwise_5fold.py --help
```

**不要假定重构后的工作区会自动重定向原始脚本中硬编码 / 配置内的相对路径。** 请先审查这些路径引用，再按需使用原始训练入口。`incident-align prepare-train` **只会生成一份交接 manifest，不会训练五折**。

> 提示：该脚本支持 `--folds 1,2,3,4,5`（指定折号）与 `--skip-train`（跳过训练、直接走解码）。默认会读取脚本自身推算的 `BASE_DIR` 下的 `data/eval_cases.jsonl`、`data/eval_structure.json`、`outputs/recall.jsonl`、`outputs/embeddings`、`outputs/prepared_5fold_cv`。

## 1b. 完全从零复现（评委路径）

Git 检出只包含金标、评测 case 与五折划分（约 52 MiB）。其余资产按下面顺序重建——这就是
「从零训练」的路径：

```bash
# (1) BGE-M3 权重 —— 需你自行下载（约 2.3 GiB，第三方权重，不随包分发）
python scripts/download_bge_m3.py                     # HuggingFace
# python scripts/download_bge_m3.py --source modelscope    # 大陆网络
python scripts/download_bge_m3.py --verify-only       # 校验完整性

# (2) 用随包 case 生成字段 embedding
python research_pipeline/Incident_Align_Method/candidate_generation/build_embeddings.py \
    --config config/Incident_Align_Method-build_embeddings-config.ini

# (3) 构建向量索引
python research_pipeline/Incident_Align_Method/candidate_generation/build_faiss_index.py \
    --config config/Incident_Align_Method-build_faiss_index-config.ini

# (4) 双通道候选召回
python research_pipeline/Incident_Align_Method/candidate_generation/run_recall.py \
    --config config/Incident_Align_Method-run_recall-config.ini

# (5) 五折训练（同时生成已准备 pair，并按 dev 选最优 checkpoint）
python research_pipeline/Incident_Align_Method/pairwise_and_clustering/train_deepwide_pairwise_5fold.py

# (6) 生成每折评测子集（英文各折 Test 互不重叠）
python scripts/make_fold_subsets.py --root . --folds 1,2,3,4,5 --splits test,dev --chinese

# (7) Dev 选阈 -> 锁定 -> 评测（见下文第 2–5 节）
```

中文资产用 `config/chinese_eval_templates.ini` 以同样方式重建（拆分方法见
[config/README.md](../config/README.md)）。

**务必保持 `topk_per_route = 150` 与训练脚本的 `TOPM_OUT = 200` 不变。** 这一对参数决定了
已发表阈值所依据的候选池；只改其中一个会静默改变 dev/test pair 集合，使锁定阈值不可比。

## 2. 导出 Dev 的 pair 分数

对每一折，用**已训练好的** checkpoint，跑冻结的 Dev cases、召回与 embedding：

```bash
python scripts/export_pair_scores.py \
  --checkpoint checkpoints/deepwide/fold_01.pt \
  --cases data/english/folds/fold_01/dev/cases.jsonl \
  --recall data/english/folds/fold_01/dev/recall_topk.jsonl \
  --embeddings data/english/folds/fold_01/dev/embeddings \
  --output outputs/dev/fold_01_scores.jsonl --device cuda:0
```

02–05 折请用**对应折的** checkpoint、Dev 文件与实际文件名重复执行。核对分数 manifest 与行数。**阈值选择期间不得使用中文 / 英文留出集的 gold。**

### 方式 B：从随包的 Dev 概率缓存转换（无需推理）

```bash
python scripts/npz_to_score_jsonl.py --all --npz-dir outputs/dev/probs --out-dir outputs/dev
```

`outputs/dev/probs/fold_0X_dev.npz` 已保存冻结的 Dev DeepWide 概率（`qc` 为
(query, candidate) 对，`p` 为概率）。转换器是严格模式：自环、重复有向对、
非有限值、越界概率都会报错，并写出含两端 SHA-256 的 manifest。
**存在 npz 缓存时应优先用方式 B**——它与历史 Dev 选阈所用的概率逐值一致，
因此得到的 θ 可与既有已发表数值直接比较。

## 3. Dev 阈值选择

```bash
incident-align select-dev --gold data/english/folds/fold_01/dev/gold.csv \
  --scores outputs/dev/fold_01_scores.jsonl \
  --out outputs/dev/fold_01_scan.json --min 0.10 --max 0.95 --step 0.01
```

五折都要重复执行。**只有五折 Dev 结果齐备后**才进行锁定：

```bash
incident-align lock-thetas --dev-run 1=outputs/dev/fold_01_scan.json --dev-run 2=outputs/dev/fold_02_scan.json --dev-run 3=outputs/dev/fold_03_scan.json --dev-run 4=outputs/dev/fold_04_scan.json --dev-run 5=outputs/dev/fold_05_scan.json --out outputs/dev/locked_thetas.json
```

## 4. 英文留出集评测

用同一个 `export_pair_scores.py` 导出五折 checkpoint 的分数，但把 `--cases`、`--recall`、`--embeddings` 指向 `data/english/test/`，并且每折分别输出。然后执行：

```bash
incident-align evaluate-folds --gold data/english/test/gold.csv --dataset english --locked outputs/dev/locked_thetas.json --fold-scores 1=outputs/english/fold_01_scores.jsonl --fold-scores 2=outputs/english/fold_02_scores.jsonl --fold-scores 3=outputs/english/fold_03_scores.jsonl --fold-scores 4=outputs/english/fold_04_scores.jsonl --fold-scores 5=outputs/english/fold_05_scores.jsonl --out outputs/english/evaluation
```

## 5. 中文 Gold v2.0.2 评测

用**同一份冻结的**中文 cases / 召回 / embedding 导出五份模型分数文件；导出时传入 `--gold data/chinese_gold/gold_v2.0.2_mapping.csv`，以便做严格的域校验。然后执行：

```bash
incident-align evaluate-folds --gold data/chinese_gold/gold_v2.0.2_mapping.csv --dataset chinese --locked outputs/dev/locked_thetas.json --expected-reports 1283 --expected-events 185 --fold-scores 1=outputs/chinese/fold_01_scores.jsonl --fold-scores 2=outputs/chinese/fold_02_scores.jsonl --fold-scores 3=outputs/chinese/fold_03_scores.jsonl --fold-scores 4=outputs/chinese/fold_04_scores.jsonl --fold-scores 5=outputs/chinese/fold_05_scores.jsonl --out outputs/chinese/evaluation
```

## 5b. 中文零样本评测（推理 + 评测）

中文是**跨语言零样本**：五个 checkpoint 全部只在英文五折上训练，从未见过中文。每折对**同一份**
1,283 篇中文域打分，因此中文只用**单份 Gold**（不用 `--fold-gold`），跨折报 mean ± std。

### 第 1 步 —— checkpoint → 分数（本包提供；CPU 即可，约 2.5 分钟/折）

```bash
python scripts/chinese_inference.py --folds 1,2,3,4,5 --device cpu
```

产出 `outputs/chinese/fold_0X_scores.jsonl`（每折 250,370 对）与
`outputs/chinese/chinese_inference_manifest.json`（每折 checkpoint 的 SHA-256、行数、概率区间、
分数文件 SHA-256）。

脚本是严格模式：会校验 recall query 域、cases 域、Gold 域三者都恰好是同一批 1,283 篇，
校验 `embedding case_ids` 一致，并检查每个概率有限且在 `[0,1]` 内。它**不做聚类、不算指标**。

> **不要改 `--topm`（默认 250）。** 中文召回平均每 query 195 个候选，250 基本不截断；
> 改成 200 会丢掉约 9,000 对，结果与已发表数值不可比。

### 第 2 步 —— 分数 → 指标

```bash
incident-align evaluate-folds --dataset chinese \
  --gold data/chinese_gold/gold_v2.0.2_mapping.csv \
  --expected-reports 1283 --expected-events 185 \
  --locked outputs/dev/locked_thetas.json \
  --fold-scores 1=outputs/chinese/fold_01_scores.jsonl \
  --fold-scores 2=outputs/chinese/fold_02_scores.jsonl \
  --fold-scores 3=outputs/chinese/fold_03_scores.jsonl \
  --fold-scores 4=outputs/chinese/fold_04_scores.jsonl \
  --fold-scores 5=outputs/chinese/fold_05_scores.jsonl \
  --out outputs/chinese/evaluation
```

`--gold` 是单份共用 Gold；`--expected-*` 锁定 1,283 / 185 的冻结契约。
输出 `outputs/chinese/evaluation/aggregate.json`（`gold_mode: "single"`），含逐折指标与 mean ± std。

### 与历史运行逐值等价（已验证）

`scripts/chinese_inference.py` 能精确复现历史中文结果。fold_01（θ = 0.52，228 簇，1,283 篇）：

| 指标 | 本包 | 历史 | 差值 |
|---|---|---|---|
| `hungarian_f1_g2p` | 0.755600 | 0.755600 | −4.7e−09 |
| `b_cubed_f1` | 0.821689 | 0.821689 | −2.2e−15 |
| `induced_pair_f1` | 0.767204 | 0.767204 | −1.1e−16 |
| `ari` | 0.763554 | 0.763554 | 0 |
| `nmi` | 0.939953 | 0.939953 | −1.1e−16 |

残差仅为浮点舍入误差。

## 6. 冻结指标与输入

把真实结果路径填进去后执行：

```bash
incident-align freeze --asset gold=... --asset locked=... \
  --asset english_results=... --asset chinese_results=... \
  --version ... --out outputs/releases/manifest.json
```

随后执行 `incident-align verify --manifest outputs/releases/manifest.json`。**manifest 哈希校验通过是必要条件，但并不等价于端到端的数学正确性验证。**

## 全量语料工作流（可选）

原始可运行脚本保留在 `research_pipeline/Incident_Align_Method/full_application/`，包括 FAISS 构建、双路召回、推理与聚类。它们旧的配置路径与数据库集成仍属项目特定内容。详见 `docs/FULL_PIPELINE_zh-CN.md`；**不在**默认复现序列之内。
