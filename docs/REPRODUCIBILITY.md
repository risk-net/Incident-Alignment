# Reproduce training and held-out evaluation

> 📖 中文版: [REPRODUCIBILITY_zh-CN.md](./REPRODUCIBILITY_zh-CN.md)

These commands are **separate researcher-run steps**, not an automated full rerun. The published package includes executable core workflows, but the original five-fold training and legacy preprocessing have their own input contracts and must be matched to your existing files. No external datasets or model checkpoints are shipped in this ZIP.

## Setup
```bash
python -m pip install -e '.[test,deepwide]'
python -m pytest -q tests
python scripts/check_assets.py --profile chinese
```

## 1. Training: fixed five-fold artifacts

Original validated training entry point (its flags and paths must be checked against `--help` or source):

```bash
python research_pipeline/Incident_Align_Method/pairwise_and_clustering/train_deepwide_pairwise_5fold.py --help
```

**Do not assume the restructured workspace automatically redirects original hardcoded/config relative paths.** Review these references and use the original training entry as appropriate. `incident-align prepare-train` only creates a hand-off manifest; it does not train a five-fold run.

## 1b. Full from-scratch reproduction (reviewer path)

The packaged Git checkout contains only the gold files, evaluation cases and fold splits (~52 MiB).
Everything else is rebuilt in this order — this is the "train from scratch" path a reviewer follows:

```bash
# (1) BGE-M3 weights — you download them yourself (~2.3 GiB, third-party, not redistributed)
python scripts/download_bge_m3.py                    # HuggingFace
# python scripts/download_bge_m3.py --source modelscope   # 大陆网络
python scripts/download_bge_m3.py --verify-only      # 校验完整性

# (2) field embeddings from the shipped cases
python research_pipeline/Incident_Align_Method/candidate_generation/build_embeddings.py \
    --config config/Incident_Align_Method-build_embeddings-config.ini

# (3) vector indices
python research_pipeline/Incident_Align_Method/candidate_generation/build_faiss_index.py \
    --config config/Incident_Align_Method-build_faiss_index-config.ini

# (4) dual-channel candidate recall
python research_pipeline/Incident_Align_Method/candidate_generation/run_recall.py \
    --config config/Incident_Align_Method-run_recall-config.ini

# (5) five-fold training (also builds prepared pairs and selects the dev-best checkpoint)
python research_pipeline/Incident_Align_Method/pairwise_and_clustering/train_deepwide_pairwise_5fold.py

# (6) per-fold evaluation subsets (disjoint English Test domains)
python scripts/make_fold_subsets.py --root . --folds 1,2,3,4,5 --splits test,dev --chinese

# (7) dev-only thresholds -> lock -> evaluate (see sections 2–5 below)
```

Chinese assets are rebuilt the same way with `config/chinese_eval_templates.ini`
(see [config/README.md](../config/README.md) for how to split it into three files).

**Keep `topk_per_route = 150` and the training script's `TOPM_OUT = 200`.** That pair defines the
candidate pool the published thresholds were selected on; changing either silently changes the
dev/test pair sets and makes the locked thetas incomparable.

## 2. Export Dev pair scores

`incident-align` consumes one JSON object per line (`query_id`, `candidate_id`, `probability`).
There are two ways to produce it for Dev.

**(a) From a trained checkpoint** (re-runs DeepWide inference):

```bash
python scripts/export_pair_scores.py \
  --checkpoint checkpoints/deepwide/fold_01.pt \
  --cases data/english/folds/fold_01/dev/cases.jsonl \
  --recall data/english/folds/fold_01/dev/recall_topk.jsonl \
  --embeddings data/english/folds/fold_01/dev/embeddings \
  --output outputs/dev/fold_01_scores.jsonl --device cuda:0
```

**(b) From the shipped Dev probability cache** (no inference; `.npz` → JSONL):

```bash
python scripts/npz_to_score_jsonl.py --all \
  --npz-dir outputs/dev/probs --out-dir outputs/dev
```

Each `outputs/dev/probs/fold_0X_dev.npz` already holds the frozen Dev DeepWide
probabilities (`qc` = query/candidate pairs, `p` = probability). The converter is
strict: it rejects self-pairs, duplicate directed pairs, non-finite values and
probabilities outside `[0,1]`, and writes a manifest with the SHA-256 of both
files. **Always prefer (b) when the npz cache exists** — it is bit-identical to
what the historical Dev threshold selection used, so the resulting thetas stay
comparable to previously published values.

Repeat for 02–05 with the **correct fold's** checkpoint, Dev files and actual filenames. Confirm score manifest and row counts. Do not use Chinese/English held-out gold during selection.

## 3. Dev threshold selection

```bash
incident-align select-dev --gold data/english/folds/fold_01/dev/gold.csv \
  --scores outputs/dev/fold_01_scores.jsonl \
  --out outputs/dev/fold_01_scan.json --min 0.10 --max 0.95 --step 0.01
```

Repeat for all folds. Lock only after five Dev results exist:

```bash
incident-align lock-thetas --dev-run 1=outputs/dev/fold_01_scan.json --dev-run 2=outputs/dev/fold_02_scan.json --dev-run 3=outputs/dev/fold_03_scan.json --dev-run 4=outputs/dev/fold_04_scan.json --dev-run 5=outputs/dev/fold_05_scan.json --out outputs/dev/locked_thetas.json
```

## 4. Held-out English evaluation

The five English folds have **disjoint** Test subsets (~1,200 cases each). Each fold must therefore be
scored **and evaluated against its own Gold** (`data/english/folds/fold_0X/test_gold.csv`). Using one
full 5,998-case Gold file for every fold is wrong: the ~4,800 unscored cases of the other folds become
singletons and dilute the metrics.

Export 5 fold checkpoint scores with the same `export_pair_scores.py`, but point `--cases`, `--recall`,
`--embeddings` to that fold's `data/english/folds/fold_0X/test/` directory and output each fold
separately. Then:

```bash
incident-align evaluate-folds --dataset english --locked outputs/dev/locked_thetas.json \
  --fold-gold 1=data/english/folds/fold_01/test_gold.csv \
  --fold-gold 2=data/english/folds/fold_02/test_gold.csv \
  --fold-gold 3=data/english/folds/fold_03/test_gold.csv \
  --fold-gold 4=data/english/folds/fold_04/test_gold.csv \
  --fold-gold 5=data/english/folds/fold_05/test_gold.csv \
  --fold-scores 1=outputs/english/fold_01_scores.jsonl --fold-scores 2=outputs/english/fold_02_scores.jsonl \
  --fold-scores 3=outputs/english/fold_03_scores.jsonl --fold-scores 4=outputs/english/fold_04_scores.jsonl \
  --fold-scores 5=outputs/english/fold_05_scores.jsonl --out outputs/english/evaluation
```

`--fold-gold` and `--gold` are mutually exclusive. `--expected-reports` / `--expected-events` are only
valid in single-Gold mode (per-fold Golds have different sizes). The output `aggregate.json` records
`gold_mode: "per_fold"` and the SHA-256 of each fold's Gold.

## 5. Chinese Gold v2.0.2 evaluation

All five folds score the **same** 1,283-report Chinese domain, so Chinese evaluation uses a **single**
Gold file (`--gold`), not `--fold-gold`.

Export 5 model score files using the *same frozen* Chinese cases/recall/embeddings, pass `--gold data/chinese_gold/gold_v2.0.2_mapping.csv` to export for strict domain verification. Then:

```bash
incident-align evaluate-folds --gold data/chinese_gold/gold_v2.0.2_mapping.csv --dataset chinese --locked outputs/dev/locked_thetas.json --expected-reports 1283 --expected-events 185 --fold-scores 1=outputs/chinese/fold_01_scores.jsonl --fold-scores 2=outputs/chinese/fold_02_scores.jsonl --fold-scores 3=outputs/chinese/fold_03_scores.jsonl --fold-scores 4=outputs/chinese/fold_04_scores.jsonl --fold-scores 5=outputs/chinese/fold_05_scores.jsonl --out outputs/chinese/evaluation
```

## 5b. Chinese zero-shot evaluation (inference + evaluation)

Chinese is **cross-lingual zero-shot**: all five checkpoints were trained on English folds only and
have never seen Chinese. Each fold scores the **same** 1,283-report Chinese domain, so Chinese uses a
**single** Gold file (not `--fold-gold`) and reports mean ± std across folds.

### Step 1 — checkpoint → scores (this repo; CPU is fine, ~2.5 min/fold)

```bash
python scripts/chinese_inference.py --folds 1,2,3,4,5 --device cpu
```

Writes `outputs/chinese/fold_0X_scores.jsonl` (250,370 pairs each) plus
`outputs/chinese/chinese_inference_manifest.json` (per-fold checkpoint SHA-256, row counts,
probability range, score file SHA-256).

The script is strict: it verifies that the recall query domain, the cases domain and the Gold
domain are all exactly the same 1,283 reports, that `embedding case_ids` matches, and that every
probability is finite and inside `[0,1]`. It performs **no clustering and no metrics**.

> **Do not change `--topm` (default 250).** Chinese recall averages 195 candidates per query, so 250
> barely truncates. Using 200 would drop ~9,000 pairs and make the results incomparable with the
> published numbers.

### Step 2 — scores → metrics

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

`--gold` is the single shared Gold; `--expected-*` pin the frozen 1,283 / 185 contract.
Output: `outputs/chinese/evaluation/aggregate.json` (`gold_mode: "single"`) with per-fold metrics
and the mean ± std aggregate.

### Verified equivalence with the historical run

`scripts/chinese_inference.py` reproduces the historical Chinese numbers exactly. Fold 01
(θ = 0.52, 228 clusters, 1,283 scored reports):

| Metric | This package | Historical | Δ |
|---|---|---|---|
| `hungarian_f1_g2p` | 0.755600 | 0.755600 | −4.7e−09 |
| `b_cubed_f1` | 0.821689 | 0.821689 | −2.2e−15 |
| `induced_pair_f1` | 0.767204 | 0.767204 | −1.1e−16 |
| `ari` | 0.763554 | 0.763554 | 0 |
| `nmi` | 0.939953 | 0.939953 | −1.1e−16 |

Residual differences are floating-point rounding only.

## 6. Freeze metrics and inputs

Use `incident-align freeze --asset gold=... --asset locked=... --asset english_results=... --asset chinese_results=... --version ... --out outputs/releases/manifest.json` with actual result paths, then `incident-align verify --manifest outputs/releases/manifest.json`. A successful manifest hash verification is necessary but not equivalent to end-to-end mathematical validation.

## Full-corpus workflow (optional)

Original runnable scripts are preserved under `research_pipeline/Incident_Align_Method/full_application/`, including FAISS setup, dual recall, inference and clustering. Their old configuration paths and database integration remain project-specific. See `docs/FULL_PIPELINE.md`; **not included** in default reproduction sequence.
