> 📖 中文版: [README_zh-CN.md](./README_zh-CN.md)

Full placement rules, directory tree and verification: [docs/DATA_LAYOUT.md](../docs/DATA_LAYOUT.md).

## What is actually shipped here

This directory **does** contain evaluation assets (~5.8 GB in total):

| Path | Size | Content |
|---|---|---|
| `chinese_gold/gold_v2.0.2_mapping.csv` | 27 KB | Chinese Gold, 1,283 reports / 185 events |
| `chinese_gold/gold_v2.0.2_structure.json` | 29 KB | event structure view of the same mapping |
| `chinese_eval/cases.jsonl` | 7.4 MB | 1,283 Chinese cases, **with article body text** |
| `chinese_eval/recall_topk.jsonl` | 48 MB | Chinese candidate recall, 1,283 queries |
| `chinese_eval/embeddings/` | 361 MB | per-field embeddings + masks + `case_ids.txt` |
| `precomputed/eval_structure.json` | 191 KB | English Gold structure, 1,742 events / 5,998 reports |
| `english/test/` | 1.7 GB | English cases (**body text included**), full recall, per-field embeddings |
| `english/folds/` | 3.3 GB | five-fold splits, prepared pairwise inputs, per-fold `dev`/`test` subsets and gold |

`checkpoints/deepwide/fold_0X.pt` (5 × 81 MB) lives one level up.

## What is deliberately *not* here

- The full-corpus news database and the production FAISS index (`data/full_pipeline/` is an empty
  placeholder; the optional full pipeline is not part of the reproduction).
- Model weights other than the five DeepWide fold checkpoints.
- Database credentials / API keys.

## What is committed to Git vs. distributed separately

Total package ≈ 5.7 GiB, but **only ≈52 MiB is tracked in Git** — the minimum a reviewer needs to
run the pipeline end-to-end from scratch. Everything else is either rebuildable or must be hosted
separately.

### Tracked in Git (43 files under `data/`, 51.8 MiB)

| File | Size | Why it must be in Git |
|---|---|---|
| `chinese_gold/gold_v2.0.2_mapping.csv` | 27 KB | **the** Chinese gold — 1,283 reports / 185 events |
| `chinese_gold/gold_v2.0.2_structure.json` | 29 KB | structure view used by the recall script |
| `precomputed/eval_structure.json` | 191 KB | **the** English gold — 1,742 events / 5,998 reports |
| `chinese_eval/cases.jsonl` | 7.0 MiB | 1,283 Chinese cases (with body text) — input to embedding/recall |
| `english/test/cases.jsonl` | 44.2 MiB | 5,998 English cases (with body text) — input to embedding/recall |
| `english/folds/fold_splits.json` | 1.5 KB | five-fold split definition (seed 42, 1,742 / 5,998) |
| `english/folds/fold_0X/{train,dev,test}_incidents.json` | ~10 KB each | per-fold incident membership |
| `english/folds/fold_0X/{train,test}_gold.csv`, `fold_0X/{dev,test}/gold.csv` | 12–36 KB each | per-fold gold (English folds are disjoint → per-fold gold required) |

### Distributed separately / rebuilt by the reviewer

| Asset | Size | How to obtain |
|---|---|---|
| `data/english/test/embeddings/`, `data/chinese_eval/embeddings/` | 1.4 GiB + 361 MiB | rebuild with `scripts/download_bge_m3.py` + `build_embeddings.py` |
| `data/*/recall_topk.jsonl`, per-fold recall | 240 MiB + 48 MiB | rebuild with `build_faiss_index.py` + `run_recall.py` |
| `data/english/folds/fold_0X/train/` (prepared pairs) | 406 MiB | produced automatically by the training script |
| per-fold `dev`/`test` cases+recall+embeddings subsets | 3.3 GiB | `python scripts/make_fold_subsets.py` |
| `checkpoints/deepwide/fold_0X.pt` | 5 × 81 MiB | produced by training (or host separately) |
| `models/bge-m3/` | ≈2.3 GiB | **you download it yourself**: `python scripts/download_bge_m3.py` |

Git LFS filters are declared in `.gitattributes` for `data/**/*.{jsonl,csv,json}`; clone with
`git lfs install && git lfs pull` to fetch them as real files.

## Licensing note

Article body text is a frozen snapshot of third-party news articles; **copyright stays with the
original publishers**. The repository MIT licence covers code only. See `docs/DATA_LAYOUT.md` §7.

## Verify the placement

```bash
python scripts/check_assets.py --root . --profile chinese
python scripts/check_assets.py --root . --profile english
python scripts/check_assets.py --root . --profile train
python scripts/check_assets.py --root . --profile folds
```
