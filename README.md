# Incident Alignment

> 📖 中文版: [README_zh-CN.md](./README_zh-CN.md)

Reproducible DeepWide-based incident clustering, threshold selection, and English/Chinese event-level evaluation.

## What this package contains

The full working set is **~5.7 GB on disk**, but **only ~52 MiB of it is in the GitHub checkout**.
Everything else must be fetched from the dataset release or regenerated. The table marks which is which —
please do not read "shipped" as "already in the repository".

| Content | Location | In GitHub checkout? |
|---|---|---|
| Frozen English / Chinese evaluation cases (**article body text included**) | `data/english/`, `data/chinese_eval/` | ✅ yes (Git LFS for large files) |
| Gold structures (English 1,742 events; Chinese 1,283 reports / 185 events) | `data/precomputed/`, `data/chinese_gold/` | ✅ yes |
| Five-fold split assignments | `data/english/folds/fold_splits.json` | ✅ yes |
| Candidate recall (English 5,998 queries / Chinese 1,283 queries) | `data/*/recall_topk.jsonl` | ❌ no — dataset release, or regenerate (§1b) |
| Per-field BGE-M3 embeddings + validity masks | `data/*/embeddings/` | ❌ no — dataset release, or regenerate (§1b) |
| Five-fold DeepWide checkpoints | `checkpoints/deepwide/fold_0X.pt` | ❌ no — dataset release, or retrain |
| Locked thresholds | `outputs/dev/locked_thetas.json` | ✅ yes |
| Dev probability caches | `outputs/dev/probs/` | ❌ no — re-export from a checkpoint |
| English / Chinese metric outputs | `outputs/english/`, `outputs/chinese/` | ❌ no — regenerate by running the pipeline |

**Not shipped anywhere:** the full-corpus news database, the production FAISS index, BGE-M3 weights
(only needed if you *rebuild* embeddings), and any database credentials. Nothing is regenerated
automatically — every stage reads existing files.

See [data/README.md](data/README.md) for the detailed inventory and [docs/DATA_LAYOUT.md](docs/DATA_LAYOUT.md) for the directory contract.

### Reviewer / from-scratch path

The Git checkout carries only gold files, evaluation cases and fold splits (~52 MiB). Everything else
falls into one of two buckets:

1. **Large assets distributed with the dataset release (with DOI)** — recall, embeddings, checkpoints
   and the Dev probability cache. Place them at the paths in the table above.
2. **Regenerated from the checkout** — BGE-M3 embeddings, then recall, then checkpoints:

```bash
python scripts/download_bge_m3.py          # you download the third-party weights (~2.3 GiB) yourself
```

then follow [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) §1b step by step.

> ⚠️ **A clean clone alone does not reproduce the paper numbers.** `bash scripts/run_full_validation.sh`
> reads the Dev score files and locked thresholds; run it only after bucket 1 or 2 above has populated them.

## Quick start

```bash
# 0) baseline self-check — no training, no inference
bash scripts/run_full_validation.sh

# 1) Dev probabilities (.npz cache -> M6 JSONL, no inference)
python scripts/npz_to_score_jsonl.py --all --npz-dir outputs/dev/probs --out-dir outputs/dev

# 2) Dev-only threshold selection and lock
incident-align select-dev --gold data/english/folds/fold_01/dev/gold.csv \
  --scores outputs/dev/fold_01_scores.jsonl --out outputs/dev/fold_01_scan.json
#    ... repeat for folds 02–05, then:
incident-align lock-thetas --dev-run 1=outputs/dev/fold_01_scan.json \
  --dev-run 2=outputs/dev/fold_02_scan.json --dev-run 3=outputs/dev/fold_03_scan.json \
  --dev-run 4=outputs/dev/fold_04_scan.json --dev-run 5=outputs/dev/fold_05_scan.json \
  --out outputs/dev/locked_thetas.json
```

Full details, including the English per-fold Gold requirement: [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Supported tasks

| User goal | Entry |
|---|---|
| Train DeepWide on the existing five folds | `research_pipeline/Incident_Align_Method/pairwise_and_clustering/train_deepwide_pairwise_5fold.py` |
| Export frozen pair scores from a checkpoint | `scripts/export_pair_scores.py` |
| Convert cached Dev probabilities to M6 JSONL | `scripts/npz_to_score_jsonl.py` |
| Select / lock Dev-only thresholds | `incident-align select-dev` / `incident-align lock-thetas` |
| Evaluate English test and Chinese Gold v1.1 | `incident-align evaluate-folds` |
| Verify data placement | `scripts/check_assets.py --profile {chinese,english,train,folds}` |
| Regenerate checksums after changing assets | `scripts/regen_checksums.py` |
| Run full-corpus retrieval and clustering (optional, not part of the rerun) | `research_pipeline/Incident_Align_Method/full_application/` |

## Documentation

- [Data placement & validation](docs/DATA_LAYOUT.md)
- [Training, threshold search and evaluation](docs/REPRODUCIBILITY.md)
- [Optional full pipeline](docs/FULL_PIPELINE.md)
- [Validation status](docs/VALIDATION_STATUS.md)
- [Stage configs](config/README.md)

## Licensing

Code: MIT (see [LICENSE](LICENSE)). Article body text is a snapshot of third-party news and
remains © the original publishers — research reproducibility only. Citation metadata:
[CITATION.cff](CITATION.cff).

## Development validation

`python -m pytest -q tests` covers unit behaviour only; it does **not** substitute for end-to-end
equivalence tests with the shipped checkpoints and frozen research inputs.
