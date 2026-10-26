# Incident Alignment

> 📖 中文版: [README_zh-CN.md](./README_zh-CN.md)

Reproducible DeepWide-based incident clustering, threshold selection, and English/Chinese event-level evaluation.

## What this package contains (~5.7 GB)

| Content | Location |
|---|---|
| Frozen English / Chinese evaluation cases (**article body text included**) | `data/english/`, `data/chinese_eval/` |
| Candidate recall (English 5,998 queries / Chinese 1,283 queries) | `data/*/recall_topk.jsonl` |
| Per-field BGE-M3 embeddings + validity masks | `data/*/embeddings/` |
| Two gold structures (English 1,742 events; Chinese 1,283 reports / 185 events) | `data/precomputed/`, `data/chinese_gold/` |
| Five-fold DeepWide checkpoints | `checkpoints/deepwide/fold_0X.pt` |
| Dev probability caches + locked thresholds | `outputs/dev/probs/`, `outputs/dev/locked_thetas.json` |
| English / Chinese metric outputs | `outputs/english/`, `outputs/chinese/` |

**Not shipped:** the full-corpus news database, the production FAISS index, BGE-M3 weights (only
needed if you *rebuild* embeddings), and any database credentials. Nothing is regenerated
automatically — every stage reads existing files.

See [data/README.md](data/README.md) for the detailed inventory and [docs/DATA_LAYOUT.md](docs/DATA_LAYOUT.md) for the directory contract.

### Reviewer / from-scratch path

The Git checkout carries only the gold files, evaluation cases and fold splits (~52 MiB). To rebuild
embeddings, recall and checkpoints from there:

```bash
python scripts/download_bge_m3.py          # you download the third-party weights (~2.3 GiB) yourself
```

then follow [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) §1b step by step.

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
| Evaluate English test and Chinese Gold v2.0.2 | `incident-align evaluate-folds` |
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
