> 📖 Read this in [中文](./README_zh-CN.md)

# Data Overview

This directory contains two layers of data used by the alignment pipeline:

- `standard_*`: standardized full data assets
- `eval_*`: evaluation subset + gold clustering structure used in experiments

## Core Files

### 1) `standard_cases.jsonl`
Standardized case-level corpus (one JSON object per line).

- Unit: one case/news record
- Key fields (common): `id`, `title`, `text`, plus source/metadata fields
- Typical usage: upstream data source for building evaluation sets

### 2) `standard_incidents.jsonl`
Standardized incident-level structure.

- Unit: one incident/event cluster per line
- Schema: `incident_id`, `ids`
- Meaning: `ids` are case IDs belonging to the same incident

### 3) `eval_cases.jsonl`
Case-level evaluation subset used by alignment experiments.

- Unit: one evaluation case/news per line
- Includes richer structured fields (for example risk/event annotations)
- Typical usage: embedding, recall, pairwise training/evaluation inputs

### 4) `eval_structure.json`
Gold event-case structure for evaluation.

- Top-level schema: `{ "events": [...] }`
- Each event item: `incident_id`, `ids`
- Typical usage: clustering/alignment metrics against predicted clusters

## Relationship Between Files

- `standard_cases.jsonl` + `standard_incidents.jsonl` represent the broader standardized dataset layer.
- `eval_cases.jsonl` is the evaluation subset at case level.
- `eval_structure.json` is the gold clustering structure corresponding to the evaluation scope.

In short:
- `eval_cases.jsonl` provides the case content.
- `eval_structure.json` provides the ground-truth grouping of those cases into incidents.

## Notes

- ID fields may appear as numbers in files but are often normalized to strings in code.
- If you regenerate evaluation data, keep `eval_cases.jsonl` and `eval_structure.json` in sync.
