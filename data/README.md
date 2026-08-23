> 📖 Read this in [中文](./README_zh-CN.md)

# Data Overview

This directory contains two layers of data used by the alignment pipeline:

- `standard_*`: standardized full data assets
- `eval_*`: evaluation subset + gold clustering structure used in experiments

> **Public-release policy**: original news/article full-text fields (`text`,
> `description`, `summary`, etc.) have been removed from the tracked data files
> for the public repository. Files retain titles, metadata, structures, and
> RiskNet annotations. A restricted version containing the original full-text
> is available from the RiskNet authors under a data-use agreement.

## Core Files

### 1) `standard_cases.jsonl`
Standardized case-level corpus (one JSON object per line).

- Unit: one case/news record
- Key fields (common): `id`, `title`, plus source/metadata fields and annotations
  (original full-text `text` excluded per public-release policy)
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

### 5) `chinese_eval_cases.jsonl`
Chinese AI risk event benchmark (case-level), annotated from Chinese news sources.

- Unit: one evaluation case/news per line
- Same nested schema as `eval_cases.jsonl` (`event_annotation` / `ai_risk` / `ai_tech`)
- Typical usage: zero-shot cross-lingual evaluation of the English-trained model

### 6) `chinese_eval_structure.json`
Gold event-case structure for the Chinese benchmark.

- Top-level schema: `{ "events": [...], "metadata": {...} }`
- Each event item: `incident_id`, `ids`, optional `is_ai_risk`
- 200 events / 1428 annotated reports, covering 130+ event types
- Typical usage: cross-lingual clustering/alignment metrics

## Relationship Between Files

- `standard_cases.jsonl` + `standard_incidents.jsonl` represent the broader standardized dataset layer.
- `eval_cases.jsonl` is the evaluation subset at case level.
- `eval_structure.json` is the gold clustering structure corresponding to the evaluation scope.

In short:
- `eval_cases.jsonl` provides case identifiers, metadata, and RiskNet annotations
  (source full-text is available in the restricted version).
- `eval_structure.json` provides the ground-truth grouping of those cases into incidents.

## Notes

- ID fields may appear as numbers in files but are often normalized to strings in code.
- If you regenerate evaluation data, keep `eval_cases.jsonl` and `eval_structure.json` in sync.
