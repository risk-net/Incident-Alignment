#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prepare cached pairwise train/dev/test data for the no-reranker Deep+Wide pipeline.
"""

import os
from pathlib import Path

from pairwise_data_io import (
    TEXT_FIELDS_DEFAULT,
    build_cat_vocabs,
    build_all_pairs,
    case_ids_in_split,
    incident_to_gold_clusters,
    load_cases_minimal,
    load_eval_structure,
    load_recall_topk,
    sample_train_pairs,
    split_incidents,
    write_json,
    write_prepared_repeat,
)


BASE_DIR = Path(__file__).resolve().parents[3]

CASES_FILE = os.path.join(BASE_DIR, "data", "eval_cases.jsonl")
STRUCTURE_FILE = os.path.join(BASE_DIR, "data", "eval_structure.json")
RECALL_FILE = os.path.join(BASE_DIR, "outputs", "recall.jsonl")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "prepared_pairwise")

TOPM_OUT = 200
SEED = 42
TRAIN_RATIO = 0.6
DEV_RATIO = 0.2
NUM_REPEATS = 5
NEG_RATIO = 4
FALLBACK_NEG_PER_EMPTY_Q = 10

TEXT_FIELDS = TEXT_FIELDS_DEFAULT + ["text"]
SIM_FIELDS = ["domain", "event_type", "ai_risk_description", "ai_risk_subtype", "affected_actor_subtype", "text"]


def main():
    out_dir = OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    for input_path in [CASES_FILE, STRUCTURE_FILE, RECALL_FILE]:
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"输入文件不存在: {input_path}")

    text_fields = list(TEXT_FIELDS)
    sim_fields = list(SIM_FIELDS)

    incident2caseids, caseid2incident, _ = load_eval_structure(STRUCTURE_FILE)
    all_incidents = sorted(list(incident2caseids.keys()))
    case2cats = load_cases_minimal(CASES_FILE)
    query2cands, query2gt, query2minmax = load_recall_topk(RECALL_FILE, topm_out=TOPM_OUT)

    print(f"[Prepare] incidents={len(all_incidents)} queries_in_recall={len(query2cands)}")
    print(f"[Prepare] output_dir={out_dir}")
    print(f"[Prepare] text_fields={text_fields}")
    print(f"[Prepare] sim_fields={sim_fields}")

    repeat_entries = []

    for repeat_idx in range(1, NUM_REPEATS + 1):
        repeat_seed = SEED + (repeat_idx - 1) * 1000
        print(f"\n[Prepare] repeat={repeat_idx}/{NUM_REPEATS} seed={repeat_seed}")

        train_inc, dev_inc, test_inc = split_incidents(
            all_incidents,
            seed=repeat_seed,
            train_ratio=TRAIN_RATIO,
            dev_ratio=DEV_RATIO,
        )

        train_case_ids = case_ids_in_split(incident2caseids, train_inc)
        dev_case_ids = case_ids_in_split(incident2caseids, dev_inc)
        test_case_ids = case_ids_in_split(incident2caseids, test_inc)

        vocabs = build_cat_vocabs(case2cats, set(train_case_ids))

        train_pairs_all = build_all_pairs(
            query2cands,
            query2gt,
            query2minmax,
            caseid2incident,
            inc_set=train_inc,
            fallback_neg_per_empty_q=FALLBACK_NEG_PER_EMPTY_Q,
            fallback_seed=repeat_seed + 3000,
            fallback_pool_case_ids=train_case_ids,
        )
        train_pairs = sample_train_pairs(train_pairs_all, neg_ratio=NEG_RATIO, seed=repeat_seed)

        dev_pairs = build_all_pairs(
            query2cands,
            query2gt,
            query2minmax,
            caseid2incident,
            inc_set=dev_inc,
            fallback_neg_per_empty_q=FALLBACK_NEG_PER_EMPTY_Q,
            fallback_seed=repeat_seed + 1000,
            fallback_pool_case_ids=dev_case_ids,
        )
        test_pairs = build_all_pairs(
            query2cands,
            query2gt,
            query2minmax,
            caseid2incident,
            inc_set=test_inc,
            fallback_neg_per_empty_q=FALLBACK_NEG_PER_EMPTY_Q,
            fallback_seed=repeat_seed + 2000,
            fallback_pool_case_ids=test_case_ids,
        )

        gold_train = incident_to_gold_clusters(incident2caseids, train_inc)
        gold_dev = incident_to_gold_clusters(incident2caseids, dev_inc)
        gold_test = incident_to_gold_clusters(incident2caseids, test_inc)

        split_meta = {
            "repeat_index": repeat_idx,
            "repeat_seed": repeat_seed,
            "train_incidents": sorted(train_inc),
            "dev_incidents": sorted(dev_inc),
            "test_incidents": sorted(test_inc),
            "train_case_ids": train_case_ids,
            "dev_case_ids": dev_case_ids,
            "test_case_ids": test_case_ids,
            "gold_clusters_train": gold_train,
            "gold_clusters_dev": gold_dev,
            "gold_clusters_test": gold_test,
            "stats": {
                "train_all_pairs": len(train_pairs_all),
                "train_pairs": len(train_pairs),
                "dev_pairs": len(dev_pairs),
                "test_pairs": len(test_pairs),
                "train_cases": len(train_case_ids),
                "dev_cases": len(dev_case_ids),
                "test_cases": len(test_case_ids),
                "train_incidents": len(train_inc),
                "dev_incidents": len(dev_inc),
                "test_incidents": len(test_inc),
            },
        }

        paths = write_prepared_repeat(
            prepared_dir=out_dir,
            repeat_index=repeat_idx,
            train_pairs=train_pairs,
            dev_pairs=dev_pairs,
            test_pairs=test_pairs,
            vocabs=vocabs,
            split_meta=split_meta,
        )

        repeat_entries.append({
            "repeat_index": repeat_idx,
            "repeat_seed": repeat_seed,
            "repeat_dir": paths["repeat_name"],
            "files": {
                "train_pairs": f"{paths['repeat_name']}/train_pairs.jsonl",
                "dev_pairs": f"{paths['repeat_name']}/dev_pairs.jsonl",
                "test_pairs": f"{paths['repeat_name']}/test_pairs.jsonl",
                "vocabs": f"{paths['repeat_name']}/vocabs.json",
                "split_meta": f"{paths['repeat_name']}/split_meta.json",
            },
            "stats": split_meta["stats"],
        })

        print(
            f"[Prepare] repeat={repeat_idx} train_pairs={len(train_pairs)} "
            f"dev_pairs={len(dev_pairs)} test_pairs={len(test_pairs)}"
        )

    manifest = {
        "version": 1,
        "pipeline": "deepwide_pairwise_noreranker_prepared",
        "sources": {
            "cases_file": os.path.abspath(CASES_FILE),
            "structure_file": os.path.abspath(STRUCTURE_FILE),
            "recall_file": os.path.abspath(RECALL_FILE),
        },
        "prepare_config": {
            "topm_out": TOPM_OUT,
            "seed": SEED,
            "train_ratio": TRAIN_RATIO,
            "dev_ratio": DEV_RATIO,
            "num_repeats": NUM_REPEATS,
            "neg_ratio": NEG_RATIO,
            "fallback_neg_per_empty_q": FALLBACK_NEG_PER_EMPTY_Q,
        },
        "text_fields": text_fields,
        "sim_fields": sim_fields,
        "repeats": repeat_entries,
    }

    manifest_path = os.path.join(out_dir, "manifest.json")
    write_json(manifest_path, manifest)
    print(f"\n[Prepare] manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
