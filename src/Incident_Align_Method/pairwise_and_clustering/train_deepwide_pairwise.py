#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pure pairwise classifier training entry for the Deep+Wide model.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

from pairwise_data_io import CAT_FIELDS, TEXT_FIELDS_DEFAULT, load_cases_minimal, load_manifest, load_prepared_repeat
from pairwise_model import (
    ARCH_CURRENT,
    PAPER_TEXT_FIELDS_DEFAULT,
    EmbeddingsStore,
    build_widedeep_model,
    compute_aggregated_report_dim,
    compute_pair_dim,
    fit_pairwise_epoch,
    get_wide_feature_names,
    predict_probs,
    save_checkpoint,
    set_seed,
)


BASE_DIR = Path(__file__).resolve().parents[3]

PREPARED_DIR = os.path.join(BASE_DIR, "prepared_pairwise")
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "embeddings")
ARCHITECTURE_MODE = ARCH_CURRENT
OUTPUT_DIR = os.environ.get("PAIRWISE_OUTPUT_DIR", os.path.join(BASE_DIR, "outputs", "pairwise_train"))

TEXT_FIELDS = TEXT_FIELDS_DEFAULT + ["text"]
SIM_FIELDS = ["domain", "event_type", "ai_risk_description", "ai_risk_subtype", "affected_actor_subtype", "text"]
PAPER_TEXT_FIELDS = list(PAPER_TEXT_FIELDS_DEFAULT)

EPOCHS = 8
BATCH_SIZE = 512
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4

TAB_HIDDEN_DIMS = [128, 64]
DEEP_HIDDEN = 512
DEEP_HIDDEN2 = 256
DROPOUT = 0.2

PAIR_THRESHOLD = 0.5


def _validate_manifest_fields(manifest: Dict[str, Any], text_fields: List[str], sim_fields: List[str]):
    manifest_text = manifest.get("text_fields", [])
    manifest_sim = manifest.get("sim_fields", [])
    if text_fields != manifest_text:
        raise ValueError(f"text_fields mismatch with prepared manifest. script={text_fields} manifest={manifest_text}")
    if sim_fields != manifest_sim:
        raise ValueError(f"sim_fields mismatch with prepared manifest. script={sim_fields} manifest={manifest_sim}")


def _load_prepared_repeats(prepared_dir: str):
    manifest = load_manifest(prepared_dir)
    repeats = sorted(manifest["repeats"], key=lambda x: int(x["repeat_index"]))
    loaded = [load_prepared_repeat(prepared_dir, repeat_entry) for repeat_entry in repeats]
    return manifest, loaded


def _count_split_labels(jsonl_path: str):
    pos = 0
    neg = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if int(row.get("label", 0)) == 1:
                pos += 1
            else:
                neg += 1
    return pos, neg


def _safe_auc(y_true: np.ndarray, probs: np.ndarray):
    unique = np.unique(y_true)
    if unique.size < 2:
        return None
    return float(roc_auc_score(y_true, probs))


def _safe_ap(y_true: np.ndarray, probs: np.ndarray):
    unique = np.unique(y_true)
    if unique.size < 2:
        return None
    return float(average_precision_score(y_true, probs))


def _binary_log_loss(y_true: np.ndarray, probs: np.ndarray):
    if y_true.size == 0:
        return 0.0
    p = np.clip(probs.astype(np.float64), 1e-7, 1.0 - 1e-7)
    y = y_true.astype(np.float64)
    return float(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)).mean())


def _compute_pair_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float = 0.5):
    threshold = float(threshold)
    y_true = np.asarray(y_true, dtype=np.int32)
    probs = np.asarray(probs, dtype=np.float32)
    y_pred = (probs >= threshold).astype(np.int32)
    metrics = {
        "threshold": threshold,
        "num_pairs": int(len(y_true)),
        "num_pos": int(y_true.sum()),
        "num_neg": int(len(y_true) - y_true.sum()),
        "num_pred_pos_pairs": int(y_pred.sum()),
        "pair_accuracy": float(accuracy_score(y_true, y_pred)),
        "pair_macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "pair_f1_pos": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "pair_precision_pos": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "pair_recall_pos": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "prob_mean": float(probs.mean()) if probs.size else 0.0,
        "dev_loss": _binary_log_loss(y_true, probs),
    }
    roc_auc = _safe_auc(y_true, probs)
    pr_auc = _safe_ap(y_true, probs)
    metrics["roc_auc"] = roc_auc
    metrics["pr_auc"] = pr_auc
    return metrics


def _write_json(path: str, obj: Any):
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[Info] device={device}")
    print(f"[Info] architecture_mode={ARCHITECTURE_MODE}")
    print(f"[Info] paper_text_fields={PAPER_TEXT_FIELDS}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(PREPARED_DIR):
        raise FileNotFoundError(f"prepared_dir 不存在: {PREPARED_DIR}")
    if not os.path.exists(EMBEDDINGS_DIR):
        raise FileNotFoundError(f"embeddings_dir 不存在: {EMBEDDINGS_DIR}")

    text_fields = list(TEXT_FIELDS)
    sim_fields = list(SIM_FIELDS)

    manifest, prepared_repeats = _load_prepared_repeats(PREPARED_DIR)
    _validate_manifest_fields(manifest, text_fields, sim_fields)

    cases_file = manifest.get("sources", {}).get("cases_file")
    if not cases_file:
        raise ValueError("Prepared manifest is missing sources.cases_file")
    if not os.path.exists(cases_file):
        raise FileNotFoundError(f"Prepared manifest 指向的 cases_file 不存在: {cases_file}")

    case2cats = load_cases_minimal(cases_file)
    needed_fields = sorted(set(text_fields) | set(sim_fields) | set(PAPER_TEXT_FIELDS))
    store = EmbeddingsStore(EMBEDDINGS_DIR, needed_fields)

    print(f"[Prepared] dir={PREPARED_DIR} repeats={len(prepared_repeats)}")
    print(f"[Embeddings] loaded {len(needed_fields)} fields")
    for field in needed_fields:
        if field in store.dim:
            print(f"  {field}: dim={store.dim[field]}")

    all_repeat_summaries = []

    for prepared in prepared_repeats:
        split_meta = prepared["split_meta"]
        repeat_idx = int(split_meta["repeat_index"])
        current_seed = int(split_meta["repeat_seed"])
        repeat_paths = prepared["paths"]
        repeat_counts = prepared.get("counts", {})
        vocabs = prepared["vocabs"]

        print(f"\n=== Training repeat {repeat_idx}/{len(prepared_repeats)} ===")
        print(f"Using prepared seed: {current_seed}")
        set_seed(current_seed)

        if sorted(vocabs.keys()) != sorted(CAT_FIELDS):
            raise ValueError(f"Prepared vocabs keys mismatch for repeat {repeat_idx}: {sorted(vocabs.keys())}")

        train_count = int(repeat_counts.get("train_pairs", 0))
        dev_count = int(repeat_counts.get("dev_pairs", 0))
        test_count = int(repeat_counts.get("test_pairs", 0))
        if train_count == 0 or dev_count == 0 or test_count == 0:
            raise ValueError(f"repeat {repeat_idx} contains empty split data")

        tab_preprocessor = None
        pair_dim = compute_pair_dim(
            store,
            text_fields,
            architecture_mode=ARCHITECTURE_MODE,
            paper_text_fields=PAPER_TEXT_FIELDS,
        )
        wide_input_dim = len(get_wide_feature_names(PAPER_TEXT_FIELDS))
        aggregated_report_dim = compute_aggregated_report_dim(store, PAPER_TEXT_FIELDS)
        wide_feature_schema = {
            "cat_agreement_fields": list(CAT_FIELDS),
            "text_similarity_fields": list(PAPER_TEXT_FIELDS),
            "input_dim": int(wide_input_dim),
        }
        deep_feature_schema = {
            "aggregated_text_fields": list(PAPER_TEXT_FIELDS),
            "aggregated_report_dim": int(aggregated_report_dim),
            "pair_input_form": "[h(u), h(v), |h(u)-h(v)|]",
            "input_dim": int(pair_dim),
        }

        model_r, model_config = build_widedeep_model(
            pair_dim=pair_dim,
            tab_hidden_dims=TAB_HIDDEN_DIMS,
            pair_hidden=DEEP_HIDDEN,
            pair_hidden2=DEEP_HIDDEN2,
            dropout=DROPOUT,
            architecture_mode=ARCHITECTURE_MODE,
            tab_input_dim=wide_input_dim,
            paper_text_fields=PAPER_TEXT_FIELDS,
        )
        model_r = model_r.to(device)
        optimizer = torch.optim.AdamW(model_r.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

        n_pos, n_neg = _count_split_labels(repeat_paths["train_pairs"])
        pos_weight = (n_neg / max(1.0, n_pos)) if n_pos > 0 else None
        print(
            f"[Repeat {repeat_idx}] train={train_count} dev={dev_count} "
            f"test={test_count} pos={n_pos} neg={n_neg}"
        )

        repeat_dir = os.path.join(OUTPUT_DIR, f"repeat_{repeat_idx:02d}")
        os.makedirs(repeat_dir, exist_ok=True)

        history = []
        checkpoint_entries = []

        for ep in range(1, EPOCHS + 1):
            train_loss = fit_pairwise_epoch(
                model=model_r,
                pair_source=repeat_paths["train_pairs"],
                store=store,
                case2cats=case2cats,
                text_fields=text_fields,
                optimizer=optimizer,
                device=device,
                batch_size=BATCH_SIZE,
                pos_weight=pos_weight,
                epoch_idx=ep,
                total_epochs=EPOCHS,
                num_pairs=train_count,
                architecture_mode=ARCHITECTURE_MODE,
                paper_text_fields=PAPER_TEXT_FIELDS,
            )

            probs, y_true, _ = predict_probs(
                model=model_r,
                pair_source=repeat_paths["dev_pairs"],
                store=store,
                case2cats=case2cats,
                text_fields=text_fields,
                device=device,
                desc=f"Dev Predict (ep={ep})",
                batch_size=BATCH_SIZE,
                num_pairs=dev_count,
                architecture_mode=ARCHITECTURE_MODE,
                paper_text_fields=PAPER_TEXT_FIELDS,
            )
            dev_metrics = _compute_pair_metrics(y_true, probs, threshold=PAIR_THRESHOLD)
            dev_metrics.update(
                {
                    "epoch": int(ep),
                    "train_loss": float(train_loss),
                    "repeat_idx": int(repeat_idx),
                    "seed": int(current_seed),
                    "prepared_dir": os.path.abspath(PREPARED_DIR),
                }
            )

            checkpoint_name = f"deepwide_repeat{repeat_idx}_seed{current_seed}_epoch{ep}.pt"
            checkpoint_path = os.path.join(repeat_dir, checkpoint_name)
            metrics_name = f"pair_metrics_dev_repeat{repeat_idx}_seed{current_seed}_epoch{ep}.json"
            metrics_path = os.path.join(repeat_dir, metrics_name)

            save_checkpoint(
                path=checkpoint_path,
                model=model_r,
                metadata={
                    "epoch": int(ep),
                    "train_loss": float(train_loss),
                    "dev_pair_metrics": dev_metrics,
                    "architecture_mode": ARCHITECTURE_MODE,
                    "paper_text_fields": list(PAPER_TEXT_FIELDS),
                    "wide_feature_schema": wide_feature_schema,
                    "deep_feature_schema": deep_feature_schema,
                    "model_config": model_config,
                    "prepared_dir": os.path.abspath(PREPARED_DIR),
                    "prepared_repeat_index": int(repeat_idx),
                    "prepared_repeat_seed": int(current_seed),
                    "repeat_idx": int(repeat_idx),
                    "seed": int(current_seed),
                    "vocabs": vocabs,
                    "cat_fields": CAT_FIELDS,
                    "text_fields": text_fields,
                    "sim_fields": sim_fields,
                    "training_config": {
                        "epochs": EPOCHS,
                        "batch_size": BATCH_SIZE,
                        "learning_rate": LEARNING_RATE,
                        "weight_decay": WEIGHT_DECAY,
                        "tab_hidden_dims": TAB_HIDDEN_DIMS,
                        "deep_hidden": DEEP_HIDDEN,
                        "deep_hidden2": DEEP_HIDDEN2,
                        "dropout": DROPOUT,
                        "pair_threshold": PAIR_THRESHOLD,
                        "architecture_mode": ARCHITECTURE_MODE,
                        "paper_text_fields": PAPER_TEXT_FIELDS,
                        "wide_feature_schema": wide_feature_schema,
                        "deep_feature_schema": deep_feature_schema,
                    },
                    "config": {
                        "prepared_dir": os.path.abspath(PREPARED_DIR),
                        "embeddings_dir": os.path.abspath(EMBEDDINGS_DIR),
                        "output_dir": os.path.abspath(OUTPUT_DIR),
                        "text_fields": text_fields,
                        "sim_fields": sim_fields,
                        "epochs": EPOCHS,
                        "batch_size": BATCH_SIZE,
                        "learning_rate": LEARNING_RATE,
                        "weight_decay": WEIGHT_DECAY,
                        "tab_hidden_dims": TAB_HIDDEN_DIMS,
                        "deep_hidden": DEEP_HIDDEN,
                        "deep_hidden2": DEEP_HIDDEN2,
                        "dropout": DROPOUT,
                        "pair_threshold": PAIR_THRESHOLD,
                        "architecture_mode": ARCHITECTURE_MODE,
                        "paper_text_fields": PAPER_TEXT_FIELDS,
                        "wide_feature_schema": wide_feature_schema,
                        "deep_feature_schema": deep_feature_schema,
                    },
                },
            )
            _write_json(metrics_path, dev_metrics)

            history.append(dev_metrics)
            checkpoint_entries.append(
                {
                    "epoch": int(ep),
                    "checkpoint": checkpoint_path,
                    "pair_metrics_dev": metrics_path,
                }
            )

            print(
                f"[Repeat {repeat_idx} Epoch {ep}] train_loss={train_loss:.6f} "
                f"dev_loss={dev_metrics['dev_loss']:.6f} pair_macro_f1={dev_metrics['pair_macro_f1']:.4f} "
                f"pair_f1_pos={dev_metrics['pair_f1_pos']:.4f}"
            )
            print(f"[Save] checkpoint -> {checkpoint_path}")
            print(f"[Save] dev pair metrics -> {metrics_path}")

        history_path = os.path.join(repeat_dir, "train_history.json")
        _write_json(history_path, history)

        all_repeat_summaries.append(
            {
                "repeat_idx": int(repeat_idx),
                "seed": int(current_seed),
                "repeat_dir": repeat_dir,
                "num_train_pairs": train_count,
                "num_dev_pairs": dev_count,
                "num_test_pairs": test_count,
                "checkpoints": checkpoint_entries,
                "train_history": history_path,
            }
        )

    summary_path = os.path.join(OUTPUT_DIR, "training_summary.json")
    _write_json(
        summary_path,
        {
            "prepared_dir": os.path.abspath(PREPARED_DIR),
            "prepared_manifest": os.path.abspath(os.path.join(PREPARED_DIR, "manifest.json")),
            "embeddings_dir": os.path.abspath(EMBEDDINGS_DIR),
            "output_dir": os.path.abspath(OUTPUT_DIR),
            "architecture_mode": ARCHITECTURE_MODE,
            "text_fields": text_fields,
            "sim_fields": sim_fields,
            "paper_text_fields": PAPER_TEXT_FIELDS,
            "training_config": {
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "tab_hidden_dims": TAB_HIDDEN_DIMS,
                "deep_hidden": DEEP_HIDDEN,
                "deep_hidden2": DEEP_HIDDEN2,
                "dropout": DROPOUT,
                "pair_threshold": PAIR_THRESHOLD,
                "architecture_mode": ARCHITECTURE_MODE,
            },
            "repeats": all_repeat_summaries,
        },
    )
    print(f"\n[Done] training summary -> {summary_path}")


if __name__ == "__main__":
    main()
