#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整方法 (Deep+Wide) 五折交叉验证脚本。

对每个 fold 执行完整流程:
  1. 准备 pairwise 数据 (基于 outputs/prepared_5fold_cv 的 train/dev/test 划分)
  2. 训练 Deep+Wide 分类器
  3. Grid search τ + complete-link 聚类 (dev)
  4. Test 集评估

用法:
  python src/Incident_Align_Method/pairwise_and_clustering/train_deepwide_pairwise_5fold.py

依赖:
  - outputs/prepared_5fold_cv/ (由 prepare_5fold_cv_splits.py 生成)
  - outputs/embeddings/
  - outputs/recall.jsonl
  - data/eval_cases.jsonl

输出:
  outputs/pairwise_train_5fold/      # 训练 checkpoint (每折)
  outputs/graph_decode_5fold/        # 聚类解码结果 (每折)
    └── summary.json                 # 五折汇总
"""

import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

# path setup
SRC_DIR = str(Path(__file__).resolve().parents[2])
PAIRWISE_DIR = os.path.join(SRC_DIR, "Incident_Align_Method", "pairwise_and_clustering")
for d in [SRC_DIR, PAIRWISE_DIR]:
    if d not in sys.path:
        sys.path.insert(0, d)

from pairwise_data_io import (
    CAT_FIELDS, TEXT_FIELDS_DEFAULT,
    norm_id,
    load_cases_minimal, load_eval_structure, load_recall_topk,
    build_all_pairs, sample_train_pairs, build_cat_vocabs,
    case_ids_in_split, incident_to_gold_clusters,
    write_prepared_repeat, write_json,
)
from pairwise_model import (
    ARCH_CURRENT, PAPER_TEXT_FIELDS_DEFAULT,
    EmbeddingsStore, build_widedeep_model,
    compute_aggregated_report_dim, compute_pair_dim,
    fit_pairwise_epoch, predict_probs, save_checkpoint, set_seed,
    get_wide_feature_names,
)
from graph_clustering import (
    build_complete_link_clusters,
    build_directed_scores,
    hungarian_event_symmetric_macro_f1,
    b_cubed_f1, clustering_ari, induced_pairwise_f1,
    grid_search_on_dev, evaluate_with_best_config,
)

# ── 配置 ──────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[3]
CASES_FILE = str(BASE_DIR / "data" / "eval_cases.jsonl")
STRUCTURE_FILE = str(BASE_DIR / "data" / "eval_structure.json")
RECALL_FILE = str(BASE_DIR / "outputs" / "recall.jsonl")
EMBEDDINGS_DIR = str(BASE_DIR / "outputs" / "embeddings")
CV_SPLITS_DIR = str(BASE_DIR / "outputs" / "prepared_5fold_cv")
TRAIN_OUTPUT_DIR = str(BASE_DIR / "outputs" / "pairwise_train_5fold")
DECODE_OUTPUT_DIR = str(BASE_DIR / "outputs" / "graph_decode_5fold")

# 训练超参 (与现有 repeat 训练保持一致)
EPOCHS = 8
BATCH_SIZE = 512
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4
DEEP_HIDDEN = 512
DEEP_HIDDEN2 = 256
DROPOUT = 0.2
PAIR_THRESHOLD = 0.5
TAB_HIDDEN_DIMS = [128, 64]
NEG_RATIO = 4
FALLBACK_NEG_PER_EMPTY_Q = 10
TOPM_OUT = 200

# 图解码超参
THRESHOLD_STEP = 0.02
THRESHOLD_MIN = 0.50
THRESHOLD_MAX = 0.95
EDGE_RULE_GRID = ["mutual"]

TEXT_FIELDS = TEXT_FIELDS_DEFAULT + ["text"]
SIM_FIELDS = ["domain", "event_type", "ai_risk_description", "ai_risk_subtype",
              "affected_actor_subtype", "text"]
PAPER_TEXT_FIELDS = list(PAPER_TEXT_FIELDS_DEFAULT)
ARCHITECTURE_MODE = ARCH_CURRENT
# ──────────────────────────────────────────────────────────────────



def prepare_fold_data(fold_cv_dir, fold_idx):
    # type: (str, int) -> dict
    """为一个 fold 准备 pairwise 数据。"""
    # 加载 incident IDs
    with open(os.path.join(fold_cv_dir, "train_incidents.json")) as f:
        train_inc = set(json.load(f))
    with open(os.path.join(fold_cv_dir, "dev_incidents.json")) as f:
        dev_inc = set(json.load(f))
    with open(os.path.join(fold_cv_dir, "test_incidents.json")) as f:
        test_inc = set(json.load(f))

    # 加载全局数据
    incident2caseids, caseid2incident, _ = load_eval_structure(STRUCTURE_FILE)
    case2cats = load_cases_minimal(CASES_FILE)
    query2cands, query2gt, query2minmax = load_recall_topk(RECALL_FILE, topm_out=TOPM_OUT)

    # 划分 case IDs
    train_case_ids = set(case_ids_in_split(incident2caseids, train_inc))
    dev_case_ids = set(case_ids_in_split(incident2caseids, dev_inc))
    test_case_ids = set(case_ids_in_split(incident2caseids, test_inc))

    # 构建 gold clusters
    gold_train = incident_to_gold_clusters(incident2caseids, train_inc)
    gold_dev = incident_to_gold_clusters(incident2caseids, dev_inc)
    gold_test = incident_to_gold_clusters(incident2caseids, test_inc)

    # 构建 pairs
    print("    Building train pairs...")
    all_train = build_all_pairs(
        query2cands, query2gt, query2minmax, caseid2incident,
        inc_set=train_inc,
        fallback_neg_per_empty_q=FALLBACK_NEG_PER_EMPTY_Q,
        fallback_seed=42 + fold_idx,
    )
    sampled_train = sample_train_pairs(
        all_train, neg_ratio=NEG_RATIO, seed=42 + fold_idx,
        neg_per_empty_q=FALLBACK_NEG_PER_EMPTY_Q,
    )

    print("    Building dev pairs...")
    all_dev = build_all_pairs(
        query2cands, query2gt, query2minmax, caseid2incident,
        inc_set=dev_inc,
        fallback_neg_per_empty_q=0,
        fallback_seed=1042 + fold_idx,
    )

    print("    Building test pairs...")
    all_test = build_all_pairs(
        query2cands, query2gt, query2minmax, caseid2incident,
        inc_set=test_inc,
        fallback_neg_per_empty_q=0,
        fallback_seed=2042 + fold_idx,
    )

    # vocabs
    vocabs = build_cat_vocabs(case2cats, train_case_ids)

    # 保存 prepared data
    fold_prepared_dir = os.path.join(CV_SPLITS_DIR, "fold_{:02d}".format(fold_idx), "pairwise")
    paths = write_prepared_repeat(
        fold_prepared_dir, 1,
        train_pairs=sampled_train,
        dev_pairs=all_dev,
        test_pairs=all_test,
        vocabs=vocabs,
        split_meta={
            "repeat_index": 1,
            "repeat_seed": 42 + fold_idx,
            "fold_idx": fold_idx,
            "stats": {
                "train_pairs": len(sampled_train),
                "dev_pairs": len(all_dev),
                "test_pairs": len(all_test),
            },
        },
    )

    # manifest
    manifest = {
        "prepared_dir": fold_prepared_dir,
        "text_fields": TEXT_FIELDS,
        "sim_fields": SIM_FIELDS,
        "cat_fields": CAT_FIELDS,
        "sources": {"cases_file": CASES_FILE},
        "n_repeats": 1,
        "repeats": [{"repeat_index": 1, "repeat_seed": 42 + fold_idx}],
    }
    write_json(os.path.join(fold_prepared_dir, "manifest.json"), manifest)

    return {
        "fold_idx": fold_idx,
        "train_case_ids": train_case_ids,
        "dev_case_ids": dev_case_ids,
        "test_case_ids": test_case_ids,
        "gold_train": gold_train,
        "gold_dev": gold_dev,
        "gold_test": gold_test,
        "case2cats": case2cats,
        "vocabs": vocabs,
        "paths": paths,
        "train_count": len(sampled_train),
        "dev_count": len(all_dev),
        "test_count": len(all_test),
    }


def train_fold(prepared, fold_idx):
    # type: (dict, int) -> str
    """训练一个 fold 的 Deep+Wide 模型。返回最佳 checkpoint 路径。"""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    seed = 42 + fold_idx
    set_seed(seed)

    store = EmbeddingsStore(EMBEDDINGS_DIR, TEXT_FIELDS)
    case2cats = prepared["case2cats"]
    vocabs = prepared["vocabs"]
    paths = prepared["paths"]

    pair_dim = compute_pair_dim(store, TEXT_FIELDS,
                                architecture_mode=ARCHITECTURE_MODE,
                                paper_text_fields=PAPER_TEXT_FIELDS)
    wide_input_dim = len(get_wide_feature_names(PAPER_TEXT_FIELDS))

    model, model_config = build_widedeep_model(
        pair_dim=pair_dim,
        tab_hidden_dims=TAB_HIDDEN_DIMS,
        pair_hidden=DEEP_HIDDEN,
        pair_hidden2=DEEP_HIDDEN2,
        dropout=DROPOUT,
        architecture_mode=ARCHITECTURE_MODE,
        tab_input_dim=wide_input_dim,
        paper_text_fields=PAPER_TEXT_FIELDS,
    )
    model = model.to(device)

    n_pos, n_neg = _count_pos_neg(paths["train_pairs"])
    pos_weight = (n_neg / max(1.0, n_pos)) if n_pos > 0 else None

    fold_train_dir = os.path.join(TRAIN_OUTPUT_DIR, "fold_{:02d}".format(fold_idx))
    os.makedirs(fold_train_dir, exist_ok=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    best_dev_f1 = -1.0
    best_checkpoint = None

    for ep in range(1, EPOCHS + 1):
        train_loss = fit_pairwise_epoch(
            model=model, pair_source=paths["train_pairs"],
            store=store, case2cats=case2cats, text_fields=TEXT_FIELDS,
            optimizer=optimizer, device=device, batch_size=BATCH_SIZE,
            pos_weight=pos_weight, epoch_idx=ep, total_epochs=EPOCHS,
            num_pairs=prepared["train_count"],
            architecture_mode=ARCHITECTURE_MODE, paper_text_fields=PAPER_TEXT_FIELDS,
        )

        probs, y_true, _ = predict_probs(
            model=model, pair_source=paths["dev_pairs"],
            store=store, case2cats=case2cats, text_fields=TEXT_FIELDS,
            device=device, desc="Dev ep={}".format(ep),
            batch_size=BATCH_SIZE, num_pairs=prepared["dev_count"],
            architecture_mode=ARCHITECTURE_MODE, paper_text_fields=PAPER_TEXT_FIELDS,
        )

        from sklearn.metrics import f1_score
        y_pred = (probs >= PAIR_THRESHOLD).astype(np.int32)
        dev_f1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))

        ckpt_name = "deepwide_fold{:02d}_epoch{}.pt".format(fold_idx, ep)
        ckpt_path = os.path.join(fold_train_dir, ckpt_name)
        save_checkpoint(
            path=ckpt_path, model=model,
            metadata={
                "epoch": ep, "fold_idx": fold_idx, "seed": seed,
                "train_loss": float(train_loss), "dev_pair_f1_pos": dev_f1,
                "architecture_mode": ARCHITECTURE_MODE,
                "paper_text_fields": PAPER_TEXT_FIELDS,
                "vocabs": vocabs, "cat_fields": CAT_FIELDS,
                "text_fields": TEXT_FIELDS,
                "prepared_dir": os.path.dirname(paths["train_pairs"]),
            },
        )

        print("    Epoch {}: train_loss={:.6f}, dev_pair_f1_pos={:.4f}".format(
            ep, train_loss, dev_f1))

        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            best_checkpoint = ckpt_path

    return best_checkpoint if best_checkpoint else ckpt_path


def _count_pos_neg(jsonl_path):
    pos, neg = 0, 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if int(row.get("label", 0)) == 1:
                pos += 1
            else:
                neg += 1
    return pos, neg


def decode_and_evaluate_fold(prepared, fold_idx, best_ckpt):
    # type: (dict, int, str) -> dict
    """对一个 fold 做 grid search 解码和评估。"""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    seed = 42 + fold_idx
    set_seed(seed)

    store = EmbeddingsStore(EMBEDDINGS_DIR, TEXT_FIELDS)
    case2cats = prepared["case2cats"]

    # 加载模型
    model, _ = build_widedeep_model(
        pair_dim=compute_pair_dim(store, TEXT_FIELDS,
                                  architecture_mode=ARCHITECTURE_MODE,
                                  paper_text_fields=PAPER_TEXT_FIELDS),
        tab_hidden_dims=TAB_HIDDEN_DIMS,
        pair_hidden=DEEP_HIDDEN, pair_hidden2=DEEP_HIDDEN2, dropout=DROPOUT,
        architecture_mode=ARCHITECTURE_MODE,
        tab_input_dim=len(get_wide_feature_names(PAPER_TEXT_FIELDS)),
        paper_text_fields=PAPER_TEXT_FIELDS,
    )
    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device)
    model.eval()

    # Dev grid search
    print("    Grid search on dev...")
    dev_probs, dev_y_true, dev_qc = predict_probs(
        model=model, pair_source=prepared["paths"]["dev_pairs"],
        store=store, case2cats=case2cats, text_fields=TEXT_FIELDS,
        device=device, desc="Dev predict",
        batch_size=BATCH_SIZE, num_pairs=prepared["dev_count"],
        architecture_mode=ARCHITECTURE_MODE, paper_text_fields=PAPER_TEXT_FIELDS,
    )
    dev_case_ids = sorted(prepared["dev_case_ids"])

    best_dev = grid_search_on_dev(
        probs=dev_probs, y_true=dev_y_true, qc_list=dev_qc,
        dev_case_ids=dev_case_ids, gold_clusters_dev=prepared["gold_dev"],
        edge_rule_grid=EDGE_RULE_GRID,
        threshold_grid_step=THRESHOLD_STEP,
        threshold_min=THRESHOLD_MIN, threshold_max=THRESHOLD_MAX,
    )

    # Test evaluate
    print("    Evaluating on test...")
    test_probs, test_y_true, test_qc = predict_probs(
        model=model, pair_source=prepared["paths"]["test_pairs"],
        store=store, case2cats=case2cats, text_fields=TEXT_FIELDS,
        device=device, desc="Test predict",
        batch_size=BATCH_SIZE, num_pairs=prepared["test_count"],
        architecture_mode=ARCHITECTURE_MODE, paper_text_fields=PAPER_TEXT_FIELDS,
    )
    test_case_ids = sorted(prepared["test_case_ids"])

    test_metrics, pred_clusters, _ = evaluate_with_best_config(
        model=model, pair_source=prepared["paths"]["test_pairs"],
        store=store, case2cats=case2cats, text_fields=TEXT_FIELDS,
        split_case_ids=test_case_ids, gold_clusters=prepared["gold_test"],
        device=device, best_cfg=best_dev,
        batch_size=BATCH_SIZE,
        architecture_mode=ARCHITECTURE_MODE, paper_text_fields=PAPER_TEXT_FIELDS,
    )

    return {
        "best_dev": best_dev,
        "test_metrics": test_metrics,
        "num_pred_clusters": len(pred_clusters),
        "num_gold_events": len(prepared["gold_test"]),
    }


def main():
    t_start = time.time()
    print("=" * 70)
    print("完整方法 (Deep+Wide) 五折交叉验证")
    print("=" * 70)

    # 加载 CV splits
    splits_file = os.path.join(CV_SPLITS_DIR, "fold_splits.json")
    if not os.path.exists(splits_file):
        print("ERROR: {} not found".format(splits_file))
        print("Run: python src/Incident_Align_Method/data_preparation/prepare_5fold_cv_splits.py")
        sys.exit(1)
    with open(splits_file) as f:
        cv_info = json.load(f)

    n_folds = cv_info["n_folds"]
    print("Folds: {}".format(n_folds))

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("Device: {}".format(device))

    os.makedirs(TRAIN_OUTPUT_DIR, exist_ok=True)
    os.makedirs(DECODE_OUTPUT_DIR, exist_ok=True)

    fold_results = []

    for fold_info in cv_info["folds"]:
        fold_name = fold_info["fold_name"]
        fold_idx = int(fold_info["test_fold_idx"]) + 1  # 1-indexed
        fold_cv_dir = os.path.join(CV_SPLITS_DIR, fold_name)

        print("\n" + "=" * 60)
        print("{} (fold {}/{})".format(fold_name, fold_idx, n_folds))
        print("  Test: {} incidents, Dev: {} incidents, Train: {} incidents".format(
            fold_info["n_test_incidents"], fold_info["n_dev_incidents"],
            fold_info["n_train_incidents"]))

        # Step 1: Prepare data
        t_prep = time.time()
        print("[Step 1] Preparing pairwise data...")
        prepared = prepare_fold_data(fold_cv_dir, fold_idx)
        print("  Train pairs: {}, Dev pairs: {}, Test pairs: {}".format(
            prepared["train_count"], prepared["dev_count"], prepared["test_count"]))
        print("  Time: {:.0f}s".format(time.time() - t_prep))

        # Step 2: Train
        t_train = time.time()
        print("[Step 2] Training Deep+Wide model ({} epochs)...".format(EPOCHS))
        best_ckpt = train_fold(prepared, fold_idx)
        print("  Best checkpoint: {}".format(best_ckpt))
        print("  Time: {:.0f}s".format(time.time() - t_train))

        # Step 3: Decode & Evaluate
        t_decode = time.time()
        print("[Step 3] Grid search + evaluate...")
        result = decode_and_evaluate_fold(prepared, fold_idx, best_ckpt)
        print("  Time: {:.0f}s".format(time.time() - t_decode))

        tm = result["test_metrics"]
        print("  Test Hungarian F1 (G->P): {:.4f}".format(tm["event_macro_f1_hungarian"]))
        print("  Test Hungarian F1 (sym):  {:.4f}".format(tm["event_macro_f1_hungarian_symmetric"]))
        print("  Test B-cubed F1:          {:.4f}".format(tm["b_cubed_f1"]))
        print("  Test ARI:                 {:.4f}".format(tm["ari"]))
        print("  Best dev tau:             {:.4f}".format(result["best_dev"]["threshold"]))

        # Save per-fold result
        fold_out_dir = os.path.join(DECODE_OUTPUT_DIR, fold_name)
        os.makedirs(fold_out_dir, exist_ok=True)
        fold_result = {
            "fold_name": fold_name,
            "fold_idx": fold_idx,
            "n_test_incidents": fold_info["n_test_incidents"],
            "n_dev_incidents": fold_info["n_dev_incidents"],
            "n_train_incidents": fold_info["n_train_incidents"],
            "train_pairs": prepared["train_count"],
            "dev_pairs": prepared["dev_count"],
            "test_pairs": prepared["test_count"],
            "best_checkpoint": best_ckpt,
            "best_dev": result["best_dev"],
            "test_metrics": result["test_metrics"],
            "num_pred_clusters": result["num_pred_clusters"],
            "num_gold_events": result["num_gold_events"],
        }
        write_json(os.path.join(fold_out_dir, "result.json"), fold_result)
        fold_results.append(fold_result)

    # ── 汇总 ──
    print("\n" + "=" * 70)
    print("五折 CV 汇总")
    print("=" * 70)

    metric_names = [
        "event_macro_f1_hungarian",
        "event_macro_f1_hungarian_symmetric",
        "event_macro_f1_hungarian_pred_to_gold",
        "b_cubed_f1", "b_cubed_precision", "b_cubed_recall",
        "ari", "induced_pair_f1", "induced_pair_precision", "induced_pair_recall",
        "pair_macro_f1", "pair_f1_pos", "pair_precision_pos", "pair_recall_pos",
    ]

    summary_metrics = {}
    for m in metric_names:
        values = [fr["test_metrics"][m] for fr in fold_results]
        summary_metrics[m] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)),
            "values": values,
        }
        print("  {:45s} {:.4f} +- {:.4f}".format(m, summary_metrics[m]["mean"], summary_metrics[m]["std"]))

    thresholds = [fr["best_dev"]["threshold"] for fr in fold_results]
    n_clusters = [fr["num_pred_clusters"] for fr in fold_results]
    n_gold = [fr["num_gold_events"] for fr in fold_results]

    summary = {
        "method": "Deep+Wide Pairwise + Complete-Link",
        "cv_type": "5-fold incident-level cross-validation",
        "n_folds": n_folds,
        "architecture_mode": ARCHITECTURE_MODE,
        "edge_rule": "mutual",
        "merge_strategy": "complete_link",
        "threshold_grid": {"min": THRESHOLD_MIN, "max": THRESHOLD_MAX, "step": THRESHOLD_STEP},
        "training_config": {
            "epochs": EPOCHS, "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
            "deep_hidden": DEEP_HIDDEN, "deep_hidden2": DEEP_HIDDEN2, "dropout": DROPOUT,
        },
        "test_metrics_across_folds": summary_metrics,
        "best_thresholds": {
            "mean": float(np.mean(thresholds)), "std": float(np.std(thresholds, ddof=1)),
            "values": [float(t) for t in thresholds],
        },
        "num_pred_clusters": {
            "mean": float(np.mean(n_clusters)), "std": float(np.std(n_clusters, ddof=1)),
            "values": n_clusters,
        },
        "num_gold_events": {
            "mean": float(np.mean(n_gold)), "std": float(np.std(n_gold, ddof=1)),
            "values": n_gold,
        },
    }
    summary_path = os.path.join(DECODE_OUTPUT_DIR, "summary.json")
    write_json(summary_path, summary)
    print("\nSummary -> {}".format(summary_path))
    print("Total time: {:.0f}s ({:.1f}min)".format(time.time() - t_start, (time.time() - t_start) / 60))


if __name__ == "__main__":
    main()
