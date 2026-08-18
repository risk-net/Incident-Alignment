#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中文评测集推理 + 聚类 + 评估。

复用训练好的 Deep+Wide 模型，在中文评测集上做 test-only 推理。
依赖 build_embeddings / build_faiss_index / run_recall 已用中文配置跑完。

用法:
  python src/Incident_Align_Evaluation/run_chinese_eval_inference.py \
      --checkpoint outputs/your_pairwise_checkpoint.pt

说明:
  --checkpoint 是必填参数，需要指向训练好的 pairwise 模型 checkpoint。
  该 checkpoint 由 train_deepwide_pairwise.py 训练产生。
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

SRC_DIR = str(Path(__file__).resolve().parents[1])
PAIRWISE_DIR = os.path.join(SRC_DIR, "Incident_Align_Method", "pairwise_and_clustering")
for d in [SRC_DIR, PAIRWISE_DIR]:
    if d not in sys.path:
        sys.path.insert(0, d)

from pairwise_data_io import (
    TEXT_FIELDS_DEFAULT, norm_id, load_cases_minimal, load_recall_topk,
)
from pairwise_model import (
    PAPER_TEXT_FIELDS_DEFAULT, EmbeddingsStore, load_checkpoint, predict_probs,
)
from graph_clustering import (
    build_complete_link_clusters, build_directed_scores,
    hungarian_event_symmetric_macro_f1, b_cubed_f1, clustering_ari, induced_pairwise_f1,
)

# ── 配置 ──────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[2]
CASES_FILE = str(BASE_DIR / "data" / "chinese_eval_cases.jsonl")
STRUCTURE_FILE = str(BASE_DIR / "data" / "chinese_eval_structure.json")
RECALL_FILE = str(BASE_DIR / "outputs" / "chinese_eval_recall.jsonl")
EMBEDDINGS_DIR = str(BASE_DIR / "outputs" / "chinese_eval_embeddings")
OUTPUT_DIR = str(BASE_DIR / "outputs" / "chinese_eval_alignment")

THRESHOLD = 0.86
EDGE_RULE = "mutual"
TOPM_OUT = 250
TEXT_FIELDS = TEXT_FIELDS_DEFAULT + ["text"]
PAPER_TEXT_FIELDS = list(PAPER_TEXT_FIELDS_DEFAULT)
# ──────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="中文评测集推理 + 聚类 + 评估")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="训练好的 pairwise 模型 checkpoint 路径（必填）")
    parser.add_argument("--threshold", type=float, default=THRESHOLD,
                        help="聚类阈值，默认 0.86")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR,
                        help="输出目录")
    args = parser.parse_args()

    CHECKPOINT = args.checkpoint
    THRESHOLD = args.threshold
    OUTPUT_DIR = args.output_dir

    print("=" * 60)
    print("中文评测集推理 + 聚类 + 评估")
    print("=" * 60)
    print(f"checkpoint: {CHECKPOINT}")
    print(f"threshold:  {THRESHOLD}")

    # 1. 加载 case2cats（与英文训练用同一个 load_cases_minimal）
    print("\n[1] 加载 case2cats...")
    case2cats = load_cases_minimal(CASES_FILE)
    print(f"    cases: {len(case2cats)}")

    # 2. 加载 recall 候选
    print("\n[2] 加载 recall 候选...")
    query2cands, _, _ = load_recall_topk(RECALL_FILE, topm_out=TOPM_OUT)
    print(f"    queries: {len(query2cands)}")

    # 3. 加载模型 + embeddings
    print("\n[3] 加载 Deep+Wide 模型...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model, checkpoint, _ = load_checkpoint(CHECKPOINT, device=device)
    model = model.to(device)
    model.eval()
    text_fields = checkpoint.get("text_fields", TEXT_FIELDS)
    store = EmbeddingsStore(EMBEDDINGS_DIR, text_fields)
    print(f"    text_fields: {text_fields}")

    # 4. 构建 pairs + 推理
    print("\n[4] pairwise 推理...")
    pair_batch = []
    for q, cands in query2cands.items():
        for cand in cands:
            c = cand[0]  # (cid, sf, st, se)
            pair_batch.append({"q": q, "c": c, "label": 0})

    probs, _, qc_list = predict_probs(
        model=model, pair_source=pair_batch, store=store, case2cats=case2cats,
        text_fields=text_fields, device=device, desc="Chinese Eval Predict",
        batch_size=512, num_pairs=len(pair_batch),
        architecture_mode="current", paper_text_fields=PAPER_TEXT_FIELDS,
    )
    print(f"    pairs: {len(pair_batch)}")

    # 5. 聚类
    print(f"\n[5] complete-link 聚类 (threshold={THRESHOLD})...")
    directed_scores = build_directed_scores(qc_list, probs)
    all_nodes = sorted(set(case2cats.keys()))
    pred_clusters = build_complete_link_clusters(
        nodes=all_nodes,
        directed_scores=directed_scores,
        edge_rule=EDGE_RULE,
        threshold=THRESHOLD,
    )
    print(f"    predicted clusters: {len(pred_clusters)}")

    # 6. 评估
    print("\n[6] 评估...")
    with open(STRUCTURE_FILE, encoding="utf-8") as f:
        gold = json.load(f)
    gold_events = gold["events"]
    gold_sets = [set(str(norm_id(x)) for x in ev["ids"]) for ev in gold_events]
    pred_sets = [set(cl) for cl in pred_clusters]

    sym = hungarian_event_symmetric_macro_f1(gold_sets, pred_sets)
    b3 = b_cubed_f1(gold_sets, pred_sets)
    ari = clustering_ari(gold_sets, pred_sets)
    ipf = induced_pairwise_f1(gold_sets, pred_sets)

    print("\n" + "=" * 60)
    print("中文评测集对齐结果")
    print("=" * 60)
    print(f"Hungarian F1 (G→P): {sym['gold_to_pred_macro_f1']:.4f}")
    print(f"Hungarian F1 (sym):  {sym['symmetric_macro_f1']:.4f}")
    print(f"B-cubed F1:          {b3['b_cubed_f1']:.4f}")
    print(f"B-cubed P / R:       {b3['b_cubed_precision']:.4f} / {b3['b_cubed_recall']:.4f}")
    print(f"ARI:                 {ari:.4f}")
    print(f"Induced Pair F1:     {ipf['induced_pair_f1']:.4f}")
    print(f"Gold events: {len(gold_sets)}, Pred clusters: {len(pred_sets)}")

    # 保存
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    result = {
        "threshold": THRESHOLD,
        "edge_rule": EDGE_RULE,
        "checkpoint": CHECKPOINT,
        "num_cases": len(case2cats),
        "num_pairs": len(pair_batch),
        "num_gold_events": len(gold_sets),
        "num_pred_clusters": len(pred_sets),
        "hungarian_f1_g2p": sym["gold_to_pred_macro_f1"],
        "hungarian_f1_symmetric": sym["symmetric_macro_f1"],
        "b_cubed_f1": b3["b_cubed_f1"],
        "b_cubed_precision": b3["b_cubed_precision"],
        "b_cubed_recall": b3["b_cubed_recall"],
        "ari": ari,
        "induced_pair_f1": ipf["induced_pair_f1"],
    }
    with open(os.path.join(OUTPUT_DIR, "chinese_eval_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # 保存预测簇（与 accuracy.py 兼容的格式）
    with open(os.path.join(OUTPUT_DIR, "chinese_pred_clusters.json"), "w", encoding="utf-8") as f:
        json.dump([sorted(list(cl)) for cl in pred_sets], f, ensure_ascii=False)

    print(f"\n✅ 结果保存到 {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
