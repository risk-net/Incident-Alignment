#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中文评测集独立评估脚本。

读取已有的中文评测集预测簇 + 金标结构，用与英文五折 CV 完全相同的
graph_clustering 评估函数计算完整指标。无需重跑推理/聚类。

用法:
  python src/Incident_Align_Evaluation/evaluate_chinese_eval.py

输入（默认）:
  --pred_file   outputs/chinese_eval_alignment/chinese_pred_clusters.json
  --true_file   data/chinese_eval_structure.json

输出:
  outputs/chinese_eval_alignment/chinese_eval_metrics.json（完整指标）
"""

import argparse
import json
import os
import sys
from pathlib import Path

# path setup
SRC_DIR = str(Path(__file__).resolve().parents[1])
PAIRWISE_DIR = os.path.join(SRC_DIR, "Incident_Align_Method", "pairwise_and_clustering")
for d in [SRC_DIR, PAIRWISE_DIR]:
    if d not in sys.path:
        sys.path.insert(0, d)

from graph_clustering import (
    hungarian_event_symmetric_macro_f1,
    b_cubed_f1,
    clustering_ari,
    induced_pairwise_f1,
)

BASE_DIR = Path(__file__).resolve().parents[2]


def norm_id(x):
    return str(x).strip()


def load_pred_clusters(pred_file):
    """加载预测簇，兼容 list-of-lists 或 dict 格式。"""
    with open(pred_file, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        # dict 格式: {cluster_id: [case_id, ...]} 或 {"clusters": [...]}
        if "clusters" in data:
            return [set(norm_id(x) for x in c) for c in data["clusters"]]
        return [set(norm_id(x) for x in v) for v in data.values()]
    # list-of-lists
    return [set(norm_id(x) for x in c) for c in data]


def load_gold_clusters(true_file):
    """加载金标事件结构。"""
    with open(true_file, encoding="utf-8") as f:
        data = json.load(f)
    return [set(norm_id(x) for x in ev["ids"]) for ev in data["events"]]


def main():
    parser = argparse.ArgumentParser(description="中文评测集独立评估")
    parser.add_argument("--pred_file", type=str,
                        default=str(BASE_DIR / "outputs" / "chinese_eval_alignment" / "chinese_pred_clusters.json"),
                        help="预测簇文件路径")
    parser.add_argument("--true_file", type=str,
                        default=str(BASE_DIR / "data" / "chinese_eval_structure.json"),
                        help="金标结构文件路径")
    parser.add_argument("--output_file", type=str,
                        default=str(BASE_DIR / "outputs" / "chinese_eval_alignment" / "chinese_eval_metrics.json"),
                        help="指标输出路径")
    args = parser.parse_args()

    pred_sets = load_pred_clusters(args.pred_file)
    gold_sets = load_gold_clusters(args.true_file)

    print(f"预测簇: {len(pred_sets)}, 金标事件: {len(gold_sets)}")

    # 与英文五折 CV 完全相同的评估函数
    sym = hungarian_event_symmetric_macro_f1(gold_sets, pred_sets)
    b3 = b_cubed_f1(gold_sets, pred_sets)
    ari = clustering_ari(gold_sets, pred_sets)
    ipf = induced_pairwise_f1(gold_sets, pred_sets)

    print("\n" + "=" * 60)
    print("中文评测集对齐结果")
    print("=" * 60)
    print(f"Hungarian F1 (G→P): {sym['gold_to_pred_macro_f1']:.4f}")
    print(f"Hungarian F1 (P→G): {sym['pred_to_gold_macro_f1']:.4f}")
    print(f"Hungarian F1 (sym):  {sym['symmetric_macro_f1']:.4f}")
    print(f"B-cubed F1:          {b3['b_cubed_f1']:.4f}")
    print(f"B-cubed P / R:       {b3['b_cubed_precision']:.4f} / {b3['b_cubed_recall']:.4f}")
    print(f"ARI:                 {ari:.4f}")
    print(f"Induced Pair F1:     {ipf['induced_pair_f1']:.4f}")
    print(f"Induced P / R:       {ipf['induced_pair_precision']:.4f} / {ipf['induced_pair_recall']:.4f}")
    print(f"Gold events: {len(gold_sets)}, Pred clusters: {len(pred_sets)}")

    # 保存完整指标
    result = {
        "num_gold_events": len(gold_sets),
        "num_pred_clusters": len(pred_sets),
        "hungarian_f1_g2p": sym["gold_to_pred_macro_f1"],
        "hungarian_f1_p2g": sym["pred_to_gold_macro_f1"],
        "hungarian_f1_symmetric": sym["symmetric_macro_f1"],
        "b_cubed_f1": b3["b_cubed_f1"],
        "b_cubed_precision": b3["b_cubed_precision"],
        "b_cubed_recall": b3["b_cubed_recall"],
        "ari": ari,
        "induced_pair_f1": ipf["induced_pair_f1"],
        "induced_pair_precision": ipf["induced_pair_precision"],
        "induced_pair_recall": ipf["induced_pair_recall"],
    }
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 指标已保存: {args.output_file}")


if __name__ == "__main__":
    main()
