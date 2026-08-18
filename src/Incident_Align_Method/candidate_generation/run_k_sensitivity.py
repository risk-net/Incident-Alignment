#!/usr/bin/env python3
"""
K 敏感性分析：评估不同 TopK 召回参数下的候选覆盖率。
在已有 recall.jsonl 上分析，不需要重跑全流程。

使用方法:
  python src/Incident_Align_Method/candidate_generation/run_k_sensitivity.py \
      --recall_file outputs/recall.jsonl \
      --structure_file data/eval_structure.json \
      --output_dir outputs/metrics/k_sensitivity

输出:
  - k_sensitivity_summary.json: 每个 K 的每路/融合候选召回率
  - k_sensitivity_plot.json: 直接可画图的序列数据
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path


def norm_id(x):
    return str(x).strip()


def load_ground_truth(structure_file):
    """加载 gold standard: {query_case_id: {gt_case_ids}}"""
    with open(structure_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    case_to_incidents = defaultdict(list)
    incident_to_cases = {}
    for ev in data["events"]:
        inc_id = norm_id(ev["incident_id"])
        ids = [norm_id(x) for x in ev["ids"]]
        incident_to_cases[inc_id] = ids
        for cid in ids:
            case_to_incidents[cid].append(inc_id)

    # Build query → gold set
    query2gt = {}
    for inc_id, ids in incident_to_cases.items():
        for q in ids:
            gt = set()
            for c in ids:
                if c != q:
                    gt.add(c)
            query2gt[q] = gt

    return query2gt, incident_to_cases, case_to_incidents


def load_recall(recall_file):
    """加载召回结果，保持 per-candidate 的 text/event/fuse score"""
    queries = []
    with open(recall_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            q = norm_id(obj["query_case_id"])
            candidates = obj.get("candidates", [])
            packed = []
            for c in candidates:
                cid = norm_id(c["case_id"])
                st = c.get("score_text", None)
                se = c.get("score_event", None)
                sf = c.get("score_fuse", None)
                packed.append({
                    "case_id": cid,
                    "score_text": float(st) if st is not None else 0.0,
                    "score_event": float(se) if se is not None else 0.0,
                    "score_fuse": float(sf) if sf is not None else 0.0,
                })
            queries.append({"query": q, "candidates": packed})
    return queries


def compute_recall_at_k(query_entries, query2gt, K):
    """计算给定 K 的每路候选召回率"""
    text_hits, event_hits, fuse_hits = 0, 0, 0
    total_gt = 0
    n_queries = 0

    for entry in query_entries:
        q = entry["query"]
        gt = query2gt.get(q, set())
        if not gt:
            continue

        n_queries += 1
        total_gt += len(gt)

        # Sort per-view and take top-K
        text_sorted = sorted(entry["candidates"], key=lambda x: x["score_text"], reverse=True)[:K]
        event_sorted = sorted(entry["candidates"], key=lambda x: x["score_event"], reverse=True)[:K]
        fuse_sorted = sorted(entry["candidates"], key=lambda x: x["score_fuse"], reverse=True)[:K]

        text_set = {c["case_id"] for c in text_sorted}
        event_set = {c["case_id"] for c in event_sorted}
        fuse_set = {c["case_id"] for c in fuse_sorted}

        text_hits += len(text_set & gt)
        event_hits += len(event_set & gt)
        fuse_hits += len(fuse_set & gt)

    if total_gt == 0:
        return {"text_recall": 0, "event_recall": 0, "fuse_recall": 0, "n_gt": 0, "n_queries": 0}

    return {
        "K": K,
        "text_recall": round(text_hits / total_gt, 6),
        "event_recall": round(event_hits / total_gt, 6),
        "fuse_recall": round(fuse_hits / total_gt, 6),
        "text_hits": text_hits,
        "event_hits": event_hits,
        "fuse_hits": fuse_hits,
        "total_gt_pairs": total_gt,
        "n_queries": n_queries,
    }


def main():
    parser = argparse.ArgumentParser(description="K 敏感性分析")
    parser.add_argument("--recall_file", required=True, help="recall.jsonl 路径")
    parser.add_argument("--structure_file", required=True, help="eval_structure.json 路径")
    parser.add_argument("--output_dir", default="outputs/metrics/k_sensitivity", help="输出目录")
    parser.add_argument("--k_values", default="5,10,20,50,100,150", help="逗号分隔的 K 值列表")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    K_VALUES = [int(k) for k in args.k_values.split(",")]

    print(f"[K Sensitivity] Loading recall from {args.recall_file}")
    queries = load_recall(args.recall_file)

    print(f"[K Sensitivity] Loading ground truth from {args.structure_file}")
    query2gt, _, _ = load_ground_truth(args.structure_file)

    print(f"[K Sensitivity] Analyzing K ∈ {K_VALUES}")
    results = []
    for K in K_VALUES:
        r = compute_recall_at_k(queries, query2gt, K)
        results.append(r)
        print(f"  K={K:>4}: text_recall={r['text_recall']:.4f}  "
              f"event_recall={r['event_recall']:.4f}  fuse_recall={r['fuse_recall']:.4f}")

    # 输出
    summary = {
        "recall_file": str(Path(args.recall_file).resolve()),
        "structure_file": str(Path(args.structure_file).resolve()),
        "results": results,
    }

    summary_path = output_dir / "k_sensitivity_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n[K Sensitivity] Summary → {summary_path}")

    # 画图用的轻量数据
    plot_data = {
        "K": K_VALUES,
        "text_recall": [r["text_recall"] for r in results],
        "event_recall": [r["event_recall"] for r in results],
        "fuse_recall": [r["fuse_recall"] for r in results],
    }
    plot_path = output_dir / "k_sensitivity_plot.json"
    with open(plot_path, "w", encoding="utf-8") as f:
        json.dump(plot_data, f, ensure_ascii=False, indent=2)
    print(f"[K Sensitivity] Plot data → {plot_path}")


if __name__ == "__main__":
    main()
