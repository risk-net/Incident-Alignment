#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全量聚类独立入口（与推理解耦）。

从已落盘的 pair_predictions_full.jsonl 读回连续分数，运行 complete-link 聚类
（sparse 或 dense），输出 clusters.json / cluster_assignments.json / clustering_config.json。

与 run_full_inference.py 的关系：
  - run_full_inference.py 只做 pairwise 推理，产出 pair_predictions_full.jsonl；
  - 本脚本只做聚类，从该文件读回分数，不再重复推理。

因此可以反复用不同 threshold / edge_rule / merge_strategy 重跑聚类。

运行方式：
python src/Incident_Align_Method/full_application/run_full_clustering.py \
  --config config/Incident_Align_Method-full_application-config.ini \
  [--threshold 0.84] [--pair-file /path/to/pair_predictions_full.jsonl]
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

from config_utils import DEFAULT_CONFIG_PATH, get_float, load_config, require_section, resolve_path

METHOD_DIR = Path(__file__).resolve().parents[1]
PAIRWISE_DIR = METHOD_DIR / "pairwise_and_clustering"
if str(PAIRWISE_DIR) not in sys.path:
    sys.path.insert(0, str(PAIRWISE_DIR))

from sparse_complete_linkage import build_sparse_complete_link_clusters


def _norm_id(x) -> str:
    return str(x).strip()


def load_pair_scores(pair_pred_file: str, threshold: float):
    """从 pair_predictions_full.jsonl 流式读回连续分数，重建 directed_scores / scored_nodes。

    只保留 probability >= threshold 的 pair：阈值以下的边距离 > cut_distance，
    不会参与 complete-link 合并，视作缺失即可，从而大幅降低全量内存占用。
    """
    directed_scores: Dict[Tuple[str, str], float] = {}
    scored_nodes: Set[str] = set()
    total_pairs = 0
    with open(pair_pred_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            q = _norm_id(rec.get("query", rec.get("query_case_id", "")))
            c = _norm_id(rec.get("candidate", rec.get("cand_case_id", "")))
            if not q or not c or q == c:
                continue
            prob = float(rec.get("probability", 0.0))
            scored_nodes.add(q)
            scored_nodes.add(c)
            total_pairs += 1
            if prob >= threshold:
                directed_scores[(q, c)] = prob
    return directed_scores, scored_nodes, total_pairs


def materialize_clusters(clusters: List[List[str]]) -> Tuple[Dict[str, int], List[Dict[str, object]]]:
    assignments: Dict[str, int] = {}
    cluster_rows: List[Dict[str, object]] = []
    for cluster_id, case_ids in enumerate(clusters):
        ordered_case_ids = sorted(_norm_id(cid) for cid in case_ids)
        for cid in ordered_case_ids:
            assignments[cid] = cluster_id
        cluster_rows.append(
            {
                "cluster_id": cluster_id,
                "size": len(ordered_case_ids),
                "case_ids": ordered_case_ids,
            }
        )
    return assignments, cluster_rows


def main():
    parser = argparse.ArgumentParser(description="全量聚类独立入口（复用已落盘的 pairwise 推理结果）")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH), help="INI 配置文件路径")
    parser.add_argument("--threshold", type=float, default=None, help="覆盖配置里的聚类阈值")
    parser.add_argument("--pair-file", type=str, default=None, help="覆盖 pair_predictions_full.jsonl 路径")
    args = parser.parse_args()

    cfg = load_config(args.config)
    section = require_section(cfg, "RunFullInference")

    artifacts_root = resolve_path(section.get("artifacts_root", ""))
    output_dir = resolve_path(section.get("output_dir", os.path.join(artifacts_root, "full_inference")))
    threshold = get_float(section, "threshold", 0.5) if args.threshold is None else float(args.threshold)
    edge_rule = (section.get("edge_rule", "mutual") or "mutual").strip()
    merge_strategy = (section.get("merge_strategy", "sparse_complete_link") or "sparse_complete_link").strip()

    if not 0.0 < threshold <= 1.0:
        raise ValueError(f"threshold 必须在 (0,1]，得到 {threshold}")
    if edge_rule not in {"either", "mutual"}:
        raise ValueError(f"不支持的 edge_rule: {edge_rule}")
    if merge_strategy not in {"complete_link", "sparse_complete_link"}:
        raise ValueError(f"不支持的 merge_strategy: {merge_strategy}")

    pair_pred_file = Path(args.pair_file) if args.pair_file else Path(output_dir) / "pair_predictions_full.jsonl"
    if not pair_pred_file.is_file():
        raise FileNotFoundError(
            f"pair 推理结果不存在: {pair_pred_file}. 请先运行 run_full_inference.py 完成推理。"
        )

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    cluster_file = output_root / "clusters.json"
    assignment_file = output_root / "cluster_assignments.json"
    config_file = output_root / "clustering_config.json"

    print("🚀 开始全量聚类")
    print(f"   - pair_file: {pair_pred_file}")
    print(f"   - threshold: {threshold}")
    print(f"   - edge_rule: {edge_rule}")
    print(f"   - merge_strategy: {merge_strategy}")

    print("\n📂 从 pair 推理结果读回连续分数...")
    directed_scores, scored_nodes, total_pairs = load_pair_scores(str(pair_pred_file), threshold)
    if not scored_nodes:
        raise ValueError("pair 推理结果里没有可用节点，请检查 pair_predictions_full.jsonl 是否有效")
    print(f"   - 读回 pair 数: {total_pairs}")
    print(f"   - 参与聚类节点数: {len(scored_nodes)}")
    print(f"   - 保留边数(prob>=threshold): {len(directed_scores)}")

    print("\n🔗 进行图聚类...")
    sparse_stats = None
    if merge_strategy == "complete_link":
        from graph_clustering import build_complete_link_clusters  # 延迟导入，避免 torch 依赖
        print("   使用 scipy agglomerative complete-link (稠密，仅适合小数据)")
        clusters = build_complete_link_clusters(
            nodes=sorted(scored_nodes),
            directed_scores=directed_scores,
            edge_rule=edge_rule,
            threshold=threshold,
        )
    else:  # sparse_complete_link
        print("   使用 sparse complete-link (稀疏候选图，适合全量数据)")
        clusters, sparse_stats = build_sparse_complete_link_clusters(
            nodes=sorted(scored_nodes),
            directed_scores=directed_scores,
            edge_rule=edge_rule,
            threshold=threshold,
            return_stats=True,
        )

    assignments, cluster_rows = materialize_clusters(clusters)
    cluster_sizes = sorted((r["size"] for r in cluster_rows), reverse=True)
    max_cluster_size = cluster_sizes[0] if cluster_sizes else 0

    with open(cluster_file, "w", encoding="utf-8") as f:
        json.dump(cluster_rows, f, ensure_ascii=False, indent=2)
    with open(assignment_file, "w", encoding="utf-8") as f:
        json.dump(assignments, f, ensure_ascii=False, indent=2, sort_keys=True)

    clustering_config = {
        "config_path": str(Path(args.config).resolve()),
        "pair_pred_file": str(pair_pred_file.resolve()),
        "output_dir": output_dir,
        "clustering": {
            "threshold": threshold,
            "edge_rule": edge_rule,
            "merge_strategy": merge_strategy,
        },
        "stats": {
            "total_pairs_read": total_pairs,
            "total_scored_nodes": len(scored_nodes),
            "predicted_edges": len(directed_scores),
            "predicted_clusters": len(cluster_rows),
            "max_cluster_size": max_cluster_size,
            "top_cluster_sizes": cluster_sizes[:10],
            # sparse 专属监控（仅 sparse 分支有值）
            "num_edges_retained": sparse_stats["num_edges_retained"] if sparse_stats is not None else None,
            "num_components": sparse_stats["num_components"] if sparse_stats is not None else None,
            "max_component_size": sparse_stats["max_component_size"] if sparse_stats is not None else None,
            "total_states_created": sparse_stats["total_states_created"] if sparse_stats is not None else None,
            "peak_heap_size": sparse_stats["peak_heap_size"] if sparse_stats is not None else None,
        },
    }
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(clustering_config, f, ensure_ascii=False, indent=2)

    print("\n📊 聚类结果统计:")
    print(f"   - 参与聚类节点数: {len(scored_nodes)}")
    print(f"   - 保留边数(>=threshold): {len(directed_scores)}")
    print(f"   - 聚类簇数: {len(cluster_rows)}")
    print(f"   - 最大簇大小: {max_cluster_size}")
    print(f"   - Top10 簇大小: {cluster_sizes[:10]}")
    if sparse_stats is not None:
        print(f"   - 保留无向边数: {sparse_stats['num_edges_retained']}")
        print(f"   - 连通块数: {sparse_stats['num_components']}")
        print(f"   - 最大连通块大小: {sparse_stats['max_component_size']}")
        print(f"   - total states: {sparse_stats['total_states_created']}")
        print(f"   - peak heap size: {sparse_stats['peak_heap_size']}")
    print(f"\n💾 结果已保存到: {output_root}")
    print(f"   - {cluster_file}")
    print(f"   - {assignment_file}")
    print(f"   - {config_file}")


if __name__ == "__main__":
    main()
