#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""事件聚类与评测指标。

本模块提供完整的两类能力：
  * 聚类   build_directed_scores / build_complete_link_clusters
  * 指标   hungarian_* / b_cubed_f1 / clustering_ari / induced_pairwise_f1
  * 选阈值 scan_thresholds / grid_search_on_dev（在 dev 上按聚类指标选最优 tau）
  * 评测   evaluate_with_best_config（用选定 tau 在 test 上聚类并计分）

选阈值与评测都建立在同一套聚类与指标函数之上，因此放在同一模块。
"""

from collections import defaultdict

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import squareform
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    f1_score,
    precision_score,
    recall_score,
)

from pairwise_data_io import norm_id
from pairwise_model import predict_probs



class DSU:
    def __init__(self, n: int):
        self.p = list(range(n))
        self.r = [0] * n

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.r[ra] < self.r[rb]:
            ra, rb = rb, ra
        self.p[rb] = ra
        if self.r[ra] == self.r[rb]:
            self.r[ra] += 1


# ── Continuous-score complete-link (scipy-based) ──────────────────────────


def build_directed_scores(qc_list, probs):
    """保留候选 pair 的连续预测概率。

    Returns
    -------
    dict[tuple[str, str], float]
        {(query_id, cand_id): probability, ...}
    """
    out: dict[tuple[str, str], float] = {}
    for (q, c), p in zip(qc_list, probs):
        q = norm_id(q)
        c = norm_id(c)
        if q == c:
            continue
        key = (q, c)
        score = float(p)
        # 同一个 pair 因双路召回可能重复出现，保留较高分数
        if key not in out or score > out[key]:
            out[key] = score
    return out


def _build_undirected_scores(nodes, directed_scores, edge_rule):
    """将有向概率转换为无向相似度。

    Parameters
    ----------
    nodes : list[str]
    directed_scores : dict
    edge_rule : {"mutual", "either"}

    Returns
    -------
    nodes : list[str]   (with deterministic sort)
    und_scores : dict[str, dict[str, float]]
        {a: {b: similarity, ...}, ...}
    """
    nodes = sorted({norm_id(x) for x in nodes})
    node_set = set(nodes)
    und_scores: dict[str, dict[str, float]] = defaultdict(dict)

    for (a, b), score_ab in directed_scores.items():
        a = norm_id(a)
        b = norm_id(b)
        if a == b or a not in node_set or b not in node_set:
            continue
        # 每个无向 pair 只处理一次
        if a > b and (b, a) in directed_scores:
            continue

        score_ba = directed_scores.get((b, a))

        if edge_rule == "mutual":
            if score_ba is None:
                score = 0.0
            else:
                score = min(float(score_ab), float(score_ba))
        elif edge_rule == "either":
            score = max(
                float(score_ab),
                float(score_ba) if score_ba is not None else 0.0,
            )
        else:
            raise ValueError(f"Unknown edge_rule={edge_rule}")

        if score > 0.0:
            u, v = (a, b) if a < b else (b, a)
            und_scores[u][v] = score
            und_scores[v][u] = score

    return nodes, und_scores


def build_complete_link_clusters(nodes, directed_scores, edge_rule, threshold):
    """基于连续分数的 agglomerative complete-link 聚类。

    先按 threshold 过滤出连通分量，再在每个分量内用 scipy 标准
    complete-link 建树并以相同阈值切树。

    Parameters
    ----------
    nodes : list[str]
    directed_scores : dict
    edge_rule : str
    threshold : float  ∈ [0, 1]

    Returns
    -------
    list[list[str]]  每个子列表是一个事件簇
    """
    threshold = float(threshold)

    if not 0.0 < threshold <= 1.0:
        raise ValueError(f"threshold must be in (0, 1], got {threshold}")

    nodes, und_scores = _build_undirected_scores(nodes, directed_scores, edge_rule)
    n_total = len(nodes)
    idx: dict[str, int] = {node: i for i, node in enumerate(nodes)}

    # 1) 阈值图连通分量
    dsu = DSU(n_total)
    for a in nodes:
        ia = idx[a]
        for b, score in und_scores.get(a, {}).items():
            ib = idx.get(b)
            if ib is None:
                continue
            if ia < ib and score >= threshold:
                dsu.union(ia, ib)

    components: dict[int, list[str]] = defaultdict(list)
    for node in nodes:
        components[dsu.find(idx[node])].append(node)

    result: list[list[str]] = []

    # 2) 每个分量内独立 complete-link
    for members in components.values():
        members_sorted = sorted(members)
        m = len(members_sorted)

        if m == 1:
            result.append(members_sorted)
            continue

        local_idx = {node: i for i, node in enumerate(members_sorted)}

        # 默认 distance = 1（相似度 = 0）
        distance = np.ones((m, m), dtype=np.float64)
        np.fill_diagonal(distance, 0.0)

        for a in members_sorted:
            i = local_idx[a]
            for b, score in und_scores.get(a, {}).items():
                j = local_idx.get(b)
                if j is None or i >= j:
                    continue
                d = 1.0 - float(score)
                distance[i, j] = d
                distance[j, i] = d

        condensed = squareform(distance, checks=False)
        Z = linkage(condensed, method="complete", optimal_ordering=False)
        labels = fcluster(Z, t=(1.0 - threshold) + 1e-12, criterion="distance")

        label_to_members: dict[int, list[str]] = defaultdict(list)
        for node, label in zip(members_sorted, labels):
            label_to_members[int(label)].append(node)

        result.extend(sorted(c) for c in label_to_members.values())

    # 稳定输出顺序
    result.sort(key=lambda c: (str(c[0]), len(c)))
    return result


def integer_grid(tau_min: float, tau_max: float, grid_step: float):
    """整数步进生成 tau 网格，避免 float32 累积误差丢掉端点。

    例：range=(0.10, 0.95), step=0.01 -> [i/100 for i in range(10, 96)]，共 86 个。

    为什么不用 np.arange(dtype=float32)：实测 step=0.02 时会丢掉 0.95 端点
    （last = 0.9399…），导致搜索范围悄悄变窄。
    """
    step = float(grid_step)
    if step <= 0 or step > 1:
        raise ValueError(f"grid_step must be in (0,1], got {grid_step}")
    if not 0.0 < tau_min < tau_max <= 1.0:
        raise ValueError(
            f"tau range must satisfy 0 < min < max <= 1, got min={tau_min} max={tau_max}"
        )
    # 用整数 hundredths 表达，规避二进制浮点累积
    lo = int(round(tau_min * 100))
    hi = int(round(tau_max * 100))
    st = int(round(step * 100))
    if st <= 0:
        raise ValueError(f"grid_step 必须 >= 0.01，当前 {grid_step}")
    return [i / 100.0 for i in range(lo, hi + 1, st)]


def f1_set(a_set, b_set) -> float:
    inter = len(a_set & b_set)
    denom = len(a_set) + len(b_set)
    if denom == 0:
        return 0.0
    return 2.0 * inter / denom


def hungarian_event_macro_f1(gold_clusters, pred_clusters):
    if len(gold_clusters) == 0:
        return 0.0, []

    gold_sets = [set(map(norm_id, g)) for g in gold_clusters]
    pred_sets = [set(map(norm_id, p)) for p in pred_clusters]
    g_count, p_count = len(gold_sets), len(pred_sets)

    if p_count == 0:
        return 0.0, []

    scores = np.zeros((g_count, p_count), dtype=np.float32)
    for i in range(g_count):
        for j in range(p_count):
            scores[i, j] = f1_set(gold_sets[i], pred_sets[j])

    row_ind, col_ind = linear_sum_assignment(-scores)
    matched = list(zip(row_ind.tolist(), col_ind.tolist()))

    f1s = []
    for i in range(g_count):
        j = None
        for ri, cj in matched:
            if ri == i:
                j = cj
                break
        f1s.append(float(scores[i, j]) if j is not None else 0.0)

    return float(np.mean(f1s)), matched


def hungarian_event_symmetric_macro_f1(gold_clusters, pred_clusters):
    g2p, matched_g2p = hungarian_event_macro_f1(gold_clusters, pred_clusters)
    p2g, matched_p2g = hungarian_event_macro_f1(pred_clusters, gold_clusters)
    sym = 0.5 * (float(g2p) + float(p2g))
    return {
        "gold_to_pred_macro_f1": float(g2p),
        "pred_to_gold_macro_f1": float(p2g),
        "symmetric_macro_f1": float(sym),
        "matched_gold_to_pred": matched_g2p,
        "matched_pred_to_gold": matched_p2g,
    }


def b_cubed_f1(gold_clusters, pred_clusters):
    """B-cubed Precision, Recall, F1 for clustering evaluation."""
    gold_sets = [set(map(norm_id, g)) for g in gold_clusters]
    pred_sets = [set(map(norm_id, p)) for p in pred_clusters]

    # Build per-item gold/pred cluster assignments
    gold_of = {}
    for gs in gold_sets:
        for x in gs:
            gold_of[x] = gs
    pred_of = {}
    for ps in pred_sets:
        for x in ps:
            pred_of[x] = ps

    all_items = set(gold_of.keys()) | set(pred_of.keys())
    if not all_items:
        return {"b_cubed_precision": 0.0, "b_cubed_recall": 0.0, "b_cubed_f1": 0.0}

    prec_sum, rec_sum, n = 0.0, 0.0, 0
    for x in all_items:
        gs = gold_of.get(x, set())
        ps = pred_of.get(x, set())
        if not gs or not ps:
            continue
        n += 1
        prec_sum += len(gs & ps) / len(ps)  # fraction of predicted cluster in gold cluster
        rec_sum += len(gs & ps) / len(gs)    # fraction of gold cluster in predicted cluster

    p = prec_sum / n if n else 0.0
    r = rec_sum / n if n else 0.0
    f1 = 2.0 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"b_cubed_precision": p, "b_cubed_recall": r, "b_cubed_f1": f1}


def clustering_ari(gold_clusters, pred_clusters):
    """Adjusted Rand Index for clustering evaluation."""
    gold_sets = [set(map(norm_id, g)) for g in gold_clusters]
    pred_sets = [set(map(norm_id, p)) for p in pred_clusters]

    all_items = set()
    for gs in gold_sets:
        all_items.update(gs)
    for ps in pred_sets:
        all_items.update(ps)
    items = sorted(all_items)

    gold_labels = []
    pred_labels = []
    gold_map = {}
    pred_map = {}
    for gid, gs in enumerate(gold_sets):
        for x in gs:
            gold_map[x] = gid
    for pid, ps in enumerate(pred_sets):
        for x in ps:
            pred_map[x] = pid

    for x in items:
        gold_labels.append(gold_map.get(x, -1))
        pred_labels.append(pred_map.get(x, -1))

    return float(adjusted_rand_score(gold_labels, pred_labels))


def induced_pairwise_f1(gold_clusters, pred_clusters):
    """Compute pairwise F1 induced from cluster assignments."""
    gold_sets = [set(map(norm_id, g)) for g in gold_clusters]
    pred_sets = [set(map(norm_id, p)) for p in pred_clusters]

    all_items = set()
    for gs in gold_sets:
        all_items.update(gs)
    for ps in pred_sets:
        all_items.update(ps)
    items = sorted(all_items)

    y_true, y_pred = [], []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            # Gold: same cluster = 1
            same_gold = any(items[i] in gs and items[j] in gs for gs in gold_sets)
            same_pred = any(items[i] in ps and items[j] in ps for ps in pred_sets)
            y_true.append(1 if same_gold else 0)
            y_pred.append(1 if same_pred else 0)

    if not y_true:
        return {"induced_pair_precision": 0.0, "induced_pair_recall": 0.0, "induced_pair_f1": 0.0}

    return {
        "induced_pair_precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "induced_pair_recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "induced_pair_f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def grid_search_on_dev(probs, y_true, qc_list, dev_case_ids, gold_clusters_dev, edge_rule_grid, threshold_grid_step=0.01, threshold_min=0.50, threshold_max=0.95):
    step = float(threshold_grid_step)
    if step <= 0 or step > 1:
        raise ValueError(f"threshold_grid_step must be in (0,1], got {threshold_grid_step}")
    if not 0.0 < threshold_min < threshold_max <= 1.0:
        raise ValueError(
            f"threshold range must satisfy 0 < min < max <= 1, "
            f"got min={threshold_min} max={threshold_max}"
        )

    thr_values = np.arange(
        threshold_min, threshold_max + 1e-9, step, dtype=np.float32
    )

    # 预计算连续分数
    directed_scores = build_directed_scores(qc_list, probs)

    best = {
        "score": -1.0,
        "edge_rule": None,
        "threshold": None,
        "num_pred_clusters": None,
        # 键名统一到核心库命名（见 metric_keys.py）
        "hungarian_f1_g2p": None,
        "hungarian_f1_p2g": None,
        "hungarian_f1_symmetric": None,
        "b_cubed_f1": None,
        "pair_macro_f1": None,
        "pair_f1_pos": None,
        "pair_precision_pos": None,
        "pair_recall_pos": None,
        "pair_accuracy": None,
        "num_pred_pos_pairs": None,
        "num_pairs": int(len(y_true)),
    }

    for edge_rule in edge_rule_grid:
        for thr in thr_values:
            pred_clusters = build_complete_link_clusters(
                nodes=dev_case_ids,
                directed_scores=directed_scores,
                edge_rule=edge_rule,
                threshold=float(thr),
            )

            sym = hungarian_event_symmetric_macro_f1(gold_clusters_dev, pred_clusters)
            macro_f1 = sym["gold_to_pred_macro_f1"]
            b3 = b_cubed_f1(gold_clusters_dev, pred_clusters)
            ari = clustering_ari(gold_clusters_dev, pred_clusters)

            # 多指标复合比较：主指标相同则看 symmetric / B-cubed / 更高阈值
            candidate_key = (
                float(macro_f1),
                float(sym["symmetric_macro_f1"]),
                float(b3["b_cubed_f1"]),
                float(thr),          # 指标相同时倾向更高阈值（更保守）
            )
            best_key = (
                float(best["score"]),
                float(best["hungarian_f1_symmetric"] or -1.0),
                float(best.get("b_cubed_f1") or -1.0),
                float(best.get("threshold") or -1.0),
            )
            if candidate_key > best_key:
                best.update({
                    "score": float(macro_f1),
                    "edge_rule": edge_rule,
                    "threshold": float(thr),
                    "num_pred_clusters": int(len(pred_clusters)),
                    "hungarian_f1_g2p": float(macro_f1),
                    "hungarian_f1_p2g": float(sym["pred_to_gold_macro_f1"]),
                    "hungarian_f1_symmetric": float(sym["symmetric_macro_f1"]),
                    "b_cubed_f1": float(b3["b_cubed_f1"]),
                    "b_cubed_precision": float(b3["b_cubed_precision"]),
                    "b_cubed_recall": float(b3["b_cubed_recall"]),
                    "ari": float(ari),
                })

    if best["threshold"] is None:
        best["threshold"] = 0.5

    # 聚类阈值与 pairwise 指标阈值统一记录，但语义不同：
    #  - cluster_threshold: 由事件聚类指标选择，用于切 complete-link 树
    #  - pair_threshold:   同一阈值下报告 pairwise 指标（非独立优化）
    best["cluster_threshold"] = float(best["threshold"])
    best["pair_threshold"] = float(best["threshold"])

    y_pred = (probs >= float(best["threshold"])).astype(np.int32)
    best["pair_macro_f1"] = float(f1_score(y_true, y_pred, average="macro"))
    best["pair_f1_pos"] = float(f1_score(y_true, y_pred, pos_label=1))
    best["pair_precision_pos"] = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    best["pair_recall_pos"] = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    best["pair_accuracy"] = float(accuracy_score(y_true, y_pred))
    best["num_pred_pos_pairs"] = int(y_pred.sum())
    return best



def scan_thresholds(
    nodes,
    directed_scores,
    gold_clusters,
    edge_rule="mutual",
    tau_min=0.10,
    tau_max=0.95,
    grid_step=0.01,
    top_k=10,
    tie_break="repo",
    verbose=True,
):
    """在给定 tau 网格上评估完整聚类指标，返回可落盘的 scan 明细。

    与 `grid_search_on_dev` 的区别：
      * `grid_search_on_dev` 只返回最优一项（供训练脚本内部使用）；
      * `scan_thresholds` 返回**全部候选 tau 的完整指标表**，可序列化为
        `scan_fold0X.json`，用于阈值选定的审计与复现。

    两者使用**完全相同的**复合择优规则与完全相同的聚类/Hungarian 实现：
        key = (hungarian_g2p, hungarian_symmetric, b_cubed_f1, tau)
    即主指标优先、平局看对称 Hungarian、再看 B3、最后倾向更高（更保守）的 tau。

    Parameters
    ----------
    nodes : sequence of str
        评测域内的全部 case id。
    directed_scores : dict[(str,str), float]
        `build_directed_scores` 的产物。
    gold_clusters : sequence of sequence
        金标事件簇。
    tie_break : {"repo", "midpoint"}
        "repo"     与 `grid_search_on_dev` 一致（默认，保证与训练脚本口径相同）；
        "midpoint" 并列最优时取所有并列 tau 的中位数（不偏向边界）。

    Returns
    -------
    dict
        {"mode","edge_rule","n_candidate_taus","n_nodes","best","tie_info","top_k","table"}
        可直接 `json.dumps` 落盘，键名与 `src/revise/scan_fold0X.json` 一致。
    """
    nodes = [str(x).strip() for x in nodes]
    # 无向候选图与 tau 无关，只算一次
    _, und = _build_undirected_scores(nodes, directed_scores, edge_rule)

    cand = sorted(set(integer_grid(tau_min, tau_max, grid_step)))

    rows = []
    for tau in cand:
        pred = build_complete_link_clusters(nodes, directed_scores, edge_rule, tau)
        sym = hungarian_event_symmetric_macro_f1(gold_clusters, pred)
        b3 = b_cubed_f1(gold_clusters, pred)
        sizes = [len(c) for c in pred]
        rows.append({
            "tau": float(tau),
            "num_clusters": len(pred),
            "num_gold_events": len(gold_clusters),
            "max_cluster_frac": (max(sizes) / len(nodes)) if sizes and nodes else 0.0,
            "hungarian_g2p": float(sym["gold_to_pred_macro_f1"]),
            "hungarian_p2g": float(sym["pred_to_gold_macro_f1"]),
            "hungarian_symmetric": float(sym["symmetric_macro_f1"]),
            "b_cubed_f1": float(b3["b_cubed_f1"]),
            "b_cubed_precision": float(b3["b_cubed_precision"]),
            "b_cubed_recall": float(b3["b_cubed_recall"]),
            "ari": float(clustering_ari(gold_clusters, pred)),
        })

    def _key(r):
        return (r["hungarian_g2p"], r["hungarian_symmetric"], r["b_cubed_f1"], r["tau"])

    best = max(rows, key=_key) if rows else None
    ranked = sorted(rows, key=_key, reverse=True)

    tie_info = None
    if rows:
        top_val = best["hungarian_g2p"]
        tied = [r for r in rows if abs(r["hungarian_g2p"] - top_val) <= 1e-12]
        repo_pick = max(tied, key=lambda r: (r["hungarian_symmetric"],
                                             r["b_cubed_f1"], r["tau"]))["tau"]
        mid_pick = sorted(r["tau"] for r in tied)[len(tied) // 2]
        tie_info = {
            "n_tied_at_best": len(tied),
            "tied_taus": [r["tau"] for r in tied],
            "tie_break_rule": tie_break,
            "repo_rule_pick": repo_pick,
            "midpoint_rule_pick": mid_pick,
        }
        if tie_break == "midpoint":
            best = next(r for r in rows if r["tau"] == mid_pick)
        elif tie_break != "repo":
            raise ValueError(f"未知 tie_break={tie_break}")

    if verbose and best is not None:
        print(f"  [scan] tau∈[{tau_min:.2f},{tau_max:.2f}]/step{grid_step}"
              f" 候选 {len(rows)} 个")
        print(f"  [scan] best tau={best['tau']:.4f}  HuG2P={best['hungarian_g2p']:.4f}  "
              f"clusters={best['num_clusters']} (gold={best['num_gold_events']})")
        if tie_info and tie_info["n_tied_at_best"] > 1:
            print(f"  [scan] ⚠ {tie_info['n_tied_at_best']} 个 tau 并列最优："
                  f"{tie_info['tied_taus']}")

    return {
        "mode": "grid",
        "edge_rule": edge_rule,
        "tau_range": [float(tau_min), float(tau_max)],
        "grid_step": float(grid_step),
        "tie_break": tie_break,
        "n_candidate_taus": len(rows),
        "n_nodes": len(nodes),
        "best": best,
        "tie_info": tie_info,
        "top_k": ranked[:top_k],
        "table": rows,
    }



def evaluate_with_best_config(
    model,
    pair_source,
    store,
    case2cats,
    text_fields,
    split_case_ids,
    gold_clusters,
    device,
    best_cfg,
    batch_size=512,
    num_pairs=None,
    architecture_mode=None,
    paper_text_fields=None,
):
    probs, y_true, qc_list = predict_probs(
        model=model,
        pair_source=pair_source,
        store=store,
        case2cats=case2cats,
        text_fields=text_fields,
        device=device,
        desc="Eval Predict",
        batch_size=batch_size,
        num_pairs=num_pairs,
        architecture_mode=architecture_mode,
        paper_text_fields=paper_text_fields,
    )
    thr = float(best_cfg.get("cluster_threshold", best_cfg["threshold"]))

    # 预计算连续分数
    directed_scores = build_directed_scores(qc_list, probs)

    # y_pred 仍用于 pairwise 指标
    y_pred = (probs >= thr).astype(np.int32)

    pred_clusters = build_complete_link_clusters(
        nodes=split_case_ids,
        directed_scores=directed_scores,
        edge_rule=best_cfg["edge_rule"],
        threshold=thr,
    )

    sym_report = hungarian_event_symmetric_macro_f1(gold_clusters, pred_clusters)
    event_macro_f1 = sym_report["gold_to_pred_macro_f1"]
    b3 = b_cubed_f1(gold_clusters, pred_clusters)
    ari = clustering_ari(gold_clusters, pred_clusters)
    ipf = induced_pairwise_f1(gold_clusters, pred_clusters)

    metrics = {
        "edge_rule": best_cfg["edge_rule"],
        "threshold": thr,
        # 键名统一到核心库命名（见 metric_keys.py）；旧键不再写出
        "hungarian_f1_g2p": float(event_macro_f1),
        "hungarian_f1_p2g": float(sym_report["pred_to_gold_macro_f1"]),
        "hungarian_f1_symmetric": float(sym_report["symmetric_macro_f1"]),
        "b_cubed_f1": float(b3["b_cubed_f1"]),
        "b_cubed_precision": float(b3["b_cubed_precision"]),
        "b_cubed_recall": float(b3["b_cubed_recall"]),
        "ari": float(ari),
        "induced_pair_f1": float(ipf["induced_pair_f1"]),
        "induced_pair_precision": float(ipf["induced_pair_precision"]),
        "induced_pair_recall": float(ipf["induced_pair_recall"]),
        "pair_accuracy": float(accuracy_score(y_true, y_pred)),
        "pair_macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "pair_f1_pos": float(f1_score(y_true, y_pred, pos_label=1)),
        "pair_precision_pos": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "pair_recall_pos": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "num_pairs": int(len(y_true)),
        "num_pred_pos_pairs": int(y_pred.sum()),
        "num_pred_clusters": int(len(pred_clusters)),
        "num_gold_events": int(len(gold_clusters)),
    }

    return metrics, pred_clusters, (probs, y_true, qc_list, y_pred)
