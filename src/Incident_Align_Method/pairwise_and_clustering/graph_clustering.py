#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import defaultdict

import numpy as np
from sklearn.metrics import accuracy_score, adjusted_rand_score, f1_score, precision_score, recall_score

from pairwise_data_io import norm_id
from pairwise_model import predict_probs

try:
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial.distance import squareform

    SCIPY_OK = True
except Exception:
    SCIPY_OK = False


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

    if not SCIPY_OK:
        raise ImportError(
            "scipy is required for agglomerative complete-link clustering. "
            "Install it with: pip install scipy"
        )
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

    if SCIPY_OK:
        row_ind, col_ind = linear_sum_assignment(-scores)
        matched = list(zip(row_ind.tolist(), col_ind.tolist()))
    else:
        matched = []
        used_j = set()
        for i in range(g_count):
            j = int(np.argmax(scores[i]))
            if j in used_j:
                continue
            used_j.add(j)
            matched.append((i, j))

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
        "event_macro_f1_hungarian_pred_to_gold": None,
        "event_macro_f1_hungarian_symmetric": None,
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
                float(best.get("event_macro_f1_hungarian_symmetric") or -1.0),
                float(best.get("b_cubed_f1") or -1.0),
                float(best.get("threshold") or -1.0),
            )
            if candidate_key > best_key:
                best.update({
                    "score": float(macro_f1),
                    "edge_rule": edge_rule,
                    "threshold": float(thr),
                    "num_pred_clusters": int(len(pred_clusters)),
                    "event_macro_f1_hungarian_pred_to_gold": float(sym["pred_to_gold_macro_f1"]),
                    "event_macro_f1_hungarian_symmetric": float(sym["symmetric_macro_f1"]),
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
        "event_macro_f1_hungarian": float(event_macro_f1),
        "event_macro_f1_hungarian_pred_to_gold": float(sym_report["pred_to_gold_macro_f1"]),
        "event_macro_f1_hungarian_symmetric": float(sym_report["symmetric_macro_f1"]),
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
