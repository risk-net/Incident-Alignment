#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import defaultdict

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from pairwise_data_io import norm_id
from pairwise_model import predict_probs

try:
    from scipy.optimize import linear_sum_assignment

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


def build_pred_clusters_sparse(nodes, directed_pred, edge_rule="mutual", merge_strategy="closure", k_core=2, k_attach=2, missing_policy="ignore"):
    nodes = [norm_id(x) for x in nodes]
    node_set = set(nodes)
    idx = {cid: i for i, cid in enumerate(nodes)}
    und = defaultdict(set)

    if edge_rule == "either":
        for (a, b), y in directed_pred.items():
            if y != 1:
                continue
            a = norm_id(a)
            b = norm_id(b)
            if a in node_set and b in node_set and a != b:
                und[a].add(b)
                und[b].add(a)
    elif edge_rule == "mutual":
        for (a, b), y in directed_pred.items():
            if y != 1:
                continue
            a = norm_id(a)
            b = norm_id(b)
            if a == b or a not in node_set or b not in node_set:
                continue
            if directed_pred.get((b, a), 0) == 1:
                und[a].add(b)
                und[b].add(a)
    else:
        raise ValueError(f"Unknown edge_rule={edge_rule}")

    if merge_strategy == "closure":
        dsu = DSU(len(nodes))
        for a, nbrs in und.items():
            ia = idx[a]
            for b in nbrs:
                dsu.union(ia, idx[b])
        comp = defaultdict(list)
        for cid in nodes:
            comp[dsu.find(idx[cid])].append(cid)
        return list(comp.values())

    if merge_strategy == "k_support":
        deg = {a: len(und[a]) for a in nodes}
        core_edges = []
        for a in nodes:
            if deg[a] < k_core:
                continue
            for b in und[a]:
                if deg.get(b, 0) >= k_core and idx[a] < idx[b]:
                    core_edges.append((a, b))

        dsu = DSU(len(nodes))
        for a, b in core_edges:
            dsu.union(idx[a], idx[b])

        root2members = defaultdict(list)
        for cid in nodes:
            root2members[dsu.find(idx[cid])].append(cid)

        core_roots = set()
        for root, members in root2members.items():
            if len(members) >= 2 and any(deg.get(m, 0) >= k_core for m in members):
                core_roots.add(root)

        core_rep_idx = {}
        for root in core_roots:
            rep_cid = root2members[root][0]
            core_rep_idx[root] = idx[rep_cid]

        for cid in nodes:
            root = dsu.find(idx[cid])
            if root in core_roots:
                continue

            best_root = None
            best_sup = -1
            nbrs = und[cid]

            for core_root in core_roots:
                members = root2members[core_root]
                sup = sum(1 for m in members if m in nbrs)
                if sup > best_sup:
                    best_sup = sup
                    best_root = core_root

            if best_root is not None and best_sup >= k_attach:
                dsu.union(idx[cid], core_rep_idx[best_root])

        comp = defaultdict(list)
        for cid in nodes:
            comp[dsu.find(idx[cid])].append(cid)
        return list(comp.values())

    if merge_strategy == "complete_link":
        clusters = []
        for cid in nodes:
            placed = False
            clusters.sort(key=lambda x: len(x), reverse=True)
            nbrs = und[cid]
            for cl in clusters:
                if all(m in nbrs for m in cl):
                    cl.append(cid)
                    placed = True
                    break
            if not placed:
                clusters.append([cid])
        return clusters

    raise ValueError(f"Unknown merge_strategy={merge_strategy}")


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


def build_directed_pred(qc_list, probs, thr):
    out = {}
    threshold = float(thr)
    for (q, c), p in zip(qc_list, probs):
        out[(norm_id(q), norm_id(c))] = 1 if float(p) >= threshold else 0
    return out


def grid_search_on_dev(probs, y_true, qc_list, dev_case_ids, gold_clusters_dev, edge_rule_grid, merge_strategy_grid, k_core_grid, k_attach_grid, missing_policy, threshold_grid_step=0.01):
    step = float(threshold_grid_step)
    if step <= 0 or step > 1:
        raise ValueError(f"threshold_grid_step must be in (0,1], got {threshold_grid_step}")

    thr_values = np.arange(0.0, 1.0 + 1e-9, step, dtype=np.float32)

    best = {
        "score": -1.0,
        "edge_rule": None,
        "merge_strategy": None,
        "k_core": None,
        "k_attach": None,
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
        for merge_strategy in merge_strategy_grid:
            if merge_strategy == "k_support":
                k_core_list = k_core_grid
                k_attach_list = k_attach_grid
            else:
                k_core_list = [0]
                k_attach_list = [0]

            for k_core in k_core_list:
                for k_attach in k_attach_list:
                    for thr in thr_values:
                        directed = build_directed_pred(qc_list, probs, float(thr))
                        pred_clusters = build_pred_clusters_sparse(
                            nodes=dev_case_ids,
                            directed_pred=directed,
                            edge_rule=edge_rule,
                            merge_strategy=merge_strategy,
                            k_core=int(k_core),
                            k_attach=int(k_attach),
                            missing_policy=missing_policy,
                        )

                        sym = hungarian_event_symmetric_macro_f1(gold_clusters_dev, pred_clusters)
                        macro_f1 = sym["gold_to_pred_macro_f1"]

                        if float(macro_f1) > float(best["score"]):
                            best.update({
                                "score": float(macro_f1),
                                "edge_rule": edge_rule,
                                "merge_strategy": merge_strategy,
                                "k_core": int(k_core),
                                "k_attach": int(k_attach),
                                "threshold": float(thr),
                                "num_pred_clusters": int(len(pred_clusters)),
                                "event_macro_f1_hungarian_pred_to_gold": float(sym["pred_to_gold_macro_f1"]),
                                "event_macro_f1_hungarian_symmetric": float(sym["symmetric_macro_f1"]),
                            })

    if best["threshold"] is None:
        best["threshold"] = 0.5

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
    missing_policy,
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
    thr = float(best_cfg["threshold"])
    y_pred = (probs >= thr).astype(np.int32)

    directed = {(norm_id(q), norm_id(c)): int(yp) for (q, c), yp in zip(qc_list, y_pred.tolist())}
    pred_clusters = build_pred_clusters_sparse(
        nodes=split_case_ids,
        directed_pred=directed,
        edge_rule=best_cfg["edge_rule"],
        merge_strategy=best_cfg["merge_strategy"],
        k_core=int(best_cfg["k_core"]),
        k_attach=int(best_cfg["k_attach"]),
        missing_policy=missing_policy,
    )

    sym_report = hungarian_event_symmetric_macro_f1(gold_clusters, pred_clusters)
    event_macro_f1 = sym_report["gold_to_pred_macro_f1"]

    metrics = {
        "edge_rule": best_cfg["edge_rule"],
        "merge_strategy": best_cfg["merge_strategy"],
        "k_core": int(best_cfg["k_core"]),
        "k_attach": int(best_cfg["k_attach"]),
        "threshold": thr,
        "event_macro_f1_hungarian": float(event_macro_f1),
        "event_macro_f1_hungarian_pred_to_gold": float(sym_report["pred_to_gold_macro_f1"]),
        "event_macro_f1_hungarian_symmetric": float(sym_report["symmetric_macro_f1"]),
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
