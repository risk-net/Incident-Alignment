#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sparse complete-linkage 聚类模块（面向全量数据的可扩展实现）。

在不显式构建 n×n 稠密距离矩阵的前提下，实现与 complete-linkage 数学等价的
聚类逻辑（至多存在 tie-breaking 差异）。

背景
----
全量数据（约 26 万新闻）上，标准 complete-linkage 需要 O(n^2) 的稠密距离矩阵，
不可行。本模块在召回候选边构成的稀疏图上实现 complete-linkage：

    - 已召回且有连续分数的 pair：有真实距离 d = 1 - score；
    - 未召回 pair：视为大距离 M（M > cut_distance，不参与阈值内合并）。

在该假设下，一个簇要形成，必须满足“完全连接”约束：
簇内任意两个节点之间都有候选边，且边距离 <= cut_distance。

与 scipy dense complete-link 在小样本上逐例对比验证过（见 ScientificData-revise/
test_sparse_complete_linkage.py），结果完全一致（至多 tie-breaking 差异）。

仅依赖标准库（heapq / collections / typing），无 numpy / scipy 依赖。
"""

from __future__ import annotations

import heapq
import math
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

# 浮点边界容差：score/距离来自模型或浮点运算，边界判定加容差避免误删
EPS = 1e-12


def _norm_id(x) -> str:
    return str(x).strip()


def score_to_distance(score: float) -> float:
    """DeepWide 同事件分数 -> 距离。分数越大越相似，距离越小越相似。

    假设 score ∈ [0, 1]（概率）。若为 logit，请先 sigmoid 再传入。
    """
    return 1.0 - float(score)


# ── 模块 1: 边预处理 ────────────────────────────────────────────────────────


def _normalize_directed_scores(directed_scores: Dict[Tuple, float]) -> Dict[Tuple[str, str], float]:
    """将 directed_scores 的 key 统一 normalize 成 (str, str)，并去重（保留较高分数）。

    同时过滤无效分数（NaN/Inf），并把 score clamp 到 [0, 1]。
    避免下游用原始 key 做 `(b, a) in directed_scores` 判断时，因 key 类型不一致
    （如 int、带空格字符串）而失效。
    """
    norm: Dict[Tuple[str, str], float] = {}
    for key, score in directed_scores.items():
        if not isinstance(key, (tuple, list)) or len(key) != 2:
            continue
        a = _norm_id(key[0])
        b = _norm_id(key[1])
        if a == b:
            continue
        try:
            s = float(score)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(s):
            continue
        if s < 0.0:
            s = 0.0
        elif s > 1.0:
            s = 1.0
        k = (a, b)
        prev = norm.get(k)
        if prev is None or s > prev:
            norm[k] = s
    return norm


def symmetrize_undirected_scores(
    nodes: Sequence,
    directed_scores: Dict[Tuple, float],
    edge_rule: str,
) -> Tuple[List[str], Dict[str, Dict[str, float]]]:
    """有向分数 -> 无向相似度。

    Parameters
    ----------
    nodes : 节点 id 集合
    directed_scores : {(query_id, cand_id): probability}
    edge_rule : {"mutual", "either"}
        - mutual: 只保留双向都召回的 pair，score = min(s_ab, s_ba)（高纯度）。
        - either: 保留任一方向召回的 pair，score = max(s_ab, s_ba)（高召回）。

    Returns
    -------
    nodes : 排序后的节点 id 列表
    und_scores : {a: {b: similarity}}，双向对称。
    """
    nodes = sorted({_norm_id(x) for x in nodes})
    node_set = set(nodes)
    # 先统一 normalize + 去重，后续所有 `(b, a) in directed_scores` 判断都基于规范 key
    directed_scores = _normalize_directed_scores(directed_scores)
    und_scores: Dict[str, Dict[str, float]] = defaultdict(dict)

    for (a, b), score_ab in directed_scores.items():
        if a not in node_set or b not in node_set:
            continue
        # 每个无向 pair 只处理一次（以 (min, max) 顺序处理）
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


# ── 模块 2: 连通块划分 ──────────────────────────────────────────────────────


def find_connected_components(n: int, edges: Iterable[Tuple[int, int, float]]) -> List[List[int]]:
    """根据保留边找连通块（Union-Find）。edges 可为 list 或生成器（只遍历一次）。"""
    parent = list(range(n))
    rank = [0] * n

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1

    for u, v, _ in edges:
        union(u, v)

    comps: Dict[int, List[int]] = defaultdict(list)
    for i in range(n):
        comps[find(i)].append(i)
    return list(comps.values())


# ── 模块 3: sparse complete-linkage 核心 ────────────────────────────────────


class _State:
    __slots__ = ("a", "b", "maxdist", "active")

    def __init__(self, a: int, b: int, maxdist: float):
        self.a = a
        self.b = b
        self.maxdist = maxdist
        self.active = True


def sparse_complete_linkage_core(
    n: int,
    edges: List[Tuple[int, int, float]],
    cut_distance: float,
):
    """在一个连通块内部运行 sparse complete-linkage。

    Parameters
    ----------
    n : 该连通块内节点数量（局部索引 0..n-1）
    edges : [(i, j, distance)]，i < j，distance <= cut_distance
    cut_distance : 切树阈值

    Returns
    -------
    labels : List[int]，节点局部簇标签
    merge_records : List[(a, b, c, maxdist)]
    stats : dict
    """
    cap = 2 * n  # 簇 id 上界：n + 合并次数 <= 2n
    parent = list(range(cap))
    size = [1] * cap
    active = [False] * cap
    for i in range(n):
        active[i] = True

    # 邻接表按需创建（稀疏图避免预分配 2n 个空 dict）
    adj: Dict[int, Dict[int, int]] = defaultdict(dict)
    states: List[_State] = []
    heap: List[Tuple[float, int, int, int]] = []
    peak_heap_size = 0

    next_cluster_id = n
    next_state_id = 0

    def add_state(a: int, b: int, maxdist: float) -> int:
        nonlocal next_state_id
        sid = next_state_id
        next_state_id += 1
        states.append(_State(a, b, maxdist))
        adj[a][b] = sid
        adj[b][a] = sid
        return sid

    def push_state(a: int, b: int, maxdist: float):
        nonlocal peak_heap_size
        sid = add_state(a, b, maxdist)
        lo, hi = (a, b) if a < b else (b, a)
        heapq.heappush(heap, (maxdist, lo, hi, sid))
        if len(heap) > peak_heap_size:
            peak_heap_size = len(heap)

    # 初始化：每条保留边是一个 complete state；先批量构造 state，再 O(m) heapify
    for u, v, d in edges:
        if d <= cut_distance + EPS:
            sid = add_state(u, v, d)
            lo, hi = (u, v) if u < v else (v, u)
            heap.append((d, lo, hi, sid))
    heapq.heapify(heap)
    peak_heap_size = len(heap)

    merge_records: List[Tuple[int, int, int, float]] = []

    while heap:
        maxdist, _t1, _t2, sid = heapq.heappop(heap)
        if maxdist > cut_distance + EPS:
            break

        st = states[sid]
        if not st.active:
            continue

        a, b = st.a, st.b
        if not active[a] or not active[b]:
            continue

        # 创建新簇
        c = next_cluster_id
        next_cluster_id += 1
        parent[a] = c
        parent[b] = c
        active[a] = False
        active[b] = False
        active[c] = True
        size[c] = size[a] + size[b]
        merge_records.append((a, b, c, maxdist))

        neighbors_a = adj[a]
        neighbors_b = adj[b]

        # 找共同完整邻居：x 与 a、b 都 complete 时，c 与 x 才 complete
        if len(neighbors_a) <= len(neighbors_b):
            small, large = neighbors_a, neighbors_b
        else:
            small, large = neighbors_b, neighbors_a

        common: List[Tuple[int, float]] = []
        for x, sid_small in small.items():
            if x == a or x == b:
                continue
            if not active[x]:
                continue
            sid_large = large.get(x)
            if sid_large is None:
                continue
            st_small = states[sid_small]
            st_large = states[sid_large]
            if not st_small.active or not st_large.active:
                continue
            nm = st_small.maxdist if st_small.maxdist > st_large.maxdist else st_large.maxdist
            if nm <= cut_distance + EPS:
                common.append((x, nm))

        # 删除 a、b 的所有旧 state
        for x, sid in list(neighbors_a.items()):
            states[sid].active = False
            if active[x]:
                adj[x].pop(a, None)
        neighbors_a.clear()
        adj.pop(a, None)

        for x, sid in list(neighbors_b.items()):
            states[sid].active = False
            if active[x]:
                adj[x].pop(b, None)
        neighbors_b.clear()
        adj.pop(b, None)

        # 创建新 complete state
        adj[c] = {}
        for x, nm in common:
            if not active[x]:
                continue
            push_state(c, x, nm)

    # 生成最终标签
    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    labels = [-1] * n
    root_to_label: Dict[int, int] = {}
    cur = 0
    for i in range(n):
        r = find(i)
        if r not in root_to_label:
            root_to_label[r] = cur
            cur += 1
        labels[i] = root_to_label[r]

    stats = {
        "num_initial_nodes": n,
        "num_merges": len(merge_records),
        "num_final_clusters": cur,
        "num_states_created": next_state_id,
        "peak_heap_size": peak_heap_size,
    }
    return labels, merge_records, stats


# ── 高层入口 ────────────────────────────────────────────────────────────────


def build_sparse_complete_link_clusters(
    nodes: Sequence,
    directed_scores: Dict[Tuple, float],
    edge_rule: str,
    threshold: float,
    return_stats: bool = False,
    validate: bool = False,
    max_pairs_to_check: int = 100000,
):
    """基于连续分数的 sparse complete-link 聚类。

    流程：对称化 -> 分数转距离 -> 阈值过滤 -> 连通块 ->
          每块内 sparse complete-linkage -> 全局标签。

    Parameters
    ----------
    nodes : 节点 id 集合
    directed_scores : {(query_id, cand_id): probability}
    edge_rule : {"mutual", "either"}
    threshold : float ∈ (0, 1]，切树阈值（同事件分数下限）
    return_stats : 若为 True，返回 (clusters, stats)，否则只返回 clusters
    validate : 若为 True，在释放 und_scores 前抽样验证每个簇是 clique（仅小样本用）
    max_pairs_to_check : validate 时最多检查的簇内 pair 数

    Returns
    -------
    List[List[str]] 或 (List[List[str]], dict)
    """
    threshold = float(threshold)
    if not 0.0 < threshold <= 1.0:
        raise ValueError(f"threshold must be in (0, 1], got {threshold}")

    cut_distance = 1.0 - threshold

    nodes, und_scores = symmetrize_undirected_scores(nodes, directed_scores, edge_rule)

    n = len(nodes)
    idx: Dict[str, int] = {node: i for i, node in enumerate(nodes)}

    # 全局无向邻接（距离），仅保留 distance <= cut_distance 的边
    gadj: Dict[int, Dict[int, float]] = defaultdict(dict)
    num_edges_retained = 0
    for a in nodes:
        ia = idx[a]
        for b, score in und_scores.get(a, {}).items():
            ib = idx.get(b)
            if ib is None or ia >= ib:
                continue
            d = score_to_distance(score)
            if d <= cut_distance + EPS:
                gadj[ia][ib] = d
                gadj[ib][ia] = d
                num_edges_retained += 1

    # idx 仅用于构建 gadj，构建完即可释放
    del idx

    # 连通块（直接在 gadj 上 Union-Find，用生成器避免再物化一份边表）
    components = find_connected_components(
        n,
        ((ia, ib, d) for ia, nbrs in gadj.items() for ib, d in nbrs.items() if ia < ib),
    )

    result: List[List[str]] = []
    component_stats = []

    for members in components:
        m = len(members)
        if m == 1:
            result.append([nodes[members[0]]])
            continue

        local_of = {orig: li for li, orig in enumerate(members)}
        local_edges: List[Tuple[int, int, float]] = []
        for orig in members:
            li = local_of[orig]
            for other, d in gadj[orig].items():
                lj = local_of.get(other)
                if lj is None or li >= lj:
                    continue
                local_edges.append((li, lj, d))

        labels, _records, comp_stats = sparse_complete_linkage_core(m, local_edges, cut_distance)
        component_stats.append(comp_stats)

        label_to_members: Dict[int, List[str]] = defaultdict(list)
        for li, lab in enumerate(labels):
            label_to_members[lab].append(nodes[members[li]])
        for c in label_to_members.values():
            result.append(sorted(c))

    # 稳定输出顺序
    result.sort(key=lambda c: (str(c[0]), len(c)))

    # 可选：在释放 und_scores 前抽样验证完全连接约束（仅小样本）
    if validate:
        if not validate_clusters_are_cliques(result, und_scores, threshold, max_pairs_to_check):
            raise RuntimeError("Cluster clique validation failed")

    # 释放字符串邻接表，降低峰值内存（后续只用 int 邻接 gadj）
    del und_scores

    if return_stats:
        cluster_sizes = [len(c) for c in result]
        stats = {
            "num_nodes": n,
            "num_edges_retained": num_edges_retained,
            "num_components": len(components),
            "max_component_size": max((len(c) for c in components), default=0),
            "num_final_clusters": len(result),
            "max_cluster_size": max(cluster_sizes, default=0),
            "total_merges": sum(s["num_merges"] for s in component_stats),
            "total_states_created": sum(s["num_states_created"] for s in component_stats),
            "peak_heap_size": max((s["peak_heap_size"] for s in component_stats), default=0),
        }
        return result, stats
    return result


# ── 验证：最终簇必须满足完全连接（clique）约束 ───────────────────────────────


def validate_clusters_are_cliques(
    clusters: Sequence[Sequence[str]],
    und_scores: Dict[str, Dict[str, float]],
    threshold: float,
    max_pairs_to_check: int = 100000,
) -> bool:
    """验证每个最终簇内任意两点之间都有保留边且距离 <= cut_distance。

    Parameters
    ----------
    clusters : 聚类结果（build_sparse_complete_link_clusters 的输出）
    und_scores : {a: {b: similarity}}，即 symmetrize_undirected_scores 的输出
    threshold : 切树阈值（同事件分数下限）
    max_pairs_to_check : 最多检查的簇内 pair 数（大簇全量检查很贵，抽样即可）

    Returns
    -------
    bool  所有被检查的 pair 都满足完全连接约束
    """
    cut_distance = 1.0 - threshold
    checked = 0
    for cluster in clusters:
        m = len(cluster)
        for i in range(m):
            nbrs = und_scores.get(cluster[i], {})
            for j in range(i + 1, m):
                score = nbrs.get(cluster[j])
                if score is None or 1.0 - float(score) > cut_distance + EPS:
                    return False
                checked += 1
                if checked >= max_pairs_to_check:
                    return True
    return True
