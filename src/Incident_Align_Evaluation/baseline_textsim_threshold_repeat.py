#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Baseline (Text-Embedding) with Stability Evaluation:

1) Load text embeddings (emb_text.npy or emb_txt.npy) for all case_ids in eval_structure.
2) Load cached cosine similarity matrix (memmap). If missing -> compute once and cache.
   Cache validation is strict with ids_hash & valid_hash to avoid mismatch reuse.
3) Repeat R times:
   - split incidents into DEV/TEST (optionally train_ratio=0)
   - DEV grid-search ONLY threshold, metric = event-level Hungarian Macro-F1
   - TEST evaluate with best DEV threshold
   - report pairwise diagnostics on all pairs inside DEV/TEST
4) Save per-run metrics + summary (mean/std, quantiles).

Usage:
python baseline_textsim_threshold_repeat.py \
  --structure_file data/eval_structure.json \
  --embeddings_dir embeddings \
  --output_dir outputs/event_align_evaluation/textsim_threshold_repeat \
  --threshold_grid_step 0.01 \
  --train_ratio 0.0 \
  --dev_ratio 0.2 \
  --repeat 10

Optional:
  --use_gpu_for_sim    (only affects first run if sim cache not exists)
"""

import json
import random
import argparse
import hashlib
from pathlib import Path
from collections import defaultdict

import numpy as np
from tqdm import tqdm

import torch
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

try:
    from scipy.optimize import linear_sum_assignment
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parents[1]
DEFAULT_STRUCTURE_FILE = PROJECT_ROOT / "data" / "eval_structure.json"
DEFAULT_EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "Event_Align_Evaluation" / "textsim_threshold_repeat"


# -------------------------
# Utils
# -------------------------
def norm_id(x) -> str:
    return str(x).strip()

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def l2_normalize_np(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, eps)
    return x / n

def human_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    v = float(n)
    while v >= 1024 and i < len(units) - 1:
        v /= 1024
        i += 1
    return f"{v:.2f}{units[i]}"

def sha1_str(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def sha1_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()

def quantiles(x, qs=(0.1, 0.5, 0.9)):
    x = np.asarray(x, dtype=np.float64)
    return {f"q{int(q*100):02d}": float(np.quantile(x, q)) for q in qs}


# -------------------------
# DSU
# -------------------------
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


# -------------------------
# Load eval structure
# -------------------------
def load_eval_structure(structure_file: str):
    with open(structure_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    incident2caseids = {}
    caseid2incident = {}
    all_ids = set()

    for ev in data["events"]:
        inc = norm_id(ev["incident_id"])
        ids = [norm_id(x) for x in ev["ids"]]
        incident2caseids[inc] = ids
        for cid in ids:
            caseid2incident[cid] = inc
            all_ids.add(cid)

    return incident2caseids, caseid2incident, sorted(all_ids)

def split_incidents(incident_ids, seed=42, train_ratio=0.0, dev_ratio=0.2):
    ids = list(incident_ids)
    rnd = random.Random(seed)
    rnd.shuffle(ids)
    n = len(ids)
    n_train = int(n * train_ratio)
    n_dev = int(n * dev_ratio)
    train = set(ids[:n_train])
    dev = set(ids[n_train:n_train + n_dev])
    test = set(ids[n_train + n_dev:])
    return train, dev, test

def case_ids_in_split(incident2caseids, inc_set):
    s = set()
    for inc, ids in incident2caseids.items():
        if inc in inc_set:
            for x in ids:
                s.add(norm_id(x))
    return sorted(s)

def incident_to_gold_clusters_filtered(incident2caseids, inc_set, allowed_case_ids_set):
    """
    IMPORTANT FIX:
    Gold clusters are filtered to nodes that are present in this split AND embedding-valid/available.
    Empty clusters are dropped.
    """
    gold = []
    for inc, ids in incident2caseids.items():
        if inc not in inc_set:
            continue
        uniq, seen = [], set()
        for x in ids:
            x = norm_id(x)
            if x not in allowed_case_ids_set:
                continue
            if x not in seen:
                seen.add(x)
                uniq.append(x)
        if len(uniq) > 0:
            gold.append(uniq)
    return gold


# -------------------------
# Load text embeddings (compatible with emb_txt / emb_text)
# -------------------------
def load_text_embeddings_for_eval(embeddings_dir: str, eval_case_ids: list):
    """
    Expected files in embeddings_dir:
      - case_ids.txt
      - emb_text.npy or emb_txt.npy
      - valid_mask_text.npy or valid_mask_txt.npy
    """
    emb_dir = Path(embeddings_dir)
    case_ids_path = emb_dir / "case_ids.txt"
    if not case_ids_path.exists():
        raise FileNotFoundError(f"Missing {case_ids_path}")

    all_case_ids = [norm_id(x) for x in case_ids_path.read_text(encoding="utf-8").splitlines() if x.strip()]
    cid2idx = {cid: i for i, cid in enumerate(all_case_ids)}

    cand_emb = [emb_dir / "emb_text.npy", emb_dir / "emb_txt.npy"]
    cand_msk = [emb_dir / "valid_mask_text.npy", emb_dir / "valid_mask_txt.npy"]

    emb_path = next((p for p in cand_emb if p.exists()), None)
    msk_path = next((p for p in cand_msk if p.exists()), None)
    if emb_path is None or msk_path is None:
        raise FileNotFoundError(f"Missing text embedding files. Need one of {cand_emb} and one of {cand_msk}")

    emb = np.load(emb_path, mmap_mode="r")
    msk = np.load(msk_path, mmap_mode="r")
    if emb.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got {emb.shape}")

    D = int(emb.shape[1])
    X = np.zeros((len(eval_case_ids), D), dtype=np.float32)
    valid = np.zeros((len(eval_case_ids),), dtype=np.bool_)
    ids_used = [norm_id(cid) for cid in eval_case_ids]

    missing_ids = 0
    invalid_ids = 0

    for i, cid in enumerate(eval_case_ids):
        idx = cid2idx.get(norm_id(cid), None)
        if idx is None:
            missing_ids += 1
            continue
        ok = bool(msk[idx])
        if not ok:
            invalid_ids += 1
            continue
        X[i] = np.asarray(emb[idx], dtype=np.float32)
        valid[i] = True

    print(f"[Embeddings] view=text | file={emb_path.name} | N(eval)={len(eval_case_ids)} | D={D}")
    print(f"[Embeddings] missing_in_case_ids.txt={missing_ids} | invalid_mask={invalid_ids} | valid={int(valid.sum())}")

    return ids_used, X, valid, emb_path.name, msk_path.name


# -------------------------
# Similarity matrix compute/load (memmap cache)
# -------------------------
def compute_or_load_similarity(
    X: np.ndarray,
    valid: np.ndarray,
    ids_used: list,
    cache_path: Path,
    use_gpu: bool = True,
    dtype_store: str = "float16",
    block: int = 1024,
    extra_meta: dict | None = None,
):
    """
    cosine similarity for all eval ids. Cache as N x N memmap.
    If cache exists & meta matches strictly -> load; else compute and save.
    """
    N, D = X.shape
    meta_path = cache_path.with_suffix(cache_path.suffix + ".meta.json")

    # strict hashes to avoid mismatch
    ids_hash = sha1_str("\n".join(ids_used))
    valid_hash = sha1_bytes(np.asarray(valid, dtype=np.uint8).tobytes())

    if cache_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        ok = (
            meta.get("N") == N and
            meta.get("D") == D and
            meta.get("dtype_store") == dtype_store and
            meta.get("ids_hash") == ids_hash and
            meta.get("valid_hash") == valid_hash
        )
        if ok:
            print(f"[SimCache] load memmap: {cache_path} ({human_bytes(cache_path.stat().st_size)})")
            S = np.memmap(cache_path, mode="r", dtype=np.dtype(dtype_store), shape=(N, N))
            return S
        else:
            print("[SimCache] cache meta mismatch -> recompute")

    # compute
    Xn = X.copy()
    Xn[~valid] = 0.0
    Xn = l2_normalize_np(Xn)

    print(f"[SimCompute] computing cosine similarity: N={N}, D={D}")
    print(f"[SimCompute] cache -> {cache_path} (dtype={dtype_store})")

    S = np.memmap(cache_path, mode="w+", dtype=np.dtype(dtype_store), shape=(N, N))

    device = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
    print(f"[SimCompute] device={device} | block={block}")

    if device == "cuda":
        X_t = torch.from_numpy(Xn).to(device=device, dtype=torch.float16 if dtype_store == "float16" else torch.float32)
        for i0 in tqdm(range(0, N, block), desc="Sim GPU blocks", dynamic_ncols=True):
            i1 = min(N, i0 + block)
            A = X_t[i0:i1]  # [b, D]
            for j0 in range(0, N, block):
                j1 = min(N, j0 + block)
                B = X_t[j0:j1].T  # [D, b2]
                sim = (A @ B)
                S[i0:i1, j0:j1] = sim.detach().cpu().to(
                    torch.float16 if dtype_store == "float16" else torch.float32
                ).numpy()
        del X_t
    else:
        XnT = Xn.T
        for i0 in tqdm(range(0, N, block), desc="Sim CPU blocks", dynamic_ncols=True):
            i1 = min(N, i0 + block)
            S[i0:i1, :] = (Xn[i0:i1] @ XnT).astype(np.dtype(dtype_store), copy=False)

    # invalid rows/cols -> 0
    if (~valid).any():
        bad = np.where(~valid)[0]
        for idx in bad:
            S[idx, :] = 0
            S[:, idx] = 0

    meta = {
        "N": N,
        "D": D,
        "dtype_store": dtype_store,
        "block": int(block),
        "device_used": device,
        "ids_hash": ids_hash,
        "valid_hash": valid_hash,
    }
    if extra_meta:
        meta.update(extra_meta)

    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    S.flush()
    print("[SimCompute] done.")
    return S


# -------------------------
# Clustering (closure via DSU)
# -------------------------
def build_clusters_closure(sim: np.ndarray, ids: list, thr: float):
    n = len(ids)
    dsu = DSU(n)
    thr = float(thr)

    for i in range(n):
        row = sim[i]
        js = np.where(row[i+1:] >= thr)[0]
        if js.size == 0:
            continue
        js = js + (i + 1)
        for j in js.tolist():
            dsu.union(i, j)

    comp = defaultdict(list)
    for i, cid in enumerate(ids):
        comp[dsu.find(i)].append(cid)
    return list(comp.values())


# -------------------------
# Hungarian macro-F1 on events
# -------------------------
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
    G, P = len(gold_sets), len(pred_sets)
    if P == 0:
        return 0.0, []

    S = np.zeros((G, P), dtype=np.float32)
    for i in range(G):
        for j in range(P):
            S[i, j] = f1_set(gold_sets[i], pred_sets[j])

    if SCIPY_OK:
        row_ind, col_ind = linear_sum_assignment(-S)
        matched = list(zip(row_ind.tolist(), col_ind.tolist()))
    else:
        matched = []
        used_j = set()
        for i in range(G):
            j = int(np.argmax(S[i]))
            if j in used_j:
                continue
            used_j.add(j)
            matched.append((i, j))

    f1s = []
    for i in range(G):
        j = None
        for (ri, cj) in matched:
            if ri == i:
                j = cj
                break
        f1s.append(float(S[i, j]) if j is not None else 0.0)

    return float(np.mean(f1s)), matched
def hungarian_event_symmetric_macro_f1(gold_clusters, pred_clusters):
    """
    对称化诊断指标：
    - gold->pred: 以 gold 事件为基准的一一匹配后 macro-F1（你现有主指标）
    - pred->gold: 以 pred 簇为基准的一一匹配后 macro-F1（诊断：惩罚过碎/冗余簇）
    - symmetric: 二者平均

    返回：
      {
        "gold_to_pred_macro_f1": float,
        "pred_to_gold_macro_f1": float,
        "symmetric_macro_f1": float,
        "matched_gold_to_pred": list[(gi, pj)],
        "matched_pred_to_gold": list[(pi, gj)],
      }
    """
    # gold -> pred（复用你现有函数）
    g2p, matched_g2p = hungarian_event_macro_f1(gold_clusters, pred_clusters)

    # pred -> gold（交换输入即可）
    p2g, matched_p2g = hungarian_event_macro_f1(pred_clusters, gold_clusters)

    sym = 0.5 * (float(g2p) + float(p2g))

    return {
        "gold_to_pred_macro_f1": float(g2p),
        "pred_to_gold_macro_f1": float(p2g),
        "symmetric_macro_f1": float(sym),
        "matched_gold_to_pred": matched_g2p,
        "matched_pred_to_gold": matched_p2g,
    }


# -------------------------
# Pairwise diagnostics on full split pairs
# -------------------------
def pairwise_metrics_from_sim(sim: np.ndarray, ids: list, caseid2incident: dict, thr: float):
    n = len(ids)
    y_true = []
    y_pred = []

    for i in range(n):
        inc_i = caseid2incident.get(ids[i], None)
        row = sim[i]
        for j in range(i + 1, n):
            inc_j = caseid2incident.get(ids[j], None)
            if inc_i is None or inc_j is None:
                continue
            y = 1 if inc_i == inc_j else 0
            yh = 1 if float(row[j]) >= float(thr) else 0
            y_true.append(y)
            y_pred.append(yh)

    if len(y_true) == 0:
        return {
            "num_pairs": 0,
            "pair_accuracy": 0.0,
            "pair_macro_f1": 0.0,
            "pair_f1_pos": 0.0,
            "pair_precision_pos": 0.0,
            "pair_recall_pos": 0.0,
            "num_pred_pos_pairs": 0,
        }

    y_true = np.asarray(y_true, dtype=np.int32)
    y_pred = np.asarray(y_pred, dtype=np.int32)

    return {
        "num_pairs": int(len(y_true)),
        "pair_accuracy": float(accuracy_score(y_true, y_pred)),
        "pair_macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "pair_f1_pos": float(f1_score(y_true, y_pred, pos_label=1)),
        "pair_precision_pos": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "pair_recall_pos": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "num_pred_pos_pairs": int(y_pred.sum()),
    }


# -------------------------
# Grid search on DEV: ONLY threshold
# -------------------------
def grid_search_threshold_only(sim_dev: np.ndarray, dev_ids: list, gold_dev: list,
                               caseid2incident: dict, threshold_grid_step: float):
    thr_values = np.arange(0.0, 1.0 + 1e-9, float(threshold_grid_step), dtype=np.float32)

    best = {
        "score": -1.0,                 # gold->pred（主指标）
        "threshold": None,
        "num_pred_clusters": None,
        # 新增：对称化诊断
        "score_pred_to_gold": None,    # pred->gold
        "score_symmetric": None,       # 0.5*(g2p+p2g)
    }

    pbar = tqdm(thr_values, desc="DEV GridSearch(threshold)", dynamic_ncols=True)
    for thr in pbar:
        pred_clusters = build_clusters_closure(sim_dev, dev_ids, float(thr))

        sym = hungarian_event_symmetric_macro_f1(gold_dev, pred_clusters)
        macro_f1 = sym["gold_to_pred_macro_f1"]  # 主指标仍用 gold->pred

        if float(macro_f1) > float(best["score"]):
            best.update({
                "score": float(macro_f1),
                "threshold": float(thr),
                "num_pred_clusters": int(len(pred_clusters)),
                "score_pred_to_gold": float(sym["pred_to_gold_macro_f1"]),
                "score_symmetric": float(sym["symmetric_macro_f1"]),
            })

        pbar.set_postfix({
            "bestF1": f"{best['score']:.4f}",
            "bestSym": f"{best['score_symmetric']:.4f}" if best["score_symmetric"] is not None else "NA",
            "bestThr": f"{best['threshold']:.2f}" if best["threshold"] is not None else "NA",
            "predK": best["num_pred_clusters"] if best["num_pred_clusters"] is not None else "NA",
        })
    pbar.close()

    best.update(pairwise_metrics_from_sim(sim_dev, dev_ids, caseid2incident, best["threshold"]))
    best["num_gold_events"] = int(len(gold_dev))
    return best



def run_one_split(S_memmap, id2pos, incident2caseids, caseid2incident, all_incidents,
                  seed, train_ratio, dev_ratio, threshold_grid_step):
    _, dev_inc, test_inc = split_incidents(all_incidents, seed=seed, train_ratio=train_ratio, dev_ratio=dev_ratio)

    dev_ids_raw = case_ids_in_split(incident2caseids, dev_inc)
    test_ids_raw = case_ids_in_split(incident2caseids, test_inc)

    # filter to embedding-available positions
    dev_ids = [cid for cid in dev_ids_raw if cid in id2pos]
    test_ids = [cid for cid in test_ids_raw if cid in id2pos]

    dev_idx = np.array([id2pos[cid] for cid in dev_ids], dtype=np.int64)
    test_idx = np.array([id2pos[cid] for cid in test_ids], dtype=np.int64)

    sim_dev = np.asarray(S_memmap[np.ix_(dev_idx, dev_idx)], dtype=np.float32)
    sim_test = np.asarray(S_memmap[np.ix_(test_idx, test_idx)], dtype=np.float32)

    # IMPORTANT FIX: gold filtered by allowed ids in the split
    dev_allowed = set(dev_ids)
    test_allowed = set(test_ids)
    gold_dev = incident_to_gold_clusters_filtered(incident2caseids, dev_inc, dev_allowed)
    gold_test = incident_to_gold_clusters_filtered(incident2caseids, test_inc, test_allowed)

    best_cfg = grid_search_threshold_only(
        sim_dev=sim_dev,
        dev_ids=dev_ids,
        gold_dev=gold_dev,
        caseid2incident=caseid2incident,
        threshold_grid_step=threshold_grid_step,
    )

    thr = float(best_cfg["threshold"])
    pred_clusters_test = build_clusters_closure(sim_test, test_ids, thr)

    sym_test = hungarian_event_symmetric_macro_f1(gold_test, pred_clusters_test)
    event_macro_f1_test = sym_test["gold_to_pred_macro_f1"]

    test_pair_diag = pairwise_metrics_from_sim(sim_test, test_ids, caseid2incident, thr)

    test_metrics = {
        "threshold": thr,
        "event_macro_f1_hungarian": float(event_macro_f1_test),

        # 新增：对称化诊断
        "event_macro_f1_hungarian_pred_to_gold": float(sym_test["pred_to_gold_macro_f1"]),
        "event_macro_f1_hungarian_symmetric": float(sym_test["symmetric_macro_f1"]),

        "num_pred_clusters": int(len(pred_clusters_test)),
        "num_gold_events": int(len(gold_test)),
        **test_pair_diag,
        "num_nodes": int(len(test_ids)),
        "num_incidents": int(len(test_inc)),
    }


    dev_metrics = dict(best_cfg)
    dev_metrics["num_nodes"] = int(len(dev_ids))
    dev_metrics["num_incidents"] = int(len(dev_inc))

    return dev_metrics, test_metrics


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--structure_file", type=str, default=str(DEFAULT_STRUCTURE_FILE))
    parser.add_argument("--embeddings_dir", type=str, default=str(DEFAULT_EMBEDDINGS_DIR))
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.0)
    parser.add_argument("--dev_ratio", type=float, default=0.3)

    parser.add_argument("--threshold_grid_step", type=float, default=0.01)

    parser.add_argument("--repeat", type=int, default=10,
                        help="Repeat random splits with seeds = seed + k.")

    # similarity cache behavior
    parser.add_argument("--sim_cache_name", type=str, default="sim_text_float16.mmap",
                        help="Similarity memmap filename (stored in output_dir). Reused if exists.")
    parser.add_argument("--sim_dtype_store", type=str, default="float16", choices=["float16", "float32"])
    parser.add_argument("--sim_block", type=int, default=1024)
    parser.add_argument("--use_gpu_for_sim", action="store_true",
                        help="If set and CUDA available, compute similarity on GPU blocks (first run only).")

    args = parser.parse_args()
    set_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    incident2caseids, caseid2incident, all_case_ids = load_eval_structure(args.structure_file)
    all_incidents = sorted(list(incident2caseids.keys()))
    print(f"[Data] incidents={len(all_incidents)} | cases(eval_structure)={len(all_case_ids)}")

    # load embeddings for ALL eval ids
    ids_used, X, valid, emb_name, msk_name = load_text_embeddings_for_eval(args.embeddings_dir, all_case_ids)
    id2pos = {cid: i for i, cid in enumerate(ids_used)}

    # similarity cache load-or-compute
    sim_cache_path = out_dir / args.sim_cache_name
    extra_meta = {
        "embedding_view": "text",
        "emb_file": emb_name,
        "mask_file": msk_name,
        "structure_file": str(args.structure_file),
    }
    S = compute_or_load_similarity(
        X=X,
        valid=valid,
        ids_used=ids_used,
        cache_path=sim_cache_path,
        use_gpu=args.use_gpu_for_sim,
        dtype_store=args.sim_dtype_store,
        block=int(args.sim_block),
        extra_meta=extra_meta,
    )

    # repeat splits
    runs = []
    for k in range(int(args.repeat)):
        split_seed = int(args.seed) + k
        print(f"\n[Run {k+1}/{args.repeat}] seed={split_seed} | train/dev/test={args.train_ratio:.2f}/{args.dev_ratio:.2f}/{1-args.train_ratio-args.dev_ratio:.2f}")
        dev_m, test_m = run_one_split(
            S_memmap=S,
            id2pos=id2pos,
            incident2caseids=incident2caseids,
            caseid2incident=caseid2incident,
            all_incidents=all_incidents,
            seed=split_seed,
            train_ratio=float(args.train_ratio),
            dev_ratio=float(args.dev_ratio),
            threshold_grid_step=float(args.threshold_grid_step),
        )
        runs.append({
            "seed": split_seed,
            "best_config_from_dev": dev_m,
            "test_metrics_at_best_config": test_m,
        })
        print(
    f"[Run {k+1}] DEV bestF1={dev_m['score']:.4f} (sym={dev_m.get('score_symmetric', None):.4f}) "
    f"thr={dev_m['threshold']:.2f} | "
    f"TEST F1={test_m['event_macro_f1_hungarian']:.4f} (sym={test_m.get('event_macro_f1_hungarian_symmetric', None):.4f})"
)


    # summary
    test_f1s = [r["test_metrics_at_best_config"]["event_macro_f1_hungarian"] for r in runs]
    dev_f1s = [r["best_config_from_dev"]["score"] for r in runs]
    thrs = [r["best_config_from_dev"]["threshold"] for r in runs]
    test_sym_f1s = [r["test_metrics_at_best_config"]["event_macro_f1_hungarian_symmetric"] for r in runs]
    dev_sym_f1s = [r["best_config_from_dev"]["score_symmetric"] for r in runs]

    summary = {
        "repeat": int(args.repeat),
        "split": {
            "train_ratio": float(args.train_ratio),
            "dev_ratio": float(args.dev_ratio),
            "test_ratio": float(1.0 - args.train_ratio - args.dev_ratio),
        },
        "grid": {"threshold_grid_step": float(args.threshold_grid_step)},
        "sim_cache": str(sim_cache_path),
        "embedding_view": "text",
        "dev_event_macro_f1": {
            "mean": float(np.mean(dev_f1s)),
            "std": float(np.std(dev_f1s, ddof=1)) if len(dev_f1s) > 1 else 0.0,
            **quantiles(dev_f1s),
        },
        "test_event_macro_f1": {
            "mean": float(np.mean(test_f1s)),
            "std": float(np.std(test_f1s, ddof=1)) if len(test_f1s) > 1 else 0.0,
            **quantiles(test_f1s),
        },
        "best_threshold": {
            "mean": float(np.mean(thrs)),
            "std": float(np.std(thrs, ddof=1)) if len(thrs) > 1 else 0.0,
            **quantiles(thrs),
        },
        "dev_event_macro_f1_symmetric": {
        "mean": float(np.mean(dev_sym_f1s)),
        "std": float(np.std(dev_sym_f1s, ddof=1)) if len(dev_sym_f1s) > 1 else 0.0,
        **quantiles(dev_sym_f1s),
        },
        "test_event_macro_f1_symmetric": {
            "mean": float(np.mean(test_sym_f1s)),
            "std": float(np.std(test_sym_f1s, ddof=1)) if len(test_sym_f1s) > 1 else 0.0,
            **quantiles(test_sym_f1s),
        }
    }

    # save
    per_run_path = out_dir / "repeat_runs.json"
    summary_path = out_dir / "repeat_summary.json"
    csv_path = out_dir / "repeat_runs.csv"

    with open(per_run_path, "w", encoding="utf-8") as f:
        json.dump({"runs": runs}, f, indent=2, ensure_ascii=False)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # lightweight CSV
    header = [
    "seed",
    "dev_best_thr",
    "dev_event_f1", "dev_event_f1_pred_to_gold", "dev_event_f1_symmetric",
    "dev_pair_f1_pos", "dev_pair_precision_pos", "dev_pair_recall_pos",
    "dev_pred_clusters", "dev_gold_events", "dev_nodes",

    "test_event_f1", "test_event_f1_pred_to_gold", "test_event_f1_symmetric",
    "test_pair_f1_pos", "test_pair_precision_pos", "test_pair_recall_pos",
    "test_pred_clusters", "test_gold_events", "test_nodes",
]

    rows = []
    for r in runs:
        d = r["best_config_from_dev"]
        t = r["test_metrics_at_best_config"]
        rows.append([
            r["seed"],

            d["threshold"],
            d["score"], d.get("score_pred_to_gold", None), d.get("score_symmetric", None),
            d["pair_f1_pos"], d["pair_precision_pos"], d["pair_recall_pos"],
            d["num_pred_clusters"], d.get("num_gold_events", None), d.get("num_nodes", None),

            t["event_macro_f1_hungarian"], t.get("event_macro_f1_hungarian_pred_to_gold", None), t.get("event_macro_f1_hungarian_symmetric", None),
            t["pair_f1_pos"], t["pair_precision_pos"], t["pair_recall_pos"],
            t["num_pred_clusters"], t.get("num_gold_events", None), t.get("num_nodes", None),
        ])


    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(str(x) for x in row) + "\n")

    print("\n[Save]")
    print(f"  per-run -> {per_run_path}")
    print(f"  summary -> {summary_path}")
    print(f"  csv     -> {csv_path}")
    print("\n[Summary]")
    print(f"  DEV  event_macro_f1: mean={summary['dev_event_macro_f1']['mean']:.4f}, std={summary['dev_event_macro_f1']['std']:.4f}")
    print(f"  TEST event_macro_f1: mean={summary['test_event_macro_f1']['mean']:.4f}, std={summary['test_event_macro_f1']['std']:.4f}")
    print(f"  Best threshold      : mean={summary['best_threshold']['mean']:.4f}, std={summary['best_threshold']['std']:.4f}")
    print(f"  DEV  event_macro_f1_sym: mean={summary['dev_event_macro_f1_symmetric']['mean']:.4f}, std={summary['dev_event_macro_f1_symmetric']['std']:.4f}")
    print(f"  TEST event_macro_f1_sym: mean={summary['test_event_macro_f1_symmetric']['mean']:.4f}, std={summary['test_event_macro_f1_symmetric']['std']:.4f}")


if __name__ == "__main__":
    main()
