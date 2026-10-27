#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""逻辑回归基线：Dual Recall + 17 维手工特征 + Complete-Link + 五折 CV。

目的
----
与主方法（Deep + Wide）构成**唯一变量**对照：只把分类器从
``Deep+Wide`` 换成 ``nn.Linear(17, 1)``（即对 17 个手工特征做逻辑回归），
其余（数据、划分、pair 构建、负采样、选 checkpoint 规则、τ 网格、聚类、
指标）**全部保持一致**。用于回答"深度交互特征建模是否必要"。

17 维特征（``pairwise_model.get_wide_feature_names``）：
  7 个类别一致位 eq_<cat_field>
 10 个文本字段余弦 sim_<text_field>

与主方法对齐的超参（详见 config/Incident_Align_Method-baseline-logistic-regression-config.ini）
------------------------------------------------------------------------
  EPOCHS=8, BATCH_SIZE=512, LR=2e-4, WD=1e-4, NEG_RATIO=4, TOPM_OUT=200
  τ 网格：0.10–0.95 / step 0.01（86 个候选点，与 DeepWide 完全一致）
  选 checkpoint：dev 上 pair-level positive F1（固定阈值 0.5）

流程（每折）
------------
  1) 复用 ``data/english/folds/fold_0X/train/repeat_01/*.jsonl``
     （与 DeepWide 共享同一份 prepared pairs；若不存在则用
      ``pairwise_data_io`` 从 recall 现构建）
  2) 训练 8 epoch，每 epoch 在 dev 上算 pair F1，保留最优
  3) 用最优权重在 dev 上扫 τ（grid_search_on_dev）
  4) 用选中 τ 在 test 上聚类并计算全部指标

用法
----
    # 英文五折（默认）
    python research_pipeline/Incident_Align_Method/baselines/baseline_logistic_regression_5fold.py

    # 中文零样本（用英文五折训练出的 LR 权重直接打中文分数）
    python research_pipeline/Incident_Align_Method/baselines/baseline_logistic_regression_5fold.py \
        --channel chinese --folds 1,2,3,4,5

    # 只跑单折 / 指定设备
    python .../baseline_logistic_regression_5fold.py --folds 1 --device cpu

输出
----
    outputs/benchmarks/logistic_regression/
      ├── fold_0X/result.json          每折 dev 选择 + test 指标
      ├── summary.json                 五折 mean ± std
      └── chinese/                     （--channel chinese 时）
          ├── fold_0X_scores.jsonl     M6 分数 JSONL
          ├── fold_0X_result.json
          └── summary.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

# ------------------------------------------------------------------ 线程数
# 本模型是 nn.Linear(17,1)，矩阵极小。BLAS 默认开几十上百个线程时，
# 线程同步开销远大于并行收益（实测 121 线程只跑出 1.76 核）。
# 必须在 import numpy/torch **之前**设置，否则不生效。
# 可通过环境变量 THREADS=N 覆盖；THREADS=0 表示不限制。
def _setup_threads():
    raw = os.environ.get("THREADS", "1")
    try:
        n = int(raw)
    except ValueError:
        n = 1
    if n <= 0:
        return None
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(k, str(n))
    return n


_THREADS = _setup_threads()

import numpy as np  # noqa: E402

# ------------------------------------------------------------------ 路径
BASE_DIR = Path(__file__).resolve().parents[3]
PAIRWISE_DIR = BASE_DIR / "research_pipeline" / "Incident_Align_Method" / "pairwise_and_clustering"
sys.path.insert(0, str(PAIRWISE_DIR))

from pairwise_data_io import (  # noqa: E402
    CAT_FIELDS,
    TEXT_FIELDS_DEFAULT,
    norm_id,
    load_cases_minimal,
    load_eval_structure,
    load_recall_topk,
    build_all_pairs,
    sample_train_pairs,
    build_cat_vocabs,
    case_ids_in_split,
    incident_to_gold_clusters,
    write_prepared_repeat,
    write_json,
)
from pairwise_model import (  # noqa: E402
    PAPER_TEXT_FIELDS_DEFAULT,
    EmbeddingsStore,
    set_seed,
    get_wide_feature_names,
    build_pair_features_current,
    cosine,
)
from graph_clustering import (  # noqa: E402
    build_complete_link_clusters,
    build_directed_scores,
    hungarian_event_symmetric_macro_f1,
    b_cubed_f1,
    clustering_ari,
    induced_pairwise_f1,
    grid_search_on_dev,
)

# 与 DeepWide 完全一致的口径（改这里会破坏可比性）
# EPOCHS 默认 8 = 主方法同预算；可通过 --epochs 覆盖。
# 注意：LR 只有 17 个参数，在 8 epoch 下常常**尚未收敛**（dev F1 仍在上升），
# 因此补跑"收敛版"时用 --epochs 20，并在报告里注明两套预算。
EPOCHS = 8
BATCH_SIZE = 512
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4
NEG_RATIO = 4
FALLBACK_NEG_PER_EMPTY_Q = 10
TOPM_OUT = 200
PAIR_THRESHOLD = 0.5

# τ 网格：与 DeepWide 一致（0.10–0.95，step 0.01）
THRESHOLD_MIN = 0.10
THRESHOLD_MAX = 0.95
THRESHOLD_STEP = 0.01
EDGE_RULE_GRID = ["mutual"]

TEXT_FIELDS = list(TEXT_FIELDS_DEFAULT) + ["text"]
SIM_FIELDS = ["domain", "event_type", "ai_risk_description", "ai_risk_subtype",
              "affected_actor_subtype", "text"]
PAPER_TEXT_FIELDS = list(PAPER_TEXT_FIELDS_DEFAULT)

LABEL = "Dual Recall + Wide-Only (Logistic Regression) + Complete-Link"


# ------------------------------------------------------------------ 模型
def _build_model(in_dim: int):
    import torch.nn as nn

    class LogisticRegressionModel(nn.Module):
        """17 维手工特征 -> 1 logit。刻意不含任何 Deep MLP。"""

        def __init__(self, d: int):
            super().__init__()
            self.linear = nn.Linear(d, 1)

        def forward(self, x_tab):
            return self.linear(x_tab.float()).squeeze(-1)

    return LogisticRegressionModel(in_dim)


# ------------------------------------------------------------------ IO
def _iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _read_jsonl(path):
    return list(_iter_jsonl(path))


def _count_pos_neg(path):
    pos = neg = 0
    for r in _iter_jsonl(path):
        if int(r.get("label", 0)) == 1:
            pos += 1
        else:
            neg += 1
    return pos, neg


def _read_jsonl_and_count(path):
    """一次遍历同时完成"读取"和"正负样本计数"，避免把大文件读两遍。"""
    rows, pos, neg = [], 0, 0
    for r in _iter_jsonl(path):
        rows.append(r)
        if int(r.get("label", 0)) == 1:
            pos += 1
        else:
            neg += 1
    return rows, pos, neg


def _json_write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


# ------------------------------------------------------------------ 数据准备
def fold_dirs(fold_idx: int):
    """返回 (fold 根目录, prepared repeat 目录)。"""
    fold_root = BASE_DIR / "data" / "english" / "folds" / f"fold_{fold_idx:02d}"
    prepared = fold_root / "train" / "repeat_01"
    return fold_root, prepared


def ensure_prepared_pairs(fold_idx, q2c, q2gt, q2mm, i2c, c2i, c2cats):
    """返回 (prepared_repeat_dir, {train,dev,test}_pairs 路径, 计数)。

    优先复用已存在的 prepared pairs（与 DeepWide 训练用的是同一份文件），
    不存在时才用 pairwise_data_io 现场构建，保证两个方法见到的 pair 完全一致。
    """
    fold_root, prepared = fold_dirs(fold_idx)
    paths = {
        "train_pairs": prepared / "train_pairs.jsonl",
        "dev_pairs": prepared / "dev_pairs.jsonl",
        "test_pairs": prepared / "test_pairs.jsonl",
        "vocabs": prepared / "vocabs.json",
    }
    if all(p.is_file() for p in paths.values()):
        print(f"    复用已有 prepared pairs: {prepared.relative_to(BASE_DIR)}")
        counts = {k: sum(1 for _ in _iter_jsonl(v)) for k, v in paths.items()
                  if k.endswith("_pairs")}
        return prepared, paths, counts

    # --- 现场构建（与 train_deepwide_pairwise_5fold.prepare_fold_data 同逻辑） ---
    print(f"    prepared pairs 不存在，现场构建（与 DeepWide 同逻辑）")
    tr_i = set(json.loads((fold_root / "train_incidents.json").read_text(encoding="utf-8")))
    dv_i = set(json.loads((fold_root / "dev_incidents.json").read_text(encoding="utf-8")))
    te_i = set(json.loads((fold_root / "test_incidents.json").read_text(encoding="utf-8")))
    tr_c = set(case_ids_in_split(i2c, tr_i))

    a_tr = build_all_pairs(q2c, q2gt, q2mm, c2i, inc_set=tr_i,
                           fallback_neg_per_empty_q=FALLBACK_NEG_PER_EMPTY_Q,
                           fallback_seed=42 + fold_idx)
    s_tr = sample_train_pairs(a_tr, neg_ratio=NEG_RATIO, seed=42 + fold_idx,
                              neg_per_empty_q=FALLBACK_NEG_PER_EMPTY_Q)
    a_dv = build_all_pairs(q2c, q2gt, q2mm, c2i, inc_set=dv_i,
                           fallback_neg_per_empty_q=0, fallback_seed=1042 + fold_idx)
    a_te = build_all_pairs(q2c, q2gt, q2mm, c2i, inc_set=te_i,
                           fallback_neg_per_empty_q=0, fallback_seed=2042 + fold_idx)
    voc = build_cat_vocabs(c2cats, tr_c)
    pdir = fold_root / "train_baseline_lr"
    paths = write_prepared_repeat(
        str(pdir), 1, train_pairs=s_tr, dev_pairs=a_dv, test_pairs=a_te, vocabs=voc,
        split_meta={"repeat_index": 1, "repeat_seed": 42 + fold_idx, "fold_idx": fold_idx,
                    "model": "logistic_regression",
                    "stats": {"train_pairs": len(s_tr), "dev_pairs": len(a_dv),
                              "test_pairs": len(a_te)}})
    paths = {k: Path(v) for k, v in paths.items()}
    counts = {"train_pairs": len(s_tr), "dev_pairs": len(a_dv), "test_pairs": len(a_te)}
    return pdir, paths, counts


# ------------------------------------------------------------------ 特征预构建
def build_wide_features_cached(rows, store, case2cats, cache_path=None):
    """一次性构建 17 维手工特征，供所有 epoch 复用。

    与 ``build_pair_features_current`` 的 ``X_tab`` **逐位等价**，只是不构建
    用不到的 33,792 维 Deep 特征。这是 LR 训练的主要提速点：
    原实现每个 epoch 都要为 12.9 万对 pair 重建一次 Deep 特征（约 17 GB/epoch），
    而 LR 只用其中 17 维。

    Parameters
    ----------
    cache_path : Path | None
        若给出则落盘为 ``.npz``，下次直接加载（特征只依赖输入数据，跨运行不变）。

    Returns
    -------
    dict with keys X (float32 [N,17]), y (int32 [N]), qc (list[(q,c)])
    """
    if cache_path is not None and cache_path.is_file():
        try:
            z = np.load(cache_path, allow_pickle=True)
            if int(z["X"].shape[0]) == len(rows):
                return {"X": z["X"], "y": z["y"], "qc": [tuple(x) for x in z["qc"]]}
        except Exception:
            pass  # 缓存损坏则重建

    fields = list(PAPER_TEXT_FIELDS)
    n = len(rows)
    n_cat = len(CAT_FIELDS)
    X = np.zeros((n, n_cat + len(fields)), dtype=np.float32)
    y = np.zeros(n, dtype=np.int32)
    qc = [None] * n

    # 类别 token 只取一次，避免每对 pair 重复做字符串处理
    tok = {}
    for cid, row in case2cats.items():
        tok[cid] = [None if row is None else (
            str(row.get(f, "__MISSING__")).strip() if row.get(f) is not None
            and str(row.get(f)).strip() else "__MISSING__") for f in CAT_FIELDS]

    for i, p in enumerate(rows):
        q = norm_id(p["q"])
        c = norm_id(p["c"])
        tq = tok.get(q)
        tc = tok.get(c)
        if tq is None:
            tq = ["__MISSING__"] * n_cat
        if tc is None:
            tc = ["__MISSING__"] * n_cat
        for j in range(n_cat):
            X[i, j] = 1.0 if tq[j] == tc[j] else 0.0

        qi = store.get_idx(q)
        ci = store.get_idx(c)
        for k, f in enumerate(fields):
            qv, qok = store.get_vec(f, qi)
            cv, cok = store.get_vec(f, ci)
            X[i, n_cat + k] = cosine(qv, cv) if (qok and cok) else 0.0

        y[i] = int(float(p.get("label", 0.0)))
        qc[i] = (q, c)

    out = {"X": X, "y": y, "qc": qc}
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, X=X, y=y,
                            qc=np.asarray(qc, dtype=object))
    return out


# ------------------------------------------------------------------ 推理
def predict_probs(model, rows, store, case2cats, device, batch_size=BATCH_SIZE,
                  feat_cache=None):
    """返回 (probs, y_true, qc_list)。

    feat_cache 不为 None 时直接用预构建好的 17 维特征矩阵，跳过
    ``build_pair_features_current``（后者会为每对 pair 构建 33,792 维的
    Deep 特征，而 LR 只用其中 17 维）。
    """
    import torch

    model.eval()
    all_p, all_y, all_qc = [], [], []
    with torch.no_grad():
        if feat_cache is not None:
            for s in range(0, len(rows), batch_size):
                x = torch.from_numpy(feat_cache["X"][s:s + batch_size]).to(device)
                logits = model(x)
                all_p.append(torch.sigmoid(logits).cpu().numpy())
            probs = np.concatenate([np.atleast_1d(v) for v in all_p]).astype(np.float64)
            y = feat_cache["y"].astype(np.int32)
            all_qc = list(feat_cache["qc"])
            return probs, y, all_qc
        for s in range(0, len(rows), batch_size):
            batch = rows[s:s + batch_size]
            feats = build_pair_features_current(
                batch, store, case2cats, TEXT_FIELDS, paper_text_fields=PAPER_TEXT_FIELDS)
            x = torch.from_numpy(feats["X_tab"]).to(device)
            logits = model(x)
            all_p.append(torch.sigmoid(logits).cpu().numpy())
            all_y.append(np.asarray(feats["y"], dtype=np.float32))
            all_qc.extend(feats["qc_list"])
    probs = np.concatenate([np.atleast_1d(x) for x in all_p]).astype(np.float64)
    y = np.concatenate(all_y).astype(np.int32)
    return probs, y, all_qc


# ------------------------------------------------------------------ 训练
def train_fold(prep, fold_idx, store, case2cats, device, out_dir):
    import torch
    import torch.nn as nn
    from sklearn.metrics import f1_score

    set_seed(42 + fold_idx)
    in_dim = len(get_wide_feature_names(PAPER_TEXT_FIELDS))
    model = _build_model(in_dim).to(device)

    train_rows, n_pos, n_neg = _read_jsonl_and_count(prep["paths"]["train_pairs"])
    dev_rows = _read_jsonl(prep["paths"]["dev_pairs"])
    pos_weight = torch.tensor([n_neg / max(1, n_pos)], dtype=torch.float32).to(device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    seed = 42 + fold_idx
    fold_out = out_dir / f"fold_{fold_idx:02d}"
    fold_out.mkdir(parents=True, exist_ok=True)

    # —— 一次性预构建 17 维特征（原实现每个 epoch 重建 3.4 万维 Deep 特征，是主要瓶颈）
    t_feat = time.time()
    cache_dir = fold_out / "features"
    tr_feat = build_wide_features_cached(train_rows, store, case2cats,
                                         cache_dir / "train.npz")
    dv_feat = build_wide_features_cached(dev_rows, store, case2cats,
                                         cache_dir / "dev.npz")
    print(f"    特征已就绪: train={tr_feat['X'].shape} dev={dv_feat['X'].shape}  "
          f"({time.time() - t_feat:.1f}s, 已缓存)", flush=True)
    print(f"    数据就绪: train={len(train_rows)} (pos={n_pos}, neg={n_neg}, "
          f"pos_weight={n_neg / max(1, n_pos):.3f})  dev={len(dev_rows)}", flush=True)

    X_tr = tr_feat["X"]
    y_tr = torch.from_numpy(tr_feat["y"].astype(np.float32)).to(device)

    history = []
    best_f1, best_ckpt, best_ep = -1.0, None, None
    bsz = BATCH_SIZE
    n_batches = (len(train_rows) + bsz - 1) // bsz
    log_every = max(1, n_batches // 10)
    perm = np.arange(len(train_rows), dtype=np.int64)

    for ep in range(1, EPOCHS + 1):
        model.train()
        rng = random.Random(seed + ep * 100)
        rng.shuffle(perm)
        total_loss, n_batch = 0.0, 0
        t_ep = time.time()
        for s in range(0, len(perm), bsz):
            idx = perm[s:s + bsz]
            x = torch.from_numpy(X_tr[idx]).to(device)
            y = y_tr[idx]
            loss = crit(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batch += 1
            if n_batch % log_every == 0 or n_batch == n_batches:
                # stderr 是无缓冲的，即使 stdout 被 tee 缓冲住也能实时看到进度
                el = time.time() - t_ep
                print(f"      [epoch {ep}] batch {n_batch}/{n_batches}  "
                      f"loss={total_loss / n_batch:.4f}  {el:.0f}s", file=sys.stderr, flush=True)

        dev_p, dev_y, _ = predict_probs(model, dev_rows, store, case2cats, device,
                                        feat_cache=dv_feat)
        dev_f1 = float(f1_score(dev_y, (dev_p >= PAIR_THRESHOLD).astype(np.int32),
                                pos_label=1, zero_division=0))
        print(f"    Epoch {ep}: loss={total_loss / max(1, n_batch):.6f}  "
              f"dev_pair_f1_pos={dev_f1:.4f}  ({time.time() - t_ep:.0f}s)", flush=True)
        history.append({"epoch": ep, "train_loss": total_loss / max(1, n_batch),
                        "dev_pair_f1_pos": dev_f1})

        ckpt = fold_out / f"lr_fold{fold_idx:02d}_epoch{ep}.pt"
        torch.save({"framework": "wide-only",
                    "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "epoch": ep, "fold_idx": fold_idx, "seed": seed,
                    "in_dim": in_dim, "dev_pair_f1_pos": dev_f1,
                    "feature_names": list(get_wide_feature_names(PAPER_TEXT_FIELDS))}, ckpt)
        if dev_f1 > best_f1:
            best_f1, best_ckpt, best_ep = dev_f1, ckpt, ep

    print(f"    Best: epoch {best_ep}  dev_pair_f1_pos={best_f1:.4f}  -> {best_ckpt.name}")
    return best_ckpt, best_ep, best_f1, history, {"train": tr_feat, "dev": dv_feat}


# ------------------------------------------------------------------ 评测
def evaluate_fold(prep, model, store, case2cats, device, feat_cache=None):
    import torch
    from sklearn.metrics import f1_score, precision_score, recall_score

    dev_rows = _read_jsonl(prep["paths"]["dev_pairs"])
    test_rows = _read_jsonl(prep["paths"]["test_pairs"])

    print("    Dev τ grid search...")
    dev_p, dev_y, dev_qc = predict_probs(
        model, dev_rows, store, case2cats, device,
        feat_cache=(feat_cache or {}).get("dev"))
    best = grid_search_on_dev(
        probs=dev_p, y_true=dev_y, qc_list=dev_qc,
        dev_case_ids=sorted(prep["dev_case_ids"]),
        gold_clusters_dev=prep["gold_dev"],
        edge_rule_grid=EDGE_RULE_GRID,
        threshold_grid_step=THRESHOLD_STEP,
        threshold_min=THRESHOLD_MIN,
        threshold_max=THRESHOLD_MAX,
    )
    tau = float(best.get("cluster_threshold", best["threshold"]))

    print("    Test evaluation...")
    test_feat = (feat_cache or {}).get("test")
    if test_feat is None:
        test_feat = build_wide_features_cached(test_rows, store, case2cats)
    test_p, test_y, test_qc = predict_probs(
        model, test_rows, store, case2cats, device, feat_cache=test_feat)
    directed = build_directed_scores(test_qc, test_p)
    clusters = build_complete_link_clusters(
        nodes=sorted(prep["test_case_ids"]), directed_scores=directed,
        edge_rule=best["edge_rule"], threshold=tau)

    sym = hungarian_event_symmetric_macro_f1(prep["gold_test"], clusters)
    b3 = b_cubed_f1(prep["gold_test"], clusters)
    ari = clustering_ari(prep["gold_test"], clusters)
    ipf = induced_pairwise_f1(prep["gold_test"], clusters)
    yp = (test_p >= tau).astype(np.int32)

    metrics = {
        "threshold": tau,
        "edge_rule": best["edge_rule"],
        "merge_strategy": "complete_link",
        "hungarian_f1_g2p": float(sym["gold_to_pred_macro_f1"]),
        "hungarian_f1_p2g": float(sym["pred_to_gold_macro_f1"]),
        "hungarian_f1_symmetric": float(sym["symmetric_macro_f1"]),
        "b_cubed_f1": float(b3["b_cubed_f1"]),
        "b_cubed_precision": float(b3["b_cubed_precision"]),
        "b_cubed_recall": float(b3["b_cubed_recall"]),
        "ari": float(ari),
        "induced_pair_f1": float(ipf["induced_pair_f1"]),
        "induced_pair_precision": float(ipf["induced_pair_precision"]),
        "induced_pair_recall": float(ipf["induced_pair_recall"]),
        "pair_macro_f1": float(f1_score(test_y, yp, average="macro")),
        "pair_f1_pos": float(f1_score(test_y, yp, pos_label=1, zero_division=0)),
        "pair_precision_pos": float(precision_score(test_y, yp, pos_label=1, zero_division=0)),
        "pair_recall_pos": float(recall_score(test_y, yp, pos_label=1, zero_division=0)),
        "pair_accuracy": float((test_y == yp).mean()),
        "num_pairs": int(len(test_y)),
        "num_pred_pos_pairs": int(yp.sum()),
        "num_pred_clusters": int(len(clusters)),
        "num_gold_events": int(len(prep["gold_test"])),
    }
    return best, tau, metrics, clusters


# ------------------------------------------------------------------ 中文
def run_chinese(args, out_dir):
    """用英文五折训练出的 LR 权重，对中文 1,283 篇零样本打分 + 聚类。

    产出与 ``scripts/chinese_inference.py`` 同格式的分数 JSONL，
    再用同一套指标评测，便于与 DeepWide 中文结果直接对照。
    """
    import torch

    print("\n" + "=" * 70)
    print("中文零样本评测（LR 基线）")
    print("=" * 70)

    root = BASE_DIR
    gold_p = root / "data/chinese_gold/gold_v2.0.2_mapping.csv"
    cases_p = root / "data/chinese_eval/cases.jsonl"
    recall_p = root / "data/chinese_eval/recall_topk.jsonl"
    emb_p = root / "data/chinese_eval/embeddings"
    ckpt_dir = (out_dir / "fold_01")  # 每折的 LR 权重都在各自 fold_XX 目录

    import csv
    gold_ids = sorted({str(r["report_id"]).strip() for r in
                       csv.DictReader(open(gold_p, encoding="utf-8-sig"))},
                      key=lambda x: (len(x), x))
    case2cats = load_cases_minimal(str(cases_p))
    q2c, _, _ = load_recall_topk(str(recall_p), topm_out=args.topm)
    if set(q2c) != set(case2cats) or set(case2cats) != set(gold_ids):
        sys.exit("❌ 中文 cases / recall / gold 域不一致")

    # case -> gold event，用于给中文 pair 打真实标签（pair 级指标需要）
    case2event = {}
    for _r in csv.DictReader(open(gold_p, encoding="utf-8-sig")):
        case2event[str(_r["report_id"]).strip()] = str(_r["gold_v2_event_id"]).strip()

    pairs = [{"q": q, "c": e[0],
              "label": 1 if case2event.get(q) == case2event.get(e[0]) else 0}
             for q, cands in q2c.items() for e in cands]
    n_pos = sum(p["label"] for p in pairs)
    print(f"中文: {len(case2cats)} 篇 / {len(pairs)} 候选对 "
          f"(topm={args.topm}, pos={n_pos}, pos_rate={n_pos/max(1,len(pairs)):.4f})")

    store = EmbeddingsStore(str(emb_p), TEXT_FIELDS)
    nodes = sorted(gold_ids, key=lambda x: (len(x), x))
    gold_clusters = {}
    for r in csv.DictReader(open(gold_p, encoding="utf-8-sig")):
        gold_clusters.setdefault(str(r["gold_v2_event_id"]).strip(), []).append(str(r["report_id"]).strip())
    gold_list = [sorted(v) for v in gold_clusters.values()]

    locked = json.loads((root / "outputs/dev/locked_thetas.json").read_text(encoding="utf-8"))
    thetas = locked["locked_thetas"]

    zh_dir = out_dir / "chinese"
    zh_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    t_feat = time.time()
    zh_feat = build_wide_features_cached(pairs, store, case2cats,
                                         zh_dir / "features.npz")
    print(f"  中文特征已就绪: {zh_feat['X'].shape}  ({time.time() - t_feat:.1f}s, 已缓存)",
          flush=True)

    for fold in args.folds:
        in_dim = len(get_wide_feature_names(PAPER_TEXT_FIELDS))
        model = _build_model(in_dim).to(args.device)
        ckpt_path = out_dir / f"fold_{fold:02d}" / f"lr_fold{fold:02d}_epoch{args.best_epoch[fold - 1]}.pt"
        if not ckpt_path.is_file():
            # 回退：取该折目录下最新一个
            cands = sorted((out_dir / f"fold_{fold:02d}").glob("lr_fold*.pt"))
            if not cands:
                sys.exit(f"❌ 找不到 fold {fold} 的 LR 权重")
            ckpt_path = cands[-1]
        ckpt = torch.load(str(ckpt_path), map_location=args.device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        print(f"\n[fold {fold:02d}] {ckpt_path.name}")

        # 中文 pair 集合五折共用，特征只构建一次并缓存
        probs, y_true, qc = predict_probs(model, pairs, store, case2cats, args.device,
                                          feat_cache=zh_feat)
        if not np.all(np.isfinite(probs)):
            sys.exit("❌ 出现非有限概率")

        scores_path = zh_dir / f"fold_{fold:02d}_scores.jsonl"
        with scores_path.open("w", encoding="utf-8") as fo:
            for (q, c), p in zip(qc, probs):
                fo.write(json.dumps({"query_id": str(q), "candidate_id": str(c),
                                     "probability": float(min(max(p, 0.0), 1.0))},
                                    ensure_ascii=False) + "\n")

        tau = float(thetas[str(fold)])
        directed = build_directed_scores(qc, probs)
        clusters = build_complete_link_clusters(
            nodes=nodes, directed_scores=directed, edge_rule="mutual", threshold=tau)
        sym = hungarian_event_symmetric_macro_f1(gold_list, clusters)
        b3 = b_cubed_f1(gold_list, clusters)
        ari = clustering_ari(gold_list, clusters)
        ipf = induced_pairwise_f1(gold_list, clusters)

        rec = {
            "fold": fold, "threshold": tau, "checkpoint": str(ckpt_path),
            "n_scored_reports": len(nodes), "num_pred_clusters": len(clusters),
            "num_gold_events": len(gold_list),
            "hungarian_f1_g2p": float(sym["gold_to_pred_macro_f1"]),
            "hungarian_f1_p2g": float(sym["pred_to_gold_macro_f1"]),
            "hungarian_f1_symmetric": float(sym["symmetric_macro_f1"]),
            "b_cubed_f1": float(b3["b_cubed_f1"]),
            "ari": float(ari),
            "induced_pair_f1": float(ipf["induced_pair_f1"]),
        }
        # Pairwise 分类指标（与英文同口径：在聚类阈值 τ 下判定）
        from sklearn.metrics import (f1_score as _f1, precision_score as _prec,
                                     recall_score as _rec)
        _yp = (probs >= tau).astype(np.int32)
        rec.update({
            "pair_accuracy": float((y_true == _yp).mean()),
            "pair_macro_f1": float(_f1(y_true, _yp, average="macro", zero_division=0)),
            "pair_f1_pos": float(_f1(y_true, _yp, pos_label=1, zero_division=0)),
            "pair_precision_pos": float(_prec(y_true, _yp, pos_label=1, zero_division=0)),
            "pair_recall_pos": float(_rec(y_true, _yp, pos_label=1, zero_division=0)),
            "num_pairs": int(len(y_true)),
        })
        rows.append(rec)
        _json_write(zh_dir / f"fold_{fold:02d}_result.json", rec)
        print(f"  τ={tau}  簇={len(clusters)}  g2p={rec['hungarian_f1_g2p']:.6f}  "
              f"b3={rec['b_cubed_f1']:.6f}  ari={rec['ari']:.6f}")

    keys = ["hungarian_f1_g2p", "hungarian_f1_p2g", "hungarian_f1_symmetric",
            "b_cubed_f1", "ari", "induced_pair_f1",
            "pair_accuracy", "pair_macro_f1", "pair_f1_pos",
            "pair_precision_pos", "pair_recall_pos"]
    summary = {"method": LABEL + " [chinese zero-shot]", "n_folds": len(rows),
               "thetas": {str(r["fold"]): r["threshold"] for r in rows},
               "metrics_across_folds": {
                   k: {"mean": float(np.mean([r[k] for r in rows])),
                       "std": float(np.std([r[k] for r in rows], ddof=1)) if len(rows) > 1 else 0.0,
                       "values": [r[k] for r in rows]} for k in keys}}
    _json_write(zh_dir / "summary.json", summary)
    print("\n" + "=" * 70)
    for k in keys:
        print(f"  {k:34s} {summary['metrics_across_folds'][k]['mean']:.6f} "
              f"± {summary['metrics_across_folds'][k]['std']:.6f}")
    print(f"\n→ {zh_dir / 'summary.json'}")


# ------------------------------------------------------------------ 收敛判定
# 判据：dev F1 已经进入平台期 —— 最后一个 epoch 相对"历史最优"的回落不超过
# tol，且最后两个 epoch 的增量不超过 tol。两条都满足才视为收敛。
CONVERGENCE_TOL = 0.01


def _convergence_report(results, epochs: int) -> dict:
    """汇总每折 dev F1 曲线，判断 LR 是否已收敛。

    LR 只有 17 个参数，在 8 epoch 下常常仍在上升；这个报告用来回答
    "给足轮数后 LR 是否仍不及 DeepWide"，避免被质疑对照不公平。
    """
    per_fold = {}
    n_converged = 0
    for r in results:
        hist = r.get("training_history", [])
        curve = [float(h["dev_pair_f1_pos"]) for h in hist]
        if not curve:
            continue
        best_val = max(curve)
        best_ep = curve.index(best_val) + 1
        last = curve[-1]
        last_gain = (curve[-1] - curve[-2]) if len(curve) >= 2 else float("nan")
        converged = bool(
            (best_val - last) <= CONVERGENCE_TOL
            and (len(curve) < 2 or last_gain <= CONVERGENCE_TOL)
        )
        n_converged += int(converged)
        per_fold[f"fold_{r['fold']:02d}"] = {
            "dev_f1_curve": curve,
            "best_dev_f1": best_val,
            "best_epoch": best_ep,
            "last_dev_f1": last,
            "last_epoch_gain": (None if len(curve) < 2 else last_gain),
            "converged": converged,
        }
    return {
        "epochs": int(epochs),
        "tol": CONVERGENCE_TOL,
        "criterion": "last_epoch_gain <= tol 且 best-last <= tol",
        "n_folds": len(per_fold),
        "n_converged": n_converged,
        "all_converged": n_converged == len(per_fold) and bool(per_fold),
        "per_fold": per_fold,
    }


def _print_convergence(conv: dict):
    print("\n" + "-" * 70)
    print(f"收敛检查（{conv['epochs']} epoch，tol={conv['tol']}）")
    print("-" * 70)
    for name, info in conv["per_fold"].items():
        curve = info["dev_f1_curve"]
        tail = " ".join(f"{v:.4f}" for v in curve[-3:])
        gain = info["last_epoch_gain"]
        flag = "✅ 收敛" if info["converged"] else "⚠️  仍在上升"
        gtxt = "—" if gain is None else f"{gain:+.4f}"
        print(f"  {name}: best={info['best_dev_f1']:.4f}@ep{info['best_epoch']}  "
              f"tail=[{tail}]  last_gain={gtxt}  {flag}")
    print(f"  合计：{conv['n_converged']}/{conv['n_folds']} 折收敛"
          + ("" if conv["all_converged"] else "  ← 未全部收敛，建议 --epochs 更大"))


# ------------------------------------------------------------------ 主流程
def main():
    # 先声明 global，之后再读 EPOCHS 作为默认值（语法要求：默认值表达式
    # 出现在 global 声明之前会报 "used prior to global declaration"）
    global EPOCHS

    ap = argparse.ArgumentParser(description="逻辑回归基线（五折 / 中文零样本）")
    ap.add_argument("--folds", default="1,2,3,4,5")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--topm", type=int, default=TOPM_OUT)
    ap.add_argument("--out-dir", default="outputs/benchmarks/logistic_regression")
    ap.add_argument("--channel", choices=["english", "chinese", "both"], default="english",
                    help="english=只看英文五折; chinese=用已训练权重跑中文; both=两者都做")
    ap.add_argument("--skip-train", action="store_true",
                    help="跳过英文训练，直接用已有 fold_XX/lr_*.pt 跑中文")
    ap.add_argument("--epochs", type=int, default=EPOCHS,
                    help=f"训练轮数（默认 {EPOCHS} = 主方法同预算）。"
                         "LR 参数少，8 轮常未收敛；补跑收敛版用 --epochs 20")
    args = ap.parse_args()
    args.folds = [int(x) for x in args.folds.split(",") if x.strip()]

    # 用命令行 epoch 覆盖模块级常量（train_fold / summary 都读它）
    if args.epochs != EPOCHS:
        EPOCHS = int(args.epochs)

    out_dir = BASE_DIR / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch
    if _THREADS:
        torch.set_num_threads(_THREADS)
        try:
            torch.set_num_interop_threads(_THREADS)
        except RuntimeError:
            pass  # 已有并行任务启动后不可再改，忽略

    t0 = time.time()
    print("=" * 70, flush=True)
    print(LABEL, flush=True)
    print("=" * 70, flush=True)
    print(f"BASE_DIR : {BASE_DIR}", flush=True)
    print(f"OUT      : {out_dir}", flush=True)
    print(f"Folds    : {args.folds}   Device: {args.device}", flush=True)
    print(f"τ grid   : {THRESHOLD_MIN}–{THRESHOLD_MAX} step {THRESHOLD_STEP}", flush=True)
    print(f"Threads  : {_THREADS or 'unlimited'} (torch={torch.get_num_threads()})", flush=True)

    if args.channel in ("english", "both") and not args.skip_train:
        # --- 英文五折训练 + 评测 ---
        import torch
        print("\n[加载全局数据]")
        i2c, c2i, _ = load_eval_structure(str(BASE_DIR / "data/precomputed/eval_structure.json"))
        c2cats = load_cases_minimal(str(BASE_DIR / "data/english/test/cases.jsonl"))
        q2c, q2gt, q2mm = load_recall_topk(
            str(BASE_DIR / "data/english/test/recall_topk.jsonl"), topm_out=args.topm)
        print(f"  queries={len(q2c)}  wide_dim={len(get_wide_feature_names(PAPER_TEXT_FIELDS))}")

        store = EmbeddingsStore(str(BASE_DIR / "data/english/test/embeddings"), TEXT_FIELDS)
        results, best_epochs = [], []

        for fold_idx in args.folds:
            print("\n" + "-" * 60)
            print(f"fold_{fold_idx:02d}")
            fold_root, _ = fold_dirs(fold_idx)
            tr_i = set(json.loads((fold_root / "train_incidents.json").read_text(encoding="utf-8")))
            dv_i = set(json.loads((fold_root / "dev_incidents.json").read_text(encoding="utf-8")))
            te_i = set(json.loads((fold_root / "test_incidents.json").read_text(encoding="utf-8")))

            prep_dir, paths, counts = ensure_prepared_pairs(
                fold_idx, q2c, q2gt, q2mm, i2c, c2i, c2cats)
            prep = {
                "paths": paths, "counts": counts,
                "train_case_ids": set(case_ids_in_split(i2c, tr_i)),
                "dev_case_ids": set(case_ids_in_split(i2c, dv_i)),
                "test_case_ids": set(case_ids_in_split(i2c, te_i)),
                "gold_train": incident_to_gold_clusters(i2c, tr_i),
                "gold_dev": incident_to_gold_clusters(i2c, dv_i),
                "gold_test": incident_to_gold_clusters(i2c, te_i),
            }
            print(f"  pairs: train={counts['train_pairs']} dev={counts['dev_pairs']} "
                  f"test={counts['test_pairs']}")

            t = time.time()
            ckpt, best_ep, best_f1, history, feat_cache = train_fold(
                prep, fold_idx, store, c2cats, args.device, out_dir)
            print(f"  train {time.time() - t:.0f}s")
            best_epochs.append(best_ep)

            model = _build_model(len(get_wide_feature_names(PAPER_TEXT_FIELDS))).to(args.device)
            import torch
            st = torch.load(str(ckpt), map_location=args.device, weights_only=False)
            model.load_state_dict(st["state_dict"])
            model.eval()

            best_dev, tau, metrics, _ = evaluate_fold(
                prep, model, store, c2cats, args.device, feat_cache=feat_cache)
            rec = {
                "fold": fold_idx,
                "best_epoch": best_ep,
                "best_dev_pair_f1_pos": best_f1,
                "checkpoint": str(ckpt),
                "pairs": counts,
                "best_dev": best_dev,
                "test_metrics": metrics,
                "training_history": history,
            }
            _json_write(out_dir / f"fold_{fold_idx:02d}" / "result.json", rec)
            results.append(rec)
            print(f"  TEST g2p={metrics['hungarian_f1_g2p']:.6f}  "
                  f"b3={metrics['b_cubed_f1']:.6f}  τ={tau:.2f}")
            del model

        # --- 汇总 ---
        keys = ["hungarian_f1_g2p", "hungarian_f1_symmetric",
                "hungarian_f1_p2g", "b_cubed_f1", "ari",
                "induced_pair_f1", "pair_macro_f1", "pair_f1_pos",
                # Pairwise 分类表所需的完整指标
                "pair_precision_pos", "pair_recall_pos", "pair_accuracy"]
        summary = {
            "method": LABEL,
            "cv_type": "5-fold incident-level cross-validation",
            "n_folds": len(results),
            "model": "logistic_regression_17d",
            "edge_rule": "mutual",
            "merge_strategy": "complete_link",
            "threshold_grid": {"min": THRESHOLD_MIN, "max": THRESHOLD_MAX, "step": THRESHOLD_STEP},
            "training_config": {"epochs": EPOCHS, "batch_size": BATCH_SIZE,
                                "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
                                "neg_ratio": NEG_RATIO, "topm_out": args.topm},
            "best_epochs": best_epochs,
            "best_thresholds": {
                "mean": float(np.mean([r["test_metrics"]["threshold"] for r in results])),
                "std": float(np.std([r["test_metrics"]["threshold"] for r in results], ddof=1)),
                "values": [r["test_metrics"]["threshold"] for r in results]},
            "metrics_across_folds": {
                k: {"mean": float(np.mean([r["test_metrics"][k] for r in results])),
                    "std": float(np.std([r["test_metrics"][k] for r in results], ddof=1)),
                    "values": [r["test_metrics"][k] for r in results]} for k in keys},
            "convergence": _convergence_report(results, EPOCHS),
        }
        _json_write(out_dir / "summary.json", summary)
        print("\n" + "=" * 70)
        print("英文五折汇总（LR 基线）")
        print("=" * 70)
        for k in keys:
            v = summary["metrics_across_folds"][k]
            print(f"  {k:42s} {v['mean']:.6f} ± {v['std']:.6f}")
        _print_convergence(summary["convergence"])
        print(f"\n→ {out_dir / 'summary.json'}")
        args.best_epoch = best_epochs
    else:
        # 只有 chinese 且 skip-train：从已有 ckpt 反推 best_epoch
        best = []
        for fold_idx in args.folds:
            rp = out_dir / f"fold_{fold_idx:02d}" / "result.json"
            best.append(json.loads(rp.read_text())["best_epoch"] if rp.is_file() else 999)
        args.best_epoch = best

    if args.channel in ("chinese", "both"):
        run_chinese(args, out_dir)

    print(f"\n总耗时 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    # 强制 stdout 行缓冲：即使被 `| tee` 接管也能实时看到日志，
    # 不必再依赖调用方加 `python -u`。
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(line_buffering=True)
        except Exception:  # noqa: BLE001 —— 非 tty/被重定向时可能不支持，忽略即可
            pass
    main()
