#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
逻辑回归基线：Dual Recall + 逻辑回归 + Complete-Link + 五折CV

模型: nn.Linear(17, 1)，即对 17 个手工特征做逻辑回归二分类。
与完整方法的唯一区别: 将 Deep+Wide 分类器替换为逻辑回归（去掉 Deep MLP）。
用于验证深度交互特征建模的必要性。

用法:
  python src/Incident_Align_Evaluation/baseline_logistic_regression_cv.py
"""

import json, os, random, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score

SRC_DIR = str(Path(__file__).resolve().parents[1])
PAIRWISE_DIR = os.path.join(SRC_DIR, "Incident_Align_Method", "pairwise_and_clustering")
for d in [SRC_DIR, PAIRWISE_DIR]:
    if d not in sys.path:
        sys.path.insert(0, d)

from pairwise_data_io import (
    CAT_FIELDS, TEXT_FIELDS_DEFAULT, norm_id,
    load_cases_minimal, load_eval_structure, load_recall_topk,
    build_all_pairs, sample_train_pairs, build_cat_vocabs,
    case_ids_in_split, incident_to_gold_clusters,
    write_prepared_repeat, write_json,
)
from pairwise_model import (
    PAPER_TEXT_FIELDS_DEFAULT, EmbeddingsStore,
    set_seed, get_wide_feature_names, build_pair_features_current,
)
from graph_clustering import (
    build_complete_link_clusters, build_directed_scores,
    hungarian_event_symmetric_macro_f1,
    b_cubed_f1, clustering_ari, induced_pairwise_f1,
    grid_search_on_dev,
)

# ── 配置 ──────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[2]
CASES_FILE       = str(BASE_DIR / "data" / "eval_cases.jsonl")
STRUCTURE_FILE   = str(BASE_DIR / "data" / "eval_structure.json")
RECALL_FILE      = str(BASE_DIR / "outputs" / "recall.jsonl")
EMBEDDINGS_DIR   = str(BASE_DIR / "outputs" / "embeddings")
CV_SPLITS_DIR    = str(BASE_DIR / "outputs" / "prepared_5fold_cv")
TRAIN_OUT        = str(BASE_DIR / "outputs" / "pairwise_train_logistic_regression")
DECODE_OUT       = str(BASE_DIR / "outputs" / "Incident_Align_Evaluation" / "baseline_logistic_regression_cv")

EPOCHS, BATCH_SIZE, LR, WD = 8, 512, 2e-4, 1e-4
NEG_RATIO, FALLBACK_NEG, TOPM_OUT = 4, 10, 200
THR_STEP, THR_MIN, THR_MAX = 0.02, 0.50, 0.95
TEXT_FIELDS = TEXT_FIELDS_DEFAULT + ["text"]
PAPER_TEXT_FIELDS = list(PAPER_TEXT_FIELDS_DEFAULT)
# ──────────────────────────────────────────────────────────────────


class LogisticRegressionModel(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, 1)
    def forward(self, x_tab, _x_pair):
        return self.linear(x_tab.float())


def _json_write(path, obj):
    d = os.path.dirname(path)
    if d: os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _count_pos_neg(path):
    pos = neg = 0
    with open(path) as f:
        for line in f:
            if not line.strip(): continue
            lbl = int(json.loads(line).get("label", 0))
            if lbl == 1: pos += 1
            else: neg += 1
    return pos, neg


def _read_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


# ── Pipeline ──────────────────────────────────────────────────────

def prepare_fold(fold_cv_dir, fold_idx, q2c, q2gt, q2mm, i2c, c2i, c2cats):
    with open(os.path.join(fold_cv_dir, "train_incidents.json")) as f:
        tr_i = set(json.load(f))
    with open(os.path.join(fold_cv_dir, "dev_incidents.json")) as f:
        dv_i = set(json.load(f))
    with open(os.path.join(fold_cv_dir, "test_incidents.json")) as f:
        te_i = set(json.load(f))

    tr_c = set(case_ids_in_split(i2c, tr_i))
    dv_c = set(case_ids_in_split(i2c, dv_i))
    te_c = set(case_ids_in_split(i2c, te_i))

    g_tr = incident_to_gold_clusters(i2c, tr_i)
    g_dv = incident_to_gold_clusters(i2c, dv_i)
    g_te = incident_to_gold_clusters(i2c, te_i)

    print("    Building train pairs...")
    at = build_all_pairs(q2c, q2gt, q2mm, c2i, inc_set=tr_i,
                         fallback_neg_per_empty_q=FALLBACK_NEG, fallback_seed=42+fold_idx)
    st = sample_train_pairs(at, neg_ratio=NEG_RATIO, seed=42+fold_idx,
                            neg_per_empty_q=FALLBACK_NEG)

    print("    Building dev pairs...")
    ad = build_all_pairs(q2c, q2gt, q2mm, c2i, inc_set=dv_i,
                         fallback_neg_per_empty_q=0, fallback_seed=1042+fold_idx)

    print("    Building test pairs...")
    ae = build_all_pairs(q2c, q2gt, q2mm, c2i, inc_set=te_i,
                         fallback_neg_per_empty_q=0, fallback_seed=2042+fold_idx)

    voc = build_cat_vocabs(c2cats, tr_c)

    pdir = os.path.join(CV_SPLITS_DIR, "fold_{:02d}".format(fold_idx), "pairwise_logistic_regression")
    paths = write_prepared_repeat(pdir, 1,
        train_pairs=st, dev_pairs=ad, test_pairs=ae, vocabs=voc,
        split_meta={"repeat_index":1, "repeat_seed":42+fold_idx, "fold_idx":fold_idx,
                    "recall_mode":"dual", "model":"logistic_regression",
                    "stats":{"train_pairs":len(st),"dev_pairs":len(ad),"test_pairs":len(ae)}})
    write_json(os.path.join(pdir, "manifest.json"), {
        "prepared_dir": pdir, "text_fields": TEXT_FIELDS,
        "sim_fields": ["domain","event_type","ai_risk_description","ai_risk_subtype",
                       "affected_actor_subtype","text"],
        "cat_fields": CAT_FIELDS, "sources": {"cases_file": CASES_FILE},
        "n_repeats": 1, "recall_mode": "dual", "model": "logistic_regression",
        "repeats": [{"repeat_index": 1, "repeat_seed": 42+fold_idx}]})

    return {"fold_idx":fold_idx, "tr_c":tr_c, "dv_c":dv_c, "te_c":te_c,
            "g_tr":g_tr, "g_dv":g_dv, "g_te":g_te,
            "c2cats":c2cats, "voc":voc, "paths":paths,
            "n_tr":len(st), "n_dv":len(ad), "n_te":len(ae)}


def train_fold(prep, fold_idx):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    seed = 42 + fold_idx
    set_seed(seed)

    store = EmbeddingsStore(EMBEDDINGS_DIR, TEXT_FIELDS)
    c2cats = prep["c2cats"]
    p = prep["paths"]
    in_dim = len(get_wide_feature_names(PAPER_TEXT_FIELDS))
    model = LogisticRegressionModel(in_dim).to(device)

    n_pos, n_neg = _count_pos_neg(p["train_pairs"])
    pw = torch.tensor([n_neg/max(1,n_pos) if n_pos>0 else 1.0]).to(device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

    out_dir = os.path.join(TRAIN_OUT, "fold_{:02d}".format(fold_idx))
    os.makedirs(out_dir, exist_ok=True)

    train_rows = _read_jsonl(p["train_pairs"])
    dev_rows   = _read_jsonl(p["dev_pairs"])

    best_f1, best_ckpt = -1.0, None

    for ep in range(1, EPOCHS+1):
        # --- train ---
        model.train()
        rng = random.Random(seed + ep*100)
        rng.shuffle(train_rows)
        total_loss, n_batch = 0.0, 0
        for s in range(0, len(train_rows), BATCH_SIZE):
            batch = train_rows[s:s+BATCH_SIZE]
            labs = torch.tensor([int(r["label"]) for r in batch], dtype=torch.float32).to(device)
            feats = build_pair_features_current(
                batch, store, c2cats, TEXT_FIELDS,
                paper_text_fields=PAPER_TEXT_FIELDS)
            X_tab = torch.from_numpy(feats["X_tab"]).to(device)
            X_pr  = torch.zeros(X_tab.shape[0], 1).to(device)
            loss = crit(model(X_tab, X_pr).squeeze(-1), labs)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item(); n_batch += 1

        # --- dev eval ---
        model.eval()
        all_p, all_l = [], []
        for s in range(0, len(dev_rows), BATCH_SIZE):
            batch = dev_rows[s:s+BATCH_SIZE]
            labs = np.array([int(r["label"]) for r in batch], dtype=np.int32)
            feats = build_pair_features_current(
                batch, store, c2cats, TEXT_FIELDS,
                paper_text_fields=PAPER_TEXT_FIELDS)
            X_tab = torch.from_numpy(feats["X_tab"]).to(device)
            X_pr  = torch.zeros(X_tab.shape[0], 1).to(device)
            with torch.no_grad():
                p_b = torch.sigmoid(model(X_tab, X_pr).squeeze(-1)).cpu().numpy()
            all_p.append(np.atleast_1d(p_b)); all_l.append(labs)

        probs = np.concatenate([np.atleast_1d(x) for x in all_p])
        yt    = np.concatenate(all_l)
        yp    = (probs >= 0.5).astype(np.int32)
        dev_f1= float(f1_score(yt, yp, pos_label=1, zero_division=0))

        # save
        ckpt_path = os.path.join(out_dir, "lr_fold{:02d}_ep{}.pt".format(fold_idx, ep))
        torch.save({"framework":"wide-only",
                    "state_dict":{k:v.detach().cpu() for k,v in model.state_dict().items()},
                    "epoch":ep,"fold_idx":fold_idx,"seed":seed,"in_dim":in_dim}, ckpt_path)
        print("    Epoch {}: loss={:.6f} dev_f1={:.4f}".format(ep, total_loss/max(1,n_batch), dev_f1))
        if dev_f1 > best_f1:
            best_f1 = dev_f1; best_ckpt = ckpt_path

    return best_ckpt or ckpt_path


def predict_logistic_regression(model, jsonl_path, store, c2cats, device, bs=512):
    """返回 probs, y_true, qc_list"""
    model.eval()
    rows = _read_jsonl(jsonl_path)
    all_p, all_l, all_qc = [], [], []
    for s in range(0, len(rows), bs):
        batch = rows[s:s+bs]
        labs = np.array([int(r["label"]) for r in batch], dtype=np.int32)
        feats = build_pair_features_current(
            batch, store, c2cats, TEXT_FIELDS,
            paper_text_fields=PAPER_TEXT_FIELDS)
        X_tab = torch.from_numpy(feats["X_tab"]).to(device)
        X_pr  = torch.zeros(X_tab.shape[0], 1).to(device)
        with torch.no_grad():
            p_b = torch.sigmoid(model(X_tab, X_pr).squeeze(-1)).cpu().numpy()
        all_p.append(np.atleast_1d(p_b))
        all_l.append(labs)
        all_qc.extend([(norm_id(r["q"]), norm_id(r["c"])) for r in batch])
    return np.concatenate(all_p), np.concatenate(all_l), all_qc


def decode_fold(prep, fold_idx, ckpt_path):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    set_seed(42+fold_idx)
    store = EmbeddingsStore(EMBEDDINGS_DIR, TEXT_FIELDS)
    c2cats = prep["c2cats"]
    in_dim = len(get_wide_feature_names(PAPER_TEXT_FIELDS))

    model = LogisticRegressionModel(in_dim).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])

    # --- dev grid search ---
    print("    Grid search on dev...")
    dev_p, dev_y, dev_qc = predict_logistic_regression(
        model, prep["paths"]["dev_pairs"], store, c2cats, device)
    best = grid_search_on_dev(
        probs=dev_p, y_true=dev_y, qc_list=dev_qc,
        dev_case_ids=sorted(prep["dv_c"]),
        gold_clusters_dev=prep["g_dv"],
        edge_rule_grid=["mutual"],
        threshold_grid_step=THR_STEP, threshold_min=THR_MIN, threshold_max=THR_MAX)

    # --- test eval ---
    print("    Evaluating on test...")
    tst_p, tst_y, tst_qc = predict_logistic_regression(
        model, prep["paths"]["test_pairs"], store, c2cats, device)
    thr = float(best.get("cluster_threshold", best["threshold"]))
    ds  = build_directed_scores(tst_qc, tst_p)
    cl  = build_complete_link_clusters(
        nodes=sorted(prep["te_c"]), directed_scores=ds,
        edge_rule=best["edge_rule"], threshold=thr)

    sym = hungarian_event_symmetric_macro_f1(prep["g_te"], cl)
    b3  = b_cubed_f1(prep["g_te"], cl)
    ar  = clustering_ari(prep["g_te"], cl)
    ipf = induced_pairwise_f1(prep["g_te"], cl)
    yp  = (tst_p >= thr).astype(np.int32)

    return {"best_dev":best, "test_metrics":{
        "threshold":thr, "edge_rule":best["edge_rule"], "merge_strategy":"complete_link",
        "event_macro_f1_hungarian":          float(sym["gold_to_pred_macro_f1"]),
        "event_macro_f1_hungarian_pred_to_gold": float(sym["pred_to_gold_macro_f1"]),
        "event_macro_f1_hungarian_symmetric": float(sym["symmetric_macro_f1"]),
        "b_cubed_f1":float(b3["b_cubed_f1"]),
        "b_cubed_precision":float(b3["b_cubed_precision"]),
        "b_cubed_recall":float(b3["b_cubed_recall"]),
        "ari":float(ar),
        "induced_pair_f1":float(ipf["induced_pair_f1"]),
        "induced_pair_precision":float(ipf["induced_pair_precision"]),
        "induced_pair_recall":float(ipf["induced_pair_recall"]),
        "pair_macro_f1":float(f1_score(tst_y,yp,average="macro")),
        "pair_f1_pos":float(f1_score(tst_y,yp,pos_label=1,zero_division=0)),
        "pair_precision_pos":float(precision_score(tst_y,yp,pos_label=1,zero_division=0)),
        "pair_recall_pos":float(recall_score(tst_y,yp,pos_label=1,zero_division=0)),
        "num_pairs":len(tst_y), "num_pred_pos_pairs":int(yp.sum()),
        "num_pred_clusters":len(cl), "num_gold_events":len(prep["g_te"]),
    }, "n_pred":len(cl), "n_gold":len(prep["g_te"])}


# ── Main ──────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("="*60)
    print("逻辑回归基线：Dual Recall + 逻辑回归 + Complete-Link + 五折CV")
    print("="*60)

    with open(os.path.join(CV_SPLITS_DIR, "fold_splits.json")) as f:
        cv = json.load(f)
    print("Folds: {} | Device: {}".format(cv["n_folds"],
          "cuda:0" if torch.cuda.is_available() else "cpu"))

    print("\n[1] Loading data...")
    i2c, c2i, _ = load_eval_structure(STRUCTURE_FILE)
    c2cats = load_cases_minimal(CASES_FILE)
    q2c, q2gt, q2mm = load_recall_topk(RECALL_FILE, topm_out=TOPM_OUT)
    print("  Queries: {} | Wide dim: {}".format(
        len(q2c), len(get_wide_feature_names(PAPER_TEXT_FIELDS))))

    os.makedirs(TRAIN_OUT, exist_ok=True)
    os.makedirs(DECODE_OUT, exist_ok=True)

    results = []
    for fi in cv["folds"]:
        fn, fidx = fi["fold_name"], int(fi["test_fold_idx"])+1
        fdir = os.path.join(CV_SPLITS_DIR, fn)
        print("\n"+"-"*50)
        print("{} (fold {}/{}) | Test:{} Dev:{} Train:{}".format(
            fn, fidx, cv["n_folds"], fi["n_test_incidents"],
            fi["n_dev_incidents"], fi["n_train_incidents"]))

        t = time.time()
        print("[Step 1] Prepare...")
        prep = prepare_fold(fdir, fidx, q2c, q2gt, q2mm, i2c, c2i, c2cats)
        print("  Pairs: tr={} dv={} te={} ({:.0f}s)".format(prep["n_tr"], prep["n_dv"], prep["n_te"], time.time()-t))

        t = time.time()
        print("[Step 2] Train Wide-Only ({} epochs)...".format(EPOCHS))
        ckpt = train_fold(prep, fidx)
        print("  Best: {} ({:.0f}s)".format(ckpt, time.time()-t))

        t = time.time()
        print("[Step 3] Decode + Eval...")
        res = decode_fold(prep, fidx, ckpt)
        tm = res["test_metrics"]
        print("  Hungarian F1(G->P):{:.4f}  B-cubed:{:.4f}  ARI:{:.4f}  tau:{:.4f} ({:.0f}s)".format(
            tm["event_macro_f1_hungarian"], tm["b_cubed_f1"], tm["ari"],
            res["best_dev"]["threshold"], time.time()-t))

        r = {"fold_name":fn, "fold_idx":fidx,
             "n_test":fi["n_test_incidents"], "best_ckpt":ckpt,
             "best_dev":res["best_dev"], "test_metrics":tm,
             "n_pred":res["n_pred"], "n_gold":res["n_gold"]}
        _json_write(os.path.join(DECODE_OUT, fn, "result.json"), r)
        results.append(r)

    # ── Summary ──
    print("\n"+"="*60); print("五折 CV 汇总"); print("="*60)
    keys = ["event_macro_f1_hungarian","event_macro_f1_hungarian_symmetric",
            "b_cubed_f1","b_cubed_precision","b_cubed_recall",
            "ari","induced_pair_f1",
            "pair_macro_f1","pair_f1_pos","pair_precision_pos","pair_recall_pos"]
    sm = {}
    for k in keys:
        vs = [r["test_metrics"][k] for r in results]
        sm[k] = {"mean":float(np.mean(vs)),"std":float(np.std(vs,ddof=1)),"values":vs}
        print("  {:45s} {:.4f} +- {:.4f}".format(k, sm[k]["mean"], sm[k]["std"]))

    _json_write(os.path.join(DECODE_OUT, "summary.json"), {
        "method":"Dual Recall + Wide-Only + Complete-Link",
        "cv_type":"5-fold","n_folds":len(results),
        "recall_mode":"dual","model":"logistic_regression",
        "test_metrics_across_folds":sm})
    print("\nDone. {:.0f}s".format(time.time()-t0))


if __name__ == "__main__":
    main()
