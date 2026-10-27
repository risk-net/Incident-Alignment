#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""逻辑回归基线 —— 统一入口（英文五折训练 + 中文零样本，一次产出对照表）。

唯一的基线就是 LR：17 维手工特征（7 个类别一致位 + 10 个文本字段余弦）
→ ``nn.Linear(17, 1)``，刻意去掉 Deep MLP。与主方法 DeepWide 构成**唯一变量**对照：
召回、pair 构建、负采样、epoch/bs/lr/wd、dev 选 checkpoint、τ 网格、
complete-link 聚类、指标**全部相同**，只换分类器。

真实训练与评测逻辑在 ``baseline_logistic_regression_5fold.py``；
本脚本是它的驱动层，负责：

  1. 按顺序跑「英文五折」→「中文零样本」，并检查前置条件；
  2. 把两份结果汇总成一张 ``comparison.csv``，并把 DeepWide 的既有产物
     作为参照列读进来（只读，不改动 ``outputs/english``、``outputs/chinese``）；
  3. 记录 ``session.json``（命令、输入文件 sha256、运行环境），便于事后核对。

口径（与主方法一致，勿改）
--------------------------
  TOPM_OUT=200
  τ 网格 0.10–0.95 / step 0.01，edge_rule=mutual，complete-link
  英文：dev 上用 Hungarian G→P 选 τ（与 outputs/dev/locked_thetas.json 同口径）
  中文：直接复用 outputs/dev/locked_thetas.json 里锁定的每折 τ

用法
----
    # ① 英文五折训练 + 评测（约 1–2 小时 CPU）；② 中文零样本评测。一步到位：
    python research_pipeline/Incident_Align_Method/baselines/run_baselines.py --all

    # 只跑英文（先看英文结果是否合理，再决定要不要跑中文）
    python .../run_baselines.py --datasets english

    # 英文已训练完，只补跑中文零样本（秒级）
    python .../run_baselines.py --datasets chinese

    # 指定折与设备
    python .../run_baselines.py --all --folds 1,2,3,4,5 --device cuda:0

    # 只重出对照表，不重跑任何计算
    python .../run_baselines.py --reuse-only

输出
----
    outputs/benchmarks/logistic_regression/
      ├── fold_01/… fold_05/         每折训练日志权重 + result.json（英文 test 指标）
      ├── summary.json               英文五折 mean ± std
      └── chinese/
          ├── fold_0X_scores.jsonl   中文 pair 分数
          ├── fold_0X_result.json    中文每折指标
          └── summary.json           中文 mean ± std
    outputs/benchmarks/
      ├── comparison.csv  ★ 对照表：LR vs DeepWide × 英文/中文
      ├── comparison.json
      └── session.json
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# ------------------------------------------------------------------ 路径
BASE_DIR = Path(__file__).resolve().parents[3]
PAIRWISE_DIR = BASE_DIR / "research_pipeline" / "Incident_Align_Method" / "pairwise_and_clustering"
BASELINES_DIR = BASE_DIR / "research_pipeline" / "Incident_Align_Method" / "baselines"
if str(PAIRWISE_DIR) not in sys.path:
    sys.path.insert(0, str(PAIRWISE_DIR))


LR_SCRIPT = BASELINES_DIR / "baseline_logistic_regression_5fold.py"
LR_OUT_DIR = BASE_DIR / "outputs" / "benchmarks" / "logistic_regression"
BENCH_DIR = BASE_DIR / "outputs" / "benchmarks"

# 论文两张表对应的指标组
#
# ① Pairwise 分类表：给定两篇新闻，判断是否同一事件。
#    注意：这些是在**聚类阈值 τ** 下算出的 pair 判别指标（y_pred = probs >= τ），
#    τ 由 dev 的聚类指标选出，并非为 pair F1 单独调优。论文中需如实说明。
PAIRWISE_KEYS = [
    "pair_accuracy",
    "pair_macro_f1",
    "pair_f1_pos",
    "pair_precision_pos",
    "pair_recall_pos",
    "num_pairs",
]

# ② event_metrics 表：完整聚类（召回 → 打分 → Complete-Link）恢复事件集合的能力
EVENT_KEYS = [
    "hungarian_f1_g2p",             # 主指标：Hungarian G→P event macro-F1
    "hungarian_f1_p2g",
    "hungarian_f1_symmetric",
    "b_cubed_f1",
    "b_cubed_precision",
    "b_cubed_recall",
    "induced_pair_f1",
    "induced_pair_precision",
    "induced_pair_recall",
    "ari",
    "num_pred_clusters",
    "num_gold_events",
    "threshold",
]

# 兼容旧名：完整对照表 = 两张表指标的并集
METRIC_KEYS = PAIRWISE_KEYS + EVENT_KEYS


# ------------------------------------------------------------------ 小工具
def log(msg: str = ""):
    print(msg, flush=True)


def section(title: str):
    log("\n" + "=" * 78)
    log(title)
    log("=" * 78)


def json_write(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def mean_std(values):
    arr = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if arr.size == 0:
        return None
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "values": [float(x) for x in arr],
    }


# ------------------------------------------------------------------ 结果读取
def collect_lr_english(folds) -> dict | None:
    """读 LR 英文五折结果（由 baseline_logistic_regression_5fold.py 写）。"""
    per_fold = {}
    for f in folds:
        rp = LR_OUT_DIR / f"fold_{f:02d}" / "result.json"
        if not rp.is_file():
            continue
        rec = json.loads(rp.read_text(encoding="utf-8"))
        m = dict(rec["test_metrics"])
        m["best_epoch"] = rec.get("best_epoch")
        m["pairs"] = rec.get("pairs")
        per_fold[f"fold_{f:02d}"] = m
    if not per_fold:
        return None
    agg = {}
    for k in METRIC_KEYS:
        vals = [per_fold[x].get(k) for x in sorted(per_fold)]
        st = mean_std(vals)
        if st:
            agg[k] = st
    return {
        "baseline": "logistic_regression",
        "label": "B3 logistic regression on 17 handcrafted features (Deep+Wide → linear)",
        "dataset": "english",
        "dataset_label": "English 5-fold CV (each fold scored on its held-out test fold)",
        "n_folds": len(per_fold),
        "folds": sorted(per_fold),
        "aggregate": agg,
        "source": str((LR_OUT_DIR / "summary.json").relative_to(BASE_DIR)),
        "per_fold": per_fold,
        "note": "same recall / pairs / negative sampling / epochs / tau grid as DeepWide; only the classifier differs",
    }


def collect_lr_chinese(folds) -> dict | None:
    per_fold = {}
    for f in folds:
        rp = LR_OUT_DIR / "chinese" / f"fold_{f:02d}_result.json"
        if not rp.is_file():
            continue
        per_fold[f"fold_{f:02d}"] = json.loads(rp.read_text(encoding="utf-8"))
    if not per_fold:
        return None
    agg = {}
    for k in METRIC_KEYS:
        st = mean_std([per_fold[x].get(k) for x in sorted(per_fold)])
        if st:
            agg[k] = st
    return {
        "baseline": "logistic_regression",
        "label": "B3 logistic regression on 17 handcrafted features (Deep+Wide → linear)",
        "dataset": "chinese",
        "dataset_label": "Chinese zero-shot (1,283 reports / 185 gold events; weights from English folds)",
        "n_folds": len(per_fold),
        "folds": sorted(per_fold),
        "aggregate": agg,
        "source": str((LR_OUT_DIR / "chinese" / "summary.json").relative_to(BASE_DIR)),
        "per_fold": per_fold,
        "zero_shot": True,
        "note": "English-trained weights applied to Chinese; tau from outputs/dev/locked_thetas.json",
    }


def load_deepwide_reference() -> dict:
    """DeepWide 既有产物 → 参照列（只读，不重算、不改动）。"""
    ref = {}
    # 参照产物与本项目使用同一套规范键名（hungarian_f1_* 等）
    ref_keys = [
        "hungarian_f1_g2p", "hungarian_f1_p2g", "hungarian_f1_symmetric",
        "b_cubed_f1", "b_cubed_precision", "b_cubed_recall",
        "induced_pair_f1", "induced_pair_precision", "induced_pair_recall",
        "ari", "num_pred_clusters",
    ]
    locked = {}
    lp = BASE_DIR / "outputs/dev/locked_thetas.json"
    if lp.is_file():
        locked = json.loads(lp.read_text(encoding="utf-8")).get("locked_thetas", {})

    for dataset, rel, label in [
        ("english", "outputs/english/graph_decode_5fold/metrics.json",
         "DeepWide pairwise — main method (Deep+Wide MLP)"),
        ("chinese", "outputs/chinese/metrics.json",
         "DeepWide pairwise — main method, Chinese zero-shot"),
    ]:
        p = BASE_DIR / rel
        if not p.is_file():
            log(f"  （跳过参照 {dataset}：{rel} 不存在）")
            continue
        obj = json.loads(p.read_text(encoding="utf-8"))
        agg_src = obj.get("aggregate", {})
        agg = {}
        for k in ref_keys:
            if k in agg_src:
                agg[k] = {kk: agg_src[k][kk] for kk in ("n", "mean", "std", "values")
                          if kk in agg_src[k]}

        # metrics.json 只汇总了聚类指标；pair 级指标在每个 fold 的 result.json 里
        fold_dir = p.parent
        pair_vals = {}
        for r in obj.get("per_run", []):
            fj = fold_dir / f"fold_{int(r.get('fold', 0)):02d}" / "result.json"
            if not fj.is_file():
                continue
            tm = json.loads(fj.read_text(encoding="utf-8")).get("test_metrics", {})
            for k in PAIRWISE_KEYS:
                if k in tm:
                    pair_vals.setdefault(k, []).append(float(tm[k]))
        for k, vals in pair_vals.items():
            st = mean_std(vals)
            if st:
                agg[k] = st

        # 中文没有每折 result.json，pair 指标由 fold_0X_scores.jsonl 单独算好存此文件
        pw = fold_dir / "pairwise_metrics.json"
        if pw.is_file():
            pw_agg = json.loads(pw.read_text(encoding="utf-8")).get("aggregate", {})
            for k in PAIRWISE_KEYS:
                if k in pw_agg and k not in agg:
                    agg[k] = pw_agg[k]

        if locked:
            agg["threshold"] = mean_std([float(v) for v in locked.values()])
        ref[dataset] = {
            "baseline": "deepwide",
            "label": label,
            "dataset": dataset,
            "n_folds": len(obj.get("per_run", [])) or None,
            "folds": [f"fold_{r.get('fold', i + 1):02d}" for i, r in enumerate(obj.get("per_run", []))],
            "aggregate": agg,
            "source": rel,
            "note": "existing released artifact (not recomputed by this run)",
        }
    return ref


# ------------------------------------------------------------------ 对照表
def build_comparison(entries: list, out_csv: Path) -> dict:
    rows = []
    for e in entries:
        agg = e.get("aggregate", {})
        row = {
            "dataset": e["dataset"],
            "method": e["baseline"],
            "label": e["label"],
            "n_folds": e.get("n_folds"),
        }
        for k in METRIC_KEYS:
            row[k] = (agg.get(k) or {}).get("mean")
            row[f"{k}_std"] = (agg.get(k) or {}).get("std")
        row["source"] = e.get("source", "this run")
        row["note"] = e.get("note", "")
        rows.append(row)

    order = {"logistic_regression": 0, "deepwide": 1}
    rows.sort(key=lambda r: (0 if r["dataset"] == "english" else 1, order.get(r["method"], 9)))

    deltas = []
    for dataset in ("english", "chinese"):
        ds = {r["method"]: r for r in rows if r["dataset"] == dataset}
        if "deepwide" not in ds or "logistic_regression" not in ds:
            continue
        d = {"dataset": dataset,
             "comparison": "DeepWide − LR (positive = main method better)"}
        for k in METRIC_KEYS:
            a, b = ds["deepwide"].get(k), ds["logistic_regression"].get(k)
            d[k] = None if (a is None or b is None) else float(a) - float(b)
        deltas.append(d)

    out = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "metric_definitions": {
            "hungarian_f1_g2p": "Hungarian-matched event macro-F1, gold→pred (main metric)",
            "hungarian_f1_p2g": "Hungarian-matched event macro-F1, pred→gold",
            "hungarian_f1_symmetric": "symmetric Hungarian macro-F1",
            "b_cubed_f1": "B-cubed F1",
            "induced_pair_f1": "pairwise F1 induced by predicted clusters vs gold",
            "ari": "adjusted Rand index",
            "threshold": "clustering threshold tau",
        },
        "rows": rows,
        "deepwide_minus_lr": deltas,
    }
    def _write_table(keys, out_path, title):
        cols = ["dataset", "method", "label", "n_folds"]
        for k in keys:
            cols += [k, f"{k}_std"]
        cols += ["source", "note"]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow({k: ("" if r.get(k) is None else
                                (f"{r[k]:.6f}" if isinstance(r[k], float) else r[k])) for k in cols})
        return {"title": title, "csv": str(out_path.relative_to(BASE_DIR)), "metrics": keys}

    out["tables"] = {
        "pairwise": _write_table(PAIRWISE_KEYS,
                                 out_csv.parent / "comparison_pairwise.csv",
                                 "Pairwise classification (same-event judgement)"),
        "event": _write_table(EVENT_KEYS,
                              out_csv.parent / "comparison_event.csv",
                              "Event clustering (full pipeline)"),
    }
    columns = ["dataset", "method", "label", "n_folds"]
    for k in METRIC_KEYS:
        columns += [k, f"{k}_std"]
    columns += ["source", "note"]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else
                            (f"{r[k]:.6f}" if isinstance(r[k], float) else r[k])) for k in columns})
    out["csv"] = str(out_csv.relative_to(BASE_DIR))
    return out


def print_comparison(cmp: dict):
    def f4(x):
        return "—" if x is None else f"{x:.4f}"

    # ---------------- 表 ① Pairwise 分类 ----------------
    section("表 ①  Pairwise 分类（判断两篇新闻是否同一事件）")
    h1 = (f"{'dataset':9s} {'method':21s} {'acc':>8s} {'macro_f1':>9s} "
          f"{'F1(pos)':>8s} {'Prec':>8s} {'Rec':>8s}")
    log(h1)
    log("-" * len(h1))
    for r in cmp["rows"]:
        log(f"{r['dataset']:9s} {r['method']:21s} "
            f"{f4(r.get('pair_accuracy')):>8s} {f4(r.get('pair_macro_f1')):>9s} "
            f"{f4(r.get('pair_f1_pos')):>8s} {f4(r.get('pair_precision_pos')):>8s} "
            f"{f4(r.get('pair_recall_pos')):>8s}")
    log("")
    log("  ⚠️  pair 指标是在**聚类阈值 τ** 下计算的（y_pred = prob ≥ τ），τ 由 dev 的聚类指标选出，")
    log("      并非为 pair F1 单独调优。论文中需如实说明这一点。")

    # ---------------- 表 ② event_metrics ----------------
    section("表 ② event_metrics（完整聚类：召回 → 打分 → Complete-Link）")
    h2 = (f"{'dataset':9s} {'method':21s} {'folds':>5s} {'g2p':>8s} {'±':>7s} "
          f"{'sym':>8s} {'b3':>8s} {'induced':>8s} {'ari':>8s} {'n_clu':>7s} {'τ':>6s}")
    log(h2)
    log("-" * len(h2))
    for r in cmp["rows"]:
        nclu = r.get("num_pred_clusters")
        tau = r.get("threshold")
        g2p_std = r.get("hungarian_f1_g2p_std")
        log(f"{r['dataset']:9s} {r['method']:21s} {str(r['n_folds'] or '—'):>5s} "
            f"{f4(r.get('hungarian_f1_g2p')):>8s} "
            f"{('—' if g2p_std is None else f'{g2p_std:.4f}'):>7s} "
            f"{f4(r.get('hungarian_f1_symmetric')):>8s} "
            f"{f4(r.get('b_cubed_f1')):>8s} "
            f"{f4(r.get('induced_pair_f1')):>8s} "
            f"{f4(r.get('ari')):>8s} "
            f"{('—' if nclu is None else f'{nclu:.1f}'):>7s} "
            f"{('—' if tau is None else f'{tau:.2f}'):>6s}")
    log("")
    log("  g2p = Hungarian G→P event macro-F1（主指标）｜sym = 对称 macro-F1｜b3 = B-cubed F1")
    log("  induced = 预测簇诱导的 pair F1｜ari = ARI｜n_clu = 预测簇数｜τ = 聚类阈值")
    log("")
    log(f"→ {cmp['csv']}（全指标合并表）")
    for name, t in (cmp.get("tables") or {}).items():
        log(f"→ {t['csv']}")

    if cmp.get("deepwide_minus_lr"):
        log("\nDeepWide − LR（正数 = 主方法更好）")
        for d in cmp["deepwide_minus_lr"]:
            parts = []
            for k, name in [("hungarian_f1_g2p", "Δg2p"),
                            ("hungarian_f1_symmetric", "Δsym"),
                            ("b_cubed_f1", "Δb3"),
                            ("induced_pair_f1", "Δinduced"),
                            ("ari", "Δari")]:
                v = d.get(k)
                parts.append(f"{name}={'—' if v is None else f'{v:+.4f}'}")
            log(f"  {d['dataset']:8s}  " + "  ".join(parts))


# ------------------------------------------------------------------ session
def write_session(args, stages_run):
    files = {}
    for rel in [
        "data/english/test/recall_topk.jsonl",
        "data/english/test/cases.jsonl",
        "data/precomputed/eval_structure.json",
        "data/chinese_eval/recall_topk.jsonl",
        "data/chinese_eval/cases.jsonl",
        "data/chinese_gold/gold_v2.0.2_mapping.csv",
        "outputs/dev/locked_thetas.json",
    ]:
        p = BASE_DIR / rel
        if p.is_file():
            files[rel] = {"sha256": sha256_file(p), "bytes": p.stat().st_size}
    for f in range(1, 6):
        p = BASE_DIR / f"data/english/folds/fold_{f:02d}/train/repeat_01/train_pairs.jsonl"
        if p.is_file():
            files[str(p.relative_to(BASE_DIR))] = {"sha256": sha256_file(p), "bytes": p.stat().st_size}

    session = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "argv": " ".join(shlex.quote(c) for c in sys.argv),
        "stages_run": stages_run,
        "args": {k: v for k, v in vars(args).items()},
        "conventions": {
            "topm_out": 200,
            "threshold_grid": {"min": 0.10, "max": 0.95, "step": 0.01},
            "edge_rule": "mutual",
            "merge_strategy": "complete_link",
            "english_selection": "dev tau maximising Hungarian G→P",
            "chinese_selection": "outputs/dev/locked_thetas.json",
        },
        "inputs": files,
        "environment": {
            "python": sys.version.split()[0],
            "executable": sys.executable,
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
    }
    try:
        import torch

        session["environment"]["torch"] = torch.__version__
        session["environment"]["cuda_available"] = bool(torch.cuda.is_available())
    except Exception as exc:  # noqa: BLE001
        session["environment"]["torch"] = f"unavailable ({exc})"
    json_write(BENCH_DIR / "session.json", session)
    return session


# ------------------------------------------------------------------ 配置文件
CONFIG_SECTION = "BaselineLogisticRegression"
_CONFIG_DEFAULTS = {"datasets": "english,chinese", "folds": "1,2,3,4,5", "device": "cpu",
                    "threads": "1", "epochs": "8", "all": "false", "no_reference": "false", "reuse_only": "false"}


def _apply_config_file(args):
    """把 ini 里的值填入还没被命令行显式指定的参数（命令行优先）。"""
    import configparser

    raw = Path(args.config)
    path = raw if raw.is_absolute() else (BASE_DIR / raw)
    if not path.is_file():
        raise SystemExit(f"❌ 配置文件不存在：{path}")
    cp = configparser.ConfigParser(inline_comment_prefixes=(";", "#"))
    cp.read(str(path), encoding="utf-8")
    if not cp.has_section(CONFIG_SECTION):
        raise SystemExit(f"❌ 配置文件缺少段落 [{CONFIG_SECTION}]：{path}")

    explicit = _explicit_cli_flags(sys.argv[1:])
    log(f"读取配置：{path.relative_to(BASE_DIR) if path.is_relative_to(BASE_DIR) else path}")
    for key in _CONFIG_DEFAULTS:
        if f"--{key.replace('_', '-')}" in explicit:
            continue
        if cp.has_option(CONFIG_SECTION, key):
            setattr(args, key, cp.get(CONFIG_SECTION, key).strip())
    # 布尔项从字符串转回 bool
    for key in ("all", "no_reference", "reuse_only"):
        v = getattr(args, key)
        if isinstance(v, str):
            setattr(args, key, v.strip().lower() in {"1", "true", "yes", "on"})
    return args


def _explicit_cli_flags(argv):
    flags = set()
    for tok in argv:
        if tok.startswith("--"):
            flags.add(tok.split("=", 1)[0])
    return flags


# ------------------------------------------------------------------ 阶段执行
def check_prereqs():
    missing = []
    if not LR_SCRIPT.is_file():
        missing.append(str(LR_SCRIPT.relative_to(BASE_DIR)))
    for rel in ["data/english/folds/fold_01/train/repeat_01/train_pairs.jsonl",
                "data/english/test/recall_topk.jsonl",
                "data/english/test/embeddings",
                "data/precomputed/eval_structure.json",
                "outputs/dev/locked_thetas.json"]:
        if not (BASE_DIR / rel).exists():
            missing.append(rel)
    if missing:
        raise SystemExit("❌ 缺少前置资产：\n   - " + "\n   - ".join(missing))
    try:
        import torch  # noqa: F401
    except Exception:
        raise SystemExit(
            "❌ 当前 Python 环境没有 torch。\n"
            f"   当前解释器：{sys.executable}\n"
            "   可用带 torch 的环境，例如：\n"
            "   /home/nlper/miniconda3/envs/legal-DC/bin/python "
            "research_pipeline/Incident_Align_Method/baselines/run_baselines.py --all"
        )


def _subprocess_env(threads: int) -> dict:
    """给子进程注入线程数限制。

    LR 的矩阵是 512×17，BLAS 在几十上百线程下的同步开销远大于收益
    （实测 121 线程只吃到 1.76 核）。默认限制为 1。
    """
    env = dict(os.environ)
    if threads and threads > 0:
        for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                  "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            env[k] = str(threads)
        env["THREADS"] = str(threads)
    return env


def _run_lr_stage(extra_args, folds, device, threads, title, epochs=None):
    section(title)
    # `-u`：子进程 stdout 无缓冲，配合 `| tee` 也能实时看到进度
    cmd = [sys.executable, "-u", str(LR_SCRIPT),
           "--folds", ",".join(str(f) for f in folds),
           "--device", device,
           "--topm", "200",
           "--out-dir", str(LR_OUT_DIR.relative_to(BASE_DIR))]
    if epochs:
        cmd += ["--epochs", str(epochs)]
    cmd += extra_args
    env = _subprocess_env(threads)
    log("命令：")
    log(f"$ OMP_NUM_THREADS={threads} " + " ".join(shlex.quote(c) for c in cmd))
    t = time.time()
    rc = subprocess.call(cmd, cwd=str(BASE_DIR), env=env)
    log(f"阶段结束 exit={rc}，耗时 {time.time() - t:.0f}s")
    return rc


def run_english_lr(folds, device, threads=1, epochs=8):
    log(f"（{epochs} epoch × {len(folds)} 折；每 10% batch 打一条进度到 stderr，实时可见；"
        "可 Ctrl-C 后分折重跑 --folds）")
    return _run_lr_stage(["--channel", "english"], folds, device, threads,
                         "① 英文五折 LR 训练 + 评测", epochs=epochs)


def run_chinese_lr(folds, device, threads=1):
    missing = [f for f in folds if not (LR_OUT_DIR / f"fold_{f:02d}" / "result.json").is_file()]
    if missing:
        log(f"⚠️  以下折缺少英文训练结果，中文评测会跳过它们：{missing}")
        folds = [f for f in folds if f not in missing]
    if not folds:
        log("❌ 没有任何可用的英文折权重，跳过中文阶段。请先跑 --datasets english")
        return 1
    # 中文阶段只做推理，--epochs 不影响；best_epoch 从 fold_XX/result.json 读
    return _run_lr_stage(["--channel", "chinese", "--skip-train"], folds, device, threads,
                         "② 中文零样本 LR 评测（复用英文五折权重）")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(
        description="逻辑回归基线：英文五折训练 + 中文零样本，统一产出与 DeepWide 的对照表",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--datasets", default="english,chinese",
                    help="逗号分隔：english（训练+评测）, chinese（零样本）。默认两个都跑")
    ap.add_argument("--all", action="store_true", help="等价于 --datasets english,chinese")
    ap.add_argument("--folds", default="1,2,3,4,5", help="实际跑哪几折")
    ap.add_argument("--device", default="cpu", help="cpu / cuda:0")
    ap.add_argument("--epochs", type=int, default=8,
                    help="LR 训练轮数，默认 8（= 主方法同预算）；"
                         "LR 参数少、8 轮常未收敛，补跑收敛版用 20")
    ap.add_argument("--threads", type=int, default=1,
                    help="子进程 BLAS/torch 线程数。模型只有 17 维，默认 1（多线程反而更慢）；0=不限制")
    ap.add_argument("--no-reference", action="store_true", help="不读 DeepWide 既有产物做参照列")
    ap.add_argument("--reuse-only", action="store_true",
                    help="不跑任何计算，只用已有产物重出对照表")
    ap.add_argument("--config", default=None,
                    help="ini 配置文件（段落 [BaselineLogisticRegression]）；命令行参数优先级更高")
    args = ap.parse_args()

    if args.config:
        args = _apply_config_file(args)

    if args.all:
        args.datasets = "english,chinese"
    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    bad = set(datasets) - {"english", "chinese"}
    if bad:
        raise SystemExit(f"❌ 未知 dataset: {bad}（只能是 english / chinese）")
    folds = [int(x) for x in args.folds.split(",") if x.strip()]

    t0 = time.time()
    section("逻辑回归基线（LR vs DeepWide）")
    log(f"仓库   : {BASE_DIR}")
    log(f"数据   : {datasets}    折: {folds}    设备: {args.device}")
    log(f"口径   : TOPM_OUT=200  τ=0.10–0.95/0.01  edge_rule=mutual  complete-link")
    log(f"输出   : outputs/benchmarks/")

    stages = []
    rc = 0
    if not args.reuse_only:
        check_prereqs()
        if "english" in datasets:
            rc = run_english_lr(folds, args.device, args.threads, args.epochs)
            stages.append({"stage": "english_lr_train", "exit": rc,
                           "duration_s": round(time.time() - t0, 1)})
            if rc != 0:
                log("⚠️  英文阶段非零退出，仍尝试汇总已有结果。")
        if "chinese" in datasets:
            rc2 = run_chinese_lr(folds, args.device, args.threads)
            stages.append({"stage": "chinese_lr_zeroshot", "exit": rc2})
            rc = rc or rc2

    # ---------- 汇总 ----------
    section("汇总")
    entries = []
    eng = collect_lr_english(folds)
    if eng:
        log(f"LR 英文 : {eng['n_folds']} 折  ← {eng['source']}")
        entries.append(eng)
    else:
        log("LR 英文 : 未找到结果（先跑 --datasets english）")
    zh = collect_lr_chinese(folds)
    if zh:
        log(f"LR 中文 : {zh['n_folds']} 折  ← {zh['source']}")
        entries.append(zh)
    else:
        log("LR 中文 : 未找到结果（先跑 --datasets chinese）")

    ref = {} if args.no_reference else load_deepwide_reference()
    for k, v in ref.items():
        log(f"参照    : DeepWide/{k} ← {v['source']}")

    if entries:
        cmp = build_comparison(entries + list(ref.values()),
                               BENCH_DIR / "comparison.csv")
        json_write(BENCH_DIR / "comparison.json", {k: v for k, v in cmp.items() if k != "rows"}
                   | {"rows": [{kk: vv for kk, vv in r.items()} for r in cmp["rows"]]})
        json_write(LR_OUT_DIR / "comparison_with_deepwide.json", cmp)
        print_comparison(cmp)
    else:
        log("\n⚠️  还没有任何 LR 结果，无法出对照表。")
        log("   先运行：python research_pipeline/Incident_Align_Method/baselines/run_baselines.py --all")

    write_session(args, stages)

    section("完成")
    log(f"输出目录 : outputs/benchmarks/")
    log(f"总耗时   : {time.time() - t0:.0f}s")
    if not entries:
        return 1
    return rc


if __name__ == "__main__":
    # 行缓冲：被 `| tee` 接管时也能实时看到输出
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(line_buffering=True)
        except Exception:  # noqa: BLE001
            pass
    sys.exit(main())
