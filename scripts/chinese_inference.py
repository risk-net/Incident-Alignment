#!/usr/bin/env python3
"""中文零样本评测：用英文五折 checkpoint 对中文候选对打分。

这一步是「checkpoint → 分数」的桥，**只做推理，不做聚类、不做指标**。
聚类与指标交给 `incident-align evaluate-folds --dataset chinese`。

为什么要单独一个脚本
--------------------
中文评测是**跨语言零样本**：模型全部在英文五折上训练，从未见过中文。
五个 checkpoint 各自对**同一份 1,283 篇**中文域打分，因此：

  - 中文只有一份 Gold（`gold_v2.0.2_mapping.csv`），五折共用；
  - 每折产出一份独立的分数文件，评测时报 mean ± std。

流程
----
    recall + cases + embeddings + checkpoint
        → DeepWide 前向（sigmoid 概率）
        → outputs/chinese/fold_0X_scores.jsonl
          { "query_id": "...", "candidate_id": "...", "probability": 0.83 }

口径约束（务必保持默认）
------------------------
- `--topm 250`：与原始中文评测一致（中文 recall 平均 195 候选，250 基本不截断）。
  改成 200 会切掉约 9,000 对，结果与已发表数值不可比。
- checkpoint 与折号必须一一对应，不可错配。

用法
----
    # 五折全跑（CPU 可用，约 20–40 分钟）
    python scripts/chinese_inference.py

    # 只跑单折 / 指定设备
    python scripts/chinese_inference.py --folds 1 --device cpu
    python scripts/chinese_inference.py --folds 1,2,3,4,5 --device cuda:0

    # 只检查输入是否就绪，不推理
    python scripts/chinese_inference.py --dry-run
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_gold_ids(path):
    import csv
    ids = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            ids.append(str(row["report_id"]).strip())
    return sorted(set(ids), key=lambda x: (len(x), x))


def main():
    ap = argparse.ArgumentParser(description="中文零样本推理 -> M6 分数 JSONL")
    ap.add_argument("--root", default=None, help="仓库根目录，默认由脚本位置推断")
    ap.add_argument("--folds", default="1,2,3,4,5")
    ap.add_argument("--device", default="cpu", help="cpu / cuda:0 / cuda")
    ap.add_argument("--topm", type=int, default=250, help="每 query 保留候选数（默认 250，勿改）")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--out-dir", default="outputs/chinese")
    ap.add_argument("--gold", default="data/chinese_gold/gold_v2.0.2_mapping.csv")
    ap.add_argument("--cases", default="data/chinese_eval/cases.jsonl")
    ap.add_argument("--recall", default="data/chinese_eval/recall_topk.jsonl")
    ap.add_argument("--embeddings", default="data/chinese_eval/embeddings")
    ap.add_argument("--checkpoint-dir", default="checkpoints/deepwide")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = Path(args.root).resolve() if args.root else Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))

    def P(rel):
        p = Path(rel)
        return p if p.is_absolute() else root / p

    gold_p, cases_p, recall_p = P(args.gold), P(args.cases), P(args.recall)
    emb_p, ckpt_dir = P(args.embeddings), P(args.checkpoint_dir)
    out_dir = P(args.out_dir)
    folds = [int(x) for x in args.folds.split(",") if x.strip()]

    # ---------------- 输入检查 ----------------
    print(f"仓库根目录 : {root}")
    missing = [str(p) for p in (gold_p, cases_p, recall_p, emb_p / "case_ids.txt") if not p.exists()]
    for f in folds:
        ck = ckpt_dir / f"fold_{f:02d}.pt"
        if not ck.exists():
            missing.append(str(ck))
    if missing:
        sys.exit("❌ 缺少输入文件：\n" + "\n".join(f"   {m}" for m in missing))

    gold_ids = load_gold_ids(gold_p)
    print(f"Gold       : {len(gold_ids)} 篇")

    if args.dry_run:
        print("✅ 输入文件齐备（dry-run，未推理）")
        return

    # ---------------- 延迟导入 torch ----------------
    import torch
    # 推理走包内公开封装（deepwide 层），只有「数据加载」直接调用 legacy：
    # incident_alignment.data.recall 返回 RecallCandidate 对象，与 predict_probs
    # 期望的 (cid, sf, st, se) 元组不兼容，故 load_recall_topk/load_cases_minimal
    # 仍需使用 legacy 实现。
    from incident_alignment.deepwide import (
        load_model_checkpoint,
        open_embedding_store,
        predict_pairs,
    )
    from incident_alignment.deepwide.legacy import pairwise_data_io as dio
    from incident_alignment.deepwide.legacy import pairwise_model as pm

    cases = dio.load_cases_minimal(str(cases_p))
    query2cands, _, _ = dio.load_recall_topk(str(recall_p), topm_out=args.topm)

    # 严格域校验：中文评测域必须恰好是 Gold 的 1,283 篇
    if set(query2cands) != set(cases):
        sys.exit(f"❌ recall query 与 cases 域不一致："
                 f"recall={len(query2cands)}, cases={len(cases)}")
    if set(cases) != set(gold_ids):
        miss = sorted(set(gold_ids) - set(cases))[:5]
        sys.exit(f"❌ cases 与 Gold 域不一致（示例缺失: {miss}）")

    pairs = [{"q": q, "c": entry[0], "label": 0}
             for q, cands in query2cands.items() for entry in cands]
    print(f"Cases      : {len(cases)} 篇")
    print(f"召回       : {len(query2cands)} query, {len(pairs)} 候选对 (topm={args.topm})")
    print(f"Device     : {args.device}  (torch {torch.__version__}, "
          f"cuda={torch.cuda.is_available()})")

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []

    for fold in folds:
        ckpt_path = ckpt_dir / f"fold_{fold:02d}.pt"
        print(f"\n[fold {fold:02d}] {ckpt_path.name}")

        model, ckpt, _ = load_model_checkpoint(str(ckpt_path), device=args.device)
        model.eval()
        text_fields = ckpt.get("text_fields", dio.TEXT_FIELDS_DEFAULT)
        store = open_embedding_store(str(emb_p), text_fields)

        if sorted(store.case_ids) != sorted(cases):
            sys.exit(f"❌ embedding case_ids 与 cases 域不一致（fold {fold}）")

        probs, _, qc = predict_pairs(
            model=model, pair_source=pairs, store=store, cases=cases,
            text_fields=text_fields, device=args.device,
            batch_size=args.batch_size, num_pairs=len(pairs),
            paper_text_fields=ckpt.get("paper_text_fields", pm.PAPER_TEXT_FIELDS_DEFAULT),
        )
        probs = np.asarray(probs, dtype=np.float64)

        # 严格校验
        if len(qc) != len(probs):
            sys.exit(f"❌ fold {fold}: qc 与 probs 长度不一致")
        if not np.all(np.isfinite(probs)):
            sys.exit(f"❌ fold {fold}: 出现非有限概率")
        out_of_range = int(((probs < 0) | (probs > 1)).sum())
        if out_of_range:
            sys.exit(f"❌ fold {fold}: {out_of_range} 个概率越界 [0,1]")

        out_path = out_dir / f"fold_{fold:02d}_scores.jsonl"
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as fo:
            for (q, c), p in zip(qc, probs):
                fo.write(json.dumps({"query_id": str(q).strip(),
                                     "candidate_id": str(c).strip(),
                                     "probability": float(p)}, ensure_ascii=False) + "\n")
        tmp.replace(out_path)

        manifest_rows.append({
            "fold": fold,
            "checkpoint": str(ckpt_path.relative_to(root)),
            "checkpoint_sha256": sha256_file(ckpt_path),
            "rows": int(len(probs)),
            "probability_min": float(probs.min()),
            "probability_max": float(probs.max()),
            "scores_sha256": sha256_file(out_path),
        })
        print(f"  → {len(probs)} 对  p∈[{probs.min():.6f}, {probs.max():.6f}]  {out_path.relative_to(root)}")

        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    manifest = {
        "note": "中文零样本推理产物清单。五折共用同一份 1,283 篇评测域，"
                "评测时以单份 Gold 求 mean±std。",
        "gold": str(gold_p.relative_to(root)),
        "gold_sha256": sha256_file(gold_p),
        "domain_reports": len(gold_ids),
        "topm": args.topm,
        "device": args.device,
        "folds": manifest_rows,
    }
    mp = out_dir / "chinese_inference_manifest.json"
    mp.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n清单 -> {mp.relative_to(root)}")
    print("\n下一步：")
    print("  incident-align evaluate-folds --dataset chinese \\")
    print(f"    --gold {args.gold} --expected-reports 1283 --expected-events 185 \\")
    print("    --locked outputs/dev/locked_thetas.json \\")
    print("    --fold-scores 1=outputs/chinese/fold_01_scores.jsonl \\")
    print("    --fold-scores 2=outputs/chinese/fold_02_scores.jsonl \\")
    print("    --fold-scores 3=outputs/chinese/fold_03_scores.jsonl \\")
    print("    --fold-scores 4=outputs/chinese/fold_04_scores.jsonl \\")
    print("    --fold-scores 5=outputs/chinese/fold_05_scores.jsonl \\")
    print("    --out outputs/chinese/evaluation")


if __name__ == "__main__":
    main()
