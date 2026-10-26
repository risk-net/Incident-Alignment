#!/usr/bin/env python3
"""按五折划分投影出每折的评测输入子集（方案 B）。

对每一折、每一 split（test / dev），从"全量"英文资产中抽取该折对应的子集：
  - cases.jsonl         : 全量英文 cases 中属于该 split 的行（保留原始顺序）
  - recall_topk.jsonl   : 只保留 query 属于该 split 的条目；候选也裁剪到该 split 域内
  - embeddings/         : case_ids.txt + 各字段 emb_*.npy / valid_mask_*.npy，
                          行序与 case_ids.txt 完全一致（按全量原始行序抽取）

对中文（单一评测集，无折划分）：
  - 生成一份整体 self-contained 子集（gold 全覆盖），用于严格域校验

这是**只读投影**，不修改任何源文件，不重新计算 embedding，也不重建召回。

用法：
    python scripts/make_fold_subsets.py --root . --folds 1,2,3,4,5
    python scripts/make_fold_subsets.py --root . --chinese
"""
import argparse, json, shutil
from pathlib import Path

import numpy as np


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_structure(path):
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    inc2cases = {str(e["incident_id"]): [str(x) for x in e["ids"]] for e in doc["events"]}
    return inc2cases


def split_cases(inc2cases, incidents):
    out = set()
    for inc in incidents:
        if inc not in inc2cases:
            raise KeyError(f"incident {inc} 不在 eval_structure 中")
        out.update(inc2cases[inc])
    return out


def build_subset(root, out_dir, allowed_cases, *, full_cases, full_recall, full_emb, tag):
    """按 allowed_cases 抽取 cases / recall / embeddings 子集。"""
    out_dir = Path(out_dir)
    (out_dir / "embeddings").mkdir(parents=True, exist_ok=True)
    allowed_cases = {str(c) for c in allowed_cases}

    # ---- cases ----
    n_cases = 0
    with full_cases.open(encoding="utf-8") as fin, (out_dir / "cases.jsonl").open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            o = json.loads(line)
            if str(o["id"]) in allowed_cases:
                fout.write(json.dumps(o, ensure_ascii=False) + "\n")
                n_cases += 1
    if n_cases != len(allowed_cases):
        raise RuntimeError(f"{tag}: cases 抽取 {n_cases} 条，但期望 {len(allowed_cases)} 条")

    # ---- recall ----
    n_queries = 0
    n_cands_in = n_cands_kept = 0
    with full_recall.open(encoding="utf-8") as fin, (out_dir / "recall_topk.jsonl").open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            o = json.loads(line)
            q = str(o.get("query_case_id") or o.get("query_id"))
            if q not in allowed_cases:
                continue
            cands = o.get("candidates") or []
            n_cands_in += len(cands)
            kept = [c for c in cands if str(c["case_id"]) in allowed_cases and str(c["case_id"]) != q]
            n_cands_kept += len(kept)
            o["candidates"] = kept
            if "ground_truth" in o and isinstance(o["ground_truth"], list):
                o["ground_truth"] = [g for g in o["ground_truth"] if str(g) in allowed_cases]
            fout.write(json.dumps(o, ensure_ascii=False) + "\n")
            n_queries += 1
    if n_queries != len(allowed_cases):
        raise RuntimeError(f"{tag}: recall query {n_queries} 条，但期望 {len(allowed_cases)} 条")

    # ---- embeddings ----
    full_ids = [l.strip() for l in (full_emb / "case_ids.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    if len(full_ids) != len(set(full_ids)):
        raise RuntimeError("全量 case_ids.txt 存在重复")
    keep_idx = [i for i, cid in enumerate(full_ids) if cid in allowed_cases]
    kept_ids = [full_ids[i] for i in keep_idx]
    if set(kept_ids) != allowed_cases:
        raise RuntimeError(f"{tag}: embedding 覆盖 {len(kept_ids)}，期望 {len(allowed_cases)}")
    # 保持全量原始行序（不排序），保证与 recall / cases 的 ID 顺序契约一致
    (out_dir / "embeddings" / "case_ids.txt").write_text("\n".join(kept_ids) + "\n", encoding="utf-8")
    idx = np.asarray(keep_idx, dtype=np.int64)

    n_arr = 0
    for npy in sorted(full_emb.glob("*.npy")):
        if npy.name == "case_ids.npy":
            # 只读复用既有资产的 ID 索引，允许 pickle；写出的 ID 以 case_ids.txt 为准
            arr = np.load(npy, allow_pickle=True)

        else:
            try:
                arr = np.load(npy, mmap_mode="r", allow_pickle=False)
            except ValueError:
                arr = np.load(npy, allow_pickle=True)
        if arr.ndim >= 1 and arr.shape[0] == len(full_ids):
            np.save(out_dir / "embeddings" / npy.name, np.asarray(arr)[idx])
            n_arr += 1
        else:
            shutil.copy2(npy, out_dir / "embeddings" / npy.name)  # 非按行对齐的附属数组
            n_arr += 1
    for extra in full_emb.glob("*.json"):
        shutil.copy2(extra, out_dir / "embeddings" / extra.name)

    return {
        "tag": tag, "cases": n_cases, "recall_queries": n_queries,
        "candidates_in": n_cands_in, "candidates_kept": n_cands_kept,
        "embedding_rows": len(kept_ids), "npy_files": n_arr,
    }


def main():
    ap = argparse.ArgumentParser(description="按五折划分投影评测子集（只读）")
    ap.add_argument("--root", default=".")
    ap.add_argument("--folds", default=None, help="逗号分隔折号，默认 1,2,3,4,5")
    ap.add_argument("--splits", default="test,dev")
    ap.add_argument("--chinese", action="store_true", help="额外生成中文整体子集")
    a = ap.parse_args()

    root = Path(a.root).resolve()
    inc2cases = load_structure(root / "data/precomputed/eval_structure.json")
    results = []

    folds = [int(x) for x in a.folds.split(",")] if a.folds else [1, 2, 3, 4, 5]
    splits = [s.strip() for s in a.splits.split(",") if s.strip()]

    full_cases = root / "data/english/test/cases.jsonl"
    full_recall = root / "data/english/test/recall_topk.jsonl"
    full_emb = root / "data/english/test/embeddings"

    for i in folds:
        fdir = root / "data/english/folds" / f"fold_{i:02d}"
        for split in splits:
            incs = [str(x) for x in json.loads((fdir / f"{split}_incidents.json").read_text())]
            allowed = split_cases(inc2cases, incs)
            out_dir = fdir / split
            res = build_subset(root, out_dir, allowed, full_cases=full_cases,
                               full_recall=full_recall, full_emb=full_emb,
                               tag=f"fold_{i:02d}/{split}")
            results.append(res)
            print(f"[fold_{i:02d}/{split}] cases={res['cases']} recall={res['recall_queries']} "
                  f"cands {res['candidates_in']}->{res['candidates_kept']} "
                  f"emb_rows={res['embedding_rows']} npy={res['npy_files']}")

    if a.chinese:
        cdir = root / "data/chinese_eval"
        gold_ids = [l.split(",")[0].strip() for l in
                    (root / "data/chinese_gold/gold_v2.0.2_mapping.csv").read_text(encoding="utf-8-sig").splitlines()[1:]
                    if l.strip()]
        # 中文已有 cases/recall/embeddings 且已与 gold 严格对齐，这里只做一致性校验
        ids = [l.strip() for l in (cdir / "embeddings/case_ids.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
        if set(ids) != set(gold_ids):
            raise SystemExit(f"中文 embeddings 与 gold 域不一致: {len(set(ids))} vs {len(set(gold_ids))}")
        qs = {str(o.get("query_case_id") or o.get("query_id")) for o in read_jsonl(cdir / "recall_topk.jsonl")}
        if qs != set(gold_ids):
            raise SystemExit(f"中文 recall 与 gold 域不一致: {len(qs)} vs {len(set(gold_ids))}")
        print(f"[chinese] 已自洽：gold {len(set(gold_ids))} 篇 / embeddings {len(ids)} 行 / recall {len(qs)} query")

    out = root / "docs/ASSET_MANIFEST_FOLDS.json"
    if results:
        out.write_text(json.dumps({"note": "按五折划分投影的每折评测输入子集规模", "subsets": results},
                                  ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n汇总已写入 {out}")


if __name__ == "__main__":
    main()
