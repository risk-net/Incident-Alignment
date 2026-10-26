#!/usr/bin/env python3
"""DeepWide 概率缓存 (.npz) -> M6 分数 JSONL 转换器。

背景
----
仓库历史上把 Dev 的 DeepWide 输出缓存为 ``fold_0X_dev.npz``（键 ``qc``/``p``/``y``），
而 ``incident-align`` 只接受逐行 JSONL：``{"query_id","candidate_id","probability"}``。
本脚本补上这一步，**不做任何推理**，只做格式转换与域校验。

npz 结构（由历史上 ``export_dev_probs.py`` 产出）::

    qc : (N, 2) 字符串数组，列序为 (query_case_id, candidate_case_id)
    p  : (N,)  float 概率，取值 [0,1]
    y  : (N,)  int32 标签（可选，本脚本不使用，只用于核对数量）

用法
----
    python scripts/npz_to_score_jsonl.py \
        --npz outputs/dev/probs/fold_01_dev.npz \
        --out outputs/dev/fold_01_scores.jsonl

    # 五折一次转换
    python scripts/npz_to_score_jsonl.py --all --npz-dir outputs/dev/probs --out-dir outputs/dev

安全
----
- 只读 npz，不修改任何源文件。
- 任何自环、非有限值、越界概率、重复有向对都会报错，不会静默丢弃。
"""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def convert(npz_path, out_path, *, strict=True):
    import numpy as np

    data = np.load(npz_path, allow_pickle=True)
    if "qc" not in data or "p" not in data:
        raise ValueError(f"{npz_path}: 缺少 qc 或 p 字段（实际键: {sorted(data.files)}）")
    qc, probs = data["qc"], data["p"]
    if qc.ndim != 2 or qc.shape[1] != 2:
        raise ValueError(f"{npz_path}: qc 形状应为 (N,2)，实际 {qc.shape}")
    if len(qc) != len(probs):
        raise ValueError(f"{npz_path}: qc 与 p 长度不一致 ({len(qc)} vs {len(probs)})")

    rows = 0
    seen = set()
    self_pairs = 0
    dupes = 0
    bad_prob = 0
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fo:
        for (q, c), p in zip(qc, probs):
            q, c = str(q).strip(), str(c).strip()
            val = float(p)
            if not q or not c:
                raise ValueError(f"{npz_path}: 空 ID ({q!r}, {c!r})")
            if q == c:
                self_pairs += 1
                if strict:
                    raise ValueError(f"{npz_path}: 自环对 {q}->{c}")
                continue
            if not (0.0 <= val <= 1.0) or val != val:
                bad_prob += 1
                if strict:
                    raise ValueError(f"{npz_path}: 非法概率 {val} at {q}->{c}")
                continue
            key = (q, c)
            if key in seen:
                dupes += 1
                if strict:
                    raise ValueError(f"{npz_path}: 重复有向对 {q}->{c}")
                continue
            seen.add(key)
            fo.write(json.dumps({"query_id": q, "candidate_id": c, "probability": val},
                                ensure_ascii=False) + "\n")
            rows += 1
    tmp.replace(out_path)

    queries = {q for q, _ in seen}
    return {
        "npz": str(npz_path),
        "out": str(out_path),
        "rows": rows,
        "queries": len(queries),
        "candidates": len({c for _, c in seen}),
        "self_pairs_skipped": self_pairs,
        "duplicates_skipped": dupes,
        "bad_probability_skipped": bad_prob,
        "npz_sha256": sha256_file(npz_path),
        "jsonl_sha256": sha256_file(out_path),
        "probability_min": min(float(p) for p in probs) if len(probs) else None,
        "probability_max": max(float(p) for p in probs) if len(probs) else None,
    }


def main():
    ap = argparse.ArgumentParser(description="DeepWide .npz -> M6 分数 JSONL")
    ap.add_argument("--npz", help="单个 .npz 路径")
    ap.add_argument("--out", help="输出 JSONL 路径")
    ap.add_argument("--all", action="store_true", help="批量转换 --npz-dir 下所有 fold_XX_dev.npz")
    ap.add_argument("--npz-dir", default="outputs/dev/probs")
    ap.add_argument("--out-dir", default="outputs/dev")
    ap.add_argument("--lenient", action="store_true", help="跳过非法行而不是报错（默认严格）")
    a = ap.parse_args()

    strict = not a.lenient
    if a.all:
        npz_dir, out_dir = Path(a.npz_dir), Path(a.out_dir)
        npzs = sorted(npz_dir.glob("fold_*_dev.npz"))
        if not npzs:
            sys.exit(f"{npz_dir} 下没有 fold_*_dev.npz")
        reports = []
        for npz in npzs:
            fold = "_".join(npz.stem.split("_")[:2])   # fold_01_dev -> fold_01
            out = out_dir / f"{fold}_scores.jsonl"
            rep = convert(npz, out, strict=strict)
            reports.append(rep)
            print(f"{fold}: {rep['rows']} 行, {rep['queries']} query, "
                  f"p∈[{rep['probability_min']:.6f},{rep['probability_max']:.6f}] -> {out}")
        manifest = out_dir / "npz_to_jsonl_manifest.json"
        manifest.write_text(json.dumps({"conversions": reports}, ensure_ascii=False, indent=1),
                            encoding="utf-8")
        print(f"\nmanifest -> {manifest}")
        return

    if not a.npz or not a.out:
        sys.exit("需要 --npz 与 --out，或使用 --all")
    rep = convert(a.npz, a.out, strict=strict)
    print(json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
