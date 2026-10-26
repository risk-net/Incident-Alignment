#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import numpy as np
import faiss

def build_one(artifacts_root: Path, mode: str, use_idmap: bool = False):
    emb_path  = artifacts_root / "embeddings" / mode / "full" / f"emb_{mode}_full.npy"
    ids_path  = artifacts_root / "embeddings" / mode / "full" / f"ids_{mode}_full.npy"
    mask_path = artifacts_root / "embeddings" / mode / "full" / f"valid_mask_{mode}_full.npy"

    X   = np.load(emb_path).astype(np.float32, copy=False)
    ids = np.load(ids_path)
    m   = np.load(mask_path).astype(bool)

    # 输入一致性校验
    assert X.shape[0] == ids.shape[0] == m.shape[0], f"{mode}: X/ids/mask length mismatch: {X.shape[0]}/{ids.shape[0]}/{m.shape[0]}"
    assert X.ndim == 2, f"{mode}: embeddings must be 2D (N,d), got shape {X.shape}"

    # 只入库有效向量
    Xv = X[m]
    idv = ids[m]

    # 统一把 id 转成 int64（若你的 case_id 不是纯数字，这里需改成 hash->int64 + 反查表）
    idv = np.asarray([int(str(x).strip()) for x in idv], dtype=np.int64)

    # cosine：归一化 + IP
    faiss.normalize_L2(Xv)
    d = Xv.shape[1]

    # 设置FAISS线程数
    faiss.omp_set_num_threads(32)

    # 打印构建信息
    print(f"[{mode}] N_total: {X.shape[0]}, N_valid: {Xv.shape[0]}, dim: {d}, use_idmap: {use_idmap}")

    # 精确索引（最稳，先跑通）
    base = faiss.IndexFlatIP(d)

    if use_idmap:
        index = faiss.IndexIDMap2(base)
        index.add_with_ids(Xv, idv)
    else:
        index = base
        index.add(Xv)

    out_dir = artifacts_root / "faiss"
    out_dir.mkdir(parents=True, exist_ok=True)
    faiss_path = out_dir / f"{mode}.index"
    faiss.write_index(index, str(faiss_path))

    # 强制保存 ids.npy 用于 debug（即便用 IDMap2）
    ids_meta_path = out_dir / f"{mode}_ids.npy"
    np.save(ids_meta_path, idv)

    print(f"[{mode}] saved index -> {faiss_path}  ntotal={index.ntotal}")
    print(f"[{mode}] saved ids -> {ids_meta_path}  shape={idv.shape}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts_root", required=True)
    ap.add_argument("--use_idmap", action="store_true",
                    help="让 search() 直接返回 case_id（需要 ids 可转 int64）")
    args = ap.parse_args()

    root = Path(args.artifacts_root)
    build_one(root, "text",  use_idmap=args.use_idmap)
    build_one(root, "event", use_idmap=args.use_idmap)

if __name__ == "__main__":
    main()