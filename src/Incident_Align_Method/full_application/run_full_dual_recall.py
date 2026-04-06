#!/usr/bin/env python3
"""
全量数据双路召回执行脚本
基于全量数据进行E_text和E_event的双路召回 + 融合排序

使用方法：
python src/Incident_Align_Method/full_application/run_full_dual_recall.py \
    --config config/Incident_Align_Method-full_application-config.ini

依赖：
- faiss-cpu 或 faiss-gpu
- numpy
- tqdm
"""

import faiss
import numpy as np
import json
import random
from pathlib import Path
import argparse
from tqdm import tqdm

from config_utils import DEFAULT_CONFIG_PATH, get_float, get_int, load_config, require_section, resolve_path

# 设置随机种子以确保可重复性
random.seed(42)
np.random.seed(42)

# 设置FAISS CPU线程数
faiss.omp_set_num_threads(32)


def norm_id(x):
    """标准化ID：确保字符串类型，去除前后空格"""
    return str(x).strip()


def load_embeddings_and_faiss(artifacts_root: Path, mode: str):
    """
    加载指定模式的embeddings和FAISS索引
    返回: embeddings, ids, mask, faiss_index
    """
    emb_path = artifacts_root / "embeddings" / mode / "full" / f"emb_{mode}_full.npy"
    ids_path = artifacts_root / "embeddings" / mode / "full" / f"ids_{mode}_full.npy"
    mask_path = artifacts_root / "embeddings" / mode / "full" / f"valid_mask_{mode}_full.npy"
    faiss_path = artifacts_root / "faiss" / f"{mode}.index"

    # 加载embeddings
    embeddings = np.load(emb_path)
    ids = np.load(ids_path)
    mask = np.load(mask_path)

    # 加载FAISS索引
    faiss_index = faiss.read_index(str(faiss_path))

    # 一致性检查
    expected_count = np.sum(mask)
    if faiss_index.ntotal != expected_count:
        raise ValueError(f"{mode} index ntotal ({faiss_index.ntotal}) != expected ({expected_count})")

    return embeddings, ids, mask, faiss_index


def search_topk_batch(index, Q, topk):
    """
    批量topK召回
    """
    Q = Q.astype(np.float32, copy=False)
    # 防御：零向量
    norms = np.linalg.norm(Q, axis=1, keepdims=True)
    valid_mask = norms.flatten() > 0
    if np.any(valid_mask):
        Q[valid_mask] = Q[valid_mask] / norms[valid_mask]  # 归一化用于cosine相似度

    D, I = index.search(Q, topk)
    return D, I


def fuse_score(st, se, mode="max", w_text=0.5, lambda_min=0.2):
    """可配置的融合策略"""
    st_v = st if st is not None else -1e9
    se_v = se if se is not None else -1e9

    if mode == "max":
        return max(st_v, se_v)
    elif mode == "mean":
        # 缺失当成只看另一路
        if st is None: return se_v
        if se is None: return st_v
        return 0.5 * (st_v + se_v)
    elif mode == "wavg":
        if st is None: return se_v
        if se is None: return st_v
        return w_text * st_v + (1 - w_text) * se_v
    elif mode == "maxmin":
        if st is None:
            return se_v
        if se is None:
            return st_v
        mx = max(st_v, se_v)
        mn = min(st_v, se_v)
        return mx + lambda_min * mn
    else:
        raise ValueError(f"Unknown fuse mode: {mode}")


def merge_candidates(I_text, D_text, I_event, D_event, query_id,
                    fuse_mode="max", w_text=0.5, lambda_min=0.2):
    """
    双路候选合并 + 可配置融合
    """
    cand = {}

    # 处理text路结果 - ii 直接就是 case_id
    for rank, (ii, dd) in enumerate(zip(I_text, D_text), start=1):
        if ii < 0:  # 处理topk > ntotal时的-1填充
            continue
        cid = norm_id(ii)  # ii 本身就是 case_id（int64）
        if cid == query_id:  # 过滤自身
            continue
        obj = cand.setdefault(cid, {})
        obj["score_text"] = float(dd)
        obj["rank_text"] = rank

    # 处理event路结果 - ii 直接就是 case_id
    for rank, (ii, dd) in enumerate(zip(I_event, D_event), start=1):
        if ii < 0:  # 处理topk > ntotal时的-1填充
            continue
        cid = norm_id(ii)  # ii 本身就是 case_id（int64）
        if cid == query_id:  # 过滤自身
            continue
        obj = cand.setdefault(cid, {})
        obj["score_event"] = float(dd)
        obj["rank_event"] = rank

    # 融合排序
    merged = []
    for cid, obj in cand.items():
        st = obj.get("score_text")
        se = obj.get("score_event")
        score_fuse = fuse_score(st, se, fuse_mode, w_text, lambda_min)

        merged.append({
            "case_id": cid,
            "score_text": st,
            "rank_text": obj.get("rank_text"),
            "score_event": se,
            "rank_event": obj.get("rank_event"),
            "in_text": obj.get("rank_text") is not None,
            "in_event": obj.get("rank_event") is not None,
            "score_fuse": score_fuse,
        })

    merged.sort(key=lambda x: x["score_fuse"], reverse=True)
    return merged


def get_valid_query_cases(embeddings_text, ids_text, mask_text,
                         embeddings_event, ids_event, mask_event):
    """
    获取同时在text和event中都有效的案例ID
    """
    # 找到text和event中都有效的案例，并统一norm_id
    text_valid_ids = set(norm_id(cid) for cid in ids_text[mask_text])
    event_valid_ids = set(norm_id(cid) for cid in ids_event[mask_event])
    valid_case_ids = text_valid_ids & event_valid_ids

    # 排序确保可复现性
    valid_query_ids = sorted(valid_case_ids)

    # 建立case_id到索引的映射（key统一为norm_id）
    text_id2idx = {norm_id(cid): idx for idx, (cid, valid) in enumerate(zip(ids_text, mask_text)) if valid}
    event_id2idx = {norm_id(cid): idx for idx, (cid, valid) in enumerate(zip(ids_event, mask_event)) if valid}

    return valid_query_ids, text_id2idx, event_id2idx


def process_batch(query_ids, text_id2idx, event_id2idx,
                  embeddings_text, embeddings_event,
                  index_text, index_event, args):
    """
    处理一批查询的召回 - 批量优化版本
    """
    # 1) 收集能用的query
    qids = []
    t_idx = []
    e_idx = []
    for q in query_ids:
        qn = norm_id(q)
        ti = text_id2idx.get(qn)
        ei = event_id2idx.get(qn)
        if ti is None or ei is None:
            continue
        qids.append(qn)
        t_idx.append(ti)
        e_idx.append(ei)

    if not qids:
        return []

    # 2) 批量取向量
    Q_text = embeddings_text[np.array(t_idx)]
    Q_event = embeddings_event[np.array(e_idx)]

    # 3) 批量search（两路各一次）
    D_text, I_text = search_topk_batch(index_text, Q_text, args.topk_per_route)
    D_event, I_event = search_topk_batch(index_event, Q_event, args.topk_per_route)

    # 4) 逐条merge
    results = []
    for row, qid in enumerate(qids):
        candidates = merge_candidates(
            I_text[row], D_text[row],
            I_event[row], D_event[row],
            qid,  # ii 直接就是 case_id
            args.fuse_mode, args.w_text, args.lambda_min
        )

        result = {
            "query_case_id": qid,
            "meta": {
                "topk_per_route": args.topk_per_route,
                "merged_size": len(candidates),
                "fuse_mode": args.fuse_mode,
                "saved_count": min(args.topn_save, len(candidates))
            },
            "candidates": candidates[:args.topn_save],
        }

        results.append(result)

    return results


def main():
    parser = argparse.ArgumentParser(description="执行全量数据双路 embedding 召回")
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="INI 配置文件路径",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    section = require_section(cfg, "RunFullDualRecall")
    artifacts_root = Path(resolve_path(section.get("artifacts_root", "")))
    output_file = resolve_path(section.get("output_file", ""))
    topk_per_route = get_int(section, "topk_per_route", 150)
    topn_save = get_int(section, "topn_save", 300)
    fuse_mode = (section.get("fuse_mode", "max") or "max").strip()
    w_text = get_float(section, "w_text", 0.5)
    lambda_min = get_float(section, "lambda_min", 0.2)
    batch_size = get_int(section, "batch_size", 1000)
    start_idx = get_int(section, "start_idx", 0)
    end_idx = get_int(section, "end_idx", -1)
    faiss_threads = get_int(section, "faiss_threads", 32)

    if fuse_mode not in {"max", "mean", "wavg", "maxmin"}:
        raise ValueError(f"不支持的 fuse_mode: {fuse_mode}")
    faiss.omp_set_num_threads(faiss_threads)

    # 加载text和event的embeddings和FAISS索引
    print("加载text embeddings和FAISS索引...")
    emb_text, ids_text, mask_text, index_text = load_embeddings_and_faiss(artifacts_root, "text")

    print("加载event embeddings和FAISS索引...")
    emb_event, ids_event, mask_event, index_event = load_embeddings_and_faiss(artifacts_root, "event")

    print(f"text索引大小: {index_text.ntotal}")
    print(f"event索引大小: {index_event.ntotal}")

    # 获取有效的查询案例
    print("获取有效的查询案例...")
    valid_query_ids, text_id2idx, event_id2idx = get_valid_query_cases(
        emb_text, ids_text, mask_text,
        emb_event, ids_event, mask_event
    )

    print(f"有效查询案例总数: {len(valid_query_ids)}")

    # 限制查询范围
    if end_idx == -1:
        end_idx = len(valid_query_ids)
    query_ids_to_process = valid_query_ids[start_idx:end_idx]

    print(f"将处理查询案例: {len(query_ids_to_process)} (从 {start_idx} 到 {end_idx})")

    # 确保输出目录存在
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_interval = 1000  # 每1000个查询保存一次

    print("开始双路召回...")
    with tqdm(total=len(query_ids_to_process), desc="Processing") as pbar:
        processed_count = 0

        config_snapshot = {
            "config_path": str(Path(args.config).resolve()),
            "artifacts_root": str(artifacts_root),
            "output_file": output_file,
            "topk_per_route": topk_per_route,
            "topn_save": topn_save,
            "fuse_mode": fuse_mode,
            "w_text": w_text,
            "lambda_min": lambda_min,
            "batch_size": batch_size,
            "start_idx": start_idx,
            "end_idx": end_idx,
            "faiss_threads": faiss_threads,
        }
        with open(output_file, 'w', encoding='utf-8') as f:
            for i in range(0, len(query_ids_to_process), batch_size):
                batch_query_ids = query_ids_to_process[i:i+batch_size]

                batch_results = process_batch(
                    batch_query_ids, text_id2idx, event_id2idx,
                    emb_text, emb_event,
                    index_text, index_event,
                    argparse.Namespace(
                        topk_per_route=topk_per_route,
                        topn_save=topn_save,
                        fuse_mode=fuse_mode,
                        w_text=w_text,
                        lambda_min=lambda_min,
                    )
                )

                # 实时写入文件
                for result in batch_results:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')

                processed_count += len(batch_results)
                pbar.update(len(batch_query_ids))

                # 每1000个查询刷新缓冲区，确保数据写入磁盘
                if processed_count % save_interval == 0:
                    f.flush()
                    print(f"已处理并保存: {processed_count} 个查询")
    run_config_file = out_path.parent / "run_full_dual_recall_config.json"
    with open(run_config_file, "w", encoding="utf-8") as f:
        json.dump(config_snapshot, f, ensure_ascii=False, indent=2)

    print(f"保存结果到: {output_file}")
    print(f"运行配置保存到: {run_config_file}")

    print("完成!")
    print(f"处理查询数: {processed_count}")


if __name__ == "__main__":
    main()
