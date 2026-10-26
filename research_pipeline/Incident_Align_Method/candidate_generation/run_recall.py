#!/usr/bin/env python3
"""
双路召回执行脚本
基于评估子集进行E_text和E_event的双路召回 + 融合排序

使用方法：
python run_recall.py

配置文件位置：
config/Incident_Align_Method-run_recall-config.ini

依赖：
- faiss-cpu 或 faiss-gpu
- numpy
- tqdm
"""

import configparser
import faiss
import numpy as np
import json
import random
import torch
from pathlib import Path
from tqdm import tqdm
import os
BASE_DIR = Path(__file__).resolve().parents[3]

CONFIG_SECTION = "RunRecall"
DEFAULT_CONFIG_PATH = os.path.join(BASE_DIR, "config", "Incident_Align_Method-run_recall-config.ini")

# 设置随机种子以确保可重复性
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


def norm_id(x):
    """标准化ID：确保字符串类型，去除前后空格"""
    return str(x).strip()


def load_structure_data(structure_file):
    """
    加载评估结构数据
    返回: {incident_id: [case_id, ...], ...}
    """
    with open(structure_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    event_dict = {}
    for event in data["events"]:
        incident_id = norm_id(event["incident_id"])  # 标准化incident_id
        ids = [norm_id(x) for x in event["ids"]]  # 标准化ids
        event_dict[incident_id] = ids

    return event_dict


def search_topk(index, query_emb, topk=150):
    """
    单路topK召回
    返回: similarities, indices
    """
    q = query_emb.astype(np.float32).reshape(1, -1)
    # 防御：零向量
    if np.linalg.norm(q) > 0:
        faiss.normalize_L2(q)  # 归一化用于cosine相似度

    D, I = index.search(q, topk)
    return D[0], I[0]


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


def merge_candidates(I_text, D_text, I_event, D_event, idx2caseid, query_id_norm=None,
                    fuse_mode="max", w_text=0.5, lambda_min=0.2):
    """
    双路候选合并 + 可配置融合
    """
    cand = {}

    # 处理text路结果
    for rank, (ii, dd) in enumerate(zip(I_text, D_text), start=1):
        if ii < 0:  # 处理topk > ntotal时的-1填充
            continue
        cid = idx2caseid.get(str(ii), idx2caseid.get(ii))  # 兼容不同key类型
        if cid is None:
            continue
        cid = norm_id(cid)
        if cid == query_id_norm:  # 过滤自身
            continue
        obj = cand.setdefault(cid, {})
        obj["score_text"] = float(dd)
        obj["rank_text"] = rank

    # 处理event路结果
    for rank, (ii, dd) in enumerate(zip(I_event, D_event), start=1):
        if ii < 0:  # 处理topk > ntotal时的-1填充
            continue
        cid = idx2caseid.get(str(ii), idx2caseid.get(ii))  # 兼容不同key类型
        if cid is None:
            continue
        cid = norm_id(cid)
        if cid == query_id_norm:  # 过滤自身
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


def evaluate_recall_at_k(predictions, k_values=[1, 5, 10, 20, 50, 100, 150]):
    """
    计算Recall@K指标
    """
    metrics = {}

    for k in k_values:
        recall_sum = 0
        for pred in predictions:
            pred_ids = set(item["case_id"] for item in pred["candidates"][:k])
            gt_ids = set(pred["ground_truth"])

            if gt_ids:  # 避免除零
                recall = len(pred_ids & gt_ids) / len(gt_ids)
                recall_sum += recall

        metrics[f"Recall@{k}"] = recall_sum / len(predictions) if predictions else 0

    return metrics


def evaluate_hit_at_k(predictions, k_values=[1, 5, 10, 20, 50, 100, 150]):
    """
    计算Hit@K指标：前K个结果中是否至少命中一个相关项
    """
    metrics = {}

    for k in k_values:
        hit_count = 0
        for pred in predictions:
            pred_ids = set(item["case_id"] for item in pred["candidates"][:k])
            gt_ids = set(pred["ground_truth"])

            if gt_ids and len(pred_ids & gt_ids) > 0:
                hit_count += 1

        metrics[f"Hit@{k}"] = hit_count / len(predictions) if predictions else 0

    return metrics


def evaluate_mrr_at_k(predictions, k_values=[1, 5, 10, 20, 50, 100, 150]):
    """
    计算MRR@K指标：第一个相关文档的倒数排名平均值
    """
    metrics = {}

    for k in k_values:
        mrr_sum = 0
        for pred in predictions:
            pred_items = pred["candidates"][:k]
            gt_ids = set(pred["ground_truth"])

            if gt_ids:
                for rank, item in enumerate(pred_items, 1):
                    if item["case_id"] in gt_ids:
                        mrr_sum += 1.0 / rank
                        break

        metrics[f"MRR@{k}"] = mrr_sum / len(predictions) if predictions else 0

    return metrics


def resolve_path(path_value, default_value=None):
    raw_path = (path_value if path_value is not None else default_value)
    if raw_path is None:
        raise ValueError("resolve_path requires either path_value or default_value")
    raw_path = str(raw_path).strip()
    if os.path.isabs(raw_path):
        return raw_path
    return os.path.join(BASE_DIR, raw_path)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH,
                    help="配置文件路径（默认英文评测配置）")
    args = ap.parse_args()
    config_path = args.config
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    parser = configparser.ConfigParser()
    parser.read(config_path, encoding="utf-8")
    if CONFIG_SECTION not in parser:
        raise KeyError(f"配置文件缺少 [{CONFIG_SECTION}] 段: {config_path}")

    section = parser[CONFIG_SECTION]

    structure_file = resolve_path(section.get("structure_file", "data/eval_structure.json"))
    faiss_dir = resolve_path(section.get("faiss_dir", "outputs/faiss_index"))

    embeddings_value = (section.get("embeddings_dir", "") or "").strip()
    if embeddings_value:
        embeddings_dir = resolve_path(embeddings_value, embeddings_value)
    else:
        embeddings_dir = os.path.join(os.path.dirname(faiss_dir), "embeddings")

    output_file = resolve_path(section.get("output_file", "outputs/recall.jsonl"))
    metrics_file = resolve_path(section.get("metrics_file", "outputs/metrics/recall_metrics.json"))

    topk_per_route = section.getint("topk_per_route", fallback=150)
    topn_save = section.getint("topn_save", fallback=300)
    fuse_mode = section.get("fuse_mode", "max").strip()
    w_text = section.getfloat("w_text", fallback=0.5)
    lambda_min = section.getfloat("lambda_min", fallback=0.2)
    evaluate = section.getboolean("evaluate", fallback=False)

    print(f"加载配置文件: {config_path}")
    print(f"结构文件: {structure_file}")
    print(f"FAISS目录: {faiss_dir}")
    print(f"embeddings目录: {embeddings_dir if embeddings_dir else '[跟随 faiss_dir 推断]'}")
    print(f"输出文件: {output_file}")

    # 加载评估结构
    print(f"加载评估结构: {structure_file}")
    event_dict = load_structure_data(structure_file)

    # 收集所有评估case_id
    all_eval_ids = set()
    for ids in event_dict.values():
        all_eval_ids.update(ids)

    # 排序确保可复现
    all_eval_ids = sorted(all_eval_ids)

    print(f"评估案例总数: {len(all_eval_ids)}")
    print(f"事件数量: {len(event_dict)}")

    # 加载FAISS索引
    print(f"加载FAISS索引: {faiss_dir}")

    index_text = faiss.read_index(os.path.join(faiss_dir, "faiss_text.index"))
    index_event = faiss.read_index(os.path.join(faiss_dir, "faiss_event.index"))

    with open(os.path.join(faiss_dir, "idx2caseid.json"), 'r', encoding='utf-8') as f:
        idx2caseid = json.load(f)

    # 建立反向映射，避免O(N²)查找，使用标准化ID
    caseid2idx = {norm_id(cid): int(idx) for idx, cid in idx2caseid.items()}

    # 建立case_id到ground_truth的映射，避免每次扫描，使用标准化ID
    caseid2gt = {}
    for incident_id, ids in event_dict.items():
        for cid in ids:
            caseid2gt[norm_id(cid)] = [norm_id(x) for x in ids if norm_id(x) != norm_id(cid)]

    # 加载embeddings
    emb_text = np.load(os.path.join(embeddings_dir, "emb_text.npy"))
    emb_event = np.load(os.path.join(embeddings_dir, "emb_event.npy"))

    print(f"索引大小: text={index_text.ntotal}, event={index_event.ntotal}")

    # 执行召回
    results = []

    print("开始双路召回...")
    for query_id in tqdm(all_eval_ids):
        # 标准化query_id后查找
        query_id_norm = norm_id(query_id)
        query_idx = caseid2idx.get(query_id_norm)

        if query_idx is None:
            print(f"警告: 找不到query case_id {query_id} 的索引")
            continue

        # 获取query embeddings
        query_emb_text = emb_text[query_idx]
        query_emb_event = emb_event[query_idx]

        # 双路召回
        D_text, I_text = search_topk(index_text, query_emb_text, topk_per_route)
        D_event, I_event = search_topk(index_event, query_emb_event, topk_per_route)

        # 合并候选
        candidates = merge_candidates(I_text, D_text, I_event, D_event, idx2caseid, query_id_norm,
                                    fuse_mode, w_text, lambda_min)

        # 获取ground truth (同事件的其他cases)
        ground_truth = caseid2gt.get(query_id_norm, [])

        result = {
            "query_case_id": query_id_norm,
            "ground_truth": ground_truth,
            "meta": {
                "topk_per_route": topk_per_route,
                "merged_size": len(candidates),
                "fuse_mode": fuse_mode,
                "saved_count": min(topn_save, len(candidates))
            },
            "candidates": candidates[:topn_save],  # 保存指定数量的候选
        }

        results.append(result)

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # 保存结果
    print(f"保存结果到: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    # 计算评估指标
    if evaluate:
        print("计算评估指标...")
        # 保护：k值不超过保存的候选数
        k_values = [k for k in [1, 5, 10, 20, 50, 100, 150] if k <= topn_save]
        recall_metrics = evaluate_recall_at_k(results, k_values=k_values)
        hit_metrics = evaluate_hit_at_k(results, k_values=k_values)
        mrr_metrics = evaluate_mrr_at_k(results, k_values=k_values)

        # 合并指标
        metrics = {**recall_metrics, **hit_metrics, **mrr_metrics}

        print("\n评估结果:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

        # 保存评估结果
        try:
            faiss_version = faiss.__version__
        except:
            faiss_version = "unknown"

        try:
            torch_version = torch.__version__
            cuda_available = torch.cuda.is_available()
        except:
            torch_version = "unknown"
            cuda_available = False

        os.makedirs(os.path.dirname(metrics_file), exist_ok=True)

        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump({
                "metrics": metrics,
                "config_file": config_path,
                "num_queries": len(results),
                "topk_per_route": topk_per_route,
                "topn_save": topn_save,
                "fuse_mode": fuse_mode,
                "random_seed": 42,
                "config": {
                    "structure_file": structure_file,
                    "faiss_dir": faiss_dir,
                    "embeddings_dir": embeddings_dir,
                    "output_file": output_file,
                    "metrics_file": metrics_file,
                    "w_text": w_text,
                    "lambda_min": lambda_min
                },
                "versions": {
                    "faiss": faiss_version,
                    "torch": torch_version,
                    "cuda_available": cuda_available
                }
            }, f, indent=2, ensure_ascii=False)

        print(f"评估结果已保存到: {metrics_file}")

    print("完成!")
    print(f"处理查询数: {len(results)}")


if __name__ == "__main__":
    main()
