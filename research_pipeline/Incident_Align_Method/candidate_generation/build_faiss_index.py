#!/usr/bin/env python3
"""
FAISS索引构建脚本
为E_text和E_event embeddings构建索引

使用方法：
python build_faiss_index.py

配置文件位置：
config/Incident_Align_Method-build_faiss_index-config.ini

依赖：
- faiss-cpu 或 faiss-gpu
- numpy
"""

import configparser
import faiss
import numpy as np
import os
import random
import torch
from pathlib import Path
import json

BASE_DIR = Path(__file__).resolve().parents[3]

CONFIG_SECTION = "BuildFaissIndex"
DEFAULT_CONFIG_PATH = os.path.join(BASE_DIR, "config", "Incident_Align_Method-build_faiss_index-config.ini")

# 设置随机种子以确保可重复性
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


def build_faiss_ip_index(emb):
    """
    构建cosine相似度FAISS索引 (IndexFlatIP)
    """
    emb = emb.copy().astype(np.float32)
    faiss.normalize_L2(emb)  # L2归一化用于cosine相似度

    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner Product (cosine after normalization)
    index.add(emb)

    return index


def load_case_ids(embeddings_dir):
    case_ids_npy = os.path.join(embeddings_dir, "case_ids.npy")
    if os.path.exists(case_ids_npy):
        return np.load(case_ids_npy, allow_pickle=True)

    case_ids_txt = os.path.join(embeddings_dir, "case_ids.txt")
    if os.path.exists(case_ids_txt):
        with open(case_ids_txt, "r", encoding="utf-8") as f:
            case_ids = [line.strip() for line in f if line.strip()]
        return np.array(case_ids, dtype=object)

    raise FileNotFoundError(f"缺少 case_ids.npy 或 case_ids.txt: {embeddings_dir}")


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

    embeddings_dir = os.path.join(BASE_DIR, section.get("embeddings_dir", "outputs/embeddings")).strip()
    output_dir = os.path.join(BASE_DIR, section.get("output_dir", "outputs/faiss_index")).strip()
    print(f"加载配置文件: {config_path}")
    print(f"embeddings目录: {embeddings_dir}")
    print(f"输出目录: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    # 加载embeddings
    print("加载embeddings...")
    emb_text = np.load(os.path.join(embeddings_dir, "emb_text.npy"))
    emb_event = np.load(os.path.join(embeddings_dir, "emb_event.npy"))
    case_ids = load_case_ids(embeddings_dir)

    print(f"E_text embedding shape: {emb_text.shape}")
    print(f"E_event embedding shape: {emb_event.shape}")
    print(f"案例数量: {len(case_ids)}")

    # 构建索引
    print("构建E_text索引...")
    index_text = build_faiss_ip_index(emb_text)

    print("构建E_event索引...")
    index_event = build_faiss_ip_index(emb_event)

    # 保存索引
    print("保存索引...")
    faiss.write_index(index_text, os.path.join(output_dir, "faiss_text.index"))
    faiss.write_index(index_event, os.path.join(output_dir, "faiss_event.index"))

    # 保存case_ids映射，确保key和value都是字符串
    idx2caseid = {str(i): str(cid) for i, cid in enumerate(case_ids)}
    with open(os.path.join(output_dir, "idx2caseid.json"), 'w', encoding='utf-8') as f:
        json.dump(idx2caseid, f, indent=2, ensure_ascii=False)

    # 保存索引配置
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

    config = {
        "config_file": config_path,
        "embedding_dim": emb_text.shape[1],
        "num_cases": len(case_ids),
        "index_type": "IndexFlatIP",  # cosine similarity
        "random_seed": 42,
        "embeddings_dir": embeddings_dir,
        "text_index_file": "faiss_text.index",
        "event_index_file": "faiss_event.index",
        "versions": {
            "faiss": faiss_version,
            "torch": torch_version,
            "cuda_available": cuda_available
        }
    }

    with open(os.path.join(output_dir, "faiss_config.json"), 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print("完成!")
    print(f"输出目录: {output_dir}")
    print(f"索引类型: IndexFlatIP (cosine similarity)")


if __name__ == "__main__":
    main()
