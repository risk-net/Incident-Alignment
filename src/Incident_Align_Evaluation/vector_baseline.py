#!/usr/bin/env python3
"""
单次向量相似度基线入口。

该脚本负责：
1. 从 `cases.jsonl` 构建文本向量；
2. 基于相似度阈值聚类得到事件簇；
3. 调用共享评估模块输出 `clusters.json` 与 `metrics.json`。

重复划分和阈值稳定性实验请使用 `baseline_textsim_threshold_repeat.py`。
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict
import numpy as np
from tqdm import tqdm

from accuracy import EvaluationConfig, build_metrics_payload, compute_metrics

MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "Event_Align_Evaluation" / "vector_baseline"

# 向量基线相关导入 (使用本地sentence-transformers模型)
try:
    import faiss
    import torch
    from sentence_transformers import SentenceTransformer

    # 检查GPU支持
    try:
        gpu_count = faiss.get_num_gpus()
        if gpu_count > 0:
            print(f"✅ 向量基线依赖已加载 (GPU版本, {gpu_count}个GPU可用)")
        else:
            print("✅ 向量基线依赖已加载 (CPU版本)")
    except AttributeError:
        print("✅ 向量基线依赖已加载 (CPU版本)")

    VECTOR_BASELINE_AVAILABLE = True

except ImportError as e:
    VECTOR_BASELINE_AVAILABLE = False
    print(f"❌ 向量基线依赖未安装: {e}")
    print("\n安装方法:")
    print("1. 安装FAISS (CPU版本):")
    print("   conda install faiss-cpu -c pytorch")
    print("   或: pip install faiss-cpu")
    print("\n2. 安装sentence-transformers:")
    print("   pip install sentence-transformers torch")

    # 定义占位符类，避免导入时崩溃
    class TextVectorizer:
        def __init__(self, model_path="BAAI/bge-m3"):
            raise ImportError("向量基线依赖未安装")

        def encode(self, texts):
            raise ImportError("向量基线依赖未安装")

    class VectorDatabase:
        def __init__(self, dim=768, index_type="Flat", metric=None):
            raise ImportError("向量基线依赖未安装")

        def add_vectors(self, vectors, descriptions, metadata_list=None):
            raise ImportError("向量基线依赖未安装")

        def search(self, query_vector, k=10):
            raise ImportError("向量基线依赖未安装")

    def build_event_clusters_from_faiss(vector_db, high_threshold=0.78):
        raise ImportError("向量基线依赖未安装")

# ================================
# Configuration Constants (从Incident-Alignment复制)
# ================================
DEFAULT_VECTOR_DIM = 768
DEFAULT_BATCH_SIZE = 16
MAX_SEARCH_K = 200


# 已删除旧的VectorBaselineClustering类，现在直接使用Incident-Alignment的方法
# ================================
# 工具函数
# ================================

def resolve_model_path(model_path):
    """解析模型路径，仅接受显式提供的本地路径或 HF 模型名。"""
    import os

    candidate = (model_path or "BAAI/bge-m3").strip()
    if os.path.exists(candidate):
        has_modules_json = os.path.exists(os.path.join(candidate, "modules.json"))
        has_config_json = os.path.exists(os.path.join(candidate, "config.json"))
        if has_modules_json or has_config_json:
            print(f"✅ 使用本地模型: {candidate}")
            return candidate
        raise FileNotFoundError(
            f"模型路径存在但看起来不是有效的 sentence-transformers/transformers 目录: {candidate}"
        )

    print(f"📥 使用 HuggingFace 模型名: {candidate}")
    return candidate

# ================================
# 从Incident-Alignment复制的核心类
# ================================

if VECTOR_BASELINE_AVAILABLE:
    class TextVectorizer:
        """Text vectorization tool class using local sentence-transformers model"""

        def __init__(self, model_path="BAAI/bge-m3"):
            resolved_path = resolve_model_path(model_path)
            print(f"加载模型: {resolved_path}")
            try:
                self.model = SentenceTransformer(
                    resolved_path,
                    device="cuda" if torch.cuda.is_available() else "cpu"
                )
                test_vec = self.model.encode(["test"], convert_to_numpy=True)
                self.vector_dim = int(test_vec.shape[1])
                print(f"✅ 模型加载成功，嵌入维度: {self.vector_dim}")
            except Exception as e:
                print(f"❌ 模型加载失败: {e}")
                raise

        def get_embedding(self, description: str):
            if not description or not isinstance(description, str) or not description.strip():
                return [0.0] * self.vector_dim
            try:
                emb = self.model.encode([description], convert_to_numpy=True)[0]
                return emb.tolist()
            except Exception as e:
                print(f"Vectorization failed: {description[:50]}... Error: {e}")
                return [0.0] * self.vector_dim

        def batch_get_embeddings(self, descriptions, batch_size=DEFAULT_BATCH_SIZE):
            embeddings = []
            for i in tqdm(range(0, len(descriptions), batch_size), desc="生成向量"):
                batch_descriptions = descriptions[i:i + batch_size]
                try:
                    batch_emb = self.model.encode(
                        batch_descriptions,
                        batch_size=batch_size,
                        convert_to_numpy=True
                    )
                    embeddings.extend(batch_emb)
                except Exception as e:
                    print(f"批处理失败 (batch {i//batch_size}): {e}")
                    for desc in batch_descriptions:
                        embeddings.append(self.get_embedding(desc))
            return np.array(embeddings, dtype=np.float32)


    class VectorDatabase:
        """Vector database management class using FAISS for similarity search"""

        def __init__(self, dim=DEFAULT_VECTOR_DIM, index_type="Flat", metric=None):
            if metric is None:
                metric = faiss.METRIC_INNER_PRODUCT

            if index_type == "Flat":
                self.index = faiss.IndexFlatIP(dim) if metric == faiss.METRIC_INNER_PRODUCT else faiss.IndexFlatL2(dim)
            else:
                self.index = faiss.IndexFlatIP(dim)

            self.dim = dim
            self.ids_to_description = {}
            self.ids_to_metadata = {}

        def add_vectors(self, vectors, descriptions, metadata_list=None):
            if metadata_list is None:
                metadata_list = [{} for _ in range(len(descriptions))]

            start_id = len(self.ids_to_description)
            ids = np.arange(start_id, start_id + len(descriptions))

            self.index.add(vectors)

            for i, (desc, metadata) in enumerate(zip(descriptions, metadata_list)):
                self.ids_to_description[int(ids[i])] = desc
                self.ids_to_metadata[int(ids[i])] = metadata

            return ids

        def search(self, query_vector, k=10):
            distances, indices = self.index.search(np.array([query_vector], dtype=np.float32), k)
            results = []
            for i, idx in enumerate(indices[0]):
                if idx != -1:
                    idx = int(idx)
                    results.append({
                        "id": idx,
                        "description": self.ids_to_description[idx],
                        "metadata": self.ids_to_metadata[idx],
                        "distance": float(distances[0][i])
                    })
            return results

        def save(self, index_path, mapping_path):
            faiss.write_index(self.index, index_path)
            with open(mapping_path, "w", encoding="utf-8") as f:
                json.dump({
                    "ids_to_description": {int(k): v for k, v in self.ids_to_description.items()},
                    "ids_to_metadata": {int(k): v for k, v in self.ids_to_metadata.items()},
                }, f, ensure_ascii=False, indent=2)
            print(f"✅ Database saved: {index_path}")

        @classmethod
        def load(cls, index_path, mapping_path):
            index = faiss.read_index(index_path)
            with open(mapping_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            db = cls(dim=index.d)
            db.index = index
            db.ids_to_description = {int(k): v for k, v in data["ids_to_description"].items()}
            db.ids_to_metadata = {int(k): v for k, v in data["ids_to_metadata"].items()}
            return db


# ================================
# 从Incident-Alignment复制的聚类函数
# ================================

if VECTOR_BASELINE_AVAILABLE:
    def build_event_clusters_from_faiss(vector_db, high_threshold=0.78):
        """Build event clusters based on FAISS vector similarity using connected components

        Args:
            vector_db (VectorDatabase): Database containing vectors
            high_threshold (float): Similarity threshold for clustering

        Returns:
            list: List of event clusters with incident_id and cases
        """
        print("🔍 Starting event clustering...")
    
        # 🔧 关键修复：使用连通分量确保聚类闭包
        try:
            import networkx as nx
        except ImportError:
            raise ImportError("❌ 缺少networkx依赖，请运行: pip install networkx")
    
        print(f"📈 Building similarity graph (threshold: {high_threshold})")
    
        # 构建图：节点是向量ID，边是相似度≥阈值的连接
        G = nx.Graph()
    
        # 添加所有节点
        for i in range(vector_db.index.ntotal):
            G.add_node(i)
    
        # 🔧 修复MAX_SEARCH_K问题：增大搜索范围，避免漏边
        search_k = min(1000, vector_db.index.ntotal)  # 从200增大到1000
    
        # 为每个节点找到相似邻居并建边
        for i in tqdm(range(vector_db.index.ntotal), desc="Building graph"):
            query_vector = vector_db.index.reconstruct(i)
            results = vector_db.search(query_vector, k=search_k)
    
            for result in results:
                neighbor_id = result['id']
                faiss_similarity = result['distance']
    
                # 只在相似度≥阈值时建边（无向图，避免重复）
                if neighbor_id > i and faiss_similarity >= high_threshold:
                    G.add_edge(i, neighbor_id, weight=faiss_similarity)
    
        print(f"📊 相似度图构建完成: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")
    
        # 使用连通分量进行聚类（确保闭包）
        print("🔗 计算连通分量...")
        connected_components = list(nx.connected_components(G))
    
        print(f"📈 找到 {len(connected_components)} 个连通分量")
    
        # Convert to event cluster format
        event_clusters = []
        for component_id, component in enumerate(connected_components):
            if component:  # 非空分量
                case_ids = [str(vector_db.ids_to_metadata[node_id]['id'])
                            for node_id in component
                            if 'id' in vector_db.ids_to_metadata[node_id]]
    
                if case_ids:
                    case_ids.sort()
                    event_clusters.append({
                        "incident_id": f"cluster_{component_id}",
                        "cases": case_ids
                    })
    
        print(f"✅ Clustering completed, generated {len(event_clusters)} clusters")
        return event_clusters
    

# ================================
# 工具函数（不需要FAISS依赖）
# ================================

def extract_descriptions_and_metadata(data):
    """Extract text descriptions and metadata from input data

    Args:
        data (list): List of input data items

    Returns:
        tuple: (descriptions list, metadata_list list)
    """
    descriptions = []
    metadata_list = []

    for item in data:
        # 从cases.jsonl中提取text字段，优先使用title + text
        title = item.get("title", "")
        text = item.get("text") or item.get("content") or ""

        if title and text:
            description = f"{title}\n{text}"
        else:
            description = text or title or ""

        if description.strip():
            descriptions.append(description)
            metadata_list.append({
                'id': str(item.get('id'))  # 确保ID是字符串
            })

    print(f"📊 Extracted {len(descriptions)} valid descriptions")
    return descriptions, metadata_list


def load_data(data_dir="data"):
    """加载数据"""
    import jsonlines

    # 加载案例数据
    cases_data = {}
    with jsonlines.open(f"{data_dir}/cases.jsonl", 'r') as reader:
        for case in reader:
            cases_data[str(case['id'])] = case

    # 加载评估数据
    with open(f"{data_dir}/eval_structure.json", 'r') as f:
        eval_data = json.load(f)

    return cases_data, eval_data


# 尝试导入scipy，如果没有则使用贪心备选方案
try:
    from scipy.optimize import linear_sum_assignment
    import numpy as np
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("⚠️ scipy未安装，将使用贪心匹配备选方案")

# 尝试导入sklearn聚类准确率指标
try:
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️ sklearn未安装，将跳过聚类准确率指标(ARI/NMI)")

def load_ground_truth(true_file):
    """加载真实标签"""
    print(f"📖 加载真实标签: {true_file}")
    with open(true_file, 'r', encoding='utf-8') as f:
        true_data = json.load(f)
        true_clusters = true_data['events']
    return true_clusters

def prepare_cluster_sets_for_eval(pred_clusters, true_clusters):
    """准备聚类集合用于评估"""
    # 预测结果: {"incident_id": "cluster_0", "cases": ["1", "8", ...]}
    pred_sets = {}
    for cluster in pred_clusters:
        incident_id = str(cluster['incident_id'])
        cases = set(str(case_id) for case_id in cluster['cases'])
        pred_sets[incident_id] = cases

    # 真实标签: {"incident_id": 1, "ids": [1, 2, 3, ...]}
    true_sets = {}
    for cluster in true_clusters:
        incident_id = str(cluster['incident_id'])
        ids = set(str(case_id) for case_id in cluster['ids'])
        true_sets[incident_id] = ids

    print(f"📊 预测聚类数: {len(pred_sets)}, 真实聚类数: {len(true_sets)}")
    return pred_sets, true_sets

def calculate_basic_metrics(pred_sets, true_sets):
    total_pred_cases = sum(len(cases) for cases in pred_sets.values())
    total_true_cases = sum(len(ids) for ids in true_sets.values())

    print("📈 基本统计:")
    print(f"   预测聚类数: {len(pred_sets)}")
    print(f"   真实聚类数: {len(true_sets)}")
    print(f"   预测案例总数: {total_pred_cases}")
    print(f"   真实案例总数: {total_true_cases}")

    singleton_pred = sum(1 for cases in pred_sets.values() if len(cases) == 1)
    singleton_true = sum(1 for ids in true_sets.values() if len(ids) == 1)

    pred_singleton_ratio = (singleton_pred / len(pred_sets) * 100) if pred_sets else 0.0
    true_singleton_ratio = (singleton_true / len(true_sets) * 100) if true_sets else 0.0
    print(f"   预测单例簇数: {singleton_pred} ({pred_singleton_ratio:.1f}%)")
    print(f"   真实单例簇数: {singleton_true} ({true_singleton_ratio:.1f}%)")

    pred_sizes = [len(cases) for cases in pred_sets.values()]
    true_sizes = [len(ids) for ids in true_sets.values()]

    if pred_sizes:
        print(f"   预测平均簇大小: {sum(pred_sizes)/len(pred_sizes):.2f}")
        print(f"   预测最大簇大小: {max(pred_sizes)}")
    else:
        print("   预测平均簇大小: 0.00")
        print("   预测最大簇大小: 0")

    if true_sizes:
        print(f"   真实平均簇大小: {sum(true_sizes)/len(true_sizes):.2f}")
        print(f"   真实最大簇大小: {max(true_sizes)}")
    else:
        print("   真实平均簇大小: 0.00")
        print("   真实最大簇大小: 0")

def hungarian_matching_macro_metrics(pred_sets, true_sets, objective="f1"):
    """
    匈牙利匹配（Hungarian / assignment） + 宏平均

    objective:
      - "intersection": 最大化交集数
      - "f1": 最大化簇对F1（推荐）
      - "jaccard": 最大化Jaccard
    """
    if not SCIPY_AVAILABLE:
        # 如果没有scipy，回退到贪心匹配
        print("⚠️ scipy不可用，使用贪心匹配备选方案")
        return greedy_matching_fallback(pred_sets, true_sets, objective)

    pred_ids = list(pred_sets.keys())
    true_ids = list(true_sets.keys())
    P, T = len(pred_ids), len(true_ids)

    # 构造相似度矩阵 S (P x T)，再转成 cost = -S 做最小化
    S = np.zeros((P, T), dtype=np.float64)

    for i, pid in enumerate(pred_ids):
        A = pred_sets[pid]
        if not A:
            continue
        for j, tid in enumerate(true_ids):
            B = true_sets[tid]
            if not B:
                continue
            inter = len(A & B)
            if inter == 0:
                continue

            if objective == "intersection":
                score = inter
            elif objective == "jaccard":
                score = inter / len(A | B)
            else:  # objective == "f1"
                prec = inter / len(A)
                rec = inter / len(B)
                score = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

            S[i, j] = score

    # 匈牙利要求做"最小化 cost"，所以取负数；同时要变成方阵，补 dummy
    n = max(P, T)
    cost = np.zeros((n, n), dtype=np.float64)
    cost[:P, :T] = -S  # maximize S  <=> minimize -S

    row_ind, col_ind = linear_sum_assignment(cost)

    # 得到匹配对（过滤掉 dummy 和 0 交集匹配）
    matches = []
    for r, c in zip(row_ind, col_ind):
        if r < P and c < T and S[r, c] > 0:
            matches.append((pred_ids[r], true_ids[c]))

    # 为每个 pred/true 生成"匹配对象"（没匹配到为 None）
    pred_to_true = {pid: None for pid in pred_ids}
    true_to_pred = {tid: None for tid in true_ids}
    for pid, tid in matches:
        pred_to_true[pid] = tid
        true_to_pred[tid] = pid

    # 宏平均 precision：按 pred cluster 平均
    pred_precisions = []
    pred_f1s = []
    for pid in pred_ids:
        A = pred_sets[pid]
        tid = pred_to_true[pid]
        if tid is None or len(A) == 0:
            pred_precisions.append(0.0)
            pred_f1s.append(0.0)
            continue
        B = true_sets[tid]
        inter = len(A & B)
        prec = inter / len(A) if len(A) else 0.0
        rec = inter / len(B) if len(B) else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        pred_precisions.append(prec)
        pred_f1s.append(f1)

    # 宏平均 recall：按 true cluster 平均
    true_recalls = []
    true_f1s = []
    for tid in true_ids:
        B = true_sets[tid]
        pid = true_to_pred[tid]
        if pid is None or len(B) == 0:
            true_recalls.append(0.0)
            true_f1s.append(0.0)
            continue
        A = pred_sets[pid]
        inter = len(A & B)
        prec = inter / len(A) if len(A) else 0.0
        rec = inter / len(B) if len(B) else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        true_recalls.append(rec)
        true_f1s.append(f1)

    macro_precision = sum(pred_precisions) / len(pred_precisions) if pred_precisions else 0.0
    macro_recall = sum(true_recalls) / len(true_recalls) if true_recalls else 0.0

    # macro_f1 给你一个"对称"的定义：pred视角F1和true视角F1取平均
    macro_f1 = 0.5 * (
        (sum(pred_f1s) / len(pred_f1s) if pred_f1s else 0.0) +
        (sum(true_f1s) / len(true_f1s) if true_f1s else 0.0)
    )

    return {
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "matched_pairs": len(matches),
        "total_pred_clusters": P,
        "total_true_clusters": T,
        "objective": objective
    }

def greedy_matching_fallback(pred_sets, true_sets, objective="f1"):
    """贪心匹配备选方案（当scipy不可用时使用）"""
    pred_ids = list(pred_sets.keys())
    true_ids = list(true_sets.keys())

    # 计算所有可能的匹配得分
    matches = []
    for pred_id, pred_cases in pred_sets.items():
        for true_id, true_cases in true_sets.items():
            inter = len(pred_cases & true_cases)
            if inter == 0:
                continue

            if objective == "intersection":
                score = inter
            elif objective == "jaccard":
                score = inter / len(pred_cases | true_cases)
            else:  # objective == "f1"
                prec = inter / len(pred_cases) if len(pred_cases) > 0 else 0
                rec = inter / len(true_cases) if len(true_cases) > 0 else 0
                score = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

            matches.append((score, pred_id, true_id))

    # 按得分降序排序
    matches.sort(reverse=True)

    # 贪心匹配
    matched_pred = set()
    matched_true = set()
    final_matches = []

    for score, pred_id, true_id in matches:
        if pred_id not in matched_pred and true_id not in matched_true:
            final_matches.append((pred_id, true_id))
            matched_pred.add(pred_id)
            matched_true.add(true_id)

    # 构建映射
    pred_to_true = {pid: None for pid in pred_ids}
    true_to_pred = {tid: None for tid in true_ids}
    for pid, tid in final_matches:
        pred_to_true[pid] = tid
        true_to_pred[tid] = pid

    # 计算宏平均（与匈牙利版本相同）
    pred_precisions = []
    pred_f1s = []
    for pid in pred_ids:
        A = pred_sets[pid]
        tid = pred_to_true[pid]
        if tid is None or len(A) == 0:
            pred_precisions.append(0.0)
            pred_f1s.append(0.0)
            continue
        B = true_sets[tid]
        inter = len(A & B)
        prec = inter / len(A) if len(A) else 0.0
        rec = inter / len(B) if len(B) else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        pred_precisions.append(prec)
        pred_f1s.append(f1)

    true_recalls = []
    true_f1s = []
    for tid in true_ids:
        B = true_sets[tid]
        pid = true_to_pred[tid]
        if pid is None or len(B) == 0:
            true_recalls.append(0.0)
            true_f1s.append(0.0)
            continue
        A = pred_sets[pid]
        inter = len(A & B)
        prec = inter / len(A) if len(A) else 0.0
        rec = inter / len(B) if len(B) else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        true_recalls.append(rec)
        true_f1s.append(f1)

    macro_precision = sum(pred_precisions) / len(pred_precisions) if pred_precisions else 0.0
    macro_recall = sum(true_recalls) / len(true_recalls) if true_recalls else 0.0
    macro_f1 = 0.5 * (
        (sum(pred_f1s) / len(pred_f1s) if pred_f1s else 0.0) +
        (sum(true_f1s) / len(true_f1s) if true_f1s else 0.0)
    )

    return {
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "matched_pairs": len(final_matches),
        "total_pred_clusters": len(pred_sets),
        "total_true_clusters": len(true_sets),
        "objective": objective,
        "method": "greedy_fallback"
    }

def calculate_b3_f1(pred_sets, true_sets):
    """计算B³ F1指标（B-Cubed F1）- 对每个样本计算precision/recall然后平均"""
    # 构建样本到簇的映射
    pred_cluster_map = {}
    true_cluster_map = {}

    for cluster_id, cases in pred_sets.items():
        for case_id in cases:
            pred_cluster_map[case_id] = cluster_id

    for cluster_id, cases in true_sets.items():
        for case_id in cases:
            true_cluster_map[case_id] = cluster_id

    # 获取所有样本
    all_samples = set(pred_cluster_map.keys()) | set(true_cluster_map.keys())

    if not all_samples:
        return {'b3_precision': 0.0, 'b3_recall': 0.0, 'b3_f1': 0.0}

    precision_list = []
    recall_list = []

    for sample in all_samples:
        if sample not in pred_cluster_map or sample not in true_cluster_map:
            continue

        pred_cluster = pred_cluster_map[sample]
        true_cluster = true_cluster_map[sample]

        # 计算簇大小
        pred_cluster_size = len(pred_sets[pred_cluster])
        true_cluster_size = len(true_sets[true_cluster])

        # 计算在预测簇中同时属于真实簇的样本数
        pred_correct = sum(1 for other in pred_sets[pred_cluster]
                          if other in true_cluster_map and true_cluster_map[other] == true_cluster)

        # 计算在真实簇中同时属于预测簇的样本数
        true_correct = sum(1 for other in true_sets[true_cluster]
                          if other in pred_cluster_map and pred_cluster_map[other] == pred_cluster)

        # B³ precision/recall定义
        precision = pred_correct / pred_cluster_size if pred_cluster_size > 0 else 0
        recall = true_correct / true_cluster_size if true_cluster_size > 0 else 0

        precision_list.append(precision)
        recall_list.append(recall)

    # 计算平均
    avg_precision = sum(precision_list) / len(precision_list) if precision_list else 0
    avg_recall = sum(recall_list) / len(recall_list) if recall_list else 0

    # 计算F1
    b3_f1 = 2 * avg_precision * avg_recall / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0

    return {
        'b3_precision': avg_precision,
        'b3_recall': avg_recall,
        'b3_f1': b3_f1
    }

def calculate_clustering_accuracy_metrics(pred_clusters, true_clusters):
    """计算传统的聚类准确率指标（ARI和NMI）

    Args:
        pred_clusters: 预测的聚类结果
        true_clusters: 真实的聚类列表

    Returns:
        dict: 包含ARI和NMI的字典
    """
    if not SKLEARN_AVAILABLE:
        return {'ari': 0.0, 'nmi': 0.0}

    # 构建case_id到簇ID的映射
    pred_cluster_map = {}
    true_cluster_map = {}

    # 处理预测簇
    for cluster in pred_clusters:
        cluster_id = cluster['incident_id']
        for case_id in cluster['cases']:
            pred_cluster_map[str(case_id)] = str(cluster_id)

    # 处理真实簇
    for cluster in true_clusters:
        cluster_id = cluster['incident_id']
        for case_id in cluster['ids']:
            true_cluster_map[str(case_id)] = str(cluster_id)

    # 获取所有case_id
    all_cases = set(pred_cluster_map.keys()) | set(true_cluster_map.keys())

    if not all_cases:
        return {'ari': 0.0, 'nmi': 0.0}

    # 构建标签数组
    y_pred = []
    y_true = []

    for case_id in sorted(all_cases):
        pred_label = pred_cluster_map.get(case_id, 'unknown_pred')
        true_label = true_cluster_map.get(case_id, 'unknown_true')
        y_pred.append(pred_label)
        y_true.append(true_label)

    # 计算ARI和NMI
    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)

    return {'ari': ari, 'nmi': nmi}

def print_matching_macro_metrics(pred_sets, true_sets):
    """打印匈牙利匹配 + 宏平均指标 + B³补充指标"""
    # 使用匈牙利匹配进行评估
    matching_results = hungarian_matching_macro_metrics(pred_sets, true_sets, objective="f1")

    # 计算B³ F1作为补充指标
    b3_results = calculate_b3_f1(pred_sets, true_sets)

    method_name = "匈牙利匹配" if SCIPY_AVAILABLE else "贪心匹配"
    print(f"🎯 {method_name} + 宏平均评估 (目标: F1):")
    print(f"   宏平均精确率: {matching_results['macro_precision']:.4f}")
    print(f"   宏平均召回率: {matching_results['macro_recall']:.4f}")
    print(f"   宏平均F1: {matching_results['macro_f1']:.4f}")
    print(f"   匹配簇对数: {matching_results['matched_pairs']}")
    print(f"   预测簇总数: {matching_results['total_pred_clusters']}")
    print(f"   真实簇总数: {matching_results['total_true_clusters']}")

    print(f"\n🔍 B³补充指标 (样本级评估):")
    print(f"   B³精确率: {b3_results['b3_precision']:.4f}")
    print(f"   B³召回率: {b3_results['b3_recall']:.4f}")
    print(f"   B³ F1: {b3_results['b3_f1']:.4f}")

    return matching_results['macro_f1']

def kfold_split_incident_ids(gt_clusters, k: int = 5):
    """按incident_id进行k折交叉验证划分"""
    from collections import defaultdict

    # 统计每个incident的大小
    incident_sizes = defaultdict(list)
    for case_id, incident_id in gt_clusters.items():
        incident_sizes[incident_id].append(case_id)

    # 按大小排序，确保大簇均匀分布到各折
    sorted_incidents = sorted(incident_sizes.keys(),
                            key=lambda x: len(incident_sizes[x]),
                            reverse=True)

    folds = [[] for _ in range(k)]
    fold_sizes = [0] * k

    # 轮流分配到各折，确保大小相对均衡
    for incident_id in sorted_incidents:
        incident_size = len(incident_sizes[incident_id])
        # 选择当前最小的折
        min_fold_idx = fold_sizes.index(min(fold_sizes))
        folds[min_fold_idx].append(incident_id)
        fold_sizes[min_fold_idx] += incident_size

    # 转换为train/val格式
    kfolds = []
    for i in range(k):
        val_incidents = set(folds[i])
        train_incidents = set()
        for j in range(k):
            if j != i:
                train_incidents.update(folds[j])

        kfolds.append((train_incidents, val_incidents))

    print(f"✅ {k}折交叉验证划分完成:")
    for i, (train_incidents, val_incidents) in enumerate(kfolds):
        train_cases = sum(len(incident_sizes[inc]) for inc in train_incidents)
        val_cases = sum(len(incident_sizes[inc]) for inc in val_incidents)
        print(f"   折{i+1}: 训练 {len(train_incidents)}个簇/{train_cases}个案例, 验证 {len(val_incidents)}个簇/{val_cases}个案例")

    return kfolds

def filter_cases_by_incidents(gt_clusters, incident_ids):
    """根据incident_ids过滤case"""
    return {case_id: incident_id for case_id, incident_id in gt_clusters.items()
            if incident_id in incident_ids}

def extract_vectors_from_db(vector_db):
    """从VectorDatabase中提取向量数组、描述和元数据"""
    n_total = vector_db.index.ntotal
    vectors = np.zeros((n_total, vector_db.dim), dtype=np.float32)
    descriptions = []
    metadata_list = []

    for i in range(n_total):
        vectors[i] = vector_db.index.reconstruct(i)
        descriptions.append(vector_db.ids_to_description[i])
        metadata_list.append(vector_db.ids_to_metadata[i])

    return vectors, descriptions, metadata_list

def repeated_outer_evaluation_vector_baseline(gt_clusters, data_dir="data", output_dir="outputs",
                                             repeats=3, thresholds=None, k_folds=5, model_path="BAAI/bge-m3"):
    """向量基线的外层重复评估：验证阈值选择的可靠性"""

    if thresholds is None:
        thresholds = [0.70, 0.72, 0.74, 0.76, 0.78, 0.80, 0.82, 0.84]

    print(f"\n🔄 开始向量基线外层重复评估 (repeats={repeats})...")
    print("=" * 70)

    outer_runs = []

    for repeat_idx in range(repeats):
        seed = 42 + repeat_idx
        print(f"\n📊 外层重复 #{repeat_idx + 1}/{repeats} (seed={seed})")
        print("-" * 50)

        # 设置随机种子
        np.random.seed(seed)

        # 1. 外层数据划分（使用与主pipeline相同的逻辑）
        print("✂️ 按incident_id划分外层数据...")
        from collections import defaultdict
        incident_sizes = defaultdict(list)
        for case_id, incident_id in gt_clusters.items():
            incident_sizes[incident_id].append(case_id)

        # (size, incident_id) 列表
        items = [(len(v), inc) for inc, v in incident_sizes.items()]
        items.sort(key=lambda x: x[0], reverse=True)

        # 对相同size的incident打散（保证"按大小优先"的同时引入随机性）
        rng = np.random.default_rng(seed)
        i = 0
        shuffled = []
        while i < len(items):
            j = i
            while j < len(items) and items[j][0] == items[i][0]:
                j += 1
            bucket = items[i:j]
            rng.shuffle(bucket)
            shuffled.extend(bucket)
            i = j

        sorted_incidents = [inc for _, inc in shuffled]

        train_incidents = set()
        dev_incidents = set()
        test_incidents = set()

        total_cases = len(gt_clusters)
        train_target = int(total_cases * 0.7)  # 统一使用70/10/20
        dev_target = int(total_cases * 0.1)

        train_count = 0
        dev_count = 0

        for incident_id in sorted_incidents:
            incident_size = len(incident_sizes[incident_id])

            if train_count + incident_size <= train_target:
                train_incidents.add(incident_id)
                train_count += incident_size
            elif dev_count + incident_size <= dev_target:
                dev_incidents.add(incident_id)
                dev_count += incident_size
            else:
                test_incidents.add(incident_id)

        print(f"   训练集: {len(train_incidents)}个事件/{train_count}个案例")
        print(f"   开发集: {len(dev_incidents)}个事件/{dev_count}个案例")
        print(f"   测试集: {len(test_incidents)}个事件/{total_cases - train_count - dev_count}个案例")

        # 2. 加载向量数据（假设已预计算）
        vector_db = load_or_create_vector_db(data_dir, output_dir, model_path)

        # 3. 内层CV：在train+dev上选择最优阈值
        print("\n🎯 内层CV：在train+dev上选择最优阈值...")
        inner_gt = {}
        for case_id, incident_id in gt_clusters.items():
            if incident_id in train_incidents or incident_id in dev_incidents:
                inner_gt[case_id] = incident_id

        vectors, descriptions, metadata_list = extract_vectors_from_db(vector_db)
        best_threshold, best_score = grid_search_threshold(
            vectors, descriptions, metadata_list, inner_gt, thresholds, k_folds=k_folds
        )

        # 4. 用最优阈值在test上最终评估
        print(f"\n🧪 用最优阈值 {best_threshold} 在test上最终评估...")
        test_case_ids = set(str(k) for k in gt_clusters.keys() if gt_clusters[k] in test_incidents)

        if not test_case_ids:
            print("⚠️ 测试集为空，跳过")
            continue

        # 过滤测试集数据
        test_vectors = []
        test_descriptions = []
        test_metadata = []

        for i, meta in enumerate(metadata_list):
            if meta['id'] in test_case_ids:
                test_vectors.append(vectors[i])
                test_descriptions.append(descriptions[i])
                test_metadata.append(meta)

        # 创建测试集向量数据库并聚类
        test_vector_db = VectorDatabase(dim=vectors.shape[1])
        test_vector_db.add_vectors(np.array(test_vectors), test_descriptions, test_metadata)
        event_clusters = build_event_clusters_from_faiss(test_vector_db, high_threshold=best_threshold)

        # 构造测试集的真实标签（只包含test_incidents）
        true_clusters_test = []
        for incident_id in test_incidents:
            case_ids = [case_id for case_id, inc_id in gt_clusters.items() if inc_id == incident_id]
            if case_ids:  # 只添加非空簇
                true_clusters_test.append({
                    "incident_id": incident_id,
                    "ids": case_ids
                })

        # 评估
        vector_metrics = evaluate_clustering_results(event_clusters, true_clusters_test)

        print("\n📊 本轮测试结果:")
        print(f"   内层CV最佳得分: {best_score:.4f}")
        print(f"   测试集Macro F1: {vector_metrics['hungarian_macro_f1']:.4f}")
        if SKLEARN_AVAILABLE:
            print(f"   测试集ARI: {vector_metrics['ari']:.4f}")
            print(f"   测试集NMI: {vector_metrics['nmi']:.4f}")

        # 保存本轮结果
        run_result = {
            "repeat_idx": repeat_idx,
            "seed": seed,
            "data_split": {
                "train_events": len(train_incidents),
                "train_cases": len([k for k in gt_clusters.keys() if gt_clusters[k] in train_incidents]),
                "dev_events": len(dev_incidents),
                "dev_cases": len([k for k in gt_clusters.keys() if gt_clusters[k] in dev_incidents]),
                "test_events": len(test_incidents),
                "test_cases": len([k for k in gt_clusters.keys() if gt_clusters[k] in test_incidents])
            },
            "best_threshold_inner": best_threshold,
            "inner_cv_score": best_score,
            "test_score": vector_metrics['hungarian_macro_f1'],
            "test_ari": vector_metrics['ari'],
            "test_nmi": vector_metrics['nmi'],
            "test_metrics": vector_metrics
        }
        outer_runs.append(run_result)

    # 计算汇总统计
    print(f"\n📈 向量基线外层重复评估汇总 (repeats={repeats})")
    print("=" * 70)

    test_scores = [run["test_score"] for run in outer_runs]
    ari_scores = [run["test_ari"] for run in outer_runs]
    nmi_scores = [run["test_nmi"] for run in outer_runs]

    summary = {
        "test_score_mean": float(np.mean(test_scores)),
        "test_score_std": float(np.std(test_scores)),
        "ari_mean": float(np.mean(ari_scores)),
        "ari_std": float(np.std(ari_scores)),
        "nmi_mean": float(np.mean(nmi_scores)),
        "nmi_std": float(np.std(nmi_scores)),
    }

    print("🎯 测试集得分可靠性:")
    print(f"   Macro F1: {summary['test_score_mean']:.4f} ± {summary['test_score_std']:.4f}")
    print(f"   Adjusted Rand Index (ARI): {summary['ari_mean']:.4f} ± {summary['ari_std']:.4f}")
    print(f"   Normalized Mutual Info (NMI): {summary['nmi_mean']:.4f} ± {summary['nmi_std']:.4f}")

    # 保存报告
    reliability_report = {
        "method": "vector_baseline",
        "repeats": repeats,
        "thresholds_candidates": thresholds,
        "outer_runs": outer_runs,
        "summary": summary
    }

    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/vector_baseline_reliability_report.json", 'w', encoding='utf-8') as f:
        json.dump(reliability_report, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n💾 向量基线可靠性报告已保存: {output_dir}/vector_baseline_reliability_report.json")
    return reliability_report

def load_or_create_vector_db(data_dir="data", output_dir="outputs", model_path="BAAI/bge-m3"):
    """加载或创建向量数据库"""
    index_dir = f"{output_dir}/faiss_index"
    index_path = f"{index_dir}/faiss_index.faiss"
    mapping_path = f"{index_dir}/vector_mapping.json"

    if os.path.exists(index_path) and os.path.exists(mapping_path):
        print(f"✅ 发现已保存的FAISS索引，正在加载...")
        vector_db = VectorDatabase.load(index_path, mapping_path)
        print(f"✅ FAISS索引加载成功，包含 {vector_db.index.ntotal} 个向量")
        return vector_db

    # 创建新的向量数据库
    print("📝 未发现已保存的索引，将创建新的索引")
    import jsonlines
    with jsonlines.open(f"{data_dir}/cases.jsonl") as reader:
        data = list(reader)

    descriptions, metadata_list = extract_descriptions_and_metadata(data)
    vectorizer = TextVectorizer(model_path=model_path)
    vectors = vectorizer.batch_get_embeddings(descriptions, batch_size=DEFAULT_BATCH_SIZE)

    # L2归一化
    faiss.normalize_L2(vectors)

    vector_db = VectorDatabase(dim=vectors.shape[1])
    vector_db.add_vectors(vectors, descriptions, metadata_list)

    # 保存
    os.makedirs(index_dir, exist_ok=True)
    vector_db.save(index_path, mapping_path)
    print(f"💾 FAISS索引已保存到: {index_dir}")

    return vector_db

def grid_search_threshold(vectors, descriptions, metadata_list, gt_clusters, thresholds, k_folds=5):
    """网格搜索最优相似度阈值"""
    print(f"\n🎯 开始{k_folds}折交叉验证阈值搜索...")

    # 获取k折划分
    kfolds = kfold_split_incident_ids(gt_clusters, k=k_folds)

    best_threshold = None
    best_score = -1.0

    print(f"测试 {len(thresholds)} 个阈值 × {k_folds} 折 = {len(thresholds) * k_folds} 次评估")

    for threshold in thresholds:
        fold_scores = []

        for fold_idx, (train_incidents, val_incidents) in enumerate(kfolds):
            # 获取当前折的验证数据
            fold_val_gt = filter_cases_by_incidents(gt_clusters, val_incidents)
            val_case_ids = set(str(k) for k in fold_val_gt.keys())

            # 过滤向量和元数据到验证集
            val_vectors = []
            val_descriptions = []
            val_metadata = []

            for i, meta in enumerate(metadata_list):
                if meta['id'] in val_case_ids:
                    val_vectors.append(vectors[i])
                    val_descriptions.append(descriptions[i])
                    val_metadata.append(meta)

            if not val_vectors:
                print(f"⚠️ 折{fold_idx+1} 没有验证数据，跳过")
                continue

            # 创建验证集向量数据库
            val_vector_db = VectorDatabase(dim=vectors.shape[1])
            val_vector_db.add_vectors(np.array(val_vectors), val_descriptions, val_metadata)

            # 使用当前阈值进行聚类
            val_event_clusters = build_event_clusters_from_faiss(val_vector_db, high_threshold=threshold)

            # 评估当前折的结果
            pred_sets = {cluster['incident_id']: set(cluster['cases']) for cluster in val_event_clusters}
            true_sets = {str(inc_id): set(str(case_id) for case_id, inc in fold_val_gt.items() if inc == inc_id)
                        for inc_id in val_incidents}

            if pred_sets and true_sets:
                matching_results = hungarian_matching_macro_metrics(pred_sets, true_sets, objective="f1")
                fold_scores.append(matching_results["macro_f1"])

        # 计算当前阈值的平均分数
        if fold_scores:
            avg_score = np.mean(fold_scores)
            std_score = np.std(fold_scores)

            if avg_score > best_score:
                best_score = avg_score
                best_threshold = threshold

                print(f"🆕 新最佳阈值: {threshold}")
                print(f"   平均Macro F1: {avg_score:.4f} ± {std_score:.4f}")
                print(f"   各折分数: {[f'{s:.3f}' for s in fold_scores]}")

    print(f"🏆 {k_folds}折交叉验证阈值搜索完成!")
    print(f"🏆 最佳阈值: {best_threshold}, 最佳平均Macro F1: {best_score:.4f}")

    return best_threshold, best_score

def evaluate_clustering_results(pred_clusters, true_clusters):
    """Evaluate predicted clusters with the shared evaluation module."""
    config = EvaluationConfig(pred_file="", true_file="", output_file=None, use_ari=SKLEARN_AVAILABLE, use_nmi=True)
    return compute_metrics(pred_clusters, true_clusters, config)


def main():
    """Public baseline entrypoint."""
    parser = argparse.ArgumentParser(description="Run the single vector-similarity baseline for event alignment")
    parser.add_argument("--model_path", type=str, default="BAAI/bge-m3",
                       help="模型路径或HuggingFace模型名称 (默认: BAAI/bge-m3，会自动下载)")
    parser.add_argument("--threshold", type=float, default=None,
                       help="固定相似度阈值 (如果不指定，将使用k折交叉验证搜索最优阈值)")
    parser.add_argument("--thresholds", type=str, default="0.70,0.72,0.74,0.76,0.78,0.80,0.82,0.84",
                       help="阈值搜索范围，用逗号分隔 (默认: 0.70-0.84)")
    parser.add_argument("--k_folds", type=int, default=5,
                       help="k折交叉验证的折数 (默认: 5)")
    parser.add_argument("--use_cv", action="store_true",
                       help="强制使用k折交叉验证搜索最优阈值")
    parser.add_argument("--data_dir", type=str, default=str(DEFAULT_DATA_DIR),
                       help="数据目录")
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
                       help="输出目录")

    args = parser.parse_args()
    return main_single_run(args)

def main_single_run(args):
    """Run the public single baseline experiment."""

    # 解析阈值搜索范围
    thresholds = [float(x.strip()) for x in args.thresholds.split(',')]

    # 检查依赖
    if not VECTOR_BASELINE_AVAILABLE:
        print("❌ 依赖未安装，请运行: pip install faiss-cpu sentence-transformers torch")
        return

    try:
        print("🚀 开始向量相似度基线实验 (使用本地sentence-transformers模型 + k折交叉验证)...")
        print(f"配置: 模型={args.model_path}, k折={args.k_folds}")

        # 加载ground truth数据用于交叉验证
        print("📖 加载评估数据...")
        gt_clusters = {}
        with open(f"{args.data_dir}/eval_structure.json", 'r') as f:
            gt_data = json.load(f)
            for event in gt_data['events']:
                for case_id in event['ids']:
                    gt_clusters[case_id] = event['incident_id']

        print(f"📊 加载了 {len(gt_clusters)} 个案例的ground truth")

        # 检查是否已有保存的FAISS索引
        index_dir = f"{args.output_dir}/faiss_index"
        index_path = f"{index_dir}/faiss_index.faiss"
        mapping_path = f"{index_dir}/vector_mapping.json"

        if os.path.exists(index_path) and os.path.exists(mapping_path):
            print(f"✅ 发现已保存的FAISS索引，正在加载...")
            try:
                vector_db = VectorDatabase.load(index_path, mapping_path)
                print(f"✅ FAISS索引加载成功，包含 {vector_db.index.ntotal} 个向量")
                use_existing_index = True
            except Exception as e:
                print(f"⚠️ 索引加载失败: {e}, 将重新创建")
                use_existing_index = False
        else:
            print("📝 未发现已保存的索引，将创建新的索引")
            use_existing_index = False

        if not use_existing_index:
            # 加载数据 - 直接从cases.jsonl加载
            import jsonlines
            print(f"📖 读取数据文件: {args.data_dir}/cases.jsonl")
            with jsonlines.open(f"{args.data_dir}/cases.jsonl") as reader:
                data = list(reader)
            print(f"📊 加载了 {len(data)} 个案例")

            # 提取描述和元数据
            descriptions, metadata_list = extract_descriptions_and_metadata(data)

            # 初始化向量化器
            vectorizer = TextVectorizer(model_path=args.model_path)

            # 生成向量
            print("🤖 开始向量化...")
            vectors = vectorizer.batch_get_embeddings(descriptions, batch_size=DEFAULT_BATCH_SIZE)
            print(f"✅ 生成 {len(vectors)} 个向量，维度: {vectors.shape[1]}")

            # 🔧 关键修复：L2归一化，让内积等价余弦相似度
            faiss.normalize_L2(vectors)
            print("✅ 向量已L2归一化，内积距离等价余弦相似度")

            # 创建向量数据库
            vector_db = VectorDatabase(dim=vectors.shape[1])
            vector_db.add_vectors(vectors, descriptions, metadata_list)
            print(f"✅ 向量数据库创建完成，包含 {vector_db.index.ntotal} 个向量")

            # 保存FAISS索引 (可选，用于重用)
            index_dir = f"{args.output_dir}/faiss_index"
            os.makedirs(index_dir, exist_ok=True)
            index_path = f"{index_dir}/faiss_index.faiss"
            mapping_path = f"{index_dir}/vector_mapping.json"
            vector_db.save(index_path, mapping_path)
            print(f"💾 FAISS索引已保存到: {index_dir}")

        # 确定最优阈值
        if args.threshold is None or args.use_cv:
            print(f"\n🎯 使用{args.k_folds}折交叉验证搜索最优阈值...")
            vectors, descriptions, metadata_list = extract_vectors_from_db(vector_db)
            best_threshold, best_score = grid_search_threshold(
                vectors, descriptions, metadata_list, gt_clusters, thresholds, k_folds=args.k_folds
            )
            print(f"🏆 最优阈值: {best_threshold}, 交叉验证F1: {best_score:.4f}")
        else:
            best_threshold = args.threshold
            print(f"📋 使用固定阈值: {best_threshold}")

        # 🔧 关键修复：支持inductive测试评估（与主pipeline对齐）
        # 按incident_id划分train/dev/test（确保与主pipeline相同的数据划分）
        print("\n✂️ 按incident_id划分数据集（与主pipeline对齐）...")
        # 使用与主pipeline相同的划分逻辑（7:1:2）
        from collections import defaultdict
        incident_sizes = defaultdict(list)
        for case_id, incident_id in gt_clusters.items():
            incident_sizes[incident_id].append(case_id)

        sorted_incidents = sorted(incident_sizes.keys(),
                                key=lambda x: len(incident_sizes[x]),
                                reverse=True)

        train_incidents = set()
        dev_incidents = set()
        test_incidents = set()

        total_cases = len(gt_clusters)
        train_target = int(total_cases * 0.7)
        dev_target = int(total_cases * 0.1)

        train_count = 0
        dev_count = 0

        for incident_id in sorted_incidents:
            incident_size = len(incident_sizes[incident_id])

            if train_count + incident_size <= train_target:
                train_incidents.add(incident_id)
                train_count += incident_size
            elif dev_count + incident_size <= dev_target:
                dev_incidents.add(incident_id)
                dev_count += incident_size
            else:
                test_incidents.add(incident_id)

        print(f"✅ 数据划分: 训练集 {len(train_incidents)}个事件/{train_count}个案例, "
              f"开发集 {len(dev_incidents)}个事件/{dev_count}个案例, "
              f"测试集 {len(test_incidents)}个事件/{total_cases - train_count - dev_count}个案例")

        # 使用最优阈值在测试集上进行最终评估
        print(f"\n🏆 使用最优阈值 {best_threshold} 在测试集上进行最终评估...")

        # 过滤测试集的向量数据
        test_case_ids = set(str(k) for k in gt_clusters.keys() if gt_clusters[k] in test_incidents)
        test_vectors = []
        test_descriptions = []
        test_metadata = []

        vectors, descriptions, metadata_list = extract_vectors_from_db(vector_db)
        for i, meta in enumerate(metadata_list):
            if meta['id'] in test_case_ids:
                test_vectors.append(vectors[i])
                test_descriptions.append(descriptions[i])
                test_metadata.append(meta)

        if not test_vectors:
            print("⚠️ 测试集为空，使用全量数据评估")
            event_clusters = build_event_clusters_from_faiss(vector_db, high_threshold=best_threshold)
        else:
            # 创建测试集向量数据库
            test_vector_db = VectorDatabase(dim=vectors.shape[1])
            test_vector_db.add_vectors(np.array(test_vectors), test_descriptions, test_metadata)
            event_clusters = build_event_clusters_from_faiss(test_vector_db, high_threshold=best_threshold)

        # 构造测试集的真实标签（只包含test_incidents）
        true_clusters_test = []
        for incident_id in test_incidents:
            case_ids = [case_id for case_id, inc_id in gt_clusters.items() if inc_id == incident_id]
            if case_ids:  # 只添加非空簇
                true_clusters_test.append({
                    "incident_id": incident_id,
                    "ids": case_ids
                })

        # 评估结果 (完整的评估)
        print("\n🧪 开始评估聚类结果...")
        vector_metrics = evaluate_clustering_results(event_clusters, true_clusters_test)

        # 保存结果
        os.makedirs(args.output_dir, exist_ok=True)

        # 保存聚类结果
        clusters_path = f"{args.output_dir}/clusters.json"
        metrics_path = f"{args.output_dir}/metrics.json"
        with open(clusters_path, 'w', encoding='utf-8') as f:
            json.dump(event_clusters, f, ensure_ascii=False, indent=2)

        payload = build_metrics_payload(
            vector_metrics,
            EvaluationConfig(
                pred_file=clusters_path,
                true_file=f"{args.data_dir}/eval_structure.json",
                output_file=metrics_path,
                use_ari=SKLEARN_AVAILABLE,
                use_nmi=True,
            ),
        )
        payload["baseline"] = {
            'method': 'vector_baseline_local_model_with_cv',
            'config': {
                'model_path': args.model_path,
                'similarity_threshold': best_threshold,
                'original_threshold': args.threshold,
                'used_cv': args.threshold is None or args.use_cv,
                'k_folds': args.k_folds,
                'threshold_candidates': thresholds,
                'device': 'cuda' if torch.cuda.is_available() else 'cpu'
            },
        }
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        print("\n✅ 向量基线实验完成!")
        print("=" * 70)
        print(f"📊 实验配置:")
        threshold_info = f"交叉验证选择 (k={args.k_folds})" if args.threshold is None or args.use_cv else "固定阈值"
        print(f"   🔧 阈值选择: {threshold_info}")
        print(f"   🎯 最优阈值: {best_threshold}")
        print(f"   🤖 模型: {args.model_path}")

        print(f"\n📈 最终评估结果:")
        print(f"   🎯 匈牙利宏平均F1: {vector_metrics['matching']['macro_f1']*100:.1f}%")
        print(f"   🔍 B³ F1: {vector_metrics['b3']['b3_f1']*100:.1f}%")
        if SKLEARN_AVAILABLE and 'ari' in vector_metrics['standardized']:
            print(f"   📊 Adjusted Rand Index (ARI): {vector_metrics['standardized']['ari']:.4f}")
        if 'nmi' in vector_metrics['standardized']:
            print(f"   📊 Normalized Mutual Info (NMI): {vector_metrics['standardized']['nmi']:.4f}")
        coverage = (
            vector_metrics['matching']['matched_pairs'] / vector_metrics['matching']['total_true_clusters'] * 100
            if vector_metrics['matching']['total_true_clusters'] else 0.0
        )
        print(f"   📈 真实事件覆盖率: {coverage:.1f}% ({vector_metrics['matching']['matched_pairs']}/{vector_metrics['matching']['total_true_clusters']})")
        print(f"   📊 预测簇数: {vector_metrics['stats']['total_pred_clusters']}")
        print(f"   🎪 单例簇比例: {vector_metrics['stats']['pred_singleton_ratio']*100:.1f}%")

        print(f"\n📁 输出文件:")
        print(f"   - {clusters_path}: 聚类结果")
        print(f"   - {metrics_path}: 统一评估指标与baseline配置")

    except Exception as e:
        print(f"❌ 实验失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
