#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_db_embeddings_and_faiss.py  (B1 Route)

从 PostgreSQL(ai_risk_events_news) 生成 embeddings，并构建/更新 FAISS 向量库，同时维护 SQLite 索引
以支持“按新闻 id 精确定位 embedding（full 或 shards）”。

运行方式：
python src/Incident_Align_Method/full_application/build_db_embeddings_and_faiss.py \
  --config config/Incident_Align_Method-full_application-config.ini

核心特性：
- 数据筛选：由配置文件中的 `classification_result` 控制
- 按月增量：严格使用 archive_year + archive_month（不是事件时间，不是 release_date）
- 主键对齐：使用 ai_risk_events_news.news_id 作为向量 ID（重复文本不影响，因为 news_id 唯一）
- 输出：
  artifacts_root/
    embeddings/
      text/
        full/emb_text_full.npy, ids_text_full.npy, valid_mask_text_full.npy
        shards/emb_text_YYYY-MM.npy, ids_text_YYYY-MM.npy, valid_mask_text_YYYY-MM.npy
      event/...
      elements/
        actor_main/full/emb_actor_main_full.npy ...
        actor_list/shards/emb_actor_list_YYYY-MM.npy (3D)
        ...
    faiss/
      text.index, text_ids.npy, text_meta.json
      event.index, event_ids.npy, event_meta.json
      shards/ (可选 shard 索引)
    embedding_index/
      embeddings.sqlite
    runs/
      run_YYYYmmdd_HHMMSS_config.json

要素向量化（默认开启）：
- 字符串字段（2D）：[N, D]
- 列表字段（3D）：[N, max_items, D]  —— 不做 pooling

依赖：
- psycopg2
- sentence-transformers
- torch
- numpy
- tqdm
- faiss-cpu (import faiss)

"""

import argparse
import json
import os
import shutil
import time
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# DB
try:
    import psycopg2
    import psycopg2.extras
except Exception as e:
    print("❌ 缺少数据库依赖：", repr(e))
    raise SystemExit(1)

# FAISS
try:
    import faiss  # type: ignore
except Exception as e:
    print("❌ 缺少 faiss 依赖（建议安装 faiss-cpu）：", repr(e))
    raise SystemExit(1)

from config_utils import (
    DEFAULT_CONFIG_PATH,
    get_bool,
    get_int,
    load_config,
    load_database_config,
    require_section,
    resolve_path,
    split_csv,
)

# =========================
# Reproducibility
# =========================
import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# =========================
# Constants: element fields
# =========================
# 列表字段：保留 3D，不 pooling
ELEMENT_LIST_FIELDS = [
    "actor_list",
    "ai_system_list",
    "ai_system_type_list",
    "ai_system_domain_list",
]

# 字符串字段：2D
ELEMENT_STRING_FIELDS = [
    "actor_main",
    "actor_main_type",
    "ai_system",
    "domain",
    "event_type",
    "event_cause",
    "event_process",
    "event_result",
    "ai_risk_description",
    "ai_risk_type",
    "ai_risk_subtype",
    "harm_type",
    "harm_severity",
    "affected_actor_type",
    "affected_actor_subtype",
    "realized_or_potential",
    "risk_stage",
]

# =========================
# Helpers
# =========================
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def now_ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())

def to_str(x) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    try:
        return json.dumps(x, ensure_ascii=False)
    except Exception:
        return str(x)

def clip(x, n: int) -> str:
    s = to_str(x).strip()
    return s[:n]

def parse_month(s: str) -> Tuple[int, int]:
    try:
        y_str, m_str = s.split("-")
        y = int(y_str)
        m = int(m_str)
    except Exception:
        raise ValueError(f"Invalid month format: {s}. Expected YYYY-MM")
    if not (1 <= m <= 12):
        raise ValueError(f"Invalid month: {s}. Month must be 1..12")
    return y, m

def month_to_str(y: int, m: int) -> str:
    return f"{y:04d}-{m:02d}"

def iter_months_inclusive(start: str, end: str) -> List[str]:
    sy, sm = parse_month(start)
    ey, em = parse_month(end)
    out = []
    y, m = sy, sm
    while (y < ey) or (y == ey and m <= em):
        out.append(month_to_str(y, m))
        m += 1
        if m == 13:
            m = 1
            y += 1
    return out

def resolve_model_dir(model_path: str) -> str:
    p = Path(model_path)
    if not p.exists():
        return model_path

    if p.name == "snapshots":
        snaps = sorted(p.glob("*"), key=lambda x: x.stat().st_mtime, reverse=True)
        if not snaps:
            raise RuntimeError(f"No snapshots found in: {p}")
        return str(snaps[0])

    snap_dir = p / "snapshots"
    if snap_dir.exists():
        snaps = sorted(snap_dir.glob("*"), key=lambda x: x.stat().st_mtime, reverse=True)
        if not snaps:
            raise RuntimeError(f"No snapshots found in: {snap_dir}")
        return str(snaps[0])

    return str(p)

def save_npy_atomic(path: Path, arr: np.ndarray):
    # 确保目标目录存在
    path.parent.mkdir(parents=True, exist_ok=True)
    # 修复：确保tmp文件名以.npy结尾，避免np.save自动追加.npz
    tmp = path.with_suffix(".tmp.npy")
    np.save(tmp, arr)
    os.replace(tmp, path)

# =========================
# Views: E_text / E_event
# =========================
def build_text_view(row: Dict) -> str:
    title = clip(row.get("title", ""), 200)
    content = clip(row.get("content", ""), 6000)
    s = f"Title: {title}\nBody: {content}".strip()
    return s[:7000]

def build_event_view(row: Dict) -> str:
    title = clip(row.get("title", ""), 200)

    # 事件时间仅作为文本内容展示，不用于按月筛选
    event_time = ""
    if row.get("event_time_start"):
        event_time = str(row["event_time_start"])[:10]
    elif row.get("release_date"):
        event_time = str(row["release_date"])[:10]

    event_subject = clip(row.get("event_actor_main", ""), 200)
    event_type = clip(row.get("event_type", ""), 100)

    actor_list = row.get("event_actor_list", []) or []
    if isinstance(actor_list, str):
        try:
            actor_list = json.loads(actor_list)
        except Exception:
            actor_list = [actor_list]
    if not isinstance(actor_list, list):
        actor_list = [actor_list] if actor_list else []
    actors_str = ", ".join(clip(a, 100) for a in actor_list[:10])

    ai_system = clip(row.get("event_ai_system", ""), 100)
    ai_system_list = row.get("ai_system_list", []) or []
    if isinstance(ai_system_list, str):
        try:
            ai_system_list = json.loads(ai_system_list)
        except Exception:
            ai_system_list = [ai_system_list]
    if not isinstance(ai_system_list, list):
        ai_system_list = [ai_system_list] if ai_system_list else []
    if ai_system and ai_system not in ai_system_list:
        ai_system_list.insert(0, ai_system)
    ai_tech_str = ", ".join(clip(s, 100) for s in ai_system_list[:10])

    ai_risk_type = clip(row.get("ai_risk_type", ""), 100)
    ai_risk_desc = clip(row.get("ai_risk_description", ""), 200)
    ai_risk_str = " | ".join([p for p in (ai_risk_type, ai_risk_desc) if p])

    event_cause = clip(row.get("event_cause", ""), 400)
    event_process = clip(row.get("event_process", ""), 400)
    event_result = clip(row.get("event_result", ""), 200)

    result = (
        f"Title: {title}\n"
        f"Time: {event_time}\n"
        f"EventType: {event_type}\n"
        f"Subject: {event_subject}\n"
        f"Actors: {actors_str}\n"
        f"AITech: {ai_tech_str}\n"
        f"AIRisk: {ai_risk_str}\n"
        f"Cause: {event_cause}\n"
        f"Process: {event_process}\n"
        f"Result: {event_result}"
    ).strip()
    return result[:7000]

def _parse_list(x) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return []
        try:
            v = json.loads(s)
            if isinstance(v, list):
                return v
            return [str(v)]
        except Exception:
            return [s]
    return [str(x)]

def build_element_views(row: Dict, max_items: int) -> Dict[str, object]:
    """
    返回 dict:
    - 字符串字段 -> str
    - 列表字段 -> List[str]  (后续编码为 3D)
    """
    views: Dict[str, object] = {}

    # event_annotation 映射
    views["actor_main"] = clip(row.get("event_actor_main", ""), 200)
    views["actor_main_type"] = clip(row.get("event_actor_main_type", ""), 80)
    actors = _parse_list(row.get("event_actor_list", []))
    actors = [clip(a, 100) for a in actors if str(a).strip()][:max_items]
    views["actor_list"] = actors

    views["ai_system"] = clip(row.get("event_ai_system", ""), 150)
    views["domain"] = clip(row.get("event_domain", ""), 80)
    views["event_type"] = clip(row.get("event_type", ""), 150)
    views["event_cause"] = clip(row.get("event_cause", ""), 800)
    views["event_process"] = clip(row.get("event_process", ""), 800)
    views["event_result"] = clip(row.get("event_result", ""), 800)

    # ai_tech
    ai_system_list = _parse_list(row.get("ai_system_list", []))
    ai_system_list = [clip(a, 100) for a in ai_system_list if str(a).strip()][:max_items]
    views["ai_system_list"] = ai_system_list

    ai_system_type_list = _parse_list(row.get("ai_system_type_list", []))
    ai_system_type_list = [clip(a, 100) for a in ai_system_type_list if str(a).strip()][:max_items]
    views["ai_system_type_list"] = ai_system_type_list

    ai_system_domain_list = _parse_list(row.get("ai_system_domain_list", []))
    ai_system_domain_list = [clip(a, 100) for a in ai_system_domain_list if str(a).strip()][:max_items]
    views["ai_system_domain_list"] = ai_system_domain_list

    # ai_risk
    views["ai_risk_description"] = clip(row.get("ai_risk_description", ""), 1200)
    views["ai_risk_type"] = clip(row.get("ai_risk_type", ""), 150)
    views["ai_risk_subtype"] = clip(row.get("ai_risk_subtype", ""), 150)
    views["harm_type"] = clip(row.get("harm_type", ""), 150)
    views["harm_severity"] = clip(row.get("harm_severity", ""), 80)
    views["affected_actor_type"] = clip(row.get("affected_actor_type", ""), 150)
    views["affected_actor_subtype"] = clip(row.get("affected_actor_subtype", ""), 150)
    views["realized_or_potential"] = clip(row.get("realized_or_potential", ""), 80)
    views["risk_stage"] = clip(row.get("risk_stage", ""), 150)

    return views

# =========================
# DB query
# =========================
SELECT_COLUMNS = """
    news_id, title, content, release_date, event_time_start,
    event_actor_main, event_actor_main_type, event_actor_list,
    event_ai_system, event_domain, event_type, event_cause, event_process, event_result,
    ai_system_list, ai_system_type_list, ai_system_domain_list,
    ai_risk_description, ai_risk_type, ai_risk_subtype,
    harm_type, harm_severity, affected_actor_type, affected_actor_subtype,
    realized_or_potential, risk_stage,
    archive_year, archive_month
"""
def where_for_scope(month: Optional[str], classification_result: str) -> Tuple[str, Tuple]:
    base_where = "classification_result = %s"
    if month is None:
        return base_where, (classification_result,)
    y, m = parse_month(month)
    where = f"{base_where} AND archive_year = %s AND archive_month = %s"
    return where, (classification_result, y, m)

def count_records(conn, source_relation: str, where_sql: str, params: Tuple) -> int:
    sql = f"SELECT COUNT(*) FROM {source_relation} WHERE {where_sql}"
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return int(cur.fetchone()[0])

def stream_records(conn, source_relation: str, where_sql: str, params: Tuple, fetch_size: int = 2000) -> Iterable[Dict]:
    sql = f"""
        SELECT {SELECT_COLUMNS}
        FROM {source_relation}
        WHERE {where_sql}
        ORDER BY news_id
    """
    name = f"cur_{int(time.time()*1000)}"
    cur = conn.cursor(name=name, cursor_factory=psycopg2.extras.RealDictCursor)
    cur.itersize = fetch_size
    cur.execute(sql, params)
    try:
        for row in cur:
            yield dict(row)
    finally:
        cur.close()

# =========================
# SQLite index: news_id -> (scope,row)
# =========================
INDEX_SCHEMA = """
CREATE TABLE IF NOT EXISTS embedding_loc (
  field TEXT NOT NULL,
  id INTEGER NOT NULL,
  scope TEXT NOT NULL,
  row_idx INTEGER NOT NULL,
  is_valid INTEGER NOT NULL,
  items_count INTEGER,
  updated_at TEXT NOT NULL,
  PRIMARY KEY(field, id)
);
CREATE INDEX IF NOT EXISTS idx_embedding_loc_id ON embedding_loc(id);
"""

UPSERT_SQL = """
INSERT INTO embedding_loc(field, id, scope, row_idx, is_valid, items_count, updated_at)
VALUES(?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(field, id) DO UPDATE SET
  scope=excluded.scope,
  row_idx=excluded.row_idx,
  is_valid=excluded.is_valid,
  items_count=excluded.items_count,
  updated_at=excluded.updated_at;
"""

def init_index_db(db_path: Path) -> sqlite3.Connection:
    ensure_dir(db_path.parent)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    for stmt in INDEX_SCHEMA.strip().split(";"):
        s = stmt.strip()
        if s:
            conn.execute(s + ";")
    conn.commit()
    return conn

class IndexWriter:
    def __init__(self, conn: sqlite3.Connection, flush_size: int = 50000):
        self.conn = conn
        self.flush_size = flush_size
        self.buf: List[Tuple] = []

    def add_batch(
        self,
        field: str,
        scope: str,
        start_row_idx: int,
        ids: List[int],
        is_valid: List[bool],
        items_count: Optional[List[int]] = None,
    ):
        ts = now_ts()
        for i, rid in enumerate(ids):
            cnt = None
            if items_count is not None:
                cnt = int(items_count[i])
            self.buf.append((
                field,
                int(rid),
                scope,
                int(start_row_idx + i),
                1 if is_valid[i] else 0,
                cnt,
                ts
            ))
        if len(self.buf) >= self.flush_size:
            self.flush()

    def flush(self):
        if not self.buf:
            return
        cur = self.conn.cursor()
        cur.executemany(UPSERT_SQL, self.buf)
        self.conn.commit()
        self.buf.clear()

def reindex_from_files(conn: sqlite3.Connection, field: str, scope: str, ids_path: Path, mask_path: Path,
                      items_count_path: Optional[Path] = None, chunk: int = 200000):
    ids = np.load(ids_path).astype(np.int64, copy=False)
    mask = np.load(mask_path).astype(bool, copy=False)
    counts = None
    if items_count_path is not None and items_count_path.exists():
        counts = np.load(items_count_path).astype(np.int32, copy=False)

    writer = IndexWriter(conn, flush_size=50000)
    n = ids.shape[0]
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        ids_list = ids[s:e].tolist()
        mask_list = mask[s:e].tolist()
        cnt_list = counts[s:e].tolist() if counts is not None else None
        writer.add_batch(field, scope, s, ids_list, mask_list, cnt_list)
        writer.flush()
    writer.flush()

# =========================
# FAISS utilities
# =========================
def build_faiss_base_index(dim: int, index_type: str, metric: str,
                          hnsw_m: int = 32, ivf_nlist: int = 4096) -> faiss.Index:
    if metric == "ip":
        faiss_metric = faiss.METRIC_INNER_PRODUCT
    elif metric == "l2":
        faiss_metric = faiss.METRIC_L2
    else:
        raise ValueError("metric must be 'ip' or 'l2'")

    index_type = index_type.lower()
    if index_type == "flat":
        return faiss.IndexFlatIP(dim) if faiss_metric == faiss.METRIC_INNER_PRODUCT else faiss.IndexFlatL2(dim)

    if index_type == "hnsw":
        base = faiss.IndexHNSWFlat(dim, hnsw_m, faiss_metric)
        base.hnsw.efConstruction = 200
        base.hnsw.efSearch = 64
        return base

    if index_type == "ivf_flat":
        quantizer = faiss.IndexFlatIP(dim) if faiss_metric == faiss.METRIC_INNER_PRODUCT else faiss.IndexFlatL2(dim)
        base = faiss.IndexIVFFlat(quantizer, dim, ivf_nlist, faiss_metric)
        return base

    raise ValueError(f"Unknown index_type: {index_type}")

def wrap_idmap(index: faiss.Index) -> faiss.Index:
    if isinstance(index, faiss.IndexIDMap) or isinstance(index, faiss.IndexIDMap2):
        return index
    return faiss.IndexIDMap2(index)

def save_faiss(index: faiss.Index, index_path: Path):
    # 确保目标目录存在
    index_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = index_path.with_suffix(".index.tmp")
    faiss.write_index(index, str(tmp))
    os.replace(tmp, index_path)

def load_global_faiss(index_path: Path) -> Optional[faiss.Index]:
    if not index_path.exists():
        return None
    return faiss.read_index(str(index_path))

def load_ids(ids_path: Path) -> np.ndarray:
    if not ids_path.exists():
        return np.zeros((0,), dtype=np.int64)
    return np.load(ids_path).astype(np.int64, copy=False)

def save_ids(ids: np.ndarray, ids_path: Path):
    # 确保目标目录存在
    ids_path.parent.mkdir(parents=True, exist_ok=True)
    # 修复：确保tmp文件名以.npy结尾，避免np.save自动追加.npz
    tmp = ids_path.with_suffix(".tmp.npy")
    np.save(tmp, ids.astype(np.int64, copy=False))
    os.replace(tmp, ids_path)

def add_to_faiss_incremental(index: faiss.Index, vecs: np.ndarray, ids: np.ndarray, batch_add: int = 50000):
    n = vecs.shape[0]
    for i in range(0, n, batch_add):
        j = min(i + batch_add, n)
        index.add_with_ids(vecs[i:j], ids[i:j])

def maybe_train_ivf(index: faiss.Index, train_vecs: np.ndarray):
    base = faiss.downcast_index(index.index) if isinstance(index, faiss.IndexIDMap2) else faiss.downcast_index(index)
    if isinstance(base, faiss.IndexIVF) and (not base.is_trained):
        base.train(train_vecs)

def load_vectors_for_scope(emb_path: Path, ids_path: Path, mask_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    ids = np.load(ids_path).astype(np.int64, copy=False)
    mask = np.load(mask_path).astype(bool, copy=False)

    vecs_all = np.load(emb_path, mmap_mode="r")
    n = min(vecs_all.shape[0], ids.shape[0], mask.shape[0])

    ids = ids[:n]
    mask = mask[:n]
    vecs_all = vecs_all[:n]

    valid_idx = np.where(mask)[0]
    if valid_idx.size == 0:
        return np.zeros((0, vecs_all.shape[1]), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    vecs = np.asarray(vecs_all[valid_idx], dtype=np.float32)
    vids = ids[valid_idx]
    return vecs, vids

def update_global_faiss(
    mode: str,
    scope: str,
    emb_path: Path,
    ids_path: Path,
    mask_path: Path,
    faiss_dir: Path,
    index_type: str,
    metric: str,
    threads: int,
    rebuild: bool,
    make_month_shard_index: bool,
    hnsw_m: int,
    ivf_nlist: int,
):
    ensure_dir(faiss_dir)
    ensure_dir(faiss_dir / "shards")
    faiss.omp_set_num_threads(threads)

    global_index_path = faiss_dir / f"{mode}.index"
    global_ids_path = faiss_dir / f"{mode}_ids.npy"
    global_meta_path = faiss_dir / f"{mode}_meta.json"
    shard_index_path = faiss_dir / "shards" / f"{mode}_{scope}.index"

    vecs, vids = load_vectors_for_scope(emb_path, ids_path, mask_path)
    if vecs.shape[0] == 0:
        print(f"⚠️ scope={scope} mode={mode}: 没有有效向量，跳过 FAISS")
        return
    dim = vecs.shape[1]

    if make_month_shard_index:
        base = build_faiss_base_index(dim, index_type, metric, hnsw_m=hnsw_m, ivf_nlist=ivf_nlist)
        shard = wrap_idmap(base)
        if index_type.lower() == "ivf_flat":
            train_vecs = vecs
            if train_vecs.shape[0] > 200000:
                sel = np.random.choice(train_vecs.shape[0], size=200000, replace=False)
                train_vecs = train_vecs[sel]
            maybe_train_ivf(shard, train_vecs)
        add_to_faiss_incremental(shard, vecs, vids)
        save_faiss(shard, shard_index_path)

    global_ids = load_ids(global_ids_path)

    if rebuild or (not global_index_path.exists()):
        base = build_faiss_base_index(dim, index_type, metric, hnsw_m=hnsw_m, ivf_nlist=ivf_nlist)
        g = wrap_idmap(base)
        if index_type.lower() == "ivf_flat":
            train_vecs = vecs
            if train_vecs.shape[0] > 200000:
                sel = np.random.choice(train_vecs.shape[0], size=200000, replace=False)
                train_vecs = train_vecs[sel]
            maybe_train_ivf(g, train_vecs)
        add_to_faiss_incremental(g, vecs, vids)
        save_faiss(g, global_index_path)
        save_ids(vids.copy(), global_ids_path)
        meta = {
            "mode": mode,
            "index_type": index_type,
            "metric": metric,
            "dim": dim,
            "built_at": now_ts(),
            "note": "rebuild" if rebuild else "create",
            "count": int(vids.shape[0]),
        }
        with open(global_meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print(f"✅ 全局 FAISS {mode} 已{'重建' if rebuild else '创建'}：count={vids.shape[0]}")
        return

    g = load_global_faiss(global_index_path)
    if g is None:
        raise RuntimeError(f"Failed to load global FAISS index: {global_index_path}")

    existing = set(global_ids.tolist()) if global_ids.size > 0 else set()
    keep_mask = np.array([i not in existing for i in vids.tolist()], dtype=bool)
    if keep_mask.sum() == 0:
        print(f"⏭️ 全局 FAISS {mode} 无需更新（scope={scope} 所有 id 已存在）")
        return

    new_ids = vids[keep_mask]
    new_vecs = vecs[keep_mask]

    if index_type.lower() == "ivf_flat":
        base = faiss.downcast_index(g.index) if isinstance(g, faiss.IndexIDMap2) else faiss.downcast_index(g)
        if isinstance(base, faiss.IndexIVF) and (not base.is_trained):
            train_vecs = new_vecs
            if train_vecs.shape[0] > 200000:
                sel = np.random.choice(train_vecs.shape[0], size=200000, replace=False)
                train_vecs = train_vecs[sel]
            maybe_train_ivf(g, train_vecs)

    add_to_faiss_incremental(g, new_vecs, new_ids)
    global_ids = np.concatenate([global_ids, new_ids], axis=0) if global_ids.size > 0 else new_ids.copy()

    save_faiss(g, global_index_path)
    save_ids(global_ids, global_ids_path)

    meta = {
        "mode": mode,
        "index_type": index_type,
        "metric": metric,
        "dim": dim,
        "updated_at": now_ts(),
        "note": f"incremental from scope={scope}",
        "added": int(new_ids.shape[0]),
        "count": int(global_ids.shape[0]),
    }
    if global_meta_path.exists():
        try:
            with open(global_meta_path, "r", encoding="utf-8") as f:
                old = json.load(f)
            old.update(meta)
            meta = old
        except Exception:
            pass
    with open(global_meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"✅ 全局 FAISS {mode} 增量更新：added={new_ids.shape[0]}, total={global_ids.shape[0]}")

# =========================
# Embedding I/O specs
# =========================
@dataclass
class ChannelFiles2D:
    emb: Path
    ids: Path
    mask: Path

@dataclass
class ChannelFiles3D:
    emb: Path
    ids: Path
    mask: Path
    items_count: Path

def open_memmap_2d(path: Path, shape: Tuple[int, int]) -> np.memmap:
    return np.lib.format.open_memmap(str(path), mode="w+", dtype=np.float32, shape=shape)

def open_memmap_3d(path: Path, shape: Tuple[int, int, int]) -> np.memmap:
    return np.lib.format.open_memmap(str(path), mode="w+", dtype=np.float32, shape=shape)

# =========================
# Scope processing
# =========================
def process_scope(
    conn,
    model: SentenceTransformer,
    source_relation: str,
    scope: str,
    month: Optional[str],
    classification_result: str,
    artifacts_root: Path,
    batch_size: int,
    max_length: int,
    use_fp16: bool,
    skip_existing: bool,
    embed_elements: bool,
    max_items: int,
    build_faiss: bool,
    rebuild_faiss: bool,
    make_month_shard_index: bool,
    faiss_index_type: str,
    faiss_metric: str,
    faiss_threads: int,
    hnsw_m: int,
    ivf_nlist: int,
    index_conn: sqlite3.Connection,
    index_flush_size: int,
    fetch_size: int,
):
    where_sql, params = where_for_scope(month, classification_result)
    total = count_records(conn, source_relation, where_sql, params)
    if total <= 0:
        print(f"⏭️ scope={scope}: 没有记录，跳过")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.max_seq_length = max_length
    if use_fp16 and device == "cuda":
        model.half()

    dim = model.get_sentence_embedding_dimension()

    # paths
    emb_root = artifacts_root / "embeddings"
    faiss_root = artifacts_root / "faiss"

    # decide full vs shard dirs
    subdir = "full" if month is None else "shards"

    # --- channels to build ---
    # base modes (always)
    base_channels_2d = ["text", "event"]

    # elements channels
    elem_channels_2d = ELEMENT_STRING_FIELDS if embed_elements else []
    elem_channels_3d = ELEMENT_LIST_FIELDS if embed_elements else []

    # Prepare files map
    files_2d: Dict[str, ChannelFiles2D] = {}
    files_3d: Dict[str, ChannelFiles3D] = {}

    # base
    for ch in base_channels_2d:
        out_dir = emb_root / ch / subdir
        ensure_dir(out_dir)
        files_2d[ch] = ChannelFiles2D(
            emb=out_dir / f"emb_{ch}_{scope}.npy",
            ids=out_dir / f"ids_{ch}_{scope}.npy",
            mask=out_dir / f"valid_mask_{ch}_{scope}.npy",
        )

    # elements 2d
    for f in elem_channels_2d:
        out_dir = emb_root / "elements" / f / subdir
        ensure_dir(out_dir)
        files_2d[f] = ChannelFiles2D(
            emb=out_dir / f"emb_{f}_{scope}.npy",
            ids=out_dir / f"ids_{f}_{scope}.npy",
            mask=out_dir / f"valid_mask_{f}_{scope}.npy",
        )

    # elements 3d
    for f in elem_channels_3d:
        out_dir = emb_root / "elements" / f / subdir
        ensure_dir(out_dir)
        files_3d[f] = ChannelFiles3D(
            emb=out_dir / f"emb_{f}_{scope}.npy",
            ids=out_dir / f"ids_{f}_{scope}.npy",
            mask=out_dir / f"valid_mask_{f}_{scope}.npy",
            items_count=out_dir / f"items_count_{f}_{scope}.npy",
        )

    # Determine which channels to compute vs skip
    compute_2d = {}
    for ch, fp in files_2d.items():
        exists = fp.emb.exists() and fp.ids.exists() and fp.mask.exists()
        compute_2d[ch] = not (skip_existing and exists)

    compute_3d = {}
    for ch, fp in files_3d.items():
        exists = fp.emb.exists() and fp.ids.exists() and fp.mask.exists() and fp.items_count.exists()
        compute_3d[ch] = not (skip_existing and exists)

    # If skipping, still make sure index exists by reindexing from files
    # 但如果某个 channel 需要计算，那就走“边算边写索引”
    need_scan_reindex_2d = [ch for ch, do in compute_2d.items() if not do]
    need_scan_reindex_3d = [ch for ch, do in compute_3d.items() if not do]

    # Open memmaps/arrays for channels that need compute
    mem_2d: Dict[str, np.memmap] = {}
    ids_2d: Dict[str, np.ndarray] = {}
    mask_2d: Dict[str, np.ndarray] = {}

    mem_3d: Dict[str, np.memmap] = {}
    ids_3d: Dict[str, np.ndarray] = {}
    mask_3d: Dict[str, np.ndarray] = {}
    cnt_3d: Dict[str, np.ndarray] = {}

    for ch, do in compute_2d.items():
        if do:
            mem_2d[ch] = open_memmap_2d(files_2d[ch].emb, (total, dim))
            ids_2d[ch] = np.zeros((total,), dtype=np.int64)
            mask_2d[ch] = np.zeros((total,), dtype=bool)

    for ch, do in compute_3d.items():
        if do:
            mem_3d[ch] = open_memmap_3d(files_3d[ch].emb, (total, max_items, dim))
            ids_3d[ch] = np.zeros((total,), dtype=np.int64)
            mask_3d[ch] = np.zeros((total,), dtype=bool)
            cnt_3d[ch] = np.zeros((total,), dtype=np.int32)

    # If nothing to compute, just reindex & (optionally) update faiss from existing files
    if (not any(compute_2d.values())) and (not any(compute_3d.values())):
        print(f"⏭️ scope={scope}: 所有 embedding 文件都已存在（skip_existing 生效），将只做索引重建/FAISS增量（可选）")

        # reindex all
        for ch in need_scan_reindex_2d:
            reindex_from_files(index_conn, ch, scope, files_2d[ch].ids, files_2d[ch].mask, None)
        for ch in need_scan_reindex_3d:
            reindex_from_files(index_conn, ch, scope, files_3d[ch].ids, files_3d[ch].mask, files_3d[ch].items_count)

        # faiss update for text/event only
        if build_faiss:
            ensure_dir(faiss_root)
            # rebuild only applies to full
            for mode in ["text", "event"]:
                fp = files_2d[mode]
                update_global_faiss(
                    mode=mode,
                    scope=scope,
                    emb_path=fp.emb,
                    ids_path=fp.ids,
                    mask_path=fp.mask,
                    faiss_dir=faiss_root,
                    index_type=faiss_index_type,
                    metric=faiss_metric,
                    threads=faiss_threads,
                    rebuild=bool(rebuild_faiss and scope == "full"),
                    make_month_shard_index=bool(make_month_shard_index and scope != "full"),
                    hnsw_m=hnsw_m,
                    ivf_nlist=ivf_nlist,
                )
        return

    # Index writer (incremental)
    writer = IndexWriter(index_conn, flush_size=index_flush_size)

    # streaming + batching
    # 获取终端宽度，确保进度条适配并在一行显示
    try:
        terminal_width = shutil.get_terminal_size().columns
        ncols = min(120, terminal_width - 5)  # 自适应宽度，留出边距
    except:
        ncols = 100  # 默认宽度

    pbar = tqdm(
        total=total,
        desc=f"Scope {scope}",
        unit="rows",
        ncols=ncols,
        bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
    )

    batch_rows: List[Dict] = []
    row_idx = 0

    def flush_batch(rows: List[Dict], start_row_idx: int):
        nonlocal writer

        bs = len(rows)
        if bs == 0:
            return

        # per-record news_id
        rids = [int(r["news_id"]) for r in rows]

        # build views once per record
        need_elem = embed_elements
        elem_views_list = []
        if need_elem:
            for r in rows:
                elem_views_list.append(build_element_views(r, max_items=max_items))
        else:
            elem_views_list = [None] * bs

        # ---------------- 2D channels ----------------
        for ch, do in compute_2d.items():
            if not do:
                continue

            # build text per record
            texts: List[str] = []
            valids: List[bool] = []

            if ch == "text":
                for r in rows:
                    t = build_text_view(r)
                    ok = bool(t and t.strip())
                    texts.append(t if ok else "")
                    valids.append(ok)
            elif ch == "event":
                for r in rows:
                    t = build_event_view(r)
                    ok = bool(t and t.strip())
                    texts.append(t if ok else "")
                    valids.append(ok)
            else:
                # element string field
                for ev in elem_views_list:
                    t = ev.get(ch, "") if ev is not None else ""
                    if not isinstance(t, str):
                        t = to_str(t)
                    ok = bool(t.strip())
                    texts.append(t if ok else "")
                    valids.append(ok)

            # write ids/mask
            ids_2d[ch][start_row_idx:start_row_idx+bs] = np.array(rids, dtype=np.int64)
            mask_2d[ch][start_row_idx:start_row_idx+bs] = np.array(valids, dtype=bool)

            # encode valid only
            vecs = np.zeros((bs, dim), dtype=np.float32)
            valid_idx = [i for i, v in enumerate(valids) if v]
            if valid_idx:
                valid_texts = [texts[i] for i in valid_idx]
                enc = model.encode(
                    valid_texts,
                    batch_size=min(batch_size, len(valid_texts)),
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                ).astype(np.float32, copy=False)
                for k, i in enumerate(valid_idx):
                    vecs[i] = enc[k]

            mem_2d[ch][start_row_idx:start_row_idx+bs] = vecs

            # write index mapping
            writer.add_batch(ch, scope, start_row_idx, rids, valids, None)

        # ---------------- 3D channels ----------------
        for ch, do in compute_3d.items():
            if not do:
                continue

            items_per_row: List[List[str]] = []
            counts: List[int] = []
            valids: List[bool] = []

            for ev in elem_views_list:
                items = ev.get(ch, []) if ev is not None else []
                if not isinstance(items, list):
                    items = [to_str(items)] if items else []
                items = [clip(x, 100) for x in items if str(x).strip()][:max_items]
                items_per_row.append(items)
                counts.append(len(items))
                valids.append(len(items) > 0)

            ids_3d[ch][start_row_idx:start_row_idx+bs] = np.array(rids, dtype=np.int64)
            mask_3d[ch][start_row_idx:start_row_idx+bs] = np.array(valids, dtype=bool)
            cnt_3d[ch][start_row_idx:start_row_idx+bs] = np.array(counts, dtype=np.int32)

            # flatten items for encode
            flat_texts: List[str] = []
            pos: List[Tuple[int, int]] = []  # (i, j)
            for i, items in enumerate(items_per_row):
                for j, s in enumerate(items):
                    flat_texts.append(s)
                    pos.append((i, j))

            block = np.zeros((bs, max_items, dim), dtype=np.float32)
            if flat_texts:
                enc = model.encode(
                    flat_texts,
                    batch_size=min(batch_size, len(flat_texts)),
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                ).astype(np.float32, copy=False)
                for k, (i, j) in enumerate(pos):
                    block[i, j, :] = enc[k]

            mem_3d[ch][start_row_idx:start_row_idx+bs] = block

            # write index mapping (include items_count)
            writer.add_batch(ch, scope, start_row_idx, rids, valids, counts)

    for r in stream_records(conn, source_relation, where_sql, params, fetch_size=fetch_size):
        batch_rows.append(r)
        if len(batch_rows) >= batch_size:
            flush_batch(batch_rows, row_idx)
            row_idx += len(batch_rows)
            pbar.update(len(batch_rows))
            batch_rows = []

            # periodic flush memmap + sqlite
            if row_idx % 100000 == 0:
                for ch in mem_2d:
                    mem_2d[ch].flush()
                for ch in mem_3d:
                    mem_3d[ch].flush()
                writer.flush()

    # remaining
    if batch_rows:
        flush_batch(batch_rows, row_idx)
        row_idx += len(batch_rows)
        pbar.update(len(batch_rows))
        batch_rows = []

    pbar.close()
    writer.flush()

    # Save ids/masks/counts for computed channels
    for ch, do in compute_2d.items():
        if not do:
            continue
        save_npy_atomic(files_2d[ch].ids, ids_2d[ch])
        save_npy_atomic(files_2d[ch].mask, mask_2d[ch].astype(bool))
        mem_2d[ch].flush()

    for ch, do in compute_3d.items():
        if not do:
            continue
        save_npy_atomic(files_3d[ch].ids, ids_3d[ch])
        save_npy_atomic(files_3d[ch].mask, mask_3d[ch].astype(bool))
        save_npy_atomic(files_3d[ch].items_count, cnt_3d[ch].astype(np.int32))
        mem_3d[ch].flush()

    # For skipped channels, ensure index exists (reindex from files)
    for ch in need_scan_reindex_2d:
        reindex_from_files(index_conn, ch, scope, files_2d[ch].ids, files_2d[ch].mask, None)
    for ch in need_scan_reindex_3d:
        reindex_from_files(index_conn, ch, scope, files_3d[ch].ids, files_3d[ch].mask, files_3d[ch].items_count)

    # FAISS (only text/event, 2D)
    if build_faiss:
        ensure_dir(faiss_root)
        for mode in ["text", "event"]:
            fp = files_2d[mode]
            update_global_faiss(
                mode=mode,
                scope=scope,
                emb_path=fp.emb,
                ids_path=fp.ids,
                mask_path=fp.mask,
                faiss_dir=faiss_root,
                index_type=faiss_index_type,
                metric=faiss_metric,
                threads=faiss_threads,
                rebuild=bool(rebuild_faiss and scope == "full"),
                make_month_shard_index=bool(make_month_shard_index and scope != "full"),
                hnsw_m=hnsw_m,
                ivf_nlist=ivf_nlist,
            )

    print(f"✅ scope={scope} 完成：rows={total}, dim={dim}, elements={embed_elements}")

# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser(
        description="从 PostgreSQL 构建全量 embeddings、FAISS 索引与 SQLite news_id 定位库"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="INI 配置文件路径",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    section = require_section(cfg, "BuildDbEmbeddings")
    db_config = load_database_config(cfg)

    artifacts_root = Path(resolve_path(section.get("artifacts_root", "")))
    model_path = (section.get("model_path", "") or "").strip()
    if not model_path:
        raise ValueError("配置项缺失: [BuildDbEmbeddings] model_path")
    batch_size = get_int(section, "batch_size", 32)
    max_length = get_int(section, "max_length", 2048)
    use_fp16 = get_bool(section, "use_fp16", False)
    skip_existing = get_bool(section, "skip_existing", False)
    embed_elements = get_bool(section, "embed_elements", True)
    max_items = get_int(section, "max_items", 10)
    fetch_size = get_int(section, "fetch_size", 2000)
    index_flush_size = get_int(section, "index_flush_size", 50000)
    build_faiss = get_bool(section, "build_faiss", True)
    rebuild_faiss = get_bool(section, "rebuild_faiss", False)
    make_month_shard_index = get_bool(section, "make_month_shard_index", False)
    faiss_index_type = section.get("faiss_index_type", "hnsw").strip() or "hnsw"
    faiss_metric = section.get("faiss_metric", "ip").strip() or "ip"
    faiss_threads = get_int(section, "faiss_threads", 8)
    hnsw_m = get_int(section, "hnsw_m", 32)
    ivf_nlist = get_int(section, "ivf_nlist", 4096)
    scope_mode = (section.get("scope_mode", "full") or "full").strip().lower()
    month = (section.get("month", "") or "").strip()
    months_raw = (section.get("months", "") or "").strip()
    month_from = (section.get("month_from", "") or "").strip()
    month_to = (section.get("month_to", "") or "").strip()

    ensure_dir(artifacts_root / "embeddings")
    ensure_dir(artifacts_root / "faiss")
    ensure_dir(artifacts_root / "runs")
    ensure_dir(artifacts_root / "embedding_index")

    scopes: List[Tuple[str, Optional[str]]] = []
    if scope_mode == "month":
        parse_month(month)
        scopes.append((month, month))
    elif scope_mode == "months":
        ms = list(split_csv(months_raw))
        for mm in ms:
            parse_month(mm)
            scopes.append((mm, mm))
    elif scope_mode == "range":
        if not month_from or not month_to:
            raise ValueError("scope_mode=range 时必须设置 month_from 和 month_to")
        for mm in iter_months_inclusive(month_from, month_to):
            scopes.append((mm, mm))
    elif scope_mode == "full":
        scopes.append(("full", None))
    else:
        raise ValueError(f"未知 scope_mode: {scope_mode}")

    resolved = resolve_model_dir(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"加载模型: {resolved}  device={device}")
    model = SentenceTransformer(resolved, device=device, local_files_only=True)
    model.max_seq_length = max_length
    if use_fp16 and device == "cuda":
        model.half()

    conn = psycopg2.connect(
        host=db_config["host"],
        port=db_config["port"],
        database=db_config["database"],
        user=db_config["user"],
        password=db_config["password"],
    )

    index_db_path = artifacts_root / "embedding_index" / "embeddings.sqlite"
    index_conn = init_index_db(index_db_path)

    run_info = {
        "run_time": now_ts(),
        "config_path": str(Path(args.config).resolve()),
        "artifacts_root": str(artifacts_root),
        "model_path": model_path,
        "resolved_model_path": resolved,
        "device": device,
        "batch_size": batch_size,
        "max_length": max_length,
        "use_fp16": bool(use_fp16),
        "skip_existing": bool(skip_existing),
        "embed_elements": bool(embed_elements),
        "max_items": max_items,
        "scopes": [s[0] for s in scopes],
        "database": {
            "host": db_config["host"],
            "port": db_config["port"],
            "database": db_config["database"],
            "user": db_config["user"],
            "password_env": db_config["password_env"],
            "classification_result": db_config["classification_result"],
            "source_relation": db_config["source_relation"],
        },
        "faiss": {
            "enabled": bool(build_faiss),
            "rebuild": bool(rebuild_faiss),
            "index_type": faiss_index_type,
            "metric": faiss_metric,
            "threads": faiss_threads,
            "hnsw_m": hnsw_m,
            "ivf_nlist": ivf_nlist,
            "make_month_shard_index": bool(make_month_shard_index),
        },
        "index_db": str(index_db_path),
        "fetch_size": fetch_size,
        "index_flush_size": index_flush_size,
    }

    try:
        for scope, month in scopes:
            process_scope(
                conn=conn,
                model=model,
                source_relation=db_config["source_relation"],
                scope=scope,
                month=month,
                classification_result=db_config["classification_result"],
                artifacts_root=artifacts_root,
                batch_size=batch_size,
                max_length=max_length,
                use_fp16=use_fp16,
                skip_existing=skip_existing,
                embed_elements=embed_elements,
                max_items=max_items,
                build_faiss=build_faiss,
                rebuild_faiss=rebuild_faiss,
                make_month_shard_index=make_month_shard_index,
                faiss_index_type=faiss_index_type,
                faiss_metric=faiss_metric,
                faiss_threads=faiss_threads,
                hnsw_m=hnsw_m,
                ivf_nlist=ivf_nlist,
                index_conn=index_conn,
                index_flush_size=index_flush_size,
                fetch_size=fetch_size,
            )

        run_path = artifacts_root / "runs" / f"run_{run_info['run_time']}_config.json"
        with open(run_path, "w", encoding="utf-8") as f:
            json.dump(run_info, f, indent=2, ensure_ascii=False)

        print(f"✅ 全部完成。run config: {run_path}")
        print(f"✅ SQLite 索引库: {index_db_path}")

    finally:
        try:
            conn.close()
        except Exception:
            pass
        try:
            index_conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
