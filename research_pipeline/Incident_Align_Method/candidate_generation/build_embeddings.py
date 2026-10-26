#!/usr/bin/env python3
"""
BGE-M3 Embedding生成脚本 - 全程流式版 (支持百万级数据)
生成E_text、E_event和要素级向量表示

核心特性：
- 完全流式处理，不占用大量内存（JSONL推荐）
- 支持百万级数据量（不把case_ids/bad_cases放入内存）
- 自动失败样本标记和过滤（valid_mask）
- 实时进度显示与吞吐量统计
- bad_cases.log 全程流式写入（不积压内存）

使用方法：
# 使用固定配置文件
python build_embeddings.py

# 配置文件位置
config/Incident_Align_Method-build_embeddings-config.ini

输出：
- emb_text.npy, emb_event.npy         (memmap写入的标准.npy)
- valid_mask_text.npy, valid_mask_event.npy
- case_ids.txt                        (每行一个case_id)
- text_view.jsonl, event_view.jsonl   (为reranker准备的文本视图)
- embedding_config.json
- bad_cases_text.log, bad_cases_event.log  (分view记录无效/编码失败样本)

要素embeddings (当--embed_elements时)：
- emb_actor_list.npy, emb_ai_system_list.npy, emb_ai_system_type_list.npy, emb_ai_system_domain_list.npy: 3D数组 [case_count, max_items, embedding_dim]，列表字段的embeddings
- emb_actor_main.npy, emb_actor_main_type.npy, emb_ai_system.npy, emb_domain.npy, emb_event_type.npy, emb_event_cause.npy, emb_event_process.npy, emb_event_result.npy, emb_ai_risk_description.npy, emb_ai_risk_type.npy, emb_ai_risk_subtype.npy, emb_harm_type.npy, emb_harm_severity.npy, emb_affected_actor_type.npy, emb_affected_actor_subtype.npy, emb_realized_or_potential.npy, emb_risk_stage.npy: 2D数组 [case_count, embedding_dim]，字符串字段的embeddings
- items_count_*.npy: 每个列表字段的items数量 [case_count]
- valid_mask_*.npy: 有效性mask
- bad_cases_*.log: 失败样本日志
"""

import configparser
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set, Any

import numpy as np
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parents[3]

CONFIG_SECTION = "BuildEmbeddings"
CONFIG_PATH = os.path.join(BASE_DIR, "config", "Incident_Align_Method-build_embeddings-config.ini")


# =========================
# Reproducibility
# =========================
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


# =========================
# Utility functions
# =========================
def to_str(x) -> str:
    """安全转换为字符串"""
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    try:
        return json.dumps(x, ensure_ascii=False)
    except Exception:
        return str(x)

def clip(x, n: int) -> str:
    """安全截断字符串"""
    s = to_str(x).strip()
    return s[:n]


def get_case_id(rec, line_idx: int) -> str:
    """统一生成case_id的函数（确保三遍扫描一致性）"""
    cid = rec.get("id") or rec.get("original_id") or str(line_idx)
    return str(cid).strip()

# =========================
# View builders
# =========================
def build_text_view(rec):
    """E_text：主题/语义视角 (title + summary/body)，优先摘要"""
    title = clip(rec.get("title"), 200)
    body = clip(rec.get("text"), 6000)  # 正文也限制在合理长度

    return f"Title: {title}\nBody: {body}".strip()


def build_event_view(rec):
    """E_event：事件动作链/要素视角，兼容_filter_metadata与event_annotation/ai_risk/ai_tech"""
    title = clip(rec.get("title"), 200)

    event_time = ""
    event_subject = ""
    event_type = ""
    actor_list = []
    ai_tech_str = ""
    ai_risk_str = ""
    event_cause = ""
    event_process = ""
    event_result = ""

    filter_meta = rec.get("_filter_metadata")
    if isinstance(filter_meta, dict):
        # 处理_filter_metadata路径
        ai_tech_elements = filter_meta.get("ai_tech_elements") or []
        if not isinstance(ai_tech_elements, list):
            ai_tech_elements = [ai_tech_elements]
        ai_tech_str = ", ".join(clip(x, 200) for x in ai_tech_elements[:10])

        ai_risk_elements = filter_meta.get("ai_risk_elements") or []
        if not isinstance(ai_risk_elements, list):
            ai_risk_elements = [ai_risk_elements]
        ai_risk_str = ", ".join(clip(x, 200) for x in ai_risk_elements[:10])

        incident_elements = filter_meta.get("incident_elements") or {}
        if not isinstance(incident_elements, dict):
            incident_elements = {}

        event_time = clip(incident_elements.get("Time"), 50)
        event_subject = clip(incident_elements.get("Subject"), 200)
        event_cause = clip(incident_elements.get("Cause"), 500)
        event_process = clip(incident_elements.get("Process"), 500)
        event_result = clip(incident_elements.get("Result"), 500)

        if not event_type and event_subject:
            if ai_risk_elements:
                first_risk = ai_risk_elements[0]
                event_type = (
                    first_risk.split("（")[0] if "（" in first_risk else first_risk[:20]
                )

        if not actor_list and event_subject:
            actor_list = [event_subject]

        if not ai_tech_str:
            ai_tech = rec.get("ai_tech") or {}
            if isinstance(ai_tech, dict):
                systems = ai_tech.get("ai_system_list") or []
                if not isinstance(systems, list):
                    systems = [systems]
                ai_tech_str = ", ".join(clip(s, 100) for s in systems[:10])

    else:
        # 处理标准标注路径
        ea = rec.get("event_annotation") or {}
        ar = rec.get("ai_risk") or {}
        at = rec.get("ai_tech") or {}

        if not isinstance(ea, dict): ea = {}
        if not isinstance(ar, dict): ar = {}
        if not isinstance(at, dict): at = {}

        event_time = clip(ea.get("event_time_start"), 50)
        event_subject = clip(ea.get("actor_main"), 200)
        event_cause = clip(ea.get("event_cause"), 400)
        event_process = clip(ea.get("event_process"), 400)
        event_result = clip(ea.get("event_result"), 200)

        event_type = clip(ea.get("event_type"), 100)

        actor_list = ea.get("actor_list") or []
        if not isinstance(actor_list, list):
            actor_list = [actor_list]

        ai_system = clip(ea.get("ai_system"), 100)

        systems = at.get("ai_system_list") or []
        if not isinstance(systems, list):
            systems = [systems]
        if ai_system and ai_system not in systems:
            systems.insert(0, ai_system)
        ai_tech_str = ", ".join(clip(s, 100) for s in systems[:10])

        risk_type = clip(ar.get("ai_risk_type"), 100)
        risk_sub = clip(ar.get("ai_risk_subtype"), 100)
        risk_desc = clip(ar.get("ai_risk_description"), 200)
        risk_parts = [p for p in (risk_type, risk_sub, risk_desc) if p]
        ai_risk_str = " | ".join(risk_parts)

    if not event_time:
        event_time = clip(rec.get("release_date"), 50)
    if event_time and len(str(event_time)) > 10:
        event_time = str(event_time)[:10]

    actors_str = ", ".join(clip(actor, 100) for actor in actor_list[:10]) if actor_list else ""

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

    # 最后的总长度保护
    return result[:6000]


def build_element_views(rec):
    """为聚类要素构建各种文本表示"""
    views = {}

    # 从各个部分提取要素

    # 1. event_annotation部分
    ea = rec.get("event_annotation", {})
    if isinstance(ea, dict):
        # actor_main - 字符串
        views["actor_main"] = clip(ea.get("actor_main", ""), 100)

        # actor_main_type - 字符串
        views["actor_main_type"] = clip(ea.get("actor_main_type", ""), 50)

        # actor_list - 字符串列表
        actors = ea.get("actor_list", [])
        if not isinstance(actors, list):
            actors = [actors] if actors else []
        views["actor_list"] = [clip(actor, 100) for actor in actors[:10] if actor and str(actor).strip()]

        # ai_system - 字符串
        views["ai_system"] = clip(ea.get("ai_system", ""), 100)

        # domain - 字符串
        views["domain"] = clip(ea.get("domain", ""), 50)

        # event_type - 字符串
        views["event_type"] = clip(ea.get("event_type", ""), 100)

        # event_cause - 字符串
        views["event_cause"] = clip(ea.get("event_cause", ""), 500)

        # event_process - 字符串
        views["event_process"] = clip(ea.get("event_process", ""), 500)

        # event_result - 字符串
        views["event_result"] = clip(ea.get("event_result", ""), 500)

    # 2. ai_tech部分
    at = rec.get("ai_tech", {})
    if isinstance(at, dict):
        # ai_system_list - 字符串列表
        systems = at.get("ai_system_list", [])
        if not isinstance(systems, list):
            systems = [systems] if systems else []
        views["ai_system_list"] = [clip(sys, 100) for sys in systems[:10] if sys and str(sys).strip()]

        # ai_system_type_list - 字符串列表
        system_types = at.get("ai_system_type_list", [])
        if not isinstance(system_types, list):
            system_types = [system_types] if system_types else []
        views["ai_system_type_list"] = [clip(st, 100) for st in system_types[:10] if st and str(st).strip()]

        # ai_system_domain_list - 字符串列表
        domains = at.get("ai_system_domain_list", [])
        if not isinstance(domains, list):
            domains = [domains] if domains else []
        views["ai_system_domain_list"] = [clip(d, 100) for d in domains[:10] if d and str(d).strip()]

    # 3. ai_risk部分
    ar = rec.get("ai_risk", {})
    if isinstance(ar, dict):
        # ai_risk_description - 字符串
        views["ai_risk_description"] = clip(ar.get("ai_risk_description", ""), 500)

        # ai_risk_type - 字符串
        views["ai_risk_type"] = clip(ar.get("ai_risk_type", ""), 100)

        # ai_risk_subtype - 字符串
        views["ai_risk_subtype"] = clip(ar.get("ai_risk_subtype", ""), 100)

        # harm_type - 字符串
        views["harm_type"] = clip(ar.get("harm_type", ""), 100)

        # harm_severity - 字符串
        views["harm_severity"] = clip(ar.get("harm_severity", ""), 50)

        # affected_actor_type - 字符串
        views["affected_actor_type"] = clip(ar.get("affected_actor_type", ""), 100)

        # affected_actor_subtype - 字符串
        views["affected_actor_subtype"] = clip(ar.get("affected_actor_subtype", ""), 100)

        # realized_or_potential - 字符串
        views["realized_or_potential"] = clip(ar.get("realized_or_potential", ""), 50)

        # risk_stage - 字符串
        views["risk_stage"] = clip(ar.get("risk_stage", ""), 100)

    # 4. 兼容性处理：确保所有字段都有默认值
    field_defaults = {
        "actor_main": "",
        "actor_main_type": "",
        "actor_list": [],
        "ai_system": "",
        "domain": "",
        "event_type": "",
        "event_cause": "",
        "event_process": "",
        "event_result": "",
        "ai_system_list": [],
        "ai_system_type_list": [],
        "ai_system_domain_list": [],
        "ai_risk_description": "",
        "ai_risk_type": "",
        "ai_risk_subtype": "",
        "harm_type": "",
        "harm_severity": "",
        "affected_actor_type": "",
        "affected_actor_subtype": "",
        "realized_or_potential": "",
        "risk_stage": "",
    }

    for field, default in field_defaults.items():
        if field not in views:
            views[field] = default

    return views


def embed_list_field_streaming(
    model,
    cases_file: str,
    out_dir: str,
    field: str,
    total_cases: int,
    max_items: int = 10,
    batch_size_items: int = 16,
):
    """
    为列表字段生成embeddings，按case为单位写入3D memmap
    返回: (elapsed_time, throughput, valid_count)
    """
    dim = model.get_sentence_embedding_dimension()

    out_emb = os.path.join(out_dir, f"emb_{field}.npy")
    out_cnt = os.path.join(out_dir, f"items_count_{field}.npy")
    out_msk = os.path.join(out_dir, f"valid_mask_{field}.npy")
    out_bad = os.path.join(out_dir, f"bad_cases_{field}.log")

    emb = np.lib.format.open_memmap(
        out_emb, mode="w+", dtype=np.float32, shape=(total_cases, max_items, dim)
    )
    cnt = np.zeros(total_cases, dtype=np.int32)
    msk = np.zeros(total_cases, dtype=bool)

    case_pos = 0  # ✅ 用有效case序号作为写入行号
    valid_count = 0
    t0 = time.time()

    with open(out_bad, "w", encoding="utf-8") as bad_f, open(cases_file, "r", encoding="utf-8") as f:
        if not cases_file.endswith(".jsonl"):
            raise ValueError("建议只支持jsonl（百万级）")

        pbar = tqdm(total=total_cases, desc=f"ListField {field}", unit="cases")

        for line_idx, line in enumerate(f):
            if case_pos >= total_cases:
                break
            if not line.strip():
                continue

            try:
                rec = json.loads(line)
            except Exception as e:
                # 无效行：不计入case_pos（因为其它embedding也不会为它写一行）
                bad_f.write(f"line_{line_idx}: json error {repr(e)}\n")
                continue

            if not isinstance(rec, dict):
                bad_f.write(f"line_{line_idx}: not dict\n")
                continue

            cid = get_case_id(rec, line_idx)
            if not cid:
                bad_f.write(f"line_{line_idx}: empty cid\n")
                continue

            # 到这里：这是一个"有效case"，必须占用一行
            try:
                views = build_element_views(rec)
                items = views.get(field, [])
                if not isinstance(items, list):
                    items = [items] if items else []

                items = [str(x).strip() for x in items if x and str(x).strip()]
                if items:
                    items = items[:max_items]
                    vecs = model.encode(
                        items,
                        batch_size=min(batch_size_items, len(items)),
                        convert_to_numpy=True,
                        normalize_embeddings=True,
                        show_progress_bar=False,
                    ).astype(np.float32)

                    k = min(len(items), max_items)
                    emb[case_pos, :k, :] = vecs[:k]
                    cnt[case_pos] = k
                    msk[case_pos] = True
                    valid_count += 1
                # else: 空列表也占位，保持cnt=0, msk=False
            except Exception as e:
                # encode失败也占位，否则会错位
                bad_f.write(f"{cid}: encode/build failed {repr(e)}\n")

            case_pos += 1
            pbar.update(1)

            # 定期flush，抗宕机
            if case_pos % 10000 == 0:
                emb.flush()
                bad_f.flush()

        pbar.close()

    emb.flush()
    np.save(out_cnt, cnt)
    np.save(out_msk, msk)

    if case_pos != total_cases:
        # 说明你统计的valid_count_parsed_dicts与这里的"有效case判定"不一致
        # 一般是你两个地方的"有效case条件"不同导致的
        print(f"[WARN] list_field={field}: produced={case_pos} != expected={total_cases}")

    elapsed = time.time() - t0
    throughput = total_cases / elapsed if elapsed > 0 else 0.0
    return elapsed, throughput, valid_count


# =========================
# Streaming IO
# =========================
def count_cases_streaming(cases_file: str) -> int:
    """第一遍：统计总行数/总样本数（jsonl为行数；json数组会load，不建议百万级）"""
    count = 0
    with open(cases_file, "r", encoding="utf-8") as f:
        if cases_file.endswith(".jsonl"):
            for line in f:
                if line.strip():
                    count += 1
        else:
            data = json.load(f)
            if isinstance(data, list):
                count = len(data)
            else:
                count = len(data.get("cases", []))
    return count


def write_case_ids_streaming(cases_file: str, out_path: str) -> int:
    """第二遍：流式写case_ids.txt，返回有效ID计数（不进内存）"""
    n = 0
    with open(out_path, "w", encoding="utf-8") as w, open(cases_file, "r", encoding="utf-8") as f:
        if cases_file.endswith(".jsonl"):
            for i, line in enumerate(f):
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(rec, dict):
                    # 可选：记录到 view_failures / bad_cases
                    continue
                cid = get_case_id(rec, i)
                if cid:
                    w.write(cid + "\n")
                    n += 1
        else:
            data = json.load(f)
            cases = data if isinstance(data, list) else data.get("cases", [])
            for i, rec in enumerate(cases):
                if not isinstance(rec, dict):
                    continue
                cid = get_case_id(rec, i)
                if cid:
                    w.write(cid + "\n")
                    n += 1
    return n


def create_text_generator(cases_file: str, view_type: str = "text", element_field: str = None, debug_limit: int = 50):
    """第三遍：真正用于embedding的流式生成器：yield (case_id, text, is_text_valid) 或 (case_id, field, text, is_text_valid)"""
    debug_cnt = 0
    with open(cases_file, "r", encoding="utf-8") as f:
        if cases_file.endswith(".jsonl"):
            for i, line in enumerate(f):
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(rec, dict):
                    # 可选：记录到 view_failures / bad_cases
                    continue

                cid = get_case_id(rec, i)
                if not cid:
                    continue

                try:
                    if element_field:  # 生成要素embeddings
                        element_views = build_element_views(rec)
                        field_value = element_views.get(element_field, "")

                        if isinstance(field_value, list) and element_field in ["actor_list", "ai_system_list", "ai_system_type_list", "ai_system_domain_list"]:
                            # ⚠️ 列表字段不要用 create_text_generator 生成 item 级样本，否则与 case_ids 行序不一致
                            # 列表字段现在统一走 embed_list_field_streaming()，按 case 为单位处理
                            for i, item_text in enumerate(field_value):
                                if item_text and item_text.strip():
                                    yield (cid, f"{element_field}_{i}", item_text, True)
                        else:
                            # 其他字段正常处理
                            text = field_value if isinstance(field_value, str) else str(field_value)
                            is_valid = True  # 允许空字符串也生成embeddings
                            yield (cid, element_field, text, is_valid)
                    else:  # 生成传统view embeddings
                        text = build_text_view(rec) if view_type == "text" else build_event_view(rec)
                        is_valid = bool(text and text.strip())
                        yield (cid, text, is_valid)
                except Exception as e:
                    if debug_cnt < debug_limit:
                        print(f"[BUILD_FAIL] view={view_type} field={element_field} case_id={cid} err={repr(e)}")
                        # 打印关键字段类型用于调试
                        if view_type == "event" or element_field:
                            fm = rec.get("_filter_metadata")
                            print(f"  _filter_metadata type: {type(fm)}")
                            if isinstance(fm, dict):
                                ie = fm.get("incident_elements")
                                print(f"  incident_elements type: {type(ie)}")
                                if isinstance(ie, dict):
                                    print(f"  incident_elements keys: {list(ie.keys())}")
                            ea = rec.get("event_annotation")
                            print(f"  event_annotation type: {type(ea)}")
                    debug_cnt += 1
                    if element_field:
                        yield (cid, element_field, "", False)
                    else:
                        yield (cid, "", False)
        else:
            data = json.load(f)
            cases = data if isinstance(data, list) else data.get("cases", [])
            for i, rec in enumerate(cases):
                if not isinstance(rec, dict):
                    continue
                cid = get_case_id(rec, i)
                if not cid:
                    continue
                try:
                    if element_field:  # 生成要素embeddings
                        element_views = build_element_views(rec)
                        field_value = element_views.get(element_field, "")

                        if isinstance(field_value, list) and element_field in ["actor_list", "ai_system_list", "ai_system_type_list", "ai_system_domain_list"]:
                            # ⚠️ 列表字段不要用 create_text_generator 生成 item 级样本，否则与 case_ids 行序不一致
                            # 列表字段现在统一走 embed_list_field_streaming()，按 case 为单位处理
                            for i, item_text in enumerate(field_value):
                                if item_text and item_text.strip():
                                    yield (cid, f"{element_field}_{i}", item_text, True)
                        else:
                            # 其他字段正常处理
                            text = field_value if isinstance(field_value, str) else str(field_value)
                            is_valid = True  # 允许空字符串也生成embeddings
                            yield (cid, element_field, text, is_valid)
                    else:  # 生成传统view embeddings
                        text = build_text_view(rec) if view_type == "text" else build_event_view(rec)
                        is_valid = bool(text and text.strip())
                        yield (cid, text, is_valid)
                except Exception as e:
                    if debug_cnt < debug_limit:
                        print(f"[BUILD_FAIL] view={view_type} field={element_field} case_id={cid} err={repr(e)}")
                    debug_cnt += 1
                    if element_field:
                        yield (cid, element_field, "", False)
                    else:
                        yield (cid, "", False)


def batched(iterable, batch_size: int):
    batch = []
    for x in iterable:
        batch.append(x)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


# =========================
# Embedding (streaming)
# =========================
def embed_texts_streaming(
    model: SentenceTransformer,
    text_generator,
    total_count: int,
    batch_size: int,
    output_file: str,
    output_dir: str,
    mask_prefix: str,
    is_element_mode: bool = False,
):
    """
    全程流式生成embedding：
    - memmap写embeddings到 output_file
    - 写 valid_mask_{view}.npy
    - bad_cases.log 边写边刷（不占内存）
    """
    dim = model.get_sentence_embedding_dimension()
    out_npy = os.path.join(output_dir, output_file)
    embeddings_memmap = np.lib.format.open_memmap(
        out_npy, mode="w+", dtype=np.float32, shape=(total_count, dim)
    )

    valid_mask = np.zeros(total_count, dtype=bool)
    valid_mask_file = os.path.join(output_dir, f"valid_mask_{mask_prefix}.npy")

    bad_cases_path = os.path.join(output_dir, f"bad_cases_{mask_prefix}.log")
    bad_f = open(bad_cases_path, "w", encoding="utf-8")

    start_idx = 0
    t0 = time.time()

    try:
        text_generator = iter(text_generator)
        pbar = tqdm(desc=f"Processing {output_file}", total=total_count, unit="samples")

        for batch in batched(text_generator, batch_size):
            if start_idx >= total_count:
                print(f"[WARN] Generator produced more samples than expected. Stopping at {start_idx}/{total_count}")
                break

            if is_element_mode:
                batch_ids = [x[0] for x in batch]
                batch_texts = [x[2] for x in batch]
                batch_valids = [x[3] for x in batch]
            else:
                batch_ids = [x[0] for x in batch]
                batch_texts = [x[1] for x in batch]
                batch_valids = [x[2] for x in batch]

            bs = len(batch)
            end_idx = start_idx + bs

            if end_idx > total_count:
                print(f"[WARN] Batch would exceed memmap bounds ({end_idx} > {total_count}). Truncating.")
                bs = total_count - start_idx
                batch_ids = batch_ids[:bs]
                batch_texts = batch_texts[:bs]
                batch_valids = batch_valids[:bs]
                end_idx = start_idx + bs

            # 写mask + 记录invalid
            for i, v in enumerate(batch_valids):
                valid_mask[start_idx + i] = bool(v)
                if not v:
                    bad_f.write(f"{batch_ids[i]}: invalid text\n")

            vecs = np.zeros((bs, dim), dtype=np.float32)
            valid_indices = [i for i, v in enumerate(batch_valids) if v]

            if valid_indices:
                try:
                    valid_texts = [batch_texts[i] for i in valid_indices]
                    enc_bs = min(batch_size, len(valid_texts))
                    valid_vecs = model.encode(
                        valid_texts,
                        batch_size=enc_bs,
                        convert_to_numpy=True,
                        normalize_embeddings=True,
                        show_progress_bar=False,
                    ).astype(np.float32)

                    for local_i, global_i in enumerate(valid_indices):
                        vecs[global_i] = valid_vecs[local_i]

                except Exception as e:
                    print(f"批次编码失败 at idx {start_idx}: {e} -> fallback to single encode")
                    for i in valid_indices:
                        try:
                            one_vec = model.encode(
                                [batch_texts[i]],
                                batch_size=1,
                                convert_to_numpy=True,
                                normalize_embeddings=True,
                                show_progress_bar=False,
                            ).astype(np.float32)[0]
                            vecs[i] = one_vec
                        except Exception as e2:
                            bad_f.write(f"{batch_ids[i]}: encode failed ({repr(e2)})\n")
                            valid_mask[start_idx + i] = False

            # ✅ 关键：每个batch都要写入并推进start_idx
            embeddings_memmap[start_idx:end_idx] = vecs
            start_idx = end_idx
            pbar.update(bs)

            if start_idx % 10000 == 0:
                embeddings_memmap.flush()
                bad_f.flush()

        pbar.close()

        if start_idx != total_count:
            msg = f"[WARN] produced={start_idx} != expected_total={total_count} for {output_file}"
            print(msg)
            bad_f.write(msg + "\n")
            valid_mask[start_idx:] = False

        np.save(valid_mask_file, valid_mask)
        embeddings_memmap.flush()

        elapsed = time.time() - t0
        speed = start_idx / elapsed if elapsed > 0 else 0.0  # 用真实产出更合理
        return embeddings_memmap, valid_mask, elapsed, speed

    finally:
        bad_f.close()
def resolve_model_dir(model_path: str) -> str:
    """
    兼容三种输入：
    1) HuggingFace repo id: "BAAI/bge-m3"
    2) 本地snapshot目录: ".../snapshots/<hash>"   -> 原样返回
    3) 本地HF缓存根目录: ".../models--BAAI--bge-m3" -> 自动挑最新snapshot
    """
    p = Path(model_path)

    # repo id（不是本地路径）: 直接返回
    if not p.exists():
        return model_path

    # 如果是 .../snapshots/<hash> 格式
    if p.parent.name == "snapshots":
        return str(p)

    # 如果传入的是snapshots目录本身：选最新
    if p.name == "snapshots":
        snaps = sorted(p.glob("*"), key=lambda x: x.stat().st_mtime, reverse=True)
        if not snaps:
            raise RuntimeError(f"No snapshots found in: {p}")
        return str(snaps[0])

    # 如果包含snapshots子目录：自动选最新snapshot
    snap_dir = p / "snapshots"
    if snap_dir.exists():
        snaps = sorted(snap_dir.glob("*"), key=lambda x: x.stat().st_mtime, reverse=True)
        if not snaps:
            raise RuntimeError(f"No snapshots found in: {snap_dir}")
        return str(snaps[0])

    # 否则当作普通本地模型目录直接返回
    resolved = str(p)

    # 返回前校验目录完整性
    rp = Path(resolved)
    if rp.exists():
        has_config = os.path.exists(os.path.join(rp, "config.json"))
        has_modules = os.path.exists(os.path.join(rp, "modules.json"))
        if not (has_config or has_modules):
            print(f"[WARN] model dir may be incomplete (no config.json/modules.json): {resolved}")

    return resolved


def resolve_project_path(path_value: str, default_value: str) -> str:
    raw_path = (path_value if path_value is not None else default_value).strip()
    if os.path.isabs(raw_path):
        return raw_path
    return os.path.join(BASE_DIR, raw_path)


def resolve_model_path(model_path_value: str) -> str:
    raw_path = (model_path_value or "models/bge-m3").strip()
    expanded_path = os.path.expanduser(raw_path)
    if os.path.isabs(expanded_path) and os.path.exists(expanded_path):
        return expanded_path

    project_model_path = os.path.join(BASE_DIR, raw_path)
    if os.path.exists(project_model_path):
        return project_model_path

    return raw_path
# =========================
# Main
# =========================
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=CONFIG_PATH,
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

    cases_file = resolve_project_path(section.get("cases_file", "data/cases.jsonl"), "data/cases.jsonl")
    output_dir = resolve_project_path(section.get("output_dir", "outputs/embeddings"), "outputs/embeddings")
    model_path = resolve_model_path(section.get("model_path", "models/bge-m3"))

    batch_size = section.getint("batch_size", fallback=16)
    max_length = section.getint("max_length", fallback=2048)
    use_fp16 = section.getboolean("use_fp16", fallback=False)
    embed_elements = section.getboolean("embed_elements", fallback=True)
    skip_existing = section.getboolean("skip_existing", fallback=False)

    print(f"加载配置文件: {config_path}")
    print(f"案例文件: {cases_file}")
    print(f"输出目录: {output_dir}")
    print(f"模型路径: {model_path}")

    start_time = time.time() 
    os.makedirs(output_dir, exist_ok=True)

    # 1) count
    print(f"第一遍：统计原始行数 from {cases_file}")
    total_cases_raw_lines = count_cases_streaming(str(cases_file))
    print(f"原始行数/样本数: {total_cases_raw_lines}")

    # 2) write case_ids.txt
    print("第二遍：写入有效case_ids到 case_ids.txt（流式）")
    case_ids_path = os.path.join(output_dir, "case_ids.txt")
    valid_count_parsed_dicts = write_case_ids_streaming(str(cases_file), case_ids_path)
    print(f"有效解析案例数: {valid_count_parsed_dicts}")

    if valid_count_parsed_dicts == 0:
        print("❌ 没有找到有效的案例，退出")
        return

    # 2.5) 方案A：保存文本视图到JSONL（为reranker准备）
    print("第二遍半：保存text_view和event_view到JSONL（为reranker准备）")
    text_view_path = os.path.join(output_dir, "text_view.jsonl")
    event_view_path = os.path.join(output_dir, "event_view.jsonl")
    view_failures_path = os.path.join(output_dir, "view_failures.log")

    success_count = 0
    failure_count = 0

    with open(text_view_path, 'w', encoding='utf-8') as f_text, \
         open(event_view_path, 'w', encoding='utf-8') as f_event, \
         open(view_failures_path, 'w', encoding='utf-8') as f_fail, \
         open(cases_file, 'r', encoding='utf-8') as f_cases:

        if str(cases_file).endswith('.jsonl'):
            for i, line in enumerate(f_cases):  # i 从0开始，和其它遍一致
                line_num = i + 1
                if not line.strip():
                    continue

                cid = ""  # 先占位，避免 except 里引用未定义
                try:
                    rec = json.loads(line)
                    if not isinstance(rec, dict):
                        f_fail.write(f"NON_DICT_ERROR line_{line_num}: {line.strip()[:100]}...\n")
                        failure_count += 1
                        continue

                    cid = get_case_id(rec, i)
                    if not cid:
                        f_fail.write(f"EMPTY_CID line_{line_num}\n")
                        failure_count += 1
                        continue

                    # 构建text_view和event_view
                    text_view = build_text_view(rec)
                    event_view = build_event_view(rec)

                    # 保存到JSONL
                    text_record = {"case_id": cid, "text_view": text_view}
                    event_record = {"case_id": cid, "event_view": event_view}

                    f_text.write(json.dumps(text_record, ensure_ascii=False) + '\n')
                    f_event.write(json.dumps(event_record, ensure_ascii=False) + '\n')

                    success_count += 1

                except json.JSONDecodeError:
                    f_fail.write(f"JSON_ERROR line_{line_num}: {line.strip()[:100]}...\n")
                    failure_count += 1
                    continue
                except Exception as e:
                    f_fail.write(f"BUILD_ERROR line_{line_num} cid_{cid}: {repr(e)}\n")
                    failure_count += 1
                    continue

    print(f"✅ 文本视图已保存: {text_view_path}, {event_view_path}")
    print(f"📊 视图生成统计: 成功={success_count}, 失败={failure_count}")

    if failure_count > 0:
        print(f"⚠️  失败详情见: {view_failures_path}")

    # 检查覆盖率是否接近100%
    expected_count = valid_count_parsed_dicts
    if success_count < expected_count * 0.95:
        print(f"⚠️  警告: 视图覆盖率只有 {success_count}/{expected_count} = {success_count/expected_count:.1%}")
        print("这会影响Step2 reranker的效果!")
    else:
        print(f"✅ 视图覆盖率: {success_count}/{expected_count} = {success_count/expected_count:.1%}")

    # load model
    print(f"加载BGE模型: {model_path}")
    try:
        resolved_path = resolve_model_dir(model_path)
        print(f"📌 使用模型路径: {resolved_path}")

        model = SentenceTransformer(
            resolved_path,
            device="cuda" if torch.cuda.is_available() else "cpu",
            local_files_only=True,  # 避免偷偷联网下载
        )

        model.max_seq_length = max_length
        if use_fp16 and torch.cuda.is_available():
            model.half()

        test_vec = model.encode(
            ["hello world"],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        print(f"✅ 模型加载成功，维度: {test_vec.shape[1]}，设备: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    # embeddings
    print("生成E_text embeddings...")
    text_emb_file = os.path.join(output_dir, "emb_text.npy")
    text_mask_file = os.path.join(output_dir, "valid_mask_text.npy")

    if skip_existing and os.path.exists(text_emb_file) and os.path.exists(text_mask_file):
        print(f"⏭️ 跳过E_text embeddings（文件已存在）")
        # 加载已有的embeddings和mask用于后续统计
        emb_text = np.load(text_emb_file, mmap_mode='r')
        mask_text = np.load(text_mask_file)
        t_text = 0.0  # 无法获取耗时
        sp_text = 0.0  # 无法获取速度
    else:
        text_gen = create_text_generator(str(cases_file), "text")
        emb_text, mask_text, t_text, sp_text = embed_texts_streaming(
            model=model,
            text_generator=text_gen,
            total_count=valid_count_parsed_dicts,
            batch_size=batch_size,
            output_file="emb_text.npy",
            output_dir=output_dir,
            mask_prefix="text",
        )

    print("生成E_event embeddings...")
    event_emb_file = os.path.join(output_dir, "emb_event.npy")
    event_mask_file = os.path.join(output_dir, "valid_mask_event.npy")

    if skip_existing and os.path.exists(event_emb_file) and os.path.exists(event_mask_file):
        print(f"⏭️ 跳过E_event embeddings（文件已存在）")
        # 加载已有的embeddings和mask用于后续统计
        emb_event = np.load(event_emb_file, mmap_mode='r')
        mask_event = np.load(event_mask_file)
        t_event = 0.0
        sp_event = 0.0
    else:
        event_gen = create_text_generator(str(cases_file), "event")
        emb_event, mask_event, t_event, sp_event = embed_texts_streaming(
            model=model,
            text_generator=event_gen,
            total_count=valid_count_parsed_dicts,
            batch_size=batch_size,
            output_file="emb_event.npy",
            output_dir=output_dir,
            mask_prefix="event",
        )


    # 要素embeddings（可选）
    element_embeddings = {}
    element_masks = {}
    element_times = {}
    element_speeds = {}
    element_valid_counts = {}

    if embed_elements:
        element_fields = [
            "actor_main", "actor_main_type", "actor_list", "ai_system", "domain",
            "event_type", "event_cause", "event_process", "event_result",
            "ai_system_list", "ai_system_type_list", "ai_system_domain_list",
            "ai_risk_description", "ai_risk_type", "ai_risk_subtype",
            "harm_type", "harm_severity", "affected_actor_type", "affected_actor_subtype",
            "realized_or_potential", "risk_stage"
        ]

        for field in element_fields:
            print(f"生成{field} embeddings...")
            is_list_field = field in ["actor_list", "ai_system_list", "ai_system_type_list", "ai_system_domain_list"]

            # 检查文件是否已存在
            emb_file = os.path.join(output_dir, f"emb_{field}.npy")
            mask_file = os.path.join(output_dir, f"valid_mask_{field}.npy")
            count_file = os.path.join(output_dir, f"items_count_{field}.npy") if is_list_field else None

            files_exist = os.path.exists(emb_file) and os.path.exists(mask_file)
            if is_list_field and count_file:
                files_exist = files_exist and os.path.exists(count_file)

            if skip_existing and files_exist:
                print(f"⏭️ 跳过{field} embeddings（文件已存在）")
                # 加载已有文件用于统计
                if is_list_field:
                    element_valid_counts[field] = int(np.sum(np.load(mask_file)))
                else:
                    element_masks[field] = np.load(mask_file)
                    element_valid_counts[field] = int(np.sum(element_masks[field]))
                element_times[field] = 0.0
                element_speeds[field] = 0.0
                continue

            if is_list_field:
                # 列表字段：按case为单位写入3D memmap
                t, sp, valid_cnt = embed_list_field_streaming(
                    model=model,
                    cases_file=str(cases_file),
                    out_dir=output_dir,
                    field=field,
                    total_cases=valid_count_parsed_dicts,
                    max_items=10,
                    batch_size_items=batch_size,
                )
                element_times[field] = t
                element_speeds[field] = sp
                element_valid_counts[field] = valid_cnt
            else:
                # 字符串字段：使用原来的2D memmap逻辑
                element_gen = create_text_generator(str(cases_file), element_field=field)
                emb, mask, t, sp = embed_texts_streaming(
                    model=model,
                    text_generator=element_gen,
                    total_count=valid_count_parsed_dicts,
                    batch_size=batch_size,
                    output_file=f"emb_{field}.npy",
                    output_dir=output_dir,
                    mask_prefix=field,
                    is_element_mode=True,
                )
                element_embeddings[field] = emb
                element_masks[field] = mask
                element_times[field] = t
                element_speeds[field] = sp
                element_valid_counts[field] = int(np.sum(mask))

    # 可选：小数据才转npy（否则用case_ids.txt就行）
    if valid_count_parsed_dicts < 100000:
        with open(case_ids_path, "r", encoding="utf-8") as f:
            case_ids = [line.strip() for line in f if line.strip()]
        np.save(os.path.join(output_dir, "case_ids.npy"), np.array(case_ids, dtype=object))

    # 构建输出配置
    outputs = {
        "case_ids_txt": "case_ids.txt",
        "text_view_jsonl": "text_view.jsonl",
        "event_view_jsonl": "event_view.jsonl",
        "emb_text": "emb_text.npy",
        "emb_event": "emb_event.npy",
        "valid_mask_text": "valid_mask_text.npy",
        "valid_mask_event": "valid_mask_event.npy",
        "bad_cases_text_log": "bad_cases_text.log",
        "bad_cases_event_log": "bad_cases_event.log",
        "view_failures_log": "view_failures.log",
    }

    # 要素embeddings配置
    if embed_elements:
        element_fields = [
            "actor_main", "actor_main_type", "actor_list", "ai_system", "domain",
            "event_type", "event_cause", "event_process", "event_result",
            "ai_system_list", "ai_system_type_list", "ai_system_domain_list",
            "ai_risk_description", "ai_risk_type", "ai_risk_subtype",
            "harm_type", "harm_severity", "affected_actor_type", "affected_actor_subtype",
            "realized_or_potential", "risk_stage"
        ]
        for field in element_fields:
            outputs[f"emb_{field}"] = f"emb_{field}.npy"
            outputs[f"valid_mask_{field}"] = f"valid_mask_{field}.npy"
            outputs[f"bad_cases_{field}_log"] = f"bad_cases_{field}.log"
            if field in ["actor_list", "ai_system_list", "ai_system_type_list", "ai_system_domain_list"]:
                outputs[f"items_count_{field}"] = f"items_count_{field}.npy"

    config = {
        "config_file": str(config_path),
        "model": model_path,
        "embedding_dim": int(emb_text.shape[1]) if hasattr(emb_text, "shape") else 0,
        "num_cases": int(valid_count_parsed_dicts),
        "batch_size": int(batch_size),
        "max_length": int(max_length),
        "use_fp16": bool(use_fp16),
        "normalize_embeddings": True,
        "random_seed": 42,
        "cases_file": str(cases_file),
        "streaming_mode": True,
        "embed_elements": bool(embed_elements),
        "time_seconds": {
            "text": t_text,
            "event": t_event,
            "total": time.time() - start_time,
        },
        "throughput_samples_per_sec": {
            "text": sp_text,
            "event": sp_event,
        },
        "outputs": outputs,
    }

    # 添加要素embeddings的时间信息
    if embed_elements:
        for field in element_times.keys():
            config["time_seconds"][field] = element_times[field]
            config["throughput_samples_per_sec"][field] = element_speeds[field]
    config_file=os.path.join(output_dir, "embedding_config.json")
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - start_time
    print("完成!")
    print(f"输出目录: {output_dir}")
    print(f"处理案例: {valid_count_parsed_dicts}")
    print(f"总耗时: {elapsed:.1f}s")
    print(f"吞吐量: {valid_count_parsed_dicts / elapsed:.1f} samples/sec")

    failure_counts = f"text={int(np.sum(~mask_text))}, event={int(np.sum(~mask_event))}"
    if embed_elements:
        element_failures = []
        for field in element_valid_counts.keys():
            total_failures = valid_count_parsed_dicts - element_valid_counts[field]
            element_failures.append(f"{field}={total_failures}")
        failure_counts += ", " + ", ".join(element_failures)

    print(f"失败样本: {failure_counts}")

    if embed_elements:
        print("\n要素embeddings生成完成:")
        for field in element_valid_counts.keys():
            valid_count = element_valid_counts[field]
            print(f"  {field}: {valid_count}/{valid_count_parsed_dicts} ({valid_count/valid_count_parsed_dicts:.1%})")


if __name__ == "__main__":
    main()
