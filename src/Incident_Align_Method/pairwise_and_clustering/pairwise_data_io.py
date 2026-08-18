#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 数据加载和处理工具
#用于加载评估结构、案例数据、召回结果，并构建训练/开发/测试对数据集

import json
import os
import random
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Optional, Tuple


CAT_FIELDS = [
    "actor_main_type",
    "ai_risk_type",
    "harm_type",
    "harm_severity",
    "affected_actor_type",
    "realized_or_potential",
    "risk_stage",
]

TEXT_FIELDS_DEFAULT = [
    "actor_main",
    "ai_system",
    "domain",
    "event_type",
    "event_cause",
    "event_process",
    "event_result",
    "ai_risk_description",
    "ai_risk_subtype",
    "affected_actor_subtype",
]


def norm_id(x) -> str:
    return str(x).strip()


def normalize_event_annotation(ea):
    if ea is None:
        return {}
    if isinstance(ea, dict):
        return ea
    if isinstance(ea, list):
        if len(ea) == 0:
            return {}
        first = ea[0]
        return first if isinstance(first, dict) else {}
    return {}


def load_eval_structure(structure_file: str):
    with open(structure_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    incident2caseids = {}
    caseid2incident = {}
    all_ids = set()

    for ev in data["events"]:
        inc = norm_id(ev["incident_id"])
        ids = [norm_id(x) for x in ev["ids"]]
        incident2caseids[inc] = ids
        for cid in ids:
            caseid2incident[cid] = inc
            all_ids.add(cid)

    return incident2caseids, caseid2incident, sorted(all_ids)


def load_cases_minimal(cases_file: str):
    case2cats = {}
    with open(cases_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            cid = norm_id(obj.get("id", ""))

            event_ann_raw = obj.get("event_annotation", None)
            event_ann = normalize_event_annotation(event_ann_raw)

            ai_risk = obj.get("ai_risk", {}) or {}
            if not isinstance(ai_risk, dict):
                ai_risk = {}

            row = {}
            row["actor_main_type"] = event_ann.get("actor_main_type", None)
            row["ai_risk_type"] = ai_risk.get("ai_risk_type", None)
            row["harm_type"] = ai_risk.get("harm_type", None)
            row["harm_severity"] = ai_risk.get("harm_severity", None)
            row["affected_actor_type"] = ai_risk.get("affected_actor_type", None)
            row["realized_or_potential"] = ai_risk.get("realized_or_potential", None)
            row["risk_stage"] = ai_risk.get("risk_stage", None)

            for key in CAT_FIELDS:
                value = row.get(key)
                if value is None or (isinstance(value, str) and value.strip() == ""):
                    row[key] = "__MISSING__"
                else:
                    row[key] = str(value).strip()

            case2cats[cid] = row
    return case2cats


def split_incidents(incident_ids, seed=42, train_ratio=0.7, dev_ratio=0.1):
    ids = list(incident_ids)
    rnd = random.Random(seed)
    rnd.shuffle(ids)
    n = len(ids)
    n_train = int(n * train_ratio)
    n_dev = int(n * dev_ratio)
    train = set(ids[:n_train])
    dev = set(ids[n_train:n_train + n_dev])
    test = set(ids[n_train + n_dev:])
    return train, dev, test


def incident_to_gold_clusters(incident2caseids, inc_set):
    gold = []
    for inc, ids in incident2caseids.items():
        if inc in inc_set:
            uniq = []
            seen = set()
            for x in ids:
                x = norm_id(x)
                if x not in seen:
                    seen.add(x)
                    uniq.append(x)
            gold.append(uniq)
    return gold


def case_ids_in_split(incident2caseids, inc_set):
    s = set()
    for inc, ids in incident2caseids.items():
        if inc in inc_set:
            for x in ids:
                s.add(norm_id(x))
    return sorted(s)


def minmax_norm(x: float, mn: float, mx: float) -> float:
    if mx <= mn:
        return 0.0
    return float((x - mn) / (mx - mn))


def load_recall_topk(recall_file: str, topm_out: int = 200):
    query2cands = {}
    query2gt = {}
    score_collect = defaultdict(lambda: {"fuse": [], "text": [], "event": []})

    with open(recall_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            q = norm_id(obj["query_case_id"])
            gt = [norm_id(x) for x in obj.get("ground_truth", [])]
            cands = obj.get("candidates", [])

            def _sf(x):
                value = x.get("score_fuse", None)
                if value is None:
                    st = x.get("score_text", None)
                    se = x.get("score_event", None)
                    st = float(st) if st is not None else 0.0
                    se = float(se) if se is not None else 0.0
                    return max(st, se)
                return float(value)

            cands = sorted(cands, key=_sf, reverse=True)[:topm_out]

            packed = []
            for c in cands:
                cid = norm_id(c["case_id"])
                st = c.get("score_text", None)
                se = c.get("score_event", None)
                sf = c.get("score_fuse", None)

                st = float(st) if st is not None else 0.0
                se = float(se) if se is not None else 0.0
                sf = float(sf) if sf is not None else max(st, se)

                packed.append((cid, sf, st, se))
                score_collect[q]["fuse"].append(sf)
                score_collect[q]["text"].append(st)
                score_collect[q]["event"].append(se)

            query2cands[q] = packed
            query2gt[q] = set(gt)

    query2minmax = {}
    for q, d in score_collect.items():
        mm = {}
        for key in ["fuse", "text", "event"]:
            arr = d.get(key, [])
            if not arr:
                mm[key] = (0.0, 0.0)
            else:
                mm[key] = (float(min(arr)), float(max(arr)))
        query2minmax[q] = mm

    return query2cands, query2gt, query2minmax


def build_all_pairs(
    query2cands,
    query2gt,
    query2minmax,
    caseid2incident,
    inc_set: set,
    fallback_neg_per_empty_q: int = 10,
    fallback_seed: int = 42,
    fallback_pool_case_ids: Optional[List[str]] = None,
):
    rnd = random.Random(fallback_seed)

    if fallback_pool_case_ids is None:
        pool = [cid for cid, inc in caseid2incident.items() if inc in inc_set]
    else:
        pool = [norm_id(x) for x in fallback_pool_case_ids if norm_id(x)]
    pool = sorted(set(pool))
    pool_set = set(pool)

    pairs = []
    visited_q = set()

    for q, cands in query2cands.items():
        q = norm_id(q)
        q_inc = caseid2incident.get(q, None)
        if q_inc is None or q_inc not in inc_set:
            continue

        if q not in pool_set:
            continue

        visited_q.add(q)

        mm = query2minmax.get(q, {"fuse": (0.0, 0.0), "text": (0.0, 0.0), "event": (0.0, 0.0)})
        gt_set = query2gt.get(q, set())

        added = 0
        for c, sf, st, se in cands:
            c = norm_id(c)
            c_inc = caseid2incident.get(c, None)
            if c_inc is None:
                continue
            if c_inc not in inc_set:
                continue

            y = 1 if c in gt_set else 0

            mn_f, mx_f = mm["fuse"]
            mn_t, mx_t = mm["text"]
            mn_e, mx_e = mm["event"]

            pairs.append({
                "q": q,
                "c": c,
                "label": y,
                "rerank": float(sf),
                "rerank_norm": minmax_norm(float(sf), mn_f, mx_f),
                "recall_text": float(st),
                "recall_event": float(se),
                "recall_fuse": float(sf),
                "recall_text_norm": minmax_norm(float(st), mn_t, mx_t),
                "recall_event_norm": minmax_norm(float(se), mn_e, mx_e),
                "recall_fuse_norm": minmax_norm(float(sf), mn_f, mx_f),
                "q_inc": q_inc,
                "c_inc": c_inc,
            })
            added += 1

        if added == 0 and fallback_neg_per_empty_q > 0:
            cand_pool = [x for x in pool if x != q]
            if cand_pool:
                chosen = cand_pool if len(cand_pool) <= fallback_neg_per_empty_q else rnd.sample(cand_pool, k=fallback_neg_per_empty_q)
                for c in chosen:
                    c_inc = caseid2incident.get(c, None)
                    if c_inc is None:
                        continue
                    pairs.append({
                        "q": q,
                        "c": c,
                        "label": 0,
                        "rerank": 0.0,
                        "rerank_norm": 0.0,
                        "recall_text": 0.0,
                        "recall_event": 0.0,
                        "recall_fuse": 0.0,
                        "recall_text_norm": 0.0,
                        "recall_event_norm": 0.0,
                        "recall_fuse_norm": 0.0,
                        "q_inc": q_inc,
                        "c_inc": c_inc,
                    })

    if fallback_neg_per_empty_q > 0:
        missing_qs = [q for q in pool if q not in visited_q]
        for q in missing_qs:
            q_inc = caseid2incident.get(q, None)
            if q_inc is None or q_inc not in inc_set:
                continue

            cand_pool = [x for x in pool if x != q]
            if not cand_pool:
                continue
            chosen = cand_pool if len(cand_pool) <= fallback_neg_per_empty_q else rnd.sample(cand_pool, k=fallback_neg_per_empty_q)

            for c in chosen:
                c_inc = caseid2incident.get(c, None)
                if c_inc is None:
                    continue
                pairs.append({
                    "q": q,
                    "c": c,
                    "label": 0,
                    "rerank": 0.0,
                    "rerank_norm": 0.0,
                    "recall_text": 0.0,
                    "recall_event": 0.0,
                    "recall_fuse": 0.0,
                    "recall_text_norm": 0.0,
                    "recall_event_norm": 0.0,
                    "recall_fuse_norm": 0.0,
                    "q_inc": q_inc,
                    "c_inc": c_inc,
                })

    return pairs


def sample_train_pairs(
    pairs,
    neg_ratio: int = 4,
    seed: int = 42,
    neg_per_empty_q: int = 10,
    max_empty_q: int = 0,
    shuffle_empty_q: bool = True,
):
    rnd = random.Random(seed)

    q2pos = defaultdict(list)
    q2neg = defaultdict(list)
    for p in pairs:
        if p.get("label", 0) == 1:
            q2pos[p["q"]].append(p)
        else:
            q2neg[p["q"]].append(p)

    out = []

    for q, pos_list in q2pos.items():
        neg_list = q2neg.get(q, [])
        if neg_list:
            neg_list = sorted(neg_list, key=lambda x: x.get("rerank", 0.0), reverse=True)

        out.extend(pos_list)

        k = neg_ratio * len(pos_list)
        if k > 0 and neg_list:
            chosen = neg_list[:k]
            rnd.shuffle(chosen)
            out.extend(chosen)

    if neg_per_empty_q and neg_per_empty_q > 0:
        empty_qs = [q for q in q2neg.keys() if q not in q2pos]

        if max_empty_q and max_empty_q > 0 and len(empty_qs) > max_empty_q:
            if shuffle_empty_q:
                rnd.shuffle(empty_qs)
            empty_qs = empty_qs[:max_empty_q]

        for q in empty_qs:
            neg_list = q2neg.get(q, [])
            if not neg_list:
                continue
            neg_list = sorted(neg_list, key=lambda x: x.get("rerank", 0.0), reverse=True)
            chosen = neg_list[:neg_per_empty_q]
            rnd.shuffle(chosen)
            out.extend(chosen)

    rnd.shuffle(out)
    return out


def build_cat_vocabs(case2cats: dict, train_case_ids: set):
    vocabs = {}
    for field in CAT_FIELDS:
        tokens = set(["__MISSING__"])
        for cid in train_case_ids:
            row = case2cats.get(cid)
            if row is None:
                continue
            tokens.add(row.get(field, "__MISSING__"))
        tokens = sorted(tokens)
        vocabs[field] = {t: i for i, t in enumerate(tokens)}
    return vocabs


def parse_csv_arg(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def read_json(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing JSON file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, obj: Any):
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing JSONL file: {path}")
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def iter_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing JSONL file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def count_jsonl(path: str) -> int:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing JSONL file: {path}")
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def write_jsonl(path: str, rows: List[Dict[str, Any]]):
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def repeat_dir_name(repeat_index: int) -> str:
    return f"repeat_{int(repeat_index):02d}"


def build_repeat_paths(prepared_dir: str, repeat_index: int) -> Dict[str, str]:
    repeat_name = repeat_dir_name(repeat_index)
    repeat_dir = os.path.join(prepared_dir, repeat_name)
    return {
        "repeat_name": repeat_name,
        "repeat_dir": repeat_dir,
        "train_pairs": os.path.join(repeat_dir, "train_pairs.jsonl"),
        "dev_pairs": os.path.join(repeat_dir, "dev_pairs.jsonl"),
        "test_pairs": os.path.join(repeat_dir, "test_pairs.jsonl"),
        "vocabs": os.path.join(repeat_dir, "vocabs.json"),
        "split_meta": os.path.join(repeat_dir, "split_meta.json"),
    }


def write_prepared_repeat(
    prepared_dir: str,
    repeat_index: int,
    train_pairs: List[Dict[str, Any]],
    dev_pairs: List[Dict[str, Any]],
    test_pairs: List[Dict[str, Any]],
    vocabs: Dict[str, Dict[str, int]],
    split_meta: Dict[str, Any],
):
    paths = build_repeat_paths(prepared_dir, repeat_index)
    os.makedirs(paths["repeat_dir"], exist_ok=True)
    write_jsonl(paths["train_pairs"], train_pairs)
    write_jsonl(paths["dev_pairs"], dev_pairs)
    write_jsonl(paths["test_pairs"], test_pairs)
    write_json(paths["vocabs"], vocabs)
    write_json(paths["split_meta"], split_meta)
    return paths


def load_manifest(prepared_dir: str) -> Dict[str, Any]:
    manifest_path = os.path.join(prepared_dir, "manifest.json")
    manifest = read_json(manifest_path)
    if "repeats" not in manifest or not isinstance(manifest["repeats"], list) or len(manifest["repeats"]) == 0:
        raise ValueError(f"Prepared manifest is missing non-empty repeats: {manifest_path}")
    return manifest


def load_prepared_repeat(prepared_dir: str, repeat_entry: Dict[str, Any]) -> Dict[str, Any]:
    repeat_index = int(repeat_entry["repeat_index"])
    paths = build_repeat_paths(prepared_dir, repeat_index)

    missing = [p for key, p in paths.items() if key not in {"repeat_name", "repeat_dir"} and not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"Prepared repeat {repeat_index} is missing files: {missing}")

    vocabs = read_json(paths["vocabs"])
    split_meta = read_json(paths["split_meta"])

    stats = split_meta.get("stats", {}) if isinstance(split_meta, dict) else {}
    train_count = int(stats.get("train_pairs", count_jsonl(paths["train_pairs"])))
    dev_count = int(stats.get("dev_pairs", count_jsonl(paths["dev_pairs"])))
    test_count = int(stats.get("test_pairs", count_jsonl(paths["test_pairs"])))

    if train_count == 0:
        raise ValueError(f"Prepared repeat {repeat_index} has empty train_pairs: {paths['train_pairs']}")
    if dev_count == 0:
        raise ValueError(f"Prepared repeat {repeat_index} has empty dev_pairs: {paths['dev_pairs']}")
    if test_count == 0:
        raise ValueError(f"Prepared repeat {repeat_index} has empty test_pairs: {paths['test_pairs']}")

    return {
        "repeat_index": repeat_index,
        "paths": {k: str(v) for k, v in paths.items()},
        "vocabs": vocabs,
        "split_meta": split_meta,
        "counts": {
            "train_pairs": train_count,
            "dev_pairs": dev_count,
            "test_pairs": test_count,
        },
    }


def iter_prepared_pairs(prepared_dir: str, repeat_index: int, split: str) -> Iterator[Dict[str, Any]]:
    if split not in {"train", "dev", "test"}:
        raise ValueError(f"Unknown split={split}")
    paths = build_repeat_paths(prepared_dir, repeat_index)
    return iter_jsonl(paths[f"{split}_pairs"])
