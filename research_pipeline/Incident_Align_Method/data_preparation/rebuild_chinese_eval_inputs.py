#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按中文金标 v2.0.2 重建评测所需的输入产物。

数据源（均按 report_id 对齐）：
  1. 金标          data/chinese_gold/gold_v2.0.2_mapping.csv
  2. 正文/标题      assets/chinese_eval_frozen/frozen_eval_input.csv
  3. 10 结构化字段   assets/chinese_eval_frozen/event_features_1283.csv
  4. bge-m3 向量    assets/chinese_eval_frozen/emb_text_1283.npy

产出：
  data/chinese_eval_cases.jsonl        1,283 行
  outputs/chinese_eval_embeddings/     case_ids.txt + emb_text.npy + valid_mask_text.npy
  outputs/chinese_eval_faiss/          faiss_text.index + idx2caseid.json

用法：
  PY=/data2/conda_envs/AI_risk_event/bin/python
  $PY src/Incident_Align_Method/data_preparation/rebuild_chinese_eval_inputs.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
FROZEN = ROOT / "assets" / "chinese_eval_frozen"


def load_gold(path: Path):
    mapping = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            mapping[row["report_id"].strip()] = row["gold_v2_event_id"].strip()
    return sorted(mapping, key=lambda x: int(x)), mapping


def main() -> int:
    ap = argparse.ArgumentParser(description="按 1,283 篇金标重建中文评测输入")
    ap.add_argument("--gold", default="data/chinese_gold/gold_v2.0.2_mapping.csv")
    ap.add_argument("--assets", default=str(FROZEN))
    ap.add_argument("--cases-out", default="data/chinese_eval_cases.jsonl")
    ap.add_argument("--emb-out", default="outputs/chinese_eval_embeddings")
    ap.add_argument("--faiss-out", default="outputs/chinese_eval_faiss")
    args = ap.parse_args()

    assets = Path(args.assets)
    report_ids, gold = load_gold(ROOT / args.gold)
    print(f"[gold] {len(report_ids)} 篇 / {len(set(gold.values()))} 事件")

    frozen = {r["report_id"].strip(): r for r in
              csv.DictReader(open(assets / "frozen_eval_input.csv", encoding="utf-8"))}
    feats = {r["report_id"].strip(): r for r in
             csv.DictReader(open(assets / "event_features_1283.csv", encoding="utf-8"))}
    cid = np.load(assets / "case_ids_1283.npy", allow_pickle=True)
    emb = np.load(assets / "emb_text_1283.npy")

    if [str(x) for x in cid] != report_ids:
        raise SystemExit("[FATAL] case_ids_1283.npy 顺序与金标不一致")

    def g(row, key):
        v = row.get(key, "")
        return v if v not in ("", None) else None

    lines = []
    for rid in report_ids:
        fr, ft = frozen[rid], feats[rid]
        obj = {
            "id": int(rid), "original_id": int(rid),
            "title": fr.get("title", ""),
            "title_first_line": fr.get("title_first_line", ""),
            "release_date": None, "case_link": fr.get("source_url", ""),
            "from_database": ft.get("data_source", "wenge"),
            "event_annotation": {
                "event_time_start": None, "event_time_end": None,
                "event_country": None, "event_province": None, "event_city": None,
                "actor_main": g(ft, "actor_main"),
                "actor_main_type": g(ft, "event_actor_main_type"),
                "actor_list": [], "ai_system": g(ft, "ai_system"),
                "domain": g(ft, "domain"), "event_type": g(ft, "event_type"),
                "event_cause": g(ft, "event_cause"),
                "event_process": g(ft, "event_process"),
                "event_result": g(ft, "event_result"),
            },
            "ai_risk": {
                "ai_risk_description": g(ft, "ai_risk_description"),
                "ai_risk_type": g(ft, "ai_risk_type"),
                "ai_risk_subtype": g(ft, "ai_risk_subtype"),
                "harm_type": g(ft, "harm_type"),
                "harm_severity": g(ft, "harm_severity"),
                "affected_actor_type": g(ft, "affected_actor_type"),
                "affected_actor_subtype": g(ft, "affected_actor_subtype"),
                "realized_or_potential": g(ft, "realized_or_potential"),
                "risk_stage": g(ft, "risk_stage"),
            },
            "ai_tech": {
                "ai_system_list": [x for x in (ft.get("ai_system_list") or "").strip("{}").split(",") if x],
                "ai_system_type_list": [], "ai_system_domain_list": [],
            },
            "text": fr.get("text", ""),
            "gold_v2_event_id": gold[rid], "in_gold": True,
        }
        lines.append(json.dumps(obj, ensure_ascii=False))

    cases_out = ROOT / args.cases_out
    cases_out.parent.mkdir(parents=True, exist_ok=True)
    cases_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[cases] {len(lines)} 行 -> {cases_out.relative_to(ROOT)}")

    emb_dir = ROOT / args.emb_out
    emb_dir.mkdir(parents=True, exist_ok=True)
    arr = np.ascontiguousarray(emb.astype(np.float32))
    np.save(emb_dir / "emb_text.npy", arr)
    np.save(emb_dir / "valid_mask_text.npy", np.ones((arr.shape[0],), dtype=np.float32))
    (emb_dir / "case_ids.txt").write_text("\n".join(report_ids) + "\n", encoding="utf-8")
    print(f"[emb] {arr.shape} -> {emb_dir.relative_to(ROOT)}")

    import faiss
    faiss_dir = ROOT / args.faiss_out
    faiss_dir.mkdir(parents=True, exist_ok=True)
    vecs = np.ascontiguousarray(arr)
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    faiss.write_index(index, str(faiss_dir / "faiss_text.index"))
    (faiss_dir / "idx2caseid.json").write_text(
        json.dumps({str(i): c for i, c in enumerate(report_ids)}), encoding="utf-8")
    print(f"[faiss] ntotal={index.ntotal} -> {faiss_dir.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
