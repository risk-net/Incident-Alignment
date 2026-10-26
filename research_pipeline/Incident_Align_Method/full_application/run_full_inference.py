#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全量 pairwise 推理入口（与聚类解耦）。

只做 pairwise 推理，把每对 (query, candidate) 的概率写入 pair_predictions_full.jsonl。
聚类由 run_full_clustering.py 独立执行，从该文件读回分数，无需重复推理。

运行方式：
python src/Incident_Align_Method/full_application/run_full_inference.py \
  --config config/Incident_Align_Method-full_application-config.ini
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple

import torch

from config_utils import (
    DEFAULT_CONFIG_PATH,
    get_float,
    get_int,
    load_config,
    load_database_config,
    require_section,
    resolve_path,
)

# 数据库相关导入
try:
    import psycopg2
except ImportError:
    psycopg2 = None


METHOD_DIR = Path(__file__).resolve().parents[1]
PAIRWISE_DIR = METHOD_DIR / "pairwise_and_clustering"
if str(PAIRWISE_DIR) not in sys.path:
    sys.path.insert(0, str(PAIRWISE_DIR))

from pairwise_data_io import norm_id
from pairwise_model import EmbeddingsStore, load_checkpoint, predict_probs


def require_psycopg2():
    if psycopg2 is None:
        raise ImportError("需要安装 psycopg2-binary 才能执行数据库全量推理")


def clip01(value) -> float:
    try:
        out = float(value)
    except Exception:
        return 0.0
    if out != out:
        return 0.0
    return max(0.0, min(1.0, out))


def load_model_checkpoint(model_path: str, device: str, embeddings_dir: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"checkpoint 不存在: {model_path}")
    if not os.path.exists(embeddings_dir):
        raise FileNotFoundError(f"embeddings_dir 不存在: {embeddings_dir}")

    model, checkpoint, _tab_preprocessor = load_checkpoint(model_path, device=device)
    if "vocabs" not in checkpoint:
        raise ValueError("checkpoint 中缺少 vocabs，无法执行全量推理")

    text_fields = checkpoint.get("text_fields", [])
    sim_fields = checkpoint.get("sim_fields", [])
    needed_fields = sorted(set(text_fields) | set(sim_fields))
    store = EmbeddingsStore(embeddings_dir, needed_fields)
    return {
        "model": model,
        "checkpoint": checkpoint,
        "store": store,
        "text_fields": text_fields,
    }


def load_cases_from_database(db_config: Dict[str, object]) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
    require_psycopg2()
    source_relation = str(db_config.get("source_relation", "v_alignment_input_v1"))
    query = """
        SELECT
            news_id,
            event_actor_main_type,
            ai_system_type_list,
            event_domain,
            event_type,
            event_cause,
            event_process,
            event_result,
            ai_risk_description,
            ai_risk_type,
            ai_risk_subtype,
            harm_type,
            harm_severity,
            affected_actor_type,
            affected_actor_subtype,
            realized_or_potential,
            risk_stage
        FROM {source_relation}
        WHERE classification_result = %s
        ORDER BY news_id
    """.format(source_relation=source_relation)

    conn = None
    cursor = None
    try:
        conn = psycopg2.connect(
            host=db_config["host"],
            database=db_config["database"],
            user=db_config["user"],
            password=db_config["password"],
            port=db_config["port"],
        )
        cursor = conn.cursor(name="full_inference_cases_stream")
        cursor.itersize = 5000
        cursor.execute(query, (db_config["classification_result"],))

        cases_data: Dict[str, Dict] = {}
        case2cats: Dict[str, Dict] = {}

        def norm_list_field(raw):
            if raw is None:
                return "__MISSING__"
            if isinstance(raw, list):
                return str(raw[0]) if raw else "__MISSING__"
            try:
                parsed = json.loads(raw) if isinstance(raw, str) else raw
                if isinstance(parsed, list):
                    return str(parsed[0]) if parsed else "__MISSING__"
            except Exception:
                pass
            return str(raw)

        total = 0
        while True:
            rows = cursor.fetchmany(cursor.itersize)
            if not rows:
                break
            for row in rows:
                (
                    row_id,
                    actor_main_type,
                    ai_system_type_list,
                    domain,
                    event_type,
                    event_cause,
                    event_process,
                    event_result,
                    ai_risk_description,
                    ai_risk_type,
                    ai_risk_subtype,
                    harm_type,
                    harm_severity,
                    affected_actor_type,
                    affected_actor_subtype,
                    realized_or_potential,
                    risk_stage,
                ) = row
                cid = norm_id(row_id)
                cases_data[cid] = {"news_id": cid}
                case2cats[cid] = {
                    "actor_main_type": actor_main_type if actor_main_type is not None else "__MISSING__",
                    "ai_system_type": norm_list_field(ai_system_type_list),
                    "domain": domain if domain is not None else "__MISSING__",
                    "event_type": event_type if event_type is not None else "__MISSING__",
                    "event_cause": event_cause if event_cause is not None else "__MISSING__",
                    "event_process": event_process if event_process is not None else "__MISSING__",
                    "event_result": event_result if event_result is not None else "__MISSING__",
                    "ai_risk_description": ai_risk_description if ai_risk_description is not None else "__MISSING__",
                    "ai_risk_type": ai_risk_type if ai_risk_type is not None else "__MISSING__",
                    "ai_risk_subtype": ai_risk_subtype if ai_risk_subtype is not None else "__MISSING__",
                    "harm_type": harm_type if harm_type is not None else "__MISSING__",
                    "harm_severity": harm_severity if harm_severity is not None else "__MISSING__",
                    "affected_actor_type": affected_actor_type if affected_actor_type is not None else "__MISSING__",
                    "affected_actor_subtype": affected_actor_subtype if affected_actor_subtype is not None else "__MISSING__",
                    "realized_or_potential": realized_or_potential if realized_or_potential is not None else "__MISSING__",
                    "risk_stage": risk_stage if risk_stage is not None else "__MISSING__",
                }
                total += 1
            print(f"   - streamed {total} cases...", flush=True)

        if not cases_data:
            raise ValueError("数据库查询结果为空，请检查 classification_result 或表数据")
        print(f"   - 成功加载 {len(cases_data)} 条案例数据")
        return cases_data, case2cats
    finally:
        if cursor is not None:
            try:
                cursor.close()
            except Exception:
                pass
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def create_pair_batches_from_recall(
    recall_file: str,
    case2cats: Dict[str, Dict],
    valid_embed_ids: Optional[Set[str]],
    batch_pairs: int,
    topm_out: int,
) -> Iterator[List[Dict[str, object]]]:
    if not os.path.exists(recall_file):
        raise FileNotFoundError(f"recall_file 不存在: {recall_file}")

    print("📂 从 recall 文件流式生成 pair 批次...")
    batch: List[Dict[str, object]] = []
    with open(recall_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            query_id = norm_id(
                item.get("query_case_id", item.get("query", item.get("q", item.get("query_id", item.get("qid", "")))))
            )
            if query_id not in case2cats:
                continue
            if valid_embed_ids is not None and query_id not in valid_embed_ids:
                continue

            for cand in item.get("candidates", [])[:topm_out]:
                cand_id = norm_id(cand.get("case_id", cand.get("candidate", cand.get("c", cand.get("cand_id", cand.get("id", ""))))))
                if cand_id == query_id or cand_id not in case2cats:
                    continue
                if valid_embed_ids is not None and cand_id not in valid_embed_ids:
                    continue

                batch.append(
                    {
                        "q": query_id,
                        "c": cand_id,
                        "recall_fuse_norm": clip01(cand.get("score_fuse_norm", cand.get("score_fuse", cand.get("score", 0.0)))),
                        "recall_text_norm": clip01(cand.get("score_text_norm", cand.get("score_text", 0.0))),
                        "recall_event_norm": clip01(cand.get("score_event_norm", cand.get("score_event", 0.0))),
                        "label": 0,
                    }
                )
                if len(batch) >= batch_pairs:
                    yield batch
                    batch = []
    if batch:
        yield batch


def main():
    parser = argparse.ArgumentParser(description="统一的全量 pairwise + clustering 推理入口")
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="INI 配置文件路径",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    section = require_section(cfg, "RunFullInference")
    db_config = load_database_config(cfg)

    checkpoint_path = resolve_path(section.get("checkpoint_path", ""))
    artifacts_root = resolve_path(section.get("artifacts_root", ""))
    embeddings_dir = resolve_path(section.get("embeddings_dir", os.path.join(artifacts_root, "embeddings")))
    recall_file = resolve_path(section.get("recall_file", os.path.join(artifacts_root, "recall", "full_recall_candidates.jsonl")))
    output_dir = resolve_path(section.get("output_dir", os.path.join(artifacts_root, "outputs", "full_inference")))
    device = (section.get("device", "auto") or "auto").strip().lower()
    batch_size = get_int(section, "batch_size", 2048)
    pair_batch_size = get_int(section, "pair_batch_size", 10000)
    topm_out = get_int(section, "topm_out", 250)
    threshold = get_float(section, "threshold", 0.5)

    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print("🚀 开始全量推理")
    print(f"   - checkpoint: {checkpoint_path}")
    print(f"   - recall_file: {recall_file}")
    print(f"   - embeddings_dir: {embeddings_dir}")
    print(f"   - output_dir: {output_dir}")
    print(f"   - device: {device}")

    bundle = load_model_checkpoint(checkpoint_path, device, embeddings_dir)
    model = bundle["model"]
    checkpoint = bundle["checkpoint"]
    store = bundle["store"]
    text_fields = bundle["text_fields"]

    cases_data, case2cats = load_cases_from_database(db_config)
    valid_embed_ids = set(store.caseid2idx.keys())
    if not valid_embed_ids:
        raise ValueError("embeddings 中没有可用 case id")

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    pair_pred_file = output_root / "pair_predictions_full.jsonl"
    config_file = output_root / "inference_config.json"

    scored_nodes: Set[str] = set()
    total_pairs_scored = 0

    print("\n🔮 进行 pairwise 推理...")
    with open(pair_pred_file, "w", encoding="utf-8") as fout:
        for batch_idx, pair_batch in enumerate(
            create_pair_batches_from_recall(
                recall_file=recall_file,
                case2cats=case2cats,
                valid_embed_ids=valid_embed_ids,
                batch_pairs=pair_batch_size,
                topm_out=topm_out,
            ),
            start=1,
        ):
            probs, _, qc_list = predict_probs(
                model=model,
                pair_source=pair_batch,
                store=store,
                case2cats=case2cats,
                text_fields=text_fields,
                device=device,
                desc=f"Full Inference batch {batch_idx}",
                batch_size=batch_size,
                num_pairs=len(pair_batch),
            )

            for (q, c), prob in zip(qc_list, probs):
                qn = norm_id(q)
                cn = norm_id(c)
                scored_nodes.add(qn)
                scored_nodes.add(cn)
                probability = float(prob)
                prediction = int(probability >= threshold)
                record = {
                    "query": qn,
                    "candidate": cn,
                    "probability": probability,
                    "prediction": prediction,
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_pairs_scored += 1
            if total_pairs_scored and total_pairs_scored % 200000 == 0:
                print(f"   - scored={total_pairs_scored}", flush=True)

    if not scored_nodes:
        raise ValueError("没有生成任何 pair 预测，请检查 recall 文件、DB 数据和 embeddings 的 id 是否一致")

    print("\n✅ pairwise 推理完成。聚类请运行 run_full_clustering.py。")

    training_config = checkpoint.get("training_config", {})
    run_config = {
        "config_path": str(Path(args.config).resolve()),
        "checkpoint_path": checkpoint_path,
        "artifacts_root": artifacts_root,
        "embeddings_dir": embeddings_dir,
        "recall_file": recall_file,
        "output_dir": output_dir,
        "device": device,
        "database": {
            "host": db_config["host"],
            "port": db_config["port"],
            "database": db_config["database"],
            "user": db_config["user"],
            "password_env": db_config["password_env"],
            "classification_result": db_config["classification_result"],
        },
        "inference": {
            "batch_size": batch_size,
            "pair_batch_size": pair_batch_size,
            "topm_out": topm_out,
            "threshold": threshold,
        },
        "stats": {
            "total_db_cases": len(cases_data),
            "total_scored_nodes": len(scored_nodes),
            "total_pairs_scored": total_pairs_scored,
        },
        "checkpoint_info": {
            "best_dev_f1": float(checkpoint.get("best_dev_event_macro_f1", 0.0)),
            "best_epoch": int(checkpoint.get("best_epoch", -1)),
            "repeat_idx": int(checkpoint.get("repeat_idx", -1)),
            "seed": int(checkpoint.get("seed", -1)),
        },
        "args": {
            "text_fields": checkpoint.get("text_fields"),
            "sim_fields": checkpoint.get("sim_fields"),
            "cat_emb_dim": training_config.get("cat_emb_dim"),
            "tab_hidden_dims": training_config.get("tab_hidden_dims"),
            "deep_hidden": training_config.get("deep_hidden"),
            "deep_hidden2": training_config.get("deep_hidden2"),
            "dropout": training_config.get("dropout"),
        },
    }
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(run_config, f, ensure_ascii=False, indent=2)

    print("\n📊 推理结果统计:")
    print(f"   - 总案例数(DB): {len(cases_data)}")
    print(f"   - 参与推理节点数: {len(scored_nodes)}")
    print(f"   - 评分 pair 对数: {total_pairs_scored}")
    print(f"\n💾 结果已保存到: {output_root}")
    print(f"   - {pair_pred_file}")
    print(f"   - {config_file}")
    print(f"\n🔗 下一步: 运行 run_full_clustering.py 进行聚类。")


if __name__ == "__main__":
    main()
