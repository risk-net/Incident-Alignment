#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
这个脚本用于从pairwise模型的训练输出中，针对每个repeat的多个checkpoint，搜索最佳的图解码配置，并使用该配置评估测试集性能。最终结果包括每个repeat的最佳checkpoint、对应的dev/test指标，以及一个全局总结文件。
"""


import configparser
import json
import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from graph_clustering import SCIPY_OK, evaluate_with_best_config, grid_search_on_dev
from pairwise_data_io import load_cases_minimal, load_manifest, load_prepared_repeat, norm_id
from pairwise_model import EmbeddingsStore, load_checkpoint, predict_probs


BASE_DIR = Path(__file__).resolve().parents[3]
DEFAULT_BATCH_SIZE = 512
DEFAULT_THRESHOLD_GRID_STEP = 0.02
DEFAULT_EDGE_RULE_GRID = ["mutual"]
SUPPORTED_EDGE_RULES = {"mutual"}
LOGGER = logging.getLogger(__name__)


@dataclass
class DecodeConfig:
    prepared_dir: str
    embeddings_dir: str
    train_output_dir: str
    output_dir: str
    path_display_base: str
    device: str
    batch_size: int
    threshold_grid_step: float
    edge_rule_grid: List[str]
    overwrite: bool
    config_path: Optional[str] = None


@dataclass
class LoadedArtifacts:
    config: DecodeConfig
    device: str
    manifest: Dict[str, Any]
    training_summary_path: str
    training_summary: Dict[str, Any]
    prepared_by_repeat: Dict[int, Dict[str, Any]]
    case2cats: Dict[str, Dict[str, Any]]
    store: EmbeddingsStore
    text_fields: List[str]
    sim_fields: List[str]


def _load_json(path: str) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, obj: Any):
    path_obj = Path(path)
    if path_obj.parent:
        path_obj.parent.mkdir(parents=True, exist_ok=True)
    with path_obj.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _resolve_project_path(raw: Optional[str], field_name: str) -> str:
    if raw is None or not str(raw).strip():
        raise ValueError(f"配置项 {field_name} 不能为空。")
    candidate = Path(os.path.expanduser(str(raw).strip()))
    if candidate.is_absolute():
        return str(candidate.resolve())
    return str((BASE_DIR / candidate).resolve())


def _display_path(path: str, base_dir: str) -> str:
    path_obj = Path(path).expanduser().resolve()
    base_obj = Path(base_dir).expanduser().resolve()
    try:
        return os.path.relpath(str(path_obj), str(base_obj))
    except ValueError:
        return str(path_obj)


def _parse_csv_list(raw: str) -> List[str]:
    return [part.strip() for part in str(raw).split(",") if part.strip()]


def _normalize_str_list(raw: Any, field_name: str) -> List[str]:
    if raw is None:
        raise ValueError(f"配置项 {field_name} 不能为空。")
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"配置项 {field_name} 必须是非空列表。")
    values = [str(item).strip() for item in raw if str(item).strip()]
    if not values:
        raise ValueError(f"配置项 {field_name} 不能为空。")
    return values


def _normalize_int_list(raw: Any, field_name: str) -> List[int]:
    values = _normalize_str_list(raw, field_name)
    try:
        return [int(item) for item in values]
    except ValueError as exc:
        raise ValueError(f"配置项 {field_name} 必须全部是整数。") from exc


def _validate_choice_list(values: List[str], field_name: str, allowed: set):
    unknown = [value for value in values if value not in allowed]
    if unknown:
        raise ValueError(
            f"配置项 {field_name} 包含不支持的取值: {unknown}. "
            f"允许的取值为: {sorted(allowed)}"
        )


def _resolve_device(requested: Optional[str]) -> str:
    candidate = str(requested or "auto").strip()
    if candidate == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if candidate.startswith("cuda") and not torch.cuda.is_available():
        LOGGER.warning("CUDA device '%s' requested but CUDA is unavailable. Falling back to cpu.", candidate)
        return "cpu"
    return candidate


CONFIG_SECTION = "DecodeGraphFromPairwiseCheckpoints"
DEFAULT_CONFIG_PATH = BASE_DIR / "config" / "Incident_Align_Method-decode_graph_from_pairwise_checkpoints-config.ini"


def load_config() -> DecodeConfig:
    config_path = DEFAULT_CONFIG_PATH
    if not config_path.is_file():
        raise FileNotFoundError(
            f"配置文件不存在: {config_path}. "
            "请在 config/ 目录下创建 Incident_Align_Method-decode_graph_from_pairwise_checkpoints-config.ini。"
        )

    parser = configparser.ConfigParser()
    parser.read(str(config_path), encoding="utf-8")
    if CONFIG_SECTION not in parser:
        raise KeyError(f"配置文件缺少 [{CONFIG_SECTION}] 段: {config_path}")

    section = parser[CONFIG_SECTION]

    batch_size = section.getint("batch_size", fallback=DEFAULT_BATCH_SIZE)
    if batch_size <= 0:
        raise ValueError("配置项 batch_size 必须大于 0。")

    threshold_grid_step = section.getfloat("threshold_grid_step", fallback=DEFAULT_THRESHOLD_GRID_STEP)
    if threshold_grid_step <= 0 or threshold_grid_step > 1:
        raise ValueError("配置项 threshold_grid_step 必须在 (0, 1] 范围内。")

    edge_rule_grid = _parse_csv_list(section.get("edge_rule_grid", ",".join(DEFAULT_EDGE_RULE_GRID)))
    edge_rule_grid = _normalize_str_list(edge_rule_grid, "edge_rule_grid")
    _validate_choice_list(edge_rule_grid, "edge_rule_grid", SUPPORTED_EDGE_RULES)

    return DecodeConfig(
        prepared_dir=_resolve_project_path(section.get("prepared_dir"), "prepared_dir"),
        embeddings_dir=_resolve_project_path(section.get("embeddings_dir"), "embeddings_dir"),
        train_output_dir=_resolve_project_path(section.get("train_output_dir"), "train_output_dir"),
        output_dir=_resolve_project_path(section.get("output_dir"), "output_dir"),
        path_display_base=_resolve_project_path(section.get("path_display_base", "."), "path_display_base"),
        device=_resolve_device(section.get("device", "auto")),
        batch_size=batch_size,
        threshold_grid_step=threshold_grid_step,
        edge_rule_grid=edge_rule_grid,
        overwrite=section.getboolean("overwrite", fallback=False),
        config_path=str(config_path),
    )


def _load_training_summary(train_output_dir: str) -> Tuple[str, Dict[str, Any]]:
    summary_path = Path(train_output_dir) / "training_summary.json"
    if summary_path.exists() and not summary_path.is_file():
        raise FileNotFoundError(
            f"Expected a file at {summary_path}, but found a non-file object. "
            "Check your train_output_dir setting."
        )
    if not summary_path.is_file():
        raise FileNotFoundError(
            f"Missing training summary: {summary_path}. "
            "Make sure train_output_dir points to a completed pairwise training run."
        )
    summary = _load_json(str(summary_path))
    repeats = summary.get("repeats", [])
    if not isinstance(repeats, list) or not repeats:
        raise ValueError(
            f"Training summary is invalid: {summary_path}. "
            "Expected a non-empty 'repeats' list."
        )
    return str(summary_path), summary


def _validate_manifest_fields(manifest: Dict[str, Any], training_summary: Dict[str, Any]):
    manifest_text = manifest.get("text_fields", [])
    manifest_sim = manifest.get("sim_fields", [])
    summary_text = training_summary.get("text_fields", [])
    summary_sim = training_summary.get("sim_fields", [])
    if manifest_text != summary_text:
        raise ValueError(
            "Prepared data and training summary disagree on text_fields. "
            f"prepared={manifest_text} training={summary_text}"
        )
    if manifest_sim != summary_sim:
        raise ValueError(
            "Prepared data and training summary disagree on sim_fields. "
            f"prepared={manifest_sim} training={summary_sim}"
        )


def _group_prepared_repeats(prepared_dir: str, manifest: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    grouped = {}
    for repeat_entry in manifest["repeats"]:
        prepared = load_prepared_repeat(prepared_dir, repeat_entry)
        grouped[int(prepared["repeat_index"])] = prepared
    return grouped


def _repeat_dir(output_dir: str, repeat_idx: int) -> str:
    return str(Path(output_dir) / f"repeat_{int(repeat_idx):02d}")


def _repeat_output_paths(output_dir: str, repeat_idx: int) -> Dict[str, str]:
    repeat_dir = Path(_repeat_dir(output_dir, repeat_idx))
    return {
        "repeat_dir": str(repeat_dir),
        "metrics_dev": str(repeat_dir / "best_graph_metrics_dev.json"),
        "metrics_test": str(repeat_dir / "best_graph_metrics_test.json"),
        "pair_predictions": str(repeat_dir / "best_graph_pair_predictions_test.jsonl"),
        "clusters": str(repeat_dir / "best_graph_clusters_test.json"),
        "checkpoint_search_summary": str(repeat_dir / "checkpoint_search_summary.json"),
    }


def _best_model_dir(output_dir: str) -> str:
    return str(Path(output_dir) / "best_model")


def _best_model_output_paths(output_dir: str) -> Dict[str, str]:
    best_dir = Path(_best_model_dir(output_dir))
    return {
        "best_model_dir": str(best_dir),
        "model_selection": str(best_dir / "model_selection.json"),
        "metrics_dev": str(best_dir / "metrics_dev.json"),
        "metrics_test": str(best_dir / "metrics_test.json"),
        "clusters_test": str(best_dir / "clusters_test.json"),
        "pair_predictions_test": str(best_dir / "pair_predictions_test.jsonl"),
    }


def _config_signature(config: DecodeConfig) -> Dict[str, Any]:
    return {
        "prepared_dir": _display_path(config.prepared_dir, config.path_display_base),
        "embeddings_dir": _display_path(config.embeddings_dir, config.path_display_base),
        "train_output_dir": _display_path(config.train_output_dir, config.path_display_base),
        "output_dir": _display_path(config.output_dir, config.path_display_base),
        "path_display_base": _display_path(config.path_display_base, config.path_display_base),
        "device": config.device,
        "batch_size": int(config.batch_size),
        "threshold_grid_step": float(config.threshold_grid_step),
        "edge_rule_grid": list(config.edge_rule_grid),
    }


def _load_existing_repeat_result(config: DecodeConfig, repeat_idx: int) -> Optional[Dict[str, Any]]:
    output_dir = config.output_dir
    output_paths = _repeat_output_paths(output_dir, repeat_idx)
    if not all(os.path.exists(output_paths[key]) for key in output_paths if key != "repeat_dir"):
        return None
    run_config_path = Path(output_dir) / "run_config.json"
    if not run_config_path.is_file():
        return None

    cached_run_config = _load_json(str(run_config_path))
    expected_signature = _config_signature(config)
    cached_signature = {key: cached_run_config.get(key) for key in expected_signature}
    if cached_signature != expected_signature:
        return None

    cached_metrics_dev = _load_json(output_paths["metrics_dev"])
    cached_metrics_test = _load_json(output_paths["metrics_test"])
    cached_search_summary = _load_json(output_paths["checkpoint_search_summary"])
    best_checkpoint_raw = cached_search_summary["best_checkpoint"]["checkpoint"]
    best_checkpoint_abs = str((Path(config.path_display_base) / best_checkpoint_raw).resolve())
    return {
        "repeat_idx": int(cached_metrics_test["repeat_idx"]),
        "seed": int(cached_metrics_test["seed"]),
        "best_epoch": int(cached_search_summary["best_checkpoint"]["epoch"]),
        "best_checkpoint": best_checkpoint_abs,
        "best_dev_score": float(cached_search_summary["best_checkpoint"]["dev_best_graph_score"]),
        "best_config": cached_search_summary["best_checkpoint"]["best_config"],
        "dev_metrics": cached_metrics_dev["metrics"],
        "test_metrics": cached_metrics_test["metrics"],
        "files": {key: value for key, value in output_paths.items() if key != "repeat_dir"},
        "path_display_base": config.path_display_base,
        "status": "skipped_existing",
    }


def _is_best_model_complete(output_dir: str) -> bool:
    best_paths = _best_model_output_paths(output_dir)
    return all(os.path.exists(path) for key, path in best_paths.items() if key != "best_model_dir")


def _select_global_best_model(per_repeat_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not per_repeat_results:
        raise ValueError("Cannot select a global best model from an empty per_repeat_results list.")

    def _sort_key(result: Dict[str, Any]) -> Tuple[float, float, int, int]:
        return (
            float(result["best_dev_score"]),
            float(result["test_metrics"]["event_macro_f1_hungarian"]),
            -int(result["repeat_idx"]),
            -int(result["best_epoch"]),
        )

    return max(per_repeat_results, key=_sort_key)


def _copy_file(src: str, dst: str):
    dst_path = Path(dst)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def _display_paths_in_mapping(mapping: Dict[str, str], base_dir: str) -> Dict[str, str]:
    return {key: _display_path(value, base_dir) for key, value in mapping.items()}


def _materialize_global_best_model(output_dir: str, global_best: Dict[str, Any]) -> Dict[str, Any]:
    best_paths = _best_model_output_paths(output_dir)
    Path(best_paths["best_model_dir"]).mkdir(parents=True, exist_ok=True)
    base_dir = global_best["path_display_base"]

    _copy_file(global_best["files"]["metrics_dev"], best_paths["metrics_dev"])
    _copy_file(global_best["files"]["metrics_test"], best_paths["metrics_test"])
    _copy_file(global_best["files"]["clusters"], best_paths["clusters_test"])
    _copy_file(global_best["files"]["pair_predictions"], best_paths["pair_predictions_test"])

    model_selection = {
        "selection_policy": "global_best_on_dev_graph_score",
        "selection_metric": "dev_graph_score",
        "repeat_idx": int(global_best["repeat_idx"]),
        "seed": int(global_best["seed"]),
        "best_epoch": int(global_best["best_epoch"]),
        "best_checkpoint": _display_path(global_best["best_checkpoint"], base_dir),
        "best_dev_score": float(global_best["best_dev_score"]),
        "best_test_score": float(global_best["test_metrics"]["event_macro_f1_hungarian"]),
        "best_graph_config": global_best["best_config"],
        "path_display_base": _display_path(base_dir, base_dir),
        "source_files": _display_paths_in_mapping(
            {
                "metrics_dev": global_best["files"]["metrics_dev"],
                "metrics_test": global_best["files"]["metrics_test"],
                "clusters_test": global_best["files"]["clusters"],
                "pair_predictions_test": global_best["files"]["pair_predictions"],
            },
            base_dir,
        ),
        "materialized_files": _display_paths_in_mapping(
            {
                "metrics_dev": best_paths["metrics_dev"],
                "metrics_test": best_paths["metrics_test"],
                "clusters_test": best_paths["clusters_test"],
                "pair_predictions_test": best_paths["pair_predictions_test"],
            },
            base_dir,
        ),
    }
    _write_json(best_paths["model_selection"], model_selection)
    return {
        "selection_policy": model_selection["selection_policy"],
        "selection_metric": model_selection["selection_metric"],
        "repeat_idx": model_selection["repeat_idx"],
        "seed": model_selection["seed"],
        "best_epoch": model_selection["best_epoch"],
        "best_checkpoint": model_selection["best_checkpoint"],
        "best_dev_score": model_selection["best_dev_score"],
        "best_test_score": model_selection["best_test_score"],
        "best_graph_config": model_selection["best_graph_config"],
        "path_display_base": model_selection["path_display_base"],
        "best_model_dir": _display_path(best_paths["best_model_dir"], base_dir),
        "best_model_files": _display_paths_in_mapping(
            {
                "model_selection": best_paths["model_selection"],
                "metrics_dev": best_paths["metrics_dev"],
                "metrics_test": best_paths["metrics_test"],
                "clusters_test": best_paths["clusters_test"],
                "pair_predictions_test": best_paths["pair_predictions_test"],
            },
            base_dir,
        ),
    }


def _validate_checkpoint_against_repeat(
    checkpoint: Dict[str, Any],
    repeat_idx: int,
    seed: int,
    config: DecodeConfig,
    training_summary: Dict[str, Any],
):
    ckpt_repeat_idx = int(checkpoint.get("prepared_repeat_index", checkpoint.get("repeat_idx", -1)))
    if ckpt_repeat_idx != repeat_idx:
        raise ValueError(
            f"Checkpoint repeat mismatch for repeat {repeat_idx}: metadata says {ckpt_repeat_idx}. "
            "Check that train_output_dir matches your prepared_dir."
        )

    ckpt_seed = checkpoint.get("prepared_repeat_seed", checkpoint.get("seed"))
    if ckpt_seed is not None and int(ckpt_seed) != int(seed):
        raise ValueError(
            f"Checkpoint seed mismatch for repeat {repeat_idx}: expected {seed}, found {ckpt_seed}."
        )

    ckpt_prepared_dir = checkpoint.get("prepared_dir")
    if ckpt_prepared_dir and os.path.abspath(str(ckpt_prepared_dir)) != config.prepared_dir:
        raise ValueError(
            f"Checkpoint prepared_dir mismatch for repeat {repeat_idx}: "
            f"checkpoint={os.path.abspath(str(ckpt_prepared_dir))} config={config.prepared_dir}."
        )

    expected_text_fields = list(training_summary.get("text_fields", []))
    expected_sim_fields = list(training_summary.get("sim_fields", []))
    if checkpoint.get("text_fields") != expected_text_fields:
        raise ValueError(
            f"Checkpoint text_fields mismatch for repeat {repeat_idx}. "
            f"checkpoint={checkpoint.get('text_fields')} expected={expected_text_fields}"
        )
    if checkpoint.get("sim_fields") != expected_sim_fields:
        raise ValueError(
            f"Checkpoint sim_fields mismatch for repeat {repeat_idx}. "
            f"checkpoint={checkpoint.get('sim_fields')} expected={expected_sim_fields}"
        )

    summary_arch = training_summary.get("architecture_mode")
    ckpt_arch = checkpoint.get("architecture_mode", checkpoint.get("model_config", {}).get("architecture_mode"))
    if summary_arch is not None and ckpt_arch is not None and summary_arch != ckpt_arch:
        raise ValueError(
            f"Checkpoint architecture_mode mismatch for repeat {repeat_idx}. "
            f"checkpoint={ckpt_arch} training_summary={summary_arch}"
        )

    summary_paper_fields = training_summary.get("paper_text_fields")
    ckpt_paper_fields = checkpoint.get("paper_text_fields", checkpoint.get("model_config", {}).get("paper_text_fields"))
    if summary_paper_fields is not None and ckpt_paper_fields is not None and summary_paper_fields != ckpt_paper_fields:
        raise ValueError(
            f"Checkpoint paper_text_fields mismatch for repeat {repeat_idx}. "
            f"checkpoint={ckpt_paper_fields} training_summary={summary_paper_fields}"
        )


def load_and_validate_inputs(config: DecodeConfig) -> LoadedArtifacts:
    if not Path(config.prepared_dir).is_dir():
        raise FileNotFoundError(
            f"Prepared directory not found: {config.prepared_dir}. "
            "Set prepared_dir to the folder containing manifest.json and repeat_* subdirectories."
        )
    if not Path(config.embeddings_dir).is_dir():
        raise FileNotFoundError(
            f"Embeddings directory not found: {config.embeddings_dir}. "
            "Set embeddings_dir to the folder containing case_ids.txt and embedding arrays."
        )
    if not Path(config.train_output_dir).is_dir():
        raise FileNotFoundError(
            f"Training output directory not found: {config.train_output_dir}. "
            "Set train_output_dir to the folder containing training_summary.json."
        )

    manifest = load_manifest(config.prepared_dir)
    training_summary_path, training_summary = _load_training_summary(config.train_output_dir)
    _validate_manifest_fields(manifest, training_summary)

    cases_file = manifest.get("sources", {}).get("cases_file")
    if not cases_file:
        raise ValueError(
            "Prepared manifest is missing sources.cases_file. "
            "Rebuild prepared data or fix the manifest before decoding."
        )
    if not Path(cases_file).is_file():
        raise FileNotFoundError(
            f"Manifest points to a missing cases_file: {cases_file}. "
            "Update the manifest or place the file at the expected location."
        )

    prepared_by_repeat = _group_prepared_repeats(config.prepared_dir, manifest)
    case2cats = load_cases_minimal(cases_file)
    text_fields = list(training_summary["text_fields"])
    sim_fields = list(training_summary["sim_fields"])
    needed_fields = sorted(set(text_fields) | set(sim_fields))
    store = EmbeddingsStore(config.embeddings_dir, needed_fields)

    for repeat_info in training_summary["repeats"]:
        repeat_idx = int(repeat_info["repeat_idx"])
        seed = int(repeat_info["seed"])
        prepared = prepared_by_repeat.get(repeat_idx)
        if prepared is None:
            raise FileNotFoundError(
                f"Prepared repeat {repeat_idx} is missing in {config.prepared_dir}. "
                "Make sure prepared_dir and train_output_dir come from the same experiment."
            )

        split_meta = prepared.get("split_meta", {})
        prepared_seed = split_meta.get("repeat_seed")
        if prepared_seed is not None and int(prepared_seed) != seed:
            raise ValueError(
                f"Repeat seed mismatch for repeat {repeat_idx}: training summary says {seed}, "
                f"prepared split metadata says {prepared_seed}."
            )

        checkpoints = repeat_info.get("checkpoints", [])
        if not checkpoints:
            raise ValueError(
                f"Repeat {repeat_idx} in {training_summary_path} has no checkpoints."
            )
        for ckpt_entry in checkpoints:
            ckpt_path = ckpt_entry.get("checkpoint")
            epoch = ckpt_entry.get("epoch")
            if not ckpt_path:
                raise ValueError(
                    f"Repeat {repeat_idx} checkpoint entry is missing 'checkpoint' for epoch {epoch}."
                )
            if not Path(ckpt_path).is_file():
                raise FileNotFoundError(
                    f"Missing checkpoint for repeat {repeat_idx}, epoch {epoch}: {ckpt_path}."
                )

    return LoadedArtifacts(
        config=config,
        device=config.device,
        manifest=manifest,
        training_summary_path=training_summary_path,
        training_summary=training_summary,
        prepared_by_repeat=prepared_by_repeat,
        case2cats=case2cats,
        store=store,
        text_fields=text_fields,
        sim_fields=sim_fields,
    )


def search_best_checkpoint_for_repeat(
    artifacts: LoadedArtifacts,
    repeat_info: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    config = artifacts.config
    repeat_idx = int(repeat_info["repeat_idx"])
    seed = int(repeat_info["seed"])
    prepared = artifacts.prepared_by_repeat[repeat_idx]
    repeat_paths = prepared["paths"]
    repeat_counts = prepared.get("counts", {})
    split_meta = prepared["split_meta"]

    dev_count = int(repeat_counts.get("dev_pairs", 0))
    test_count = int(repeat_counts.get("test_pairs", 0))
    if dev_count == 0 or test_count == 0:
        raise ValueError(
            f"Prepared repeat {repeat_idx} has empty dev/test counts: dev={dev_count}, test={test_count}."
        )

    checkpoints = sorted(repeat_info.get("checkpoints", []), key=lambda x: int(x["epoch"]))
    search_records: List[Dict[str, Any]] = []
    best_for_repeat = {
        "repeat_idx": repeat_idx,
        "seed": seed,
        "best_epoch": None,
        "best_checkpoint": None,
        "best_dev_score": -1.0,
        "best_config": None,
    }

    LOGGER.info("=== Repeat %s/%s seed=%s ===", repeat_idx, len(artifacts.training_summary["repeats"]), seed)

    for ckpt_entry in checkpoints:
        epoch = int(ckpt_entry["epoch"])
        ckpt_path = ckpt_entry["checkpoint"]
        model, checkpoint, _ = load_checkpoint(ckpt_path, device=artifacts.device)
        _validate_checkpoint_against_repeat(checkpoint, repeat_idx, seed, config, artifacts.training_summary)
        architecture_mode = checkpoint.get("architecture_mode", checkpoint.get("model_config", {}).get("architecture_mode"))
        paper_text_fields = checkpoint.get("paper_text_fields", checkpoint.get("model_config", {}).get("paper_text_fields"))

        probs, y_true, qc_list = predict_probs(
            model=model,
            pair_source=repeat_paths["dev_pairs"],
            store=artifacts.store,
            case2cats=artifacts.case2cats,
            text_fields=checkpoint["text_fields"],
            device=artifacts.device,
            desc=f"Dev Predict repeat={repeat_idx} ep={epoch}",
            batch_size=config.batch_size,
            num_pairs=dev_count,
            architecture_mode=architecture_mode,
            paper_text_fields=paper_text_fields,
        )
        best_cfg = grid_search_on_dev(
            probs=probs,
            y_true=y_true,
            qc_list=qc_list,
            dev_case_ids=split_meta["dev_case_ids"],
            gold_clusters_dev=split_meta["gold_clusters_dev"],
            edge_rule_grid=config.edge_rule_grid,
            threshold_grid_step=config.threshold_grid_step,
        )
        search_record = {
            "epoch": epoch,
            "checkpoint": ckpt_path,
            "dev_best_graph_score": float(best_cfg["score"]),
            "best_config": best_cfg,
        }
        search_records.append(search_record)
        LOGGER.info(
            "[Repeat %s Epoch %s] dev_best_graph_score=%.4f | edge_rule=%s thr=%.2f",
            repeat_idx,
            epoch,
            best_cfg["score"],
            best_cfg["edge_rule"],
            best_cfg["threshold"],
        )

        if best_cfg["score"] > best_for_repeat["best_dev_score"]:
            best_for_repeat["best_dev_score"] = float(best_cfg["score"])
            best_for_repeat["best_epoch"] = epoch
            best_for_repeat["best_checkpoint"] = ckpt_path
            best_for_repeat["best_config"] = best_cfg

    if best_for_repeat["best_checkpoint"] is None or best_for_repeat["best_config"] is None:
        raise ValueError(f"No valid checkpoint search result was produced for repeat {repeat_idx}.")

    return best_for_repeat, search_records, prepared


def evaluate_repeat_with_best_checkpoint(
    artifacts: LoadedArtifacts,
    repeat_info: Dict[str, Any],
    prepared: Dict[str, Any],
    best_for_repeat: Dict[str, Any],
    search_records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    config = artifacts.config
    repeat_idx = int(repeat_info["repeat_idx"])
    seed = int(repeat_info["seed"])
    output_paths = _repeat_output_paths(config.output_dir, repeat_idx)
    path_display_base = config.path_display_base

    Path(output_paths["repeat_dir"]).mkdir(parents=True, exist_ok=True)

    repeat_paths = prepared["paths"]
    repeat_counts = prepared.get("counts", {})
    split_meta = prepared["split_meta"]
    dev_count = int(repeat_counts.get("dev_pairs", 0))
    test_count = int(repeat_counts.get("test_pairs", 0))

    model, checkpoint, _ = load_checkpoint(best_for_repeat["best_checkpoint"], device=artifacts.device)
    _validate_checkpoint_against_repeat(checkpoint, repeat_idx, seed, config, artifacts.training_summary)
    architecture_mode = checkpoint.get("architecture_mode", checkpoint.get("model_config", {}).get("architecture_mode"))
    paper_text_fields = checkpoint.get("paper_text_fields", checkpoint.get("model_config", {}).get("paper_text_fields"))

    dev_metrics, _dev_clusters, _ = evaluate_with_best_config(
        model=model,
        pair_source=repeat_paths["dev_pairs"],
        store=artifacts.store,
        case2cats=artifacts.case2cats,
        text_fields=checkpoint["text_fields"],
        split_case_ids=split_meta["dev_case_ids"],
        gold_clusters=split_meta["gold_clusters_dev"],
        device=artifacts.device,
        best_cfg=best_for_repeat["best_config"],
        batch_size=config.batch_size,
        num_pairs=dev_count,
        architecture_mode=architecture_mode,
        paper_text_fields=paper_text_fields,
    )
    test_metrics, test_pred_clusters, test_dump = evaluate_with_best_config(
        model=model,
        pair_source=repeat_paths["test_pairs"],
        store=artifacts.store,
        case2cats=artifacts.case2cats,
        text_fields=checkpoint["text_fields"],
        split_case_ids=split_meta["test_case_ids"],
        gold_clusters=split_meta["gold_clusters_test"],
        device=artifacts.device,
        best_cfg=best_for_repeat["best_config"],
        batch_size=config.batch_size,
        num_pairs=test_count,
        architecture_mode=architecture_mode,
        paper_text_fields=paper_text_fields,
    )

    probs, y_true, qc_list, y_pred = test_dump
    with Path(output_paths["pair_predictions"]).open("w", encoding="utf-8") as f:
        thr = float(best_for_repeat["best_config"]["threshold"])
        for (q, c), p, yp, yt in zip(qc_list, probs.tolist(), y_pred.tolist(), y_true.tolist()):
            f.write(
                json.dumps(
                    {
                        "query_case_id": norm_id(q),
                        "cand_case_id": norm_id(c),
                        "prob": float(p),
                        "pred": int(yp),
                        "label": int(yt),
                        "threshold": thr,
                        "repeat_idx": repeat_idx,
                        "seed": seed,
                        "best_epoch": int(best_for_repeat["best_epoch"]),
                        "best_checkpoint": _display_path(best_for_repeat["best_checkpoint"], path_display_base),
                        "best_graph_config": best_for_repeat["best_config"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    _write_json(
        output_paths["metrics_dev"],
        {
            "repeat_idx": repeat_idx,
            "seed": seed,
            "best_epoch": int(best_for_repeat["best_epoch"]),
            "best_checkpoint": _display_path(best_for_repeat["best_checkpoint"], path_display_base),
            "best_dev_graph_score": float(best_for_repeat["best_dev_score"]),
            "best_graph_config": best_for_repeat["best_config"],
        "metrics": dev_metrics,
        },
    )
    _write_json(
        output_paths["metrics_test"],
        {
            "repeat_idx": repeat_idx,
            "seed": seed,
            "best_epoch": int(best_for_repeat["best_epoch"]),
            "best_checkpoint": _display_path(best_for_repeat["best_checkpoint"], path_display_base),
            "best_graph_config": best_for_repeat["best_config"],
            "metrics": test_metrics,
        },
    )
    _write_json(
        output_paths["clusters"],
        {
            "repeat_idx": repeat_idx,
            "seed": seed,
            "best_epoch": int(best_for_repeat["best_epoch"]),
            "best_checkpoint": _display_path(best_for_repeat["best_checkpoint"], path_display_base),
            "best_graph_config": best_for_repeat["best_config"],
            "clusters": test_pred_clusters,
        },
    )
    _write_json(
        output_paths["checkpoint_search_summary"],
        {
            "repeat_idx": repeat_idx,
            "seed": seed,
            "num_checkpoints": len(search_records),
            "best_checkpoint": {
                "epoch": int(best_for_repeat["best_epoch"]),
                "checkpoint": _display_path(best_for_repeat["best_checkpoint"], path_display_base),
                "dev_best_graph_score": float(best_for_repeat["best_dev_score"]),
                "best_config": best_for_repeat["best_config"],
            },
            "checkpoint_search_results": [
                {
                    **record,
                    "checkpoint": _display_path(record["checkpoint"], path_display_base),
                }
                for record in search_records
            ],
        },
    )

    LOGGER.info("[Save] repeat %s outputs -> %s", repeat_idx, output_paths["repeat_dir"])
    LOGGER.info(
        "[Result] TEST event_macro_f1_hungarian=%.4f | pair_macro_f1=%.4f",
        test_metrics["event_macro_f1_hungarian"],
        test_metrics["pair_macro_f1"],
    )

    return {
        "repeat_idx": repeat_idx,
        "seed": seed,
        "best_epoch": int(best_for_repeat["best_epoch"]),
        "best_checkpoint": best_for_repeat["best_checkpoint"],
        "best_dev_score": float(best_for_repeat["best_dev_score"]),
        "best_config": best_for_repeat["best_config"],
        "dev_metrics": dev_metrics,
        "test_metrics": test_metrics,
        "files": {key: value for key, value in output_paths.items() if key != "repeat_dir"},
        "path_display_base": path_display_base,
        "status": "computed",
    }


def _public_repeat_result(result: Dict[str, Any], base_dir: str) -> Dict[str, Any]:
    return {
        "repeat_idx": int(result["repeat_idx"]),
        "seed": int(result["seed"]),
        "best_epoch": int(result["best_epoch"]),
        "best_checkpoint": _display_path(result["best_checkpoint"], base_dir),
        "best_dev_score": float(result["best_dev_score"]),
        "best_config": result["best_config"],
        "dev_metrics": result["dev_metrics"],
        "test_metrics": result["test_metrics"],
        "files": _display_paths_in_mapping(result["files"], base_dir),
        "path_display_base": _display_path(base_dir, base_dir),
        "status": result["status"],
    }


def build_global_summary(
    artifacts: LoadedArtifacts,
    per_repeat_results: List[Dict[str, Any]],
    global_best_model: Dict[str, Any],
) -> Dict[str, Any]:
    def _metric_vals(key):
        return [r["test_metrics"].get(key, 0.0) for r in per_repeat_results]

    test_f1_values = _metric_vals("event_macro_f1_hungarian")
    dev_scores = [r["best_dev_score"] for r in per_repeat_results]
    base_dir = artifacts.config.path_display_base

    def _mean_std(values):
        arr = np.array(values, dtype=np.float64)
        return float(np.mean(arr)) if len(arr) else 0.0, float(np.std(arr)) if len(arr) else 0.0

    # New metrics
    b3_f1_vals = _metric_vals("b_cubed_f1")
    ari_vals = _metric_vals("ari")
    ipf_vals = _metric_vals("induced_pair_f1")
    pair_macro_vals = _metric_vals("pair_macro_f1")

    b3_mean, b3_std = _mean_std(b3_f1_vals)
    ari_mean, ari_std = _mean_std(ari_vals)
    ipf_mean, ipf_std = _mean_std(ipf_vals)
    pair_mean, pair_std = _mean_std(pair_macro_vals)

    return {
        "script": _display_path(__file__, base_dir),
        "train_summary": _display_path(artifacts.training_summary_path, base_dir),
        "prepared_manifest": _display_path(os.path.join(artifacts.config.prepared_dir, "manifest.json"), base_dir),
        "config": {
            "prepared_dir": _display_path(artifacts.config.prepared_dir, base_dir),
            "embeddings_dir": _display_path(artifacts.config.embeddings_dir, base_dir),
            "train_output_dir": _display_path(artifacts.config.train_output_dir, base_dir),
            "output_dir": _display_path(artifacts.config.output_dir, base_dir),
            "path_display_base": _display_path(base_dir, base_dir),
            "device": artifacts.device,
            "batch_size": artifacts.config.batch_size,
            "threshold_grid_step": artifacts.config.threshold_grid_step,
            "edge_rule_grid": artifacts.config.edge_rule_grid,
            "overwrite": artifacts.config.overwrite,
        },
        "num_repeats": len(per_repeat_results),
        "test_metrics_across_repeats": {
            "event_macro_f1_hungarian": {"mean": _mean_std(test_f1_values)[0], "std": _mean_std(test_f1_values)[1]},
            "b_cubed_f1": {"mean": b3_mean, "std": b3_std},
            "ari": {"mean": ari_mean, "std": ari_std},
            "induced_pair_f1": {"mean": ipf_mean, "std": ipf_std},
            "pair_macro_f1": {"mean": pair_mean, "std": pair_std},
        },
        "dev_best_graph_score": {"mean": _mean_std(dev_scores)[0], "std": _mean_std(dev_scores)[1]},
        "global_best_model": global_best_model,
        "repeats": [_public_repeat_result(result, base_dir) for result in per_repeat_results],
    }


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    config = load_config()
    artifacts = load_and_validate_inputs(config)

    LOGGER.info("[Info] device=%s | scipy_hungarian=%s", artifacts.device, SCIPY_OK)
    LOGGER.info("[Prepared] dir=%s repeats=%s", config.prepared_dir, len(artifacts.training_summary["repeats"]))
    LOGGER.info("[Train] summary=%s", artifacts.training_summary_path)
    LOGGER.info("[Embeddings] loaded %s fields", len(set(artifacts.text_fields) | set(artifacts.sim_fields)))
    LOGGER.info("[Output] dir=%s", config.output_dir)
    if not config.overwrite and not _is_best_model_complete(config.output_dir):
        LOGGER.info("[Info] best_model artifacts are incomplete or missing; global-best materialization will be rebuilt.")

    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    run_config_path = str(Path(config.output_dir) / "run_config.json")
    summary_path = str(Path(config.output_dir) / "summary.json")
    best_model_paths = _best_model_output_paths(config.output_dir)

    per_repeat_results = []
    for repeat_info in artifacts.training_summary["repeats"]:
        repeat_idx = int(repeat_info["repeat_idx"])
        if not config.overwrite:
            cached_result = _load_existing_repeat_result(config, repeat_idx)
            if cached_result is not None:
                LOGGER.info("[Skip] repeat %s already has complete outputs in %s", repeat_idx, _repeat_dir(config.output_dir, repeat_idx))
                per_repeat_results.append(cached_result)
                continue
        best_for_repeat, search_records, prepared = search_best_checkpoint_for_repeat(artifacts, repeat_info)
        repeat_result = evaluate_repeat_with_best_checkpoint(
            artifacts=artifacts,
            repeat_info=repeat_info,
            prepared=prepared,
            best_for_repeat=best_for_repeat,
            search_records=search_records,
        )
        per_repeat_results.append(repeat_result)

    _write_json(
        run_config_path,
        {
            "config_path": _display_path(config.config_path, config.path_display_base),
            **_config_signature(config),
            "prepared_dir": _display_path(config.prepared_dir, config.path_display_base),
            "embeddings_dir": _display_path(config.embeddings_dir, config.path_display_base),
            "train_output_dir": _display_path(config.train_output_dir, config.path_display_base),
            "output_dir": _display_path(config.output_dir, config.path_display_base),
            "path_display_base": _display_path(config.path_display_base, config.path_display_base),
            "device": artifacts.device,
            "overwrite": config.overwrite,
            "global_best_selection_policy": "global_best_on_dev_graph_score",
            "global_best_selection_metric": "dev_graph_score",
        },
    )

    global_best_result = _select_global_best_model(per_repeat_results)
    global_best_model = _materialize_global_best_model(config.output_dir, global_best_result)
    summary = build_global_summary(artifacts, per_repeat_results, global_best_model)
    _write_json(summary_path, summary)
    LOGGER.info("[Done] graph decode summary -> %s", summary_path)
    LOGGER.info(
        "[GlobalBest] repeat=%s epoch=%s test_event_macro_f1_hungarian=%.4f -> %s",
        global_best_model["repeat_idx"],
        global_best_model["best_epoch"],
        global_best_model["best_test_score"],
        best_model_paths["best_model_dir"],
    )


if __name__ == "__main__":
    main()
