#!/usr/bin/env python3
"""Unified clustering evaluation entrypoint for Event Align results."""

import argparse
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment

MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parents[1]
DEFAULT_PRED_FILE = PROJECT_ROOT / "outputs" / "Incident_Align_Evaluation" / "baseline_logistic_regression_cv" / "chinese_pred_clusters.json"
DEFAULT_TRUE_FILE = PROJECT_ROOT / "data" / "eval_structure.json"


@dataclass
class EvaluationConfig:
    pred_file: str
    true_file: str
    output_file: Optional[str] = None
    use_ari: bool = False
    use_nmi: bool = True


@dataclass
class PredInputResolution:
    pred_source_type: str
    resolved_pred_clusters_file: str
    pred_clusters: List[Dict[str, Any]]
    best_model_selection_file: Optional[str] = None


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def norm_id(value: Any) -> str:
    return str(value).strip()


def resolve_path(file_path: str) -> Path:
    return Path(file_path).expanduser().resolve()


def display_path(file_path: str) -> str:
    path = resolve_path(file_path)
    try:
        rel = os.path.relpath(str(path), str(Path.cwd()))
    except ValueError:
        return str(path)
    return rel


def load_json_file(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_cluster_file(file_path: str) -> List[Dict[str, Any]]:
    path = resolve_path(file_path)
    logging.info("Reading clusters from %s", path)
    if path.suffix == ".json":
        data = load_json_file(path)
        if isinstance(data, dict) and "events" in data:
            return data["events"]
        if isinstance(data, dict) and "clusters" in data:
            return data["clusters"]
        if isinstance(data, list):
            return data
        return [data] if data else []

    clusters: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                clusters.append(json.loads(line))
    return clusters


def _resolve_best_model_clusters_file(selection_path: Path, data: Dict[str, Any]) -> Path:
    materialized_files = data.get("materialized_files", {})
    raw_clusters_path = materialized_files.get("clusters_test")
    if raw_clusters_path:
        clusters_path = Path(str(raw_clusters_path)).expanduser()
        if not clusters_path.is_absolute():
            clusters_path = (selection_path.parent / clusters_path).resolve()
    else:
        clusters_path = (selection_path.parent / "clusters_test.json").resolve()
    return clusters_path


def resolve_pred_input(pred_file: str) -> PredInputResolution:
    pred_path = resolve_path(pred_file)
    logging.info("Resolving prediction input from %s", pred_path)

    if pred_path.suffix != ".json":
        return PredInputResolution(
            pred_source_type="clusters_list",
            resolved_pred_clusters_file=str(pred_path),
            pred_clusters=read_cluster_file(str(pred_path)),
        )

    data = load_json_file(pred_path)
    if isinstance(data, list):
        return PredInputResolution(
            pred_source_type="clusters_list",
            resolved_pred_clusters_file=str(pred_path),
            pred_clusters=data,
        )

    if isinstance(data, dict) and "clusters" in data:
        clusters = data.get("clusters", [])
        if not isinstance(clusters, list):
            raise ValueError(f"'clusters' must be a list in prediction file: {pred_path}")
        return PredInputResolution(
            pred_source_type="wrapped_clusters",
            resolved_pred_clusters_file=str(pred_path),
            pred_clusters=clusters,
        )

    if isinstance(data, dict) and (
        "selection_policy" in data or
        "materialized_files" in data or
        "source_files" in data
    ):
        clusters_path = _resolve_best_model_clusters_file(pred_path, data)
        if not clusters_path.is_file():
            raise FileNotFoundError(
                f"Resolved clusters_test.json not found for best_model selection file: {clusters_path}"
            )
        clusters = read_cluster_file(str(clusters_path))
        return PredInputResolution(
            pred_source_type="best_model_selection",
            resolved_pred_clusters_file=str(clusters_path),
            pred_clusters=clusters,
            best_model_selection_file=str(pred_path),
        )

    if isinstance(data, dict) and "events" in data:
        raise ValueError(
            f"Prediction file looks like a gold eval_structure.json file, not a predicted clusters file: {pred_path}"
        )

    raise ValueError(f"Unsupported prediction file format: {pred_path}")


def _normalize_cluster(cluster: Any, idx: int) -> Tuple[str, set]:
    """Handle both list-of-lists and dict-with-incident_id formats."""
    if isinstance(cluster, list):
        # decode 输出的 list-of-lists 格式
        incident_id = str(idx + 1)
        case_ids = set(norm_id(c) for c in cluster)
    elif isinstance(cluster, dict):
        incident_id = norm_id(cluster.get("incident_id", str(idx + 1)))
        case_ids = set(norm_id(c) for c in cluster.get("cases", cluster.get("ids", [])))
    else:
        incident_id = str(idx + 1)
        case_ids = set()
    return incident_id, case_ids


def prepare_cluster_sets(
    pred_clusters: List[Any],
    true_clusters: List[Any],
) -> Tuple[Dict[str, set], Dict[str, set]]:
    pred_sets = {}
    for i, cluster in enumerate(pred_clusters):
        incident_id, case_ids = _normalize_cluster(cluster, i)
        pred_sets[incident_id] = case_ids

    true_sets = {}
    for i, cluster in enumerate(true_clusters):
        incident_id, case_ids = _normalize_cluster(cluster, i)
        true_sets[incident_id] = case_ids

    return pred_sets, true_sets


def create_label_mappings(
    pred_sets: Dict[str, set],
    true_sets: Dict[str, set],
) -> Tuple[np.ndarray, np.ndarray]:
    all_samples = set().union(*pred_sets.values(), *true_sets.values()) if (pred_sets or true_sets) else set()
    pred_labels = {sample: "unclustered" for sample in all_samples}
    true_labels = {sample: "unclustered" for sample in all_samples}

    for pred_id, samples in pred_sets.items():
        for sample in samples:
            pred_labels[sample] = pred_id
    for true_id, samples in true_sets.items():
        for sample in samples:
            true_labels[sample] = true_id

    sample_list = sorted(all_samples)
    unique_pred = sorted(set(pred_labels.values()))
    unique_true = sorted(set(true_labels.values()))
    pred_id_map = {value: idx for idx, value in enumerate(unique_pred)}
    true_id_map = {value: idx for idx, value in enumerate(unique_true)}

    y_pred = np.array([pred_id_map[pred_labels[sample]] for sample in sample_list], dtype=np.int32)
    y_true = np.array([true_id_map[true_labels[sample]] for sample in sample_list], dtype=np.int32)
    return y_true, y_pred


def calculate_cluster_accuracy(pred_sets: Dict[str, set], true_sets: Dict[str, set]) -> Dict[str, Any]:
    pred_ids = list(pred_sets.keys())
    true_ids = list(true_sets.keys())
    dim = max(len(pred_ids), len(true_ids))
    cost_matrix = np.zeros((dim, dim), dtype=int)

    for i, pred_id in enumerate(pred_ids):
        for j, true_id in enumerate(true_ids):
            cost_matrix[i, j] = -len(pred_sets[pred_id] & true_sets[true_id])

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    total_correct = 0
    for i, j in zip(row_ind, col_ind):
        if i < len(pred_ids) and j < len(true_ids):
            total_correct += len(pred_sets[pred_ids[i]] & true_sets[true_ids[j]])

    total_samples = len(set().union(*pred_sets.values(), *true_sets.values())) if (pred_sets or true_sets) else 0
    accuracy = total_correct / total_samples if total_samples else 0.0
    return {"accuracy": accuracy, "correct": total_correct, "total": total_samples}


def calculate_macro_accuracy(pred_sets: Dict[str, set], true_sets: Dict[str, set]) -> Dict[str, float]:
    true_accuracies = []
    for true_set in true_sets.values():
        max_intersection = max((len(pred_set & true_set) for pred_set in pred_sets.values()), default=0)
        true_accuracies.append(max_intersection / len(true_set) if true_set else 0.0)
    accuracy = sum(true_accuracies) / len(true_accuracies) if true_accuracies else 0.0
    return {"accuracy": accuracy}


def calculate_standardized_metrics(y_true: np.ndarray, y_pred: np.ndarray, config: EvaluationConfig) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    if config.use_ari:
        metrics["ari"] = adjusted_rand_score(y_true, y_pred) if len(y_true) else 0.0
    if config.use_nmi:
        metrics["nmi"] = normalized_mutual_info_score(y_true, y_pred) if len(y_true) else 0.0
    return metrics


def greedy_matching_fallback(pred_sets: Dict[str, set], true_sets: Dict[str, set], objective: str = "f1") -> Dict[str, Any]:
    pred_ids = list(pred_sets.keys())
    true_ids = list(true_sets.keys())
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
            else:
                prec = inter / len(pred_cases) if pred_cases else 0.0
                rec = inter / len(true_cases) if true_cases else 0.0
                score = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
            matches.append((score, pred_id, true_id))

    matches.sort(reverse=True)
    matched_pred = set()
    matched_true = set()
    final_matches = []
    for _, pred_id, true_id in matches:
        if pred_id not in matched_pred and true_id not in matched_true:
            final_matches.append((pred_id, true_id))
            matched_pred.add(pred_id)
            matched_true.add(true_id)

    return compute_macro_scores_from_matches(pred_sets, true_sets, final_matches, objective, method="greedy_fallback")


def compute_macro_scores_from_matches(
    pred_sets: Dict[str, set],
    true_sets: Dict[str, set],
    matches: List[Tuple[str, str]],
    objective: str,
    method: str,
) -> Dict[str, Any]:
    pred_ids = list(pred_sets.keys())
    true_ids = list(true_sets.keys())
    pred_to_true = {pid: None for pid in pred_ids}
    true_to_pred = {tid: None for tid in true_ids}
    for pid, tid in matches:
        pred_to_true[pid] = tid
        true_to_pred[tid] = pid

    pred_precisions = []
    pred_f1s = []
    for pid in pred_ids:
        pred_cases = pred_sets[pid]
        true_id = pred_to_true[pid]
        if true_id is None or not pred_cases:
            pred_precisions.append(0.0)
            pred_f1s.append(0.0)
            continue
        true_cases = true_sets[true_id]
        inter = len(pred_cases & true_cases)
        prec = inter / len(pred_cases) if pred_cases else 0.0
        rec = inter / len(true_cases) if true_cases else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        pred_precisions.append(prec)
        pred_f1s.append(f1)

    true_recalls = []
    true_f1s = []
    for tid in true_ids:
        true_cases = true_sets[tid]
        pred_id = true_to_pred[tid]
        if pred_id is None or not true_cases:
            true_recalls.append(0.0)
            true_f1s.append(0.0)
            continue
        pred_cases = pred_sets[pred_id]
        inter = len(pred_cases & true_cases)
        prec = inter / len(pred_cases) if pred_cases else 0.0
        rec = inter / len(true_cases) if true_cases else 0.0
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
        "matched_pairs": len(matches),
        "total_pred_clusters": len(pred_sets),
        "total_true_clusters": len(true_sets),
        "objective": objective,
        "method": method,
    }


def hungarian_matching_macro_metrics(pred_sets: Dict[str, set], true_sets: Dict[str, set], objective: str = "f1") -> Dict[str, Any]:
    pred_ids = list(pred_sets.keys())
    true_ids = list(true_sets.keys())
    rows, cols = len(pred_ids), len(true_ids)
    if rows == 0 or cols == 0:
        return compute_macro_scores_from_matches(pred_sets, true_sets, [], objective, method="hungarian")

    similarity = np.zeros((rows, cols), dtype=np.float64)
    for i, pred_id in enumerate(pred_ids):
        pred_cases = pred_sets[pred_id]
        for j, true_id in enumerate(true_ids):
            true_cases = true_sets[true_id]
            inter = len(pred_cases & true_cases)
            if inter == 0:
                continue
            if objective == "intersection":
                score = inter
            elif objective == "jaccard":
                score = inter / len(pred_cases | true_cases)
            else:
                prec = inter / len(pred_cases) if pred_cases else 0.0
                rec = inter / len(true_cases) if true_cases else 0.0
                score = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
            similarity[i, j] = score

    size = max(rows, cols)
    cost = np.zeros((size, size), dtype=np.float64)
    cost[:rows, :cols] = -similarity
    row_ind, col_ind = linear_sum_assignment(cost)

    matches = []
    for row, col in zip(row_ind, col_ind):
        if row < rows and col < cols and similarity[row, col] > 0:
            matches.append((pred_ids[row], true_ids[col]))
    return compute_macro_scores_from_matches(pred_sets, true_sets, matches, objective, method="hungarian")


def calculate_b3_f1(pred_sets: Dict[str, set], true_sets: Dict[str, set]) -> Dict[str, float]:
    pred_cluster_map = {}
    true_cluster_map = {}
    for cluster_id, cases in pred_sets.items():
        for case_id in cases:
            pred_cluster_map[case_id] = cluster_id
    for cluster_id, cases in true_sets.items():
        for case_id in cases:
            true_cluster_map[case_id] = cluster_id

    all_samples = set(pred_cluster_map.keys()) | set(true_cluster_map.keys())
    if not all_samples:
        return {"b3_precision": 0.0, "b3_recall": 0.0, "b3_f1": 0.0}

    precision_scores = []
    recall_scores = []
    for sample in all_samples:
        pred_cluster = pred_sets.get(pred_cluster_map.get(sample, ""), {sample})
        true_cluster = true_sets.get(true_cluster_map.get(sample, ""), {sample})
        intersection = len(pred_cluster & true_cluster)
        precision_scores.append(intersection / len(pred_cluster) if pred_cluster else 0.0)
        recall_scores.append(intersection / len(true_cluster) if true_cluster else 0.0)

    precision = float(sum(precision_scores) / len(precision_scores))
    recall = float(sum(recall_scores) / len(recall_scores))
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {"b3_precision": precision, "b3_recall": recall, "b3_f1": f1}


def calculate_basic_stats(pred_sets: Dict[str, set], true_sets: Dict[str, set]) -> Dict[str, Any]:
    pred_sizes = [len(cases) for cases in pred_sets.values()]
    true_sizes = [len(cases) for cases in true_sets.values()]
    return {
        "total_pred_clusters": len(pred_sets),
        "total_true_clusters": len(true_sets),
        "total_pred_cases": sum(pred_sizes),
        "total_true_cases": sum(true_sizes),
        "pred_singleton_ratio": (sum(1 for size in pred_sizes if size == 1) / len(pred_sizes)) if pred_sizes else 0.0,
        "true_singleton_ratio": (sum(1 for size in true_sizes if size == 1) / len(true_sizes)) if true_sizes else 0.0,
        "pred_avg_cluster_size": (sum(pred_sizes) / len(pred_sizes)) if pred_sizes else 0.0,
        "true_avg_cluster_size": (sum(true_sizes) / len(true_sizes)) if true_sizes else 0.0,
        "pred_max_cluster_size": max(pred_sizes) if pred_sizes else 0,
        "true_max_cluster_size": max(true_sizes) if true_sizes else 0,
    }


def compute_metrics(pred_clusters: List[Dict[str, Any]], true_clusters: List[Dict[str, Any]], config: EvaluationConfig) -> Dict[str, Any]:
    pred_sets, true_sets = prepare_cluster_sets(pred_clusters, true_clusters)
    y_true, y_pred = create_label_mappings(pred_sets, true_sets)
    matching = hungarian_matching_macro_metrics(pred_sets, true_sets, objective="f1")
    b3 = calculate_b3_f1(pred_sets, true_sets)

    return {
        "stats": calculate_basic_stats(pred_sets, true_sets),
        "cluster_accuracy": calculate_cluster_accuracy(pred_sets, true_sets),
        "macro": calculate_macro_accuracy(pred_sets, true_sets),
        "matching": matching,
        "b3": b3,
        "standardized": calculate_standardized_metrics(y_true, y_pred, config),
    }


def build_metrics_payload(
    metrics: Dict[str, Any],
    config: EvaluationConfig,
    pred_resolution: PredInputResolution,
) -> Dict[str, Any]:
    payload = {
        "pred_file": display_path(config.pred_file),
        "true_file": display_path(config.true_file),
        "resolved_pred_clusters_file": display_path(pred_resolution.resolved_pred_clusters_file),
        "pred_source_type": pred_resolution.pred_source_type,
        "stats": metrics["stats"],
        "cluster_accuracy": metrics["cluster_accuracy"],
        "macro": metrics["macro"],
        "hungarian_matching": metrics["matching"],
        "b3": metrics["b3"],
        "standardized": metrics["standardized"],
    }
    if pred_resolution.best_model_selection_file is not None:
        payload["best_model_selection_file"] = display_path(pred_resolution.best_model_selection_file)
    return payload


def print_evaluation_results(metrics: Dict[str, Any]) -> None:
    stats = metrics["stats"]
    matching = metrics["matching"]
    b3 = metrics["b3"]
    standardized = metrics["standardized"]
    cluster_accuracy = metrics["cluster_accuracy"]

    print("\n==================================================")
    print("Incident Align Evaluation")
    print("==================================================")
    print(f"Pred clusters: {stats['total_pred_clusters']}")
    print(f"True clusters: {stats['total_true_clusters']}")
    print(f"Hungarian Macro-F1: {matching['macro_f1']:.4f}")
    print(f"Hungarian Macro-P/R: {matching['macro_precision']:.4f} / {matching['macro_recall']:.4f}")
    print(f"B3 F1: {b3['b3_f1']:.4f}")
    print(f"Cluster Accuracy: {cluster_accuracy['accuracy']:.4f}")
    print(f"Macro Accuracy: {metrics['macro']['accuracy']:.4f}")
    if "nmi" in standardized:
        print(f"NMI: {standardized['nmi']:.4f}")
    if "ari" in standardized:
        print(f"ARI: {standardized['ari']:.4f}")


def resolve_output_file(config: EvaluationConfig) -> Optional[Path]:
    if config.output_file:
        return Path(config.output_file)
    pred_path = Path(config.pred_file)
    if pred_path.name == "clusters.json":
        return pred_path.with_name("metrics.json")
    return pred_path.parent / "metrics.json"


def evaluate_files(config: EvaluationConfig) -> Dict[str, Any]:
    pred_resolution = resolve_pred_input(config.pred_file)
    pred_clusters = pred_resolution.pred_clusters
    true_clusters = read_cluster_file(config.true_file)
    metrics = compute_metrics(pred_clusters, true_clusters, config)
    payload = build_metrics_payload(metrics, config, pred_resolution)

    output_path = resolve_output_file(config)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        logging.info("Saved metrics to %s", output_path)
    return payload


def parse_args() -> EvaluationConfig:
    parser = argparse.ArgumentParser(description="Evaluate predicted event clusters against eval_structure.json")
    parser.add_argument(
        "--pred_file",
        type=str,
        default=str(DEFAULT_PRED_FILE),
        help=(
            "Predicted clusters input. Supports baseline clusters.json, "
            "method best_model/clusters_test.json, or best_model/model_selection.json."
        ),
    )
    parser.add_argument(
        "--true_file",
        type=str,
        default=str(DEFAULT_TRUE_FILE),
        help="Ground-truth structure file in eval_structure.json format",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Metrics output path. Defaults to metrics.json next to pred_file.",
    )
    parser.add_argument("--use_ari", action="store_true", help="Also compute ARI.")
    parser.add_argument("--disable_nmi", action="store_true", help="Disable NMI computation.")
    args = parser.parse_args()
    return EvaluationConfig(
        pred_file=args.pred_file,
        true_file=args.true_file,
        output_file=args.output_file,
        use_ari=args.use_ari,
        use_nmi=not args.disable_nmi,
    )


def main() -> None:
    setup_logging()
    config = parse_args()
    payload = evaluate_files(config)
    print_evaluation_results(
        {
            "stats": payload["stats"],
            "cluster_accuracy": payload["cluster_accuracy"],
            "macro": payload["macro"],
            "matching": payload["hungarian_matching"],
            "b3": payload["b3"],
            "standardized": payload["standardized"],
        }
    )


if __name__ == "__main__":
    main()
