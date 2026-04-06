#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
from typing import Any, Dict, Iterable, Iterator, List, Optional

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from pairwise_data_io import CAT_FIELDS, TEXT_FIELDS_DEFAULT, count_jsonl, iter_jsonl, norm_id


ARCH_CURRENT = "current"
SUPPORTED_ARCHITECTURE_MODES = {ARCH_CURRENT}
PAPER_TEXT_FIELDS_DEFAULT = list(TEXT_FIELDS_DEFAULT)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class EmbeddingsStore:
    def __init__(self, embeddings_dir: str, fields: List[str]):
        self.embeddings_dir = embeddings_dir
        self.fields = fields

        ids_path = self._resolve_ids_path()
        with open(ids_path, "r", encoding="utf-8") as f:
            self.case_ids = [norm_id(x) for x in f.read().splitlines() if x.strip()]
        self.caseid2idx = {cid: i for i, cid in enumerate(self.case_ids)}

        self.emb: Dict[str, Any] = {}
        self.mask: Dict[str, Any] = {}
        self.dim: Dict[str, int] = {}

        for field in fields:
            emb_path, mask_path = self._resolve_embedding_files(field)
            emb_arr = np.load(emb_path, mmap_mode="r")
            mask_arr = np.load(mask_path, mmap_mode="r")
            if emb_arr.ndim != 2:
                raise ValueError(f"Expected 2D emb for field {field}, got shape {emb_arr.shape}")
            self.emb[field] = emb_arr
            self.mask[field] = mask_arr
            self.dim[field] = int(emb_arr.shape[1])

    def _resolve_ids_path(self) -> str:
        candidates = [
            os.path.join(self.embeddings_dir, "case_ids.txt"),
            os.path.join(self.embeddings_dir, "ids.txt"),
        ]
        for path in candidates:
            if os.path.exists(path):
                return path

        for root, _dirs, files in os.walk(self.embeddings_dir):
            for name in ["case_ids.txt", "ids.txt"]:
                if name in files:
                    return os.path.join(root, name)

        raise FileNotFoundError(f"Missing ids file under {self.embeddings_dir}")

    def _resolve_embedding_files(self, field: str):
        emb_names = [f"emb_{field}.npy", f"emb_{field}_full.npy"]
        mask_names = [f"valid_mask_{field}.npy", f"valid_mask_{field}_full.npy"]
        if field == "text":
            emb_names.extend(["emb_txt.npy", "emb_txt_full.npy"])
            mask_names.extend(["valid_mask_txt.npy", "valid_mask_txt_full.npy"])

        direct_candidates = []
        for emb_name, mask_name in zip(emb_names, mask_names):
            direct_candidates.extend(
                [
                    (os.path.join(self.embeddings_dir, emb_name), os.path.join(self.embeddings_dir, mask_name)),
                    (
                        os.path.join(self.embeddings_dir, field, "full", emb_name),
                        os.path.join(self.embeddings_dir, field, "full", mask_name),
                    ),
                    (
                        os.path.join(self.embeddings_dir, "elements", field, "full", emb_name),
                        os.path.join(self.embeddings_dir, "elements", field, "full", mask_name),
                    ),
                ]
            )

        for emb_path, mask_path in direct_candidates:
            if os.path.exists(emb_path) and os.path.exists(mask_path):
                return emb_path, mask_path

        emb_path = self._find_first_existing(emb_names)
        mask_path = self._find_first_existing(mask_names)
        if emb_path is None or mask_path is None:
            raise FileNotFoundError(
                f"Missing embedding files for field={field}: searched for {emb_names} and {mask_names} under {self.embeddings_dir}"
            )
        return emb_path, mask_path

    def _find_first_existing(self, target_names: List[str]) -> Optional[str]:
        for root, _dirs, files in os.walk(self.embeddings_dir):
            for name in target_names:
                if name in files:
                    return os.path.join(root, name)
        return None

    def get_idx(self, case_id: str):
        return self.caseid2idx.get(norm_id(case_id), None)

    def get_vec(self, field: str, idx: Optional[int]):
        valid = bool(self.mask[field][idx]) if idx is not None else False
        if not valid:
            return None, False
        vec = np.asarray(self.emb[field][idx], dtype=np.float32)
        return vec, True


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _resolve_architecture_mode(architecture_mode: Optional[str], model=None) -> str:
    if architecture_mode is not None:
        mode = str(architecture_mode)
    elif model is not None:
        mode = str(getattr(model, "architecture_mode", ARCH_CURRENT))
    else:
        mode = ARCH_CURRENT
    if mode not in SUPPORTED_ARCHITECTURE_MODES:
        raise ValueError(f"Unsupported architecture_mode={mode}")
    return mode


def _resolve_text_fields_config(paper_text_fields: Optional[List[str]]) -> List[str]:
    if paper_text_fields is None:
        return list(PAPER_TEXT_FIELDS_DEFAULT)
    out = [str(x).strip() for x in paper_text_fields if str(x).strip()]
    if not out:
        raise ValueError("paper_text_fields must not be empty")
    return out


def get_wide_feature_names(paper_text_fields: Optional[List[str]] = None):
    fields = _resolve_text_fields_config(paper_text_fields)
    eq_cols = [f"eq_{field}" for field in CAT_FIELDS]
    sim_cols = [f"sim_{field}" for field in fields]
    return eq_cols + sim_cols


def compute_aggregated_report_dim(store: EmbeddingsStore, paper_text_fields: Optional[List[str]] = None) -> int:
    fields = _resolve_text_fields_config(paper_text_fields)
    return int(sum(store.dim[field] for field in fields))


def compute_pair_dim(
    store: EmbeddingsStore,
    text_fields: List[str],
    architecture_mode: str = ARCH_CURRENT,
    paper_text_fields: Optional[List[str]] = None,
) -> int:
    _resolve_architecture_mode(architecture_mode)
    agg_dim = compute_aggregated_report_dim(store, paper_text_fields=paper_text_fields)
    return int(3 * agg_dim)


def _get_cat_token_without_vocab(row: Optional[Dict[str, Any]], field: str) -> str:
    value = "__MISSING__"
    if row is not None:
        raw = row.get(field, "__MISSING__")
        if raw is not None and str(raw).strip():
            value = str(raw).strip()
    return value


def build_pair_features_current(
    pairs,
    store: EmbeddingsStore,
    case2cats: Dict[str, Dict[str, Any]],
    text_fields: List[str],
    paper_text_fields: Optional[List[str]] = None,
):
    fields = _resolve_text_fields_config(paper_text_fields)
    wide_feature_names = get_wide_feature_names(fields)
    agg_dim = compute_aggregated_report_dim(store, fields)
    pair_dim = int(3 * agg_dim)

    wide_rows = []
    pair_vectors = []
    y_list = []
    qc_list = []

    for pair in pairs:
        q = norm_id(pair["q"])
        c = norm_id(pair["c"])
        q_idx = store.get_idx(q)
        c_idx = store.get_idx(c)

        row_q = case2cats.get(q)
        row_c = case2cats.get(c)

        wide_row = []
        for field in CAT_FIELDS:
            q_token = _get_cat_token_without_vocab(row_q, field)
            c_token = _get_cat_token_without_vocab(row_c, field)
            wide_row.append(1.0 if q_token == c_token else 0.0)

        q_report_parts = []
        c_report_parts = []
        for field in fields:
            qv, qok = store.get_vec(field, q_idx)
            cv, cok = store.get_vec(field, c_idx)
            if not qok:
                qv = np.zeros((store.dim[field],), dtype=np.float32)
            if not cok:
                cv = np.zeros((store.dim[field],), dtype=np.float32)

            q_report_parts.append(np.asarray(qv, dtype=np.float32))
            c_report_parts.append(np.asarray(cv, dtype=np.float32))
            wide_row.append(cosine(qv, cv) if (qok and cok) else 0.0)

        q_report = np.concatenate(q_report_parts, axis=0).astype(np.float32) if q_report_parts else np.zeros((0,), dtype=np.float32)
        c_report = np.concatenate(c_report_parts, axis=0).astype(np.float32) if c_report_parts else np.zeros((0,), dtype=np.float32)
        deep_vec = np.concatenate([q_report, c_report, np.abs(q_report - c_report)], axis=0).astype(np.float32)

        wide_rows.append(np.asarray(wide_row, dtype=np.float32))
        pair_vectors.append(deep_vec)
        y_list.append(float(pair.get("label", 0.0)))
        qc_list.append((q, c))

    x_wide = np.stack(wide_rows, axis=0).astype(np.float32) if wide_rows else np.zeros((0, len(wide_feature_names)), dtype=np.float32)
    x_pair = np.stack(pair_vectors, axis=0).astype(np.float32) if pair_vectors else np.zeros((0, pair_dim), dtype=np.float32)

    return {
        "X_tab": x_wide,
        "X_pair": x_pair,
        "y": np.asarray(y_list, dtype=np.float32),
        "qc_list": qc_list,
        "pair_dim": int(pair_dim),
        "wide_feature_names": list(wide_feature_names),
        "paper_text_fields": list(fields),
        "feature_schema": {
            "wide": {
                "cat_agreement_fields": list(CAT_FIELDS),
                "text_similarity_fields": list(fields),
                "input_dim": int(len(wide_feature_names)),
            },
            "deep": {
                "aggregated_text_fields": list(fields),
                "aggregated_report_dim": int(agg_dim),
                "pair_input_form": "[h(u), h(v), |h(u)-h(v)|]",
                "input_dim": int(pair_dim),
            },
        },
    }


def _iter_pair_source(pair_source) -> Iterator[Dict[str, Any]]:
    if isinstance(pair_source, str):
        yield from iter_jsonl(pair_source)
        return

    if isinstance(pair_source, list):
        for pair in pair_source:
            yield pair
        return

    if isinstance(pair_source, tuple):
        for pair in pair_source:
            yield pair
        return

    if isinstance(pair_source, Iterable):
        for pair in pair_source:
            yield pair
        return

    raise TypeError(f"Unsupported pair_source type: {type(pair_source)}")


def iter_pair_feature_batches(
    pair_source,
    store: EmbeddingsStore,
    case2cats: Dict[str, Dict[str, Any]],
    text_fields: List[str],
    batch_size: int = 512,
    limit_batches: Optional[int] = None,
    architecture_mode: Optional[str] = None,
    paper_text_fields: Optional[List[str]] = None,
):
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    batch_pairs = []
    emitted = 0
    _resolve_architecture_mode(architecture_mode)
    paper_fields = _resolve_text_fields_config(paper_text_fields)

    for pair in _iter_pair_source(pair_source):
        batch_pairs.append(pair)
        if len(batch_pairs) < batch_size:
            continue

        bundle = build_pair_features_current(
            batch_pairs,
            store=store,
            case2cats=case2cats,
            text_fields=text_fields,
            paper_text_fields=paper_fields,
        )
        yield bundle

        emitted += 1
        if limit_batches is not None and emitted >= limit_batches:
            return
        batch_pairs = []

    if batch_pairs:
        bundle = build_pair_features_current(
            batch_pairs,
            store=store,
            case2cats=case2cats,
            text_fields=text_fields,
            paper_text_fields=paper_fields,
        )
        yield bundle


class PaperSimplifiedPairwiseModel(nn.Module):
    def __init__(
        self,
        tab_input_dim: int,
        pair_dim: int,
        pair_hidden: int = 512,
        pair_hidden2: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.architecture_mode = ARCH_CURRENT
        self.wide_linear = nn.Linear(int(tab_input_dim), 1)
        self.deep_mlp = nn.Sequential(
            nn.Linear(int(pair_dim), int(pair_hidden)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(pair_hidden), int(pair_hidden2)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(pair_hidden2), 1),
        )

    def forward(self, x_tab: torch.Tensor, x_pair: torch.Tensor):
        z_wide = self.wide_linear(x_tab.float())
        z_deep = self.deep_mlp(x_pair.float())
        return z_wide + z_deep


def build_widedeep_model(
    pair_dim: int,
    tab_hidden_dims: Optional[List[int]] = None,
    pair_hidden: int = 512,
    pair_hidden2: int = 256,
    dropout: float = 0.2,
    architecture_mode: str = ARCH_CURRENT,
    tab_input_dim: Optional[int] = None,
    paper_text_fields: Optional[List[str]] = None,
):
    if tab_hidden_dims is None:
        tab_hidden_dims = [128, 64]

    architecture_mode = _resolve_architecture_mode(architecture_mode)

    if tab_input_dim is None:
        fields = _resolve_text_fields_config(paper_text_fields)
        tab_input_dim = len(CAT_FIELDS) + len(fields)
    model = PaperSimplifiedPairwiseModel(
        tab_input_dim=int(tab_input_dim),
        pair_dim=int(pair_dim),
        pair_hidden=int(pair_hidden),
        pair_hidden2=int(pair_hidden2),
        dropout=float(dropout),
    )

    model_config = {
        "architecture_mode": architecture_mode,
        "pair_dim": int(pair_dim),
        "tab_hidden_dims": [int(x) for x in tab_hidden_dims],
        "pair_hidden": int(pair_hidden),
        "pair_hidden2": int(pair_hidden2),
        "dropout": float(dropout),
        "tab_input_dim": int(tab_input_dim) if tab_input_dim is not None else None,
        "paper_text_fields": _resolve_text_fields_config(paper_text_fields) if paper_text_fields is not None else None,
    }
    return model, model_config


def _ensure_tensor(x: np.ndarray, device: str, dtype: torch.dtype):
    return torch.as_tensor(x, dtype=dtype, device=device)


def _forward_pairwise_model(model, x_tab: torch.Tensor, x_pair: torch.Tensor):
    logits = model(x_tab, x_pair)
    if isinstance(logits, tuple):
        logits = logits[0]
    if logits.ndim == 1:
        logits = logits.unsqueeze(1)
    return logits


def fit_pairwise_epoch(
    model,
    pair_source,
    store: EmbeddingsStore,
    case2cats: Dict[str, Dict[str, Any]],
    text_fields: List[str],
    optimizer,
    device: str,
    batch_size: int = 512,
    pos_weight: Optional[float] = None,
    epoch_idx: int = 1,
    total_epochs: int = 1,
    num_pairs: Optional[int] = None,
    architecture_mode: Optional[str] = None,
    paper_text_fields: Optional[List[str]] = None,
):
    model.train()
    if pos_weight is not None:
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([float(pos_weight)], device=device))
    else:
        criterion = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    seen = 0

    inferred_total = num_pairs
    if inferred_total is None and isinstance(pair_source, list):
        inferred_total = len(pair_source)
    if inferred_total is None and isinstance(pair_source, str):
        inferred_total = count_jsonl(pair_source)

    mode = _resolve_architecture_mode(architecture_mode, model=model)

    pbar = tqdm(total=inferred_total, desc=f"Train {epoch_idx}/{total_epochs}", ncols=110, leave=False)
    for feature_bundle in iter_pair_feature_batches(
        pair_source=pair_source,
        store=store,
        case2cats=case2cats,
        text_fields=text_fields,
        batch_size=batch_size,
        architecture_mode=mode,
        paper_text_fields=paper_text_fields,
    ):
        X_tab = np.asarray(feature_bundle["X_tab"], dtype=np.float32)
        X_pair = np.asarray(feature_bundle["X_pair"], dtype=np.float32)
        y = np.asarray(feature_bundle["y"], dtype=np.float32)

        if len(y) == 0:
            continue

        x_tab = _ensure_tensor(X_tab, device, torch.float32)
        x_pair = _ensure_tensor(X_pair, device, torch.float32)
        y_batch = _ensure_tensor(y.reshape(-1, 1), device, torch.float32)

        optimizer.zero_grad(set_to_none=True)
        logits = _forward_pairwise_model(model, x_tab, x_pair)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        bs = y_batch.shape[0]
        total_loss += float(loss.detach().cpu().item()) * bs
        seen += bs
        pbar.set_postfix(loss=f"{(total_loss / max(1, seen)):.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}", bs=bs)
        pbar.update(bs)

    pbar.close()
    if seen == 0:
        raise ValueError("Empty training bundle")
    return total_loss / max(1, seen)


@torch.no_grad()
def predict_probs(
    model,
    pair_source,
    store: EmbeddingsStore,
    case2cats: Dict[str, Dict[str, Any]],
    text_fields: List[str],
    device: str,
    desc: str = "Predict",
    batch_size: int = 512,
    num_pairs: Optional[int] = None,
    architecture_mode: Optional[str] = None,
    paper_text_fields: Optional[List[str]] = None,
):
    model.eval()

    all_probs = []
    y_parts = []
    qc_list = []

    inferred_total = num_pairs
    if inferred_total is None and isinstance(pair_source, list):
        inferred_total = len(pair_source)
    if inferred_total is None and isinstance(pair_source, str):
        inferred_total = count_jsonl(pair_source)

    mode = _resolve_architecture_mode(architecture_mode, model=model)

    pbar = tqdm(total=inferred_total, desc=desc, ncols=110, leave=False)
    for feature_bundle in iter_pair_feature_batches(
        pair_source=pair_source,
        store=store,
        case2cats=case2cats,
        text_fields=text_fields,
        batch_size=batch_size,
        architecture_mode=mode,
        paper_text_fields=paper_text_fields,
    ):
        X_tab = np.asarray(feature_bundle["X_tab"], dtype=np.float32)
        X_pair = np.asarray(feature_bundle["X_pair"], dtype=np.float32)
        y = np.asarray(feature_bundle.get("y", np.zeros((len(X_tab),), dtype=np.float32)), dtype=np.float32)

        if len(X_tab) == 0:
            continue

        x_tab = _ensure_tensor(X_tab, device, torch.float32)
        x_pair = _ensure_tensor(X_pair, device, torch.float32)
        logits = _forward_pairwise_model(model, x_tab, x_pair)
        probs = torch.sigmoid(logits).squeeze(1).detach().cpu().numpy()
        all_probs.append(probs.astype(np.float32))
        y_parts.append(y.astype(np.float32))
        qc_list.extend(feature_bundle.get("qc_list", []))
        if probs.size > 0:
            pbar.set_postfix(mean_p=f"{float(probs.mean()):.3f}")
            pbar.update(int(probs.size))

    probs_all = np.concatenate(all_probs, axis=0).astype(np.float32) if all_probs else np.zeros((0,), dtype=np.float32)
    y_all = np.concatenate(y_parts, axis=0).astype(np.float32) if y_parts else np.zeros((0,), dtype=np.float32)
    pbar.close()
    return probs_all, y_all, qc_list


def save_checkpoint(path: str, model, metadata: Dict[str, Any]):
    payload = {
        "framework": "pairwise-current",
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
    }
    payload.update(metadata)
    torch.save(payload, path)


def load_checkpoint(path: str, device: str = "cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("framework") not in {"pairwise-current", "pytorch-widedeep"}:
        raise ValueError(f"Unsupported checkpoint framework: {checkpoint.get('framework')}")
    if "model_config" not in checkpoint:
        raise ValueError("checkpoint 缺少 model_config")

    tab_preprocessor = checkpoint.get("tab_preprocessor")
    model_config = dict(checkpoint["model_config"])
    architecture_mode = _resolve_architecture_mode(
        checkpoint.get("architecture_mode", model_config.get("architecture_mode", ARCH_CURRENT))
    )
    model_config["architecture_mode"] = architecture_mode
    if checkpoint.get("paper_text_fields") is not None:
        model_config.setdefault("paper_text_fields", checkpoint.get("paper_text_fields"))
    model, _ = build_widedeep_model(**model_config)
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device)
    model.eval()
    return model, checkpoint, tab_preprocessor
