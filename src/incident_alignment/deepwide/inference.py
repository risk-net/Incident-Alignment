"""Pairwise probability inference (sigmoid) with original feature construction."""
from .model import load_legacy_api


def predict_pairs(model, *, pair_source, store, cases, text_fields,
                  device='cpu', batch_size=512, paper_text_fields=None, num_pairs=None):
    return load_legacy_api().predict_probs(
        model=model, pair_source=pair_source, store=store, case2cats=cases,
        text_fields=list(text_fields), device=device, batch_size=batch_size,
        num_pairs=num_pairs, paper_text_fields=paper_text_fields)
