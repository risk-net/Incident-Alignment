"""Exact preserved 2026-10-26 DeepWide architecture; no shadow reimplementation."""

def load_legacy_api():
    from .legacy import pairwise_model
    return pairwise_model


def build_model(*, pair_dim: int, tab_input_dim: int, pair_hidden: int = 512,
                pair_hidden2: int = 256, dropout: float = .2):
    api = load_legacy_api()
    return api.build_widedeep_model(pair_dim=pair_dim, tab_input_dim=tab_input_dim,
                                    pair_hidden=pair_hidden, pair_hidden2=pair_hidden2,
                                    dropout=dropout)


def load_model_checkpoint(path, device='cpu'):
    return load_legacy_api().load_checkpoint(str(path), device=device)


def save_model_checkpoint(path, model, metadata):
    return load_legacy_api().save_checkpoint(str(path), model, metadata)
