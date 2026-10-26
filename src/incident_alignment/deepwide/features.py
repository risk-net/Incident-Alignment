"""Retains the original pair feature and batch semantics without recomputation of embeddings."""
from .model import load_legacy_api


def open_embedding_store(directory, text_fields):
    return load_legacy_api().EmbeddingsStore(str(directory), list(text_fields))


def iter_feature_batches(*, pair_source, store, cases, text_fields, batch_size=512,
                         paper_text_fields=None):
    yield from load_legacy_api().iter_pair_feature_batches(
        pair_source=pair_source, store=store, case2cats=cases,
        text_fields=list(text_fields), batch_size=batch_size,
        paper_text_fields=paper_text_fields)


def feature_dimensions(store, paper_text_fields=None):
    api = load_legacy_api()
    return {
        'report_dim': api.compute_aggregated_report_dim(store, paper_text_fields),
        'pair_dim': api.compute_pair_dim(store, text_fields=store.fields, paper_text_fields=paper_text_fields),
    }
