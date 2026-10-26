"""Composable five-fold-compatible training primitives, without data reshuffling."""
from dataclasses import asdict
from pathlib import Path
from .config import DeepWideConfig
from .model import load_legacy_api, build_model, save_model_checkpoint


def train_prepared_fold(*, train_pairs, cases, embeddings_dir, text_fields,
                        paper_text_fields, tab_input_dim, pair_dim, checkpoint_out,
                        config: DeepWideConfig, checkpoint_metadata=None):
    """Train using an EXISTING pair-source and EXISTING embeddings, never resample pairs.

    The caller owns the fold split, vocabulary, negative sampling and dev selection.
    No implicit checkpoint selection by train loss is claimed to equal legacy 5-fold.
    """
    import torch
    api = load_legacy_api()
    api.set_seed(config.seed)
    store = api.EmbeddingsStore(str(embeddings_dir), list(text_fields))
    model, model_config = build_model(
        pair_dim=pair_dim, tab_input_dim=tab_input_dim,
        pair_hidden=config.pair_hidden, pair_hidden2=config.pair_hidden2,
        dropout=config.dropout)
    model = model.to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                   weight_decay=config.weight_decay)
    history = []
    for epoch in range(1, config.epochs + 1):
        loss = api.fit_pairwise_epoch(
            model=model, pair_source=str(train_pairs) if isinstance(train_pairs, Path) else train_pairs,
            store=store, case2cats=cases, text_fields=list(text_fields),
            optimizer=optimizer, device=config.device, batch_size=config.batch_size,
            pos_weight=config.pos_weight, epoch_idx=epoch, total_epochs=config.epochs,
            paper_text_fields=paper_text_fields)
        history.append({'epoch': epoch, 'train_loss': float(loss)})
    metadata = dict(checkpoint_metadata or {})
    metadata.update({'model_config': model_config, 'architecture_mode': api.ARCH_CURRENT,
                     'paper_text_fields': list(paper_text_fields),
                     'text_fields': list(text_fields), 'training_config': asdict(config),
                     'training_history': history})
    dest = Path(checkpoint_out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    save_model_checkpoint(dest, model, metadata)
    return {'checkpoint': str(dest), 'history': history, 'model_config': model_config}
