"""Checkpoint integrity and compatibility metadata."""
import hashlib
from pathlib import Path
from .model import load_model_checkpoint


def checkpoint_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def inspect_checkpoint(path, device='cpu'):
    model, payload, _ = load_model_checkpoint(path, device=device)
    config = payload['model_config']
    return {'sha256': checkpoint_sha256(path), 'framework': payload.get('framework'),
            'model_config': config, 'text_fields': payload.get('text_fields'),
            'architecture_mode': payload.get('architecture_mode'),
            'state_keys': tuple(sorted(model.state_dict()))}
