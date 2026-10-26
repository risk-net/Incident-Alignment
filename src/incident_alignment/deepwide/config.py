"""Typed training configuration; no dependency on mutable repository globals."""
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DeepWideConfig:
    epochs: int = 8
    batch_size: int = 512
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    pair_hidden: int = 512
    pair_hidden2: int = 256
    dropout: float = 0.2
    seed: int = 42
    device: str = 'cpu'
    pos_weight: float | None = None

    def __post_init__(self):
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError('epochs and batch_size must be positive')
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError('invalid optimizer parameters')
        if min(self.pair_hidden, self.pair_hidden2) <= 0:
            raise ValueError('hidden dimensions must be positive')
        if not 0 <= self.dropout < 1:
            raise ValueError('dropout must be in [0, 1)')
        if self.pos_weight is not None and self.pos_weight <= 0:
            raise ValueError('pos_weight must be positive')

@dataclass(frozen=True)
class FoldArtifacts:
    train_pairs: Path
    dev_pairs: Path
    embeddings_dir: Path
    checkpoint_out: Path
