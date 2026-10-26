"""Explicit clustering contract, independent of model and experiment workflow."""
from dataclasses import dataclass
import math
from typing import Literal

@dataclass(frozen=True)
class ClusteringConfig:
    threshold: float
    edge_rule: Literal["mutual", "either"] = "mutual"
    backend: Literal["dense", "sparse"] = "sparse"

    def __post_init__(self):
        if not math.isfinite(self.threshold) or not 0.0 < self.threshold <= 1.0:
            raise ValueError("threshold must be finite and in (0,1]")
        if self.edge_rule not in ("mutual", "either"):
            raise ValueError("edge_rule must be mutual or either")
        if self.backend not in ("dense", "sparse"):
            raise ValueError("backend must be dense or sparse")
