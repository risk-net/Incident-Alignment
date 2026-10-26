"""Immutable contracts used by train/dev/test orchestration."""
from dataclasses import dataclass
from pathlib import Path
import math

@dataclass(frozen=True)
class EvaluationInputs:
    gold_csv: Path
    scores_jsonl: Path
    dataset_name: str
    expected_reports: int | None = None
    expected_events: int | None = None

    def __post_init__(self):
        if not self.dataset_name.strip(): raise ValueError('dataset_name required')
        if self.expected_reports is not None and self.expected_reports <= 0: raise ValueError('expected_reports must be positive')
        if self.expected_events is not None and self.expected_events <= 0: raise ValueError('expected_events must be positive')

@dataclass(frozen=True)
class ThresholdSelection:
    min_theta: float = 0.10
    max_theta: float = 0.95
    step: float = 0.01
    metric: str = 'hungarian_f1_g2p'
    edge_rule: str = 'mutual'
    backend: str = 'dense'

    def __post_init__(self):
        if not (0 < self.min_theta <= self.max_theta <= 1): raise ValueError('invalid threshold interval')
        if not (0 < self.step <= 1 and math.isfinite(self.step)): raise ValueError('invalid step')
        if self.metric != 'hungarian_f1_g2p': raise ValueError('M6 threshold objective is locked to hungarian_f1_g2p')
        if self.edge_rule not in {'mutual','either'}: raise ValueError('invalid edge_rule')
        if self.backend not in {'dense','sparse'}: raise ValueError('invalid backend')

    def grid(self):
        from decimal import Decimal
        lo,hi,st = map(lambda x: Decimal(str(x)),(self.min_theta,self.max_theta,self.step))
        n=int((hi-lo)//st)
        values=[float(lo+i*st) for i in range(n+1)]
        if Decimal(str(values[-1])) < hi: values.append(float(hi))
        return values
