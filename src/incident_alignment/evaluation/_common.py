"""Shared contingency representation; partitions are strict and label-invariant."""
from dataclasses import dataclass
from collections import Counter
from incident_alignment.domain.partitions import canonicalize_partition

@dataclass(frozen=True)
class Contingency:
    gold: tuple
    pred: tuple
    counts: dict
    gold_sizes: tuple
    pred_sizes: tuple
    n: int

def build_contingency(gold, pred) -> Contingency:
    g = canonicalize_partition(gold)
    p = canonicalize_partition(pred)
    gi = {rid: i for i, group in enumerate(g) for rid in group}
    pi = {rid: j for j, group in enumerate(p) for rid in group}
    if gi.keys() != pi.keys():
        raise ValueError(f"partition domain mismatch: missing={sorted(gi.keys()-pi.keys())[:10]}, extra={sorted(pi.keys()-gi.keys())[:10]}")
    if not gi:
        raise ValueError('cannot evaluate empty partitions')
    counts = Counter((gi[x], pi[x]) for x in gi)
    return Contingency(g, p, dict(counts), tuple(map(len,g)),tuple(map(len,p)),len(gi))
