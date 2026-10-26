"""Stable identifiers and immutable input records (no external dependencies)."""
from dataclasses import dataclass


def normalize_report_id(value: object) -> str:
    if value is None:
        raise ValueError("report_id cannot be None")
    rid = str(value).strip()
    if not rid:
        raise ValueError("report_id cannot be empty")
    return rid


@dataclass(frozen=True)
class CandidatePair:
    query_id: str
    candidate_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "query_id", normalize_report_id(self.query_id))
        object.__setattr__(self, "candidate_id", normalize_report_id(self.candidate_id))
        if self.query_id == self.candidate_id:
            raise ValueError("self-pairs are not allowed")


@dataclass(frozen=True)
class ScoredPair:
    pair: CandidatePair
    score: float

    def __post_init__(self) -> None:
        import math
        if not math.isfinite(self.score) or not 0 <= self.score <= 1:
            raise ValueError("score must be finite and in [0, 1]")
