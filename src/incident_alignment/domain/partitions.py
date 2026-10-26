"""Cluster partitions represented by membership, independent of arbitrary cluster labels."""
from collections.abc import Iterable, Mapping
from .types import normalize_report_id


def canonicalize_partition(clusters: Iterable[Iterable[object]]) -> tuple[tuple[str, ...], ...]:
    """Produce a deterministic label-invariant partition signature.

    Reject empty clusters and duplicate members, including duplicates *within* a cluster.
    """
    normalized = []
    seen: set[str] = set()
    for i, group in enumerate(clusters):
        raw = [normalize_report_id(v) for v in group]
        if not raw:
            raise ValueError(f"cluster {i} is empty")
        if len(raw) != len(set(raw)):
            raise ValueError(f"cluster {i} has duplicate report IDs")
        overlap = seen.intersection(raw)
        if overlap:
            raise ValueError(f"report IDs appear in multiple clusters: {sorted(overlap)[:5]}")
        seen.update(raw)
        normalized.append(tuple(sorted(raw)))
    return tuple(sorted(normalized))


def partition_from_mapping(mapping: Mapping[object, object]) -> tuple[tuple[str, ...], ...]:
    groups: dict[str, list[str]] = {}
    for rid, event_id in mapping.items():
        eid = str(event_id).strip()
        if not eid:
            raise ValueError("event_id cannot be empty")
        groups.setdefault(eid, []).append(normalize_report_id(rid))
    return canonicalize_partition(groups.values())


def validate_partition(clusters: Iterable[Iterable[object]], expected_ids: Iterable[object]) -> tuple[tuple[str, ...], ...]:
    signature = canonicalize_partition(clusters)
    flat = {rid for group in signature for rid in group}
    expected_list = [normalize_report_id(i) for i in expected_ids]
    if len(expected_list) != len(set(expected_list)):
        raise ValueError("expected_ids contains duplicates")
    expected = set(expected_list)
    if flat != expected:
        missing = sorted(expected - flat)
        extra = sorted(flat - expected)
        raise ValueError(f"partition domain mismatch: missing={missing[:10]}, extra={extra[:10]}")
    return signature
