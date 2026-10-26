"""Input-domain assertions shared by training and locked evaluation."""
from collections.abc import Iterable, Mapping
from ..domain.types import normalize_report_id


def assert_unique_ids(ids: Iterable[object], *, label: str = "IDs") -> list[str]:
    values = [normalize_report_id(x) for x in ids]
    if len(values) != len(set(values)):
        raise ValueError(f"{label} contains duplicate report IDs")
    return values


def assert_same_domain(left: Iterable[object], right: Iterable[object]) -> None:
    a, b = set(assert_unique_ids(left, label="left")), set(assert_unique_ids(right, label="right"))
    if a != b:
        raise ValueError(f"domain mismatch: left_only={sorted(a-b)[:10]}, right_only={sorted(b-a)[:10]}")


def assert_candidate_bounds(query_to_candidates: Mapping[object, Iterable[object]], nodes: Iterable[object]) -> None:
    node_ids = set(assert_unique_ids(nodes, label="nodes"))
    query_ids = set(query_to_candidates)
    if query_ids != node_ids:
        raise ValueError("query IDs must cover exactly the node set")
    for query, candidates in query_to_candidates.items():
        for candidate in candidates:
            cid = normalize_report_id(candidate)
            if cid == query or cid not in node_ids:
                raise ValueError(f"candidate outside domain or self-loop: {query} -> {cid}")
