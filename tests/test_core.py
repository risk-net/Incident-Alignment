import pytest
from incident_alignment.domain.partitions import canonicalize_partition, validate_partition
from incident_alignment.data.validation import assert_unique_ids, assert_same_domain, assert_candidate_bounds


def test_permutation_invariance():
    assert canonicalize_partition([["b", "a"], ["c"]]) == canonicalize_partition([["c"], ["a", "b"]])


@pytest.mark.parametrize("part", [[["a", "a"]], [["a"], ["a"]], [[]]])
def test_invalid_partition(part):
    with pytest.raises(ValueError):
        canonicalize_partition(part)


def test_missing_node():
    with pytest.raises(ValueError, match="missing"):
        validate_partition([["a"]], ["a", "b"])


def test_domain_validation():
    assert_unique_ids(["a", "b"])
    assert_same_domain(["a", "b"], ["b", "a"])
    with pytest.raises(ValueError):
        assert_unique_ids(["a", "a"])
    with pytest.raises(ValueError):
        assert_candidate_bounds({"a": ["b"], "b": ["a"]}, ["a", "b", "c"])
