import random
import pytest
from incident_alignment.clustering import ClusteringConfig,cluster_events,directed_from_pairs

def run(nodes, edges, backend="sparse", rule="mutual", threshold=.7):
    return cluster_events(nodes,edges,ClusteringConfig(threshold,rule,backend))

@pytest.mark.parametrize("backend",["dense","sparse"])
def test_isolates_and_pair(backend):
    assert run(["a","b","c"],{("a","b"):.9,("b","a"):.8},backend)==[["a","b"],["c"]]

@pytest.mark.parametrize("backend",["dense","sparse"])
def test_mutual_vs_either(backend):
    edges={("a","b"):.9}
    assert run(["a","b"],edges,backend,"mutual")==[["a"],["b"]]
    assert run(["a","b"],edges,backend,"either")==[["a","b"]]

@pytest.mark.parametrize("backend",["dense","sparse"])
def test_complete_link_requires_clique(backend):
    edges={("a","b"):.91,("b","a"):.91,("b","c"):.89,("c","b"):.89}
    got=run(["a","b","c"],edges,backend,threshold=.8)
    assert sorted(map(len,got))==[1,2]

@pytest.mark.parametrize("backend",["dense","sparse"])
def test_permutation_invariance_without_ties(backend):
    edges={("a","b"):.92,("b","a"):.92,("a","c"):.81,("c","a"):.81,("b","c"):.82,("c","b"):.82}
    a=run(["a","b","c","d"],edges,backend,threshold=.8)
    b=run(["d","c","a","b"],dict(reversed(list(edges.items()))),backend,threshold=.8)
    assert a==b

def test_invalid_inputs():
    with pytest.raises(ValueError):run(["a","a"],{})
    with pytest.raises(ValueError):run(["a"],{},threshold=0)
    with pytest.raises(ValueError):run(["a","b"],{("a","b"):float("nan")})
    with pytest.raises(ValueError):run(["a","b"],{("a","outside"):.8})
    with pytest.raises(ValueError):directed_from_pairs([("a","b")],[.3,.4])

def test_backends_random_non_tied_graphs():
    rng=random.Random(42)
    for _ in range(100):
        nodes=[str(i) for i in range(7)]
        edges={}
        for i in range(7):
            for j in range(i+1,7):
                if rng.random()<.6:
                    score=rng.uniform(.2,.98)
                    edges[(str(i),str(j))]=score
                    edges[(str(j),str(i))]=score
        for threshold in [.3,.55,.75,.9]:
            dense=run(nodes,edges,"dense",threshold=threshold)
            sparse=run(nodes,edges,"sparse",threshold=threshold)
            assert dense==sparse

def test_sparse_adapter_preserves_reference_backend():
    from incident_alignment.clustering.sparse_backend import build_sparse_complete_link_clusters
    edges = {('1','2'):.94,('2','1'):.93,('2','3'):.79,('3','2'):.78,
             ('1','3'):.77,('3','1'):.76}
    for tau in (.7,.8,.9):
        expected=build_sparse_complete_link_clusters(['1','2','3','4'],edges,'mutual',tau)
        assert run(['1','2','3','4'],edges,'sparse','mutual',tau)==expected
