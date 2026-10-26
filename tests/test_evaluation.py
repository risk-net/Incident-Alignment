import math
import pytest
from incident_alignment.evaluation import (evaluate_partition,summarize_folds,validate_metric_contract,
                                           induced_pairwise_counts)

def test_perfect():
    r=evaluate_partition([['1','2'],['3']],[['3'],['2','1']])
    for k in ('hungarian_f1_g2p','hungarian_f1_p2g','induced_pair_f1', 'b_cubed_f1','ari','nmi'):
        assert r[k]==pytest.approx(1.0)
    assert validate_metric_contract(r)

def test_split_direction():
    r=evaluate_partition([['a','b']],[['a'],['b']])
    assert r['hungarian_f1_g2p']==pytest.approx(2/3)
    assert r['hungarian_f1_p2g']==pytest.approx(1/3)
    assert r['induced_pair_recall']==0
    assert r['b_cubed_precision']==1
    assert r['b_cubed_recall']==pytest.approx(.5)

def test_merge_direction():
    r=evaluate_partition([['a'],['b']],[['a','b']])
    assert r['hungarian_f1_g2p']==pytest.approx(1/3)
    assert r['hungarian_f1_p2g']==pytest.approx(2/3)
    assert r['induced_pair_precision']==0

def test_strict_domain():
    with pytest.raises(ValueError,match='domain mismatch'):
        evaluate_partition([['1','2']],[['1','3']])
    with pytest.raises(ValueError,match='multiple clusters'):
        evaluate_partition([['1','2']],[['1'],['1','2']])
    with pytest.raises(ValueError,match='empty'):
        evaluate_partition([[]],[['1']])

def test_pair_counts():
    assert induced_pairwise_counts([['1','2','3'],['4']],[['1','2'],['3','4']])=={'tp':1,'fp':1,'fn':2}

def test_summary():
    out=summarize_folds([{'x':0.3},{'x':0.5}])
    assert out['x']['mean']==pytest.approx(.4)
    assert out['x']['std']==pytest.approx(math.sqrt(.02))

def test_all_singletons():
    r=evaluate_partition([['a'],['b']],[['b'],['a']])
    assert r['induced_pair_f1']==0
    assert r['hungarian_f1_symmetric']==1
