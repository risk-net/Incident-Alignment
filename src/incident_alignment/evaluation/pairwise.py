"""Micro induced-pair classification from contingency cells, no O(N²) materialization."""
from ._common import build_contingency

def _choose2(n): return n*(n-1)//2

def induced_pairwise_counts(gold,pred):
    c=build_contingency(gold,pred)
    tp=sum(_choose2(n) for n in c.counts.values())
    predicted=sum(_choose2(n) for n in c.pred_sizes)
    actual=sum(_choose2(n) for n in c.gold_sizes)
    return {'tp':tp,'fp':predicted-tp,'fn':actual-tp}

def induced_pairwise_metrics(gold,pred):
    v=induced_pairwise_counts(gold,pred)
    tp,fp,fn=v['tp'],v['fp'],v['fn']
    precision=tp/(tp+fp) if tp+fp else 0.0
    recall=tp/(tp+fn) if tp+fn else 0.0
    f1=2*precision*recall/(precision+recall) if precision+recall else 0.0
    return {'induced_pair_precision':precision,'induced_pair_recall':recall,'induced_pair_f1':f1, **{'pair_'+k:val for k,val in v.items()}}
