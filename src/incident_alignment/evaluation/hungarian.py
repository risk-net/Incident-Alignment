"""Event-level one-to-one Hungarian macro-F1, directional denominators.

Primary contract: g2p = sum(matched F1)/n_gold; p2g = same/n_pred.
Unmatched clusters score zero. Assignment maximizes sum of per-cluster F1.
"""
from scipy.optimize import linear_sum_assignment
import numpy as np
from ._common import build_contingency

def cluster_f1(intersection: int, gold_size: int, pred_size: int) -> float:
    return 2.0*intersection/(gold_size+pred_size) if intersection else 0.0

def optimal_matching(gold, pred) -> dict:
    c = build_contingency(gold, pred)
    score = np.zeros((len(c.gold),len(c.pred)), dtype=np.float64)
    for (i,j), n in c.counts.items():
        score[i,j] = cluster_f1(n,c.gold_sizes[i],c.pred_sizes[j])
    rows,cols = linear_sum_assignment(-score)
    matches = [(int(i),int(j),float(score[i,j])) for i,j in zip(rows,cols) if score[i,j]>0]
    return {'matches':matches,'score_sum':sum(s for _,_,s in matches),
            'n_gold':len(c.gold),'n_pred':len(c.pred)}

def hungarian_metrics(gold, pred) -> dict:
    m = optimal_matching(gold,pred)
    total = m['score_sum']
    g2p = total/m['n_gold']
    p2g = total/m['n_pred']
    return {'hungarian_f1_g2p':g2p, 'hungarian_f1_p2g':p2g,
            'hungarian_f1_symmetric':(g2p+p2g)/2,
            'num_hungarian_matches':len(m['matches']),
            'n_unmatched_gold_events':m['n_gold']-len(m['matches']),
            'n_unmatched_pred_clusters':m['n_pred']-len(m['matches'])}
