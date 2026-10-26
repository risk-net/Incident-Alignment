"""ARI and arithmetic-mean NMI; canonicalized IDs avoid label dependence."""
from sklearn.metrics import adjusted_rand_score,normalized_mutual_info_score
from ._common import build_contingency

def external_metrics(gold,pred):
    c=build_contingency(gold,pred)
    gold_map={x:i for i,g in enumerate(c.gold) for x in g}
    pred_map={x:j for j,p in enumerate(c.pred) for x in p}
    ids=sorted(gold_map)
    yt=[gold_map[x] for x in ids]
    yp=[pred_map[x] for x in ids]
    return {'ari':float(adjusted_rand_score(yt,yp)),
            'nmi':float(normalized_mutual_info_score(yt,yp,average_method='arithmetic'))}
