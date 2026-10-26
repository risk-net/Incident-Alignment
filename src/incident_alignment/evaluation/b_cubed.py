"""Document-weighted B-cubed precision and recall."""
from ._common import build_contingency

def b_cubed_precision_recall_f1(gold,pred):
    c=build_contingency(gold,pred)
    precision=sum(n*n/c.pred_sizes[j] for (i,j),n in c.counts.items())/c.n
    recall=sum(n*n/c.gold_sizes[i] for (i,j),n in c.counts.items())/c.n
    f1=2*precision*recall/(precision+recall) if precision+recall else 0.0
    return {'b_cubed_precision':precision,'b_cubed_recall':recall,'b_cubed_f1':f1}
