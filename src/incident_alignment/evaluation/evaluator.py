"""Single public metrics entry point for dev, English and Chinese evaluation."""
from dataclasses import dataclass
import numpy as np
from ._common import build_contingency
from .hungarian import hungarian_metrics
from .pairwise import induced_pairwise_metrics
from .b_cubed import b_cubed_precision_recall_f1
from .external import external_metrics

METRICS_VERSION='2.0.0-directional'

def evaluate_partition(gold,pred) -> dict:
    c=build_contingency(gold,pred)
    sizes=c.pred_sizes
    return {
        'metrics_version':METRICS_VERSION,
        'n_scored_reports':c.n,'num_gold_events':len(c.gold),
        'num_pred_clusters':len(c.pred),
        'num_singleton_pred_clusters':sum(s==1 for s in sizes),
        'pred_singleton_ratio':sum(s==1 for s in sizes)/len(sizes),
        'max_cluster_frac':max(sizes)/c.n,
        'coverage':1.0,'n_extra_ids':0,'n_missing_ids':0,
        **hungarian_metrics(c.gold,c.pred),
        **induced_pairwise_metrics(c.gold,c.pred),
        **b_cubed_precision_recall_f1(c.gold,c.pred),
        **external_metrics(c.gold,c.pred),
    }

def summarize_folds(rows,keys=None):
    if not rows: raise ValueError('no folds')
    if keys is None:
        keys=[k for k,v in rows[0].items() if isinstance(v,(float,int)) and not isinstance(v,bool)]
    result={}
    for k in keys:
        values=[float(r[k]) for r in rows]
        if not all(np.isfinite(v) for v in values): raise ValueError(f'nonfinite metric: {k}')
        result[k]={'mean':float(np.mean(values)), 'std':float(np.std(values,ddof=1)) if len(values)>1 else 0.0,
                   'values':values,'n_folds':len(values)}
    return result

def validate_metric_contract(metrics):
    if metrics.get('metrics_version')!=METRICS_VERSION: raise ValueError('wrong metric version')
    for k in ['hungarian_f1_g2p','hungarian_f1_p2g','hungarian_f1_symmetric',
              'induced_pair_f1','b_cubed_f1','ari','nmi']:
        if not np.isfinite(metrics[k]) or not -1.0 <= metrics[k] <= 1.0:
            raise ValueError(f'invalid metric {k}: {metrics[k]}')
    return True
