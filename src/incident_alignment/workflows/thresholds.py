"""Select theta on DEVELOPMENT only; never read an evaluation/test dataset."""
from incident_alignment.data.gold import load_event_mapping
from incident_alignment.domain.partitions import partition_from_mapping
from incident_alignment.clustering import ClusteringConfig,cluster_events
from incident_alignment.evaluation import evaluate_partition
from .contracts import ThresholdSelection
from .io import read_scored_pairs,hash_file,write_json_atomic


def select_dev_threshold(dev_gold_csv, dev_scores_jsonl, *, config:ThresholdSelection, out_file):
    mapping=load_event_mapping(dev_gold_csv);gold=partition_from_mapping(mapping);nodes=sorted(mapping)
    directed=read_scored_pairs(dev_scores_jsonl)
    rows=[]
    for theta in config.grid():
        pred=cluster_events(nodes,directed,ClusteringConfig(theta,config.edge_rule,config.backend))
        scores=evaluate_partition(gold,pred)
        rows.append({'theta':theta,'num_pred_clusters':len(pred),
                     'hungarian_f1_g2p':scores['hungarian_f1_g2p'],
                     'hungarian_f1_symmetric':scores['hungarian_f1_symmetric'],
                     'b_cubed_f1':scores['b_cubed_f1']})
    # Deterministic tie-breaking: primary g2p, then symmetric, B3 and larger theta.
    best=max(rows,key=lambda r:(r['hungarian_f1_g2p'],r['hungarian_f1_symmetric'],r['b_cubed_f1'],r['theta']))
    result={'selected_theta':best['theta'],'selection_metric':config.metric,'best':best,'table':rows,
            'source':'dev_only', 'provenance':{'dev_gold_sha256':hash_file(dev_gold_csv),
                                            'dev_scores_sha256':hash_file(dev_scores_jsonl)},
            'config':{'min_theta':config.min_theta,'max_theta':config.max_theta,'step':config.step,
                      'edge_rule':config.edge_rule,'backend':config.backend}}
    write_json_atomic(out_file,result)
    return result


def lock_fold_thresholds(dev_runs, out_file):
    if not dev_runs:raise ValueError('no dev runs')
    locked={}
    for fold,path in sorted(dev_runs.items()):
        from .io import read_json
        doc=read_json(path)
        if doc.get('source')!='dev_only' or 'selected_theta' not in doc:raise ValueError(f'fold {fold} has no valid dev selection')
        locked[str(fold)]=float(doc['selected_theta'])
    payload={'source':'dev_only','locked_thetas':locked,
             'dev_runs_sha256':{str(k):hash_file(p) for k,p in sorted(dev_runs.items())}}
    write_json_atomic(out_file,payload)
    return payload
