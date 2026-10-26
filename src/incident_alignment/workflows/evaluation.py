"""One test-set evaluation for English or Chinese, no search or parameter estimation."""
from pathlib import Path
from incident_alignment.data.gold import load_event_mapping
from incident_alignment.domain.partitions import partition_from_mapping
from incident_alignment.clustering import ClusteringConfig,cluster_events
from incident_alignment.evaluation import evaluate_partition
from .io import read_scored_pairs,hash_file,write_json_atomic
from .contracts import EvaluationInputs


def evaluate_locked(inputs:EvaluationInputs, *, threshold:float, output_dir, edge_rule='mutual', backend='dense', provenance=None):
    mapping=load_event_mapping(inputs.gold_csv)
    gold=partition_from_mapping(mapping)
    if inputs.expected_reports is not None and len(mapping)!=inputs.expected_reports:raise ValueError('wrong number of gold reports')
    if inputs.expected_events is not None and len(gold)!=inputs.expected_events:raise ValueError('wrong number of gold events')
    nodes=sorted(mapping)
    directed=read_scored_pairs(inputs.scores_jsonl)
    pred=cluster_events(nodes,directed,ClusteringConfig(threshold=threshold,edge_rule=edge_rule,backend=backend))
    metrics=evaluate_partition(gold,pred)
    result={'dataset':inputs.dataset_name,'threshold':threshold,'edge_rule':edge_rule,'backend':backend,
            'metrics':metrics,'provenance':{'gold_sha256':hash_file(inputs.gold_csv),
            'scores_sha256':hash_file(inputs.scores_jsonl),**(provenance or {})}}
    out=Path(output_dir);write_json_atomic(out/'metrics.json',result)
    write_json_atomic(out/'pred_clusters.json',pred)
    return result


def evaluate_folds(inputs:EvaluationInputs, *, fold_scores:dict[int,Path], locked_thetas:dict[int,float], output_dir,
                   fold_gold:dict[int,Path]|None=None, edge_rule='mutual',backend='dense'):
    """五折固定阈值评测。

    两种 Gold 模式：

    - ``fold_gold=None``（单份 Gold）：所有折共用 ``inputs.gold_csv``。
      适用于中文评测——5 折打分的始终是同一份 1,283 篇域。
    - ``fold_gold={fold:csv}``（每折 Gold）：每折使用各自的 Gold。
      适用于英文五折——每折的 Test 集是互不重叠的约 1,200 篇子集，
      若用单份全量 Gold，未打分的其他折会全被判为单例，指标被严重稀释。

    每折 Gold 规模各不相同，因此 ``expected_reports`` / ``expected_events``
    只在单份 Gold 模式下生效。
    """
    from incident_alignment.evaluation import summarize_folds
    if set(fold_scores)!=set(locked_thetas) or not fold_scores:raise ValueError('fold score files and thresholds must match')
    if fold_gold is not None and set(fold_gold)!=set(fold_scores):
        raise ValueError('per-fold gold and fold score files must cover the same folds')
    rows=[];out=Path(output_dir);gold_records={}
    for fold in sorted(fold_scores):
        gold_csv=fold_gold[fold] if fold_gold is not None else inputs.gold_csv
        exp_reports=inputs.expected_reports if fold_gold is None else None
        exp_events=inputs.expected_events if fold_gold is None else None
        inp=EvaluationInputs(gold_csv,fold_scores[fold],inputs.dataset_name,exp_reports,exp_events)
        row=evaluate_locked(inp,threshold=locked_thetas[fold],output_dir=out/f'fold_{fold:02d}',edge_rule=edge_rule,backend=backend)
        rows.append({'fold':fold,**row['metrics']})
        gold_records[str(fold)]={'path':str(gold_csv),'sha256':hash_file(gold_csv),
                                 'gold_reports':len(load_event_mapping(gold_csv))}
    summary={'dataset':inputs.dataset_name,'folds':sorted(fold_scores),
             'gold_mode':'per_fold' if fold_gold is not None else 'single',
             'gold':gold_records if fold_gold is not None else
                    {'all_folds':{'path':str(inputs.gold_csv),'sha256':hash_file(inputs.gold_csv),
                                  'gold_reports':len(load_event_mapping(inputs.gold_csv))}},
             'locked_thetas':{str(k):v for k,v in locked_thetas.items()},'aggregate':summarize_folds(rows),
             'metrics_per_fold':rows}
    write_json_atomic(out/'aggregate.json',summary)
    return summary
