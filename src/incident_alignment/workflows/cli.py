"""Unified workflows CLI: dev selection / locked test / manifest / legacy train hand-off."""
import argparse
from pathlib import Path
from . import (EvaluationInputs,ThresholdSelection,evaluate_locked,evaluate_folds,
               select_dev_threshold,lock_fold_thresholds,freeze_manifest,verify_manifest,
               prepare_legacy_training)
from .io import read_json

def parse_numbered_paths(items):
    result={}
    for item in items:
        n,p=item.split('=',1)
        if int(n) in result:raise ValueError(f'duplicate fold {n}')
        result[int(n)]=Path(p)
    return result

def main(argv=None):
    p=argparse.ArgumentParser(prog='incident-align',description='Incident Alignment immutable research workflows')
    subs=p.add_subparsers(dest='command',required=True)
    d=subs.add_parser('select-dev');d.add_argument('--gold',required=True);d.add_argument('--scores',required=True)
    d.add_argument('--out',required=True);d.add_argument('--min',type=float,default=.10);d.add_argument('--max',type=float,default=.95);d.add_argument('--step',type=float,default=.01)
    for name in ('lock-thetas',):
        c=subs.add_parser(name);c.add_argument('--dev-run',action='append',required=True,help='FOLD=scan.json');c.add_argument('--out',required=True)
    e=subs.add_parser('evaluate');e.add_argument('--gold',required=True);e.add_argument('--scores',required=True)
    e.add_argument('--theta',type=float,required=True);e.add_argument('--dataset',choices=['english','chinese'],required=True)
    e.add_argument('--out',required=True);e.add_argument('--expected-reports',type=int);e.add_argument('--expected-events',type=int)
    for cmd in [d,e]:
        cmd.add_argument('--edge-rule',choices=['mutual','either'],default='mutual')
        cmd.add_argument('--backend',choices=['dense','sparse'],default='dense')
    f=subs.add_parser('evaluate-folds');f.add_argument('--gold',help='单份 Gold（中文：5 折共用同一 1,283 篇域）');f.add_argument('--fold-scores',action='append',required=True)
    f.add_argument('--fold-gold',action='append',default=[],help='FOLD=gold.csv（英文：每折各自的 Test 子集 Gold）')
    f.add_argument('--locked',required=True);f.add_argument('--dataset',choices=['english','chinese'],required=True)
    f.add_argument('--out',required=True);f.add_argument('--expected-reports',type=int);f.add_argument('--expected-events',type=int)
    f.add_argument('--edge-rule',choices=['mutual','either'],default='mutual');f.add_argument('--backend',choices=['dense','sparse'],default='dense')
    m=subs.add_parser('freeze');m.add_argument('--asset',action='append',required=True,help='NAME=path');m.add_argument('--version',required=True);m.add_argument('--out',required=True)
    v=subs.add_parser('verify');v.add_argument('--manifest',required=True)
    t=subs.add_parser('prepare-train');t.add_argument('--script',required=True);t.add_argument('--splits',required=True);t.add_argument('--out',required=True)
    args=p.parse_args(argv)
    if args.command=='select-dev':
        result=select_dev_threshold(args.gold,args.scores,config=ThresholdSelection(args.min,args.max,args.step,edge_rule=args.edge_rule,backend=args.backend),out_file=args.out)
        print(f"DEV selected theta={result['selected_theta']}")
    elif args.command=='lock-thetas':
        print(lock_fold_thresholds(parse_numbered_paths(args.dev_run),args.out))
    elif args.command=='evaluate':
        print(evaluate_locked(EvaluationInputs(Path(args.gold),Path(args.scores),args.dataset,args.expected_reports,args.expected_events),threshold=args.theta,output_dir=args.out,edge_rule=args.edge_rule,backend=args.backend)['metrics'])
    elif args.command=='evaluate-folds':
        locked=read_json(args.locked)
        if locked.get('source')!='dev_only':raise ValueError('locked thresholds must come from development')
        thetas={int(k):float(v) for k,v in locked['locked_thetas'].items()}
        fold_gold=parse_numbered_paths(args.fold_gold) if args.fold_gold else None
        if args.gold is None and fold_gold is None:
            raise ValueError('需要 --gold（单份 Gold，适用于中文）或 --fold-gold（每折 Gold，适用于英文五折）')
        if args.gold is not None and fold_gold is not None:
            raise ValueError('--gold 与 --fold-gold 不能同时使用')
        if fold_gold is not None and (args.expected_reports or args.expected_events):
            raise ValueError('--expected-reports/--expected-events 只在单份 Gold 模式下有意义')
        gold_path=Path(args.gold) if args.gold is not None else Path('.')
        inputs=EvaluationInputs(gold_path,Path('.'),args.dataset,args.expected_reports,args.expected_events)
        result=evaluate_folds(inputs,fold_scores=parse_numbered_paths(args.fold_scores),locked_thetas=thetas,
                              fold_gold=fold_gold,output_dir=args.out,edge_rule=args.edge_rule,backend=args.backend)
        print(f"Evaluated {len(result['folds'])} folds (gold_mode={result['gold_mode']})")
    elif args.command=='freeze':
        assets=dict(item.split('=',1) for item in args.asset)
        freeze_manifest(assets,out_file=args.out,version=args.version);print(args.out)
    elif args.command=='verify':
        print('manifest verified' if verify_manifest(args.manifest) else 'failed')
    elif args.command=='prepare-train':
        prepare_legacy_training(args.script,args.splits,out_file=args.out);print(args.out)
    return 0

if __name__=='__main__':raise SystemExit(main())
