"""Read frozen legacy fold_splits.json and per-fold train/dev/test JSON; never regenerate."""
from pathlib import Path
import json
from typing import Mapping

SPLIT_NAMES = ('train','dev','test')

def validate_cv_splits(folds: Mapping[int,Mapping[str,list[str]]], incident2cases: Mapping[str,list[str]] | None=None) -> dict:
    if not folds: raise ValueError('no folds')
    universe = set(incident2cases) if incident2cases is not None else None
    tests=[]
    for index,groups in folds.items():
        ss={key:set(map(str,groups[key])) for key in SPLIT_NAMES}
        for key in SPLIT_NAMES:
            if len(ss[key])!=len(groups[key]):raise ValueError(f'fold {index}: duplicate incident in {key}')
        if any(ss[a]&ss[b] for a,b in [('train','dev'),('train','test'),('dev','test')]):
            raise ValueError(f'fold {index}: train/dev/test incident overlap')
        if universe is not None and set.union(*ss.values()) != universe:
            raise ValueError(f'fold {index}: missing or unexpected incidents')
        tests.extend(ss['test'])
    if len(tests)!=len(set(tests)):raise ValueError('test folds overlap')
    if universe is not None and set(tests)!=universe:raise ValueError('test folds do not cover universe')
    return {'folds':len(folds),'test_incidents':len(tests)}

def load_legacy_cv_splits(root: str | Path, incident2cases: Mapping[str,list[str]] | None=None) -> dict[int,dict[str,list[str]]]:
    root=Path(root)
    summary=json.loads((root/'fold_splits.json').read_text(encoding='utf-8'))
    result={}
    for i in range(1,int(summary['n_folds'])+1):
        directory=root/f'fold_{i:02d}'
        groups={}
        for name in SPLIT_NAMES:
            value=json.loads((directory/f'{name}_incidents.json').read_text(encoding='utf-8'))
            if not isinstance(value,list):raise ValueError(f'invalid split: fold {i} {name}')
            groups[name]=[str(x).strip() for x in value]
        result[i]=groups
    validate_cv_splits(result,incident2cases)
    if summary.get('total_incidents') is not None and incident2cases is not None:
        if int(summary['total_incidents'])!=len(incident2cases):raise ValueError('summary incident count mismatch')
    if incident2cases is not None and summary.get('total_cases') is not None:
        if int(summary['total_cases'])!=sum(len(x) for x in incident2cases.values()):
            raise ValueError('summary case count mismatch')
    return result
