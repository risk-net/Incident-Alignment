#!/usr/bin/env python3
"""Read-only inventory and frozen Gold structural verification (no downloads)."""
import argparse, csv, json
from pathlib import Path

REQUIRED={
 'chinese':['data/chinese_gold/gold_v2.0.2_mapping.csv','data/chinese_eval/cases.jsonl','data/chinese_eval/recall_topk.jsonl','data/chinese_eval/embeddings/case_ids.txt'],
 'english':['data/english/test/cases.jsonl','data/english/test/recall_topk.jsonl','data/english/test/embeddings/case_ids.txt',
            'data/english/folds/fold_01/test_gold.csv','data/english/folds/fold_02/test_gold.csv','data/english/folds/fold_03/test_gold.csv',
            'data/english/folds/fold_04/test_gold.csv','data/english/folds/fold_05/test_gold.csv'],
 'train':['data/english/folds/fold_splits.json','checkpoints/deepwide/fold_01.pt','checkpoints/deepwide/fold_02.pt',
          'checkpoints/deepwide/fold_03.pt','checkpoints/deepwide/fold_04.pt','checkpoints/deepwide/fold_05.pt']
          + [f'data/english/folds/fold_{i:02d}/train/repeat_01/{f}'
             for i in range(1,6) for f in ('train_pairs.jsonl','dev_pairs.jsonl','test_pairs.jsonl','vocabs.json')]
          + [f'data/english/folds/fold_{i:02d}/{s}/{f}'
             for i in range(1,6) for s in ('dev','test')
             for f in ('cases.jsonl','recall_topk.jsonl','embeddings/case_ids.txt')],
 'folds':[f'data/english/folds/fold_{i:02d}/test_gold.csv' for i in range(1,6)],
 'full':[]
}
def inspect(root,profile):
 missing=[]
 for rel in REQUIRED[profile]:
  if not (root/rel).is_file():missing.append(rel)
 details={'profile':profile,'missing':missing,'checked':REQUIRED[profile]}
 if profile=='chinese' and 'data/chinese_gold/gold_v2.0.2_mapping.csv' not in missing:
  path=root/'data/chinese_gold/gold_v2.0.2_mapping.csv'
  with path.open(newline='',encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
  ids=[r.get('report_id','').strip() for r in rows]
  evs=[r.get('gold_v2_event_id','').strip() for r in rows]
  details['gold']={'rows':len(rows),'unique_reports':len(set(ids)),'unique_events':len(set(evs))}
  if not (len(rows)==1283 and len(set(ids))==1283 and len(set(evs))==185 and
          set(evs)=={f'zh_gold_{i:04d}' for i in range(1,186)} and all(ids)):
   details['gold_error']='Frozen v2.0.2 schema/count/ID mismatch'
 return details
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--root',default='.');ap.add_argument('--profile',choices=list(REQUIRED),required=True)
 a=ap.parse_args();result=inspect(Path(a.root),a.profile);print(json.dumps(result,ensure_ascii=False,indent=2))
 raise SystemExit(1 if result['missing'] or result.get('gold_error') else 0)
