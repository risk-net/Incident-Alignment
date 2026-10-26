#!/usr/bin/env python3
"""From an already trained legacy DeepWide checkpoint and frozen recall to M6 score JSONL.
This does NOT generate recall, embeddings, training data, or thresholds.
"""
import argparse, csv, hashlib, json, sys
from pathlib import Path

def checksum(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()

def serialize_scores(qc, probs):
 if len(qc)!=len(probs):raise ValueError('Pair/probability length mismatch')
 rows=[]
 for (q,c),p in zip(qc,probs):
  val=float(p)
  if not (0 <= val <= 1):raise ValueError('Invalid probability')
  rows.append({'query_id':str(q).strip(),'candidate_id':str(c).strip(),'probability':val})
 return rows

def main():
 p=argparse.ArgumentParser()
 for name in ('checkpoint','cases','recall','embeddings','output'):p.add_argument('--'+name,required=True)
 p.add_argument('--topm',type=int,default=250);p.add_argument('--device',default='cpu');p.add_argument('--batch-size',type=int,default=512)
 p.add_argument('--gold',help='Optional CSV with report_id column; strict scored domain')
 a=p.parse_args()
 from incident_alignment.deepwide.legacy.pairwise_data_io import load_cases_minimal,load_recall_topk,TEXT_FIELDS_DEFAULT
 from incident_alignment.deepwide.legacy.pairwise_model import PAPER_TEXT_FIELDS_DEFAULT,EmbeddingsStore,load_checkpoint,predict_probs
 import numpy as np
 cases=load_cases_minimal(a.cases)
 query2cands,_,_=load_recall_topk(a.recall,topm_out=a.topm)
 if a.gold:
  with open(a.gold,encoding='utf-8-sig',newline='') as f:
   gold=[row['report_id'].strip() for row in csv.DictReader(f)]
  if len(set(gold))!=len(gold):raise ValueError('Duplicate Gold IDs')
  if set(cases)!=set(gold):raise ValueError('Cases/Gold domain mismatch')
  if set(query2cands)!=set(gold):raise ValueError('Recall query/Gold domain mismatch')
 if set(query2cands)!=set(cases):raise ValueError('Recall queries and cases IDs differ')
 pairs=[]
 for q,cands in query2cands.items():
  for entry in cands:
   c=str(entry[0]).strip()
   if q==c or c not in cases:raise ValueError(f'Invalid candidate {q}->{c}')
   pairs.append({'q':q,'c':c,'label':0})
 model,ckpt,_=load_checkpoint(a.checkpoint,device=a.device)
 fields=ckpt.get('text_fields',TEXT_FIELDS_DEFAULT)
 emb=Path(a.embeddings)
 ids=[x.strip() for x in (emb/'case_ids.txt').read_text().splitlines() if x.strip()]
 # 行序由 case_ids.txt 自身决定（EmbeddingsStore 就是按它的行序建索引），
 # 因此这里只校验“域一致 + 无重复 + 数量一致”。
 # 旧断言 `ids != sorted(cases)` 在英文集上必然失败：英文 ID 是数值序
 # (1,2,3,…)，而 sorted() 是字典序 ('1','10','1000',…)；只有中文那种
 # 等长数字 ID 才会碰巧一致。
 if len(ids)!=len(set(ids)):raise ValueError('Embedding case_ids.txt contains duplicate IDs')
 if set(ids)!=set(cases):raise ValueError('Embedding IDs and cases domain differ; do not continue')
 if len(ids)!=len(cases):raise ValueError('Embedding ID count differs from cases count; do not continue')
 store=EmbeddingsStore(str(emb),fields)
 probs,_,qc=predict_probs(model=model,pair_source=pairs,store=store,case2cats=cases,text_fields=fields,
  device=a.device,desc='export',batch_size=a.batch_size,num_pairs=len(pairs),
  architecture_mode=ckpt.get('architecture_mode'),paper_text_fields=ckpt.get('paper_text_fields',PAPER_TEXT_FIELDS_DEFAULT))
 rows=serialize_scores(qc,probs)
 target=Path(a.output);target.parent.mkdir(parents=True,exist_ok=True)
 tmp=target.with_suffix(target.suffix+'.tmp')
 with tmp.open('w',encoding='utf-8') as f:
  for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
 tmp.replace(target)
 manifest={'pairs':len(rows),'checkpoint_sha256':checksum(a.checkpoint),'cases_sha256':checksum(a.cases),
  'recall_sha256':checksum(a.recall),'embedding_ids_sha256':checksum(emb/'case_ids.txt'),
  'scores_sha256':checksum(target),'device':a.device,'topm':a.topm}
 target.with_suffix(target.suffix+'.manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
 print(json.dumps(manifest,indent=2))
if __name__=='__main__': main()
