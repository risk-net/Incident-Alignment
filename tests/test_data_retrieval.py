import json
import csv
import numpy as np
import pytest
from incident_alignment.data import load_event_mapping,load_gold_partition,load_legacy_cv_splits,load_legacy_recall,validate_recall_domain,validate_embedding_assets,MemmapEmbeddings,sha256_file,verify_sha256
from incident_alignment.retrieval import BGEM3Adapter, FaissIndexAdapter

def test_gold(tmp_path):
 p=tmp_path/'gold.csv';p.write_text('report_id,gold_v2_event_id\n1,e1\n2,e1\n3,e2\n')
 assert len(load_gold_partition(p))==2
 assert load_event_mapping(p)['1']=='e1'
 p.write_text('report_id,gold_v2_event_id\n1,e1\n1,e1\n')
 with pytest.raises(ValueError):load_event_mapping(p)

def test_splits(tmp_path):
 (tmp_path/'fold_splits.json').write_text(json.dumps({'n_folds':2,'total_incidents':2,'total_cases':2}))
 for i,sets in [(1,{'test':['x'],'dev':['y'],'train':[]}),(2,{'test':['y'],'dev':['x'],'train':[]})]:
  d=tmp_path/f'fold_{i:02d}';d.mkdir()
  for key,v in sets.items():(d/f'{key}_incidents.json').write_text(json.dumps(v))
 assert len(load_legacy_cv_splits(tmp_path,{'x':['1'],'y':['2']}))==2

def test_recall(tmp_path):
 p=tmp_path/'r.jsonl';p.write_text(json.dumps({'query_case_id':'1','candidates':[{'case_id':'2','score_text':0.7}]})+'\n'+json.dumps({'query_case_id':'2','candidates':[]})+'\n')
 r=load_legacy_recall(p)
 assert validate_recall_domain(r,['1','2'])['directed_pairs']==1
 assert r['1'][0].score_fuse==0.7
 with pytest.raises(ValueError):validate_recall_domain(r,['1','2','3'])

def test_embed(tmp_path):
 (tmp_path/'case_ids.txt').write_text('1\n2\n')
 np.save(tmp_path/'emb_title.npy',np.ones((2,3),dtype='float32'))
 np.save(tmp_path/'valid_mask_title.npy',np.array([1,0]))
 assert validate_embedding_assets(tmp_path,['1','2'],['title'])['title']==(2,3)
 assert not MemmapEmbeddings(tmp_path,['title'],['1','2']).get('2','title')[1]
 with pytest.raises(ValueError):validate_embedding_assets(tmp_path,['2','1'],['title'])

def test_hash(tmp_path):
 p=tmp_path/'f';p.write_text('abc')
 assert verify_sha256(p,sha256_file(p)).path==p
 with pytest.raises(ValueError):verify_sha256(p,'0'*64)

def test_adapters():
 class FakeModel:
  def encode(self,x):return {'dense_vecs':np.ones((len(x),4))}
 class FakeIndex:
  def search(self,q,k):return np.ones((len(q),k)),np.zeros((len(q),k),dtype=int)
 assert BGEM3Adapter(FakeModel()).encode(['a','b']).shape==(2,4)
 assert FaissIndexAdapter(FakeIndex()).search(np.ones((2,4)),2)[0].shape==(2,2)

def test_legacy_triplet(tmp_path):
 from incident_alignment.data.recall import load_legacy_recall_triplet
 p=tmp_path/'r.jsonl'
 p.write_text(json.dumps({'query_case_id':'1','ground_truth':['2'], 'candidates':[{'case_id':'2','score_fuse':0.8,'score_text':0.5,'score_event':0.2},{'case_id':'3','score_fuse':0.4}]})+'\n')
 c,g,m=load_legacy_recall_triplet(p,1)
 assert c['1']==[('2',0.8,0.5,0.2)]
 assert g['1']=={'2'}
 assert m['1']['fuse']==(0.8,0.8)
