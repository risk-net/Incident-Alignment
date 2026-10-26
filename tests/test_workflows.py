import csv,json
import pytest
from pathlib import Path
from incident_alignment.workflows import EvaluationInputs, ThresholdSelection, select_dev_threshold, evaluate_locked, lock_fold_thresholds, freeze_manifest,verify_manifest
from incident_alignment.workflows.cli import main

def fixture(tmp_path):
    gold=tmp_path/'gold.csv';pairs=tmp_path/'pairs.jsonl'
    with gold.open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['report_id','gold_v2_event_id']);w.writerows([('a','e1'),('b','e1'),('c','e2')])
    pairs.write_text('\n'.join(json.dumps({'query_id':q,'candidate_id':c,'probability':s}) for q,c,s in [('a','b',.9),('b','a',.9)]))
    return gold,pairs

def test_end_to_end(tmp_path):
    gold,pairs=fixture(tmp_path)
    scan=select_dev_threshold(gold,pairs,config=ThresholdSelection(.5,.95,.05),out_file=tmp_path/'scan.json')
    assert scan['selected_theta'] >= .5
    lock=lock_fold_thresholds({1:tmp_path/'scan.json'},tmp_path/'locked.json')
    assert lock['locked_thetas']['1']==scan['selected_theta']
    res=evaluate_locked(EvaluationInputs(gold,pairs,'chinese',3,2),threshold=lock['locked_thetas']['1'],output_dir=tmp_path/'out')
    assert res['metrics']['hungarian_f1_g2p']==1.0
    assert (tmp_path/'out/pred_clusters.json').exists()

def test_invalid_domain(tmp_path):
    gold,pairs=fixture(tmp_path)
    with pairs.open('a') as f:f.write('\n'+json.dumps({'query_id':'a','candidate_id':'x','probability':.7}))
    with pytest.raises(ValueError):evaluate_locked(EvaluationInputs(gold,pairs,'chinese'),threshold=.5,output_dir=tmp_path/'out')

def test_release_hash_guard(tmp_path):
    gold,_=fixture(tmp_path)
    path=tmp_path/'manifest.json'
    freeze_manifest({'gold':str(gold)},out_file=path,version='test-1')
    assert verify_manifest(path)
    gold.write_text('tampered')
    with pytest.raises(ValueError):verify_manifest(path)

def test_cli(tmp_path):
    gold,pairs=fixture(tmp_path)
    assert main(['evaluate','--gold',str(gold),'--scores',str(pairs),'--theta','.5','--dataset','english','--out',str(tmp_path/'result')])==0

def test_grid():
    grid=ThresholdSelection().grid()
    assert len(grid)==86 and grid[0]==.1 and grid[-1]==.95
