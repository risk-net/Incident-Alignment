"""Explicit, deterministic IO for frozen pairwise scores (no labels in score files)."""
import hashlib
import json
import math
import os
from pathlib import Path
from incident_alignment.clustering.edges import directed_from_pairs


def hash_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def read_scored_pairs(path):
    """JSONL records: {query_id, candidate_id, probability}; duplicate directional scores use max."""
    pairs=[];scores=[]
    with Path(path).open(encoding='utf-8') as stream:
        for lineno,line in enumerate(stream,1):
            if not line.strip():continue
            o=json.loads(line)
            try:
                q,c=str(o['query_id']).strip(),str(o['candidate_id']).strip()
                s=float(o['probability'])
            except (KeyError,ValueError,TypeError) as exc:
                raise ValueError(f'invalid pair score at line {lineno}') from exc
            if not q or not c or q==c or not math.isfinite(s) or not 0<=s<=1:
                raise ValueError(f'invalid pair at line {lineno}')
            pairs.append((q,c));scores.append(s)
    return directed_from_pairs(pairs,scores)

def write_json_atomic(path, data):
    dest=Path(path);dest.parent.mkdir(parents=True,exist_ok=True)
    temp=dest.with_suffix(dest.suffix+'.tmp')
    temp.write_text(json.dumps(data,ensure_ascii=False,indent=2,sort_keys=True),encoding='utf-8')
    os.replace(temp,dest)

def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))
