"""Legacy JSONL recall adapter. Preserves order, score_fuse fallback and optional topM."""
import json, math
from pathlib import Path
from dataclasses import dataclass

@dataclass(frozen=True)
class RecallCandidate:
    case_id: str
    score_fuse: float
    score_text: float
    score_event: float

def _score(row,key):
    raw=row.get(key)
    value=0.0 if raw is None else float(raw)
    if not math.isfinite(value): raise ValueError(f'nonfinite {key} score')
    return value

def load_legacy_recall(path: str | Path, topm_out: int | None=250) -> dict[str,list[RecallCandidate]]:
    if topm_out is not None and topm_out<=0:raise ValueError('topm_out must be positive')
    output={}
    with Path(path).open(encoding='utf-8') as file:
        for lineno,line in enumerate(file,1):
            if not line.strip():continue
            obj=json.loads(line)
            q=str(obj['query_case_id']).strip() if 'query_case_id' in obj else str(obj['query_id']).strip() if 'query_id' in obj else str(obj['case_id']).strip()
            if not q or q in output:raise ValueError(f'empty/duplicate recall query at line {lineno}: {q}')
            candidates=[];seen=set()
            for item in obj.get('candidates',[]):
                cid=str(item['case_id']).strip()
                if not cid or cid in seen:raise ValueError(f'empty/duplicate candidate for query {q}: {cid}')
                seen.add(cid)
                st,se=_score(item,'score_text'),_score(item,'score_event')
                sf=_score(item,'score_fuse') if item.get('score_fuse') is not None else max(st,se)
                candidates.append(RecallCandidate(cid,sf,st,se))
            candidates.sort(key=lambda x:x.score_fuse,reverse=True)
            output[q]=candidates[:topm_out] if topm_out is not None else candidates
    return output

def validate_recall_domain(recall: dict[str,list[RecallCandidate]], nodes: list[str] | set[str], *, complete_queries=True) -> dict:
    domain=set(nodes)
    if complete_queries and set(recall)!=domain:raise ValueError(f'query domain mismatch missing={len(domain-set(recall))}, extra={len(set(recall)-domain)}')
    if set(recall)-domain: raise ValueError('out of domain recall queries')
    pairs=0
    for q,cands in recall.items():
        for c in cands:
            if c.case_id==q or c.case_id not in domain:raise ValueError(f'invalid candidate {q}->{c.case_id}')
            pairs+=1
    return {'queries':len(recall),'directed_pairs':pairs,'nodes':len(domain)}


def load_legacy_recall_triplet(path: str | Path, topm_out: int = 200):
    """Match original pairwise_data_io.load_recall_topk return structure.

    Returns (query2cands, query2gt, query2minmax). This adapter preserves
    ground-truth metadata and per-query normalization ranges, needed by DeepWide.
    """
    if topm_out <= 0:
        raise ValueError('topm_out must be positive')
    cands, ground_truth, minmax = {}, {}, {}
    with Path(path).open(encoding='utf-8') as file:
        for line in file:
            if not line.strip():
                continue
            obj = json.loads(line)
            query = str(obj['query_case_id']).strip()
            if not query or query in cands:
                raise ValueError(f'duplicate/empty recall query {query}')
            gt = {str(x).strip() for x in obj.get('ground_truth', [])}
            selected = sorted(obj.get('candidates', []),
                              key=lambda x: (_score(x, 'score_fuse') if x.get('score_fuse') is not None
                                             else max(_score(x, 'score_text'), _score(x, 'score_event'))),
                              reverse=True)[:topm_out]
            packed=[]
            for x in selected:
                cid=str(x['case_id']).strip()
                st,se=_score(x,'score_text'),_score(x,'score_event')
                sf=_score(x,'score_fuse') if x.get('score_fuse') is not None else max(st,se)
                packed.append((cid,sf,st,se))
            cands[query]=packed
            ground_truth[query]=gt
            minmax[query]={}
            for name,col in (('fuse',1),('text',2),('event',3)):
                vals=[r[col] for r in packed]
                minmax[query][name]=(float(min(vals)),float(max(vals))) if vals else (0.0,0.0)
    return cands,ground_truth,minmax
