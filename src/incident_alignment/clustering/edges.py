"""Score ingestion and canonical edge building."""
import math

def normalize_nodes(nodes):
    ids = [str(x).strip() for x in nodes]
    if any(not x for x in ids):
        raise ValueError("empty node ID")
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate node ID")
    return sorted(ids)

def directed_from_pairs(pairs, scores):
    """Strict pair-score ingestion. Duplicate directed edges retain maximum score."""
    pairs, scores = list(pairs), list(scores)
    if len(pairs) != len(scores):
        raise ValueError("pairs and scores length mismatch")
    out = {}
    for (u,v),score in zip(pairs,scores):
        u,v = str(u).strip(),str(v).strip()
        s = float(score)
        if not math.isfinite(s) or not 0.0 <= s <= 1.0:
            raise ValueError("scores must be finite probabilities within [0,1]")
        if not u or not v:
            raise ValueError("empty pair ID")
        if u != v:out[(u,v)] = max(s, out.get((u,v), -1.0))
    return out

def undirected_scores(nodes, directed, rule):
    """Return canonical undirected score dict; never silently accept invalid domain/scores."""
    if rule not in ("mutual","either"):
        raise ValueError("invalid edge rule")
    domain=set(nodes); directed_clean={}
    for (u,v),s in directed.items():
        u,v=str(u).strip(),str(v).strip();s=float(s)
        if not math.isfinite(s) or not 0<=s<=1:raise ValueError("invalid probability")
        if u not in domain or v not in domain:raise ValueError("edge outside input node domain")
        if u!=v:directed_clean[(u,v)] = max(s,directed_clean.get((u,v),-1.))
    out={}
    for u,v in directed_clean:
        a,b=sorted((u,v))
        if (a,b) in out:continue
        ab=directed_clean.get((a,b));ba=directed_clean.get((b,a))
        if rule=="mutual":
            if ab is None or ba is None:continue
            score=min(ab,ba)
        else:score=max(s for s in (ab,ba) if s is not None)
        if score>0:out[(a,b)]=score
    return out
