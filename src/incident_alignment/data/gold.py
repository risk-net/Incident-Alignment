"""Read-only Gold mappings; duplicates are errors rather than silently overwritten."""
import csv
from collections import defaultdict
from pathlib import Path

def load_event_mapping(path: str | Path, *, report_key='report_id', event_key='gold_v2_event_id') -> dict[str,str]:
    result = {}
    with Path(path).open(encoding='utf-8-sig',newline='') as fh:
        for line,row in enumerate(csv.DictReader(fh),start=2):
            if report_key not in row or event_key not in row:
                raise ValueError(f'missing required CSV fields: {report_key}, {event_key}')
            rid,eid = (row[report_key] or '').strip(),(row[event_key] or '').strip()
            if not rid or not eid: raise ValueError(f'empty ID on line {line}')
            if rid in result: raise ValueError(f'duplicate report ID {rid} on line {line}')
            result[rid]=eid
    if not result: raise ValueError('Gold mapping is empty')
    return result

def load_gold_partition(path: str | Path, **kwargs) -> list[list[str]]:
    grouped = defaultdict(list)
    for rid,eid in load_event_mapping(path,**kwargs).items(): grouped[eid].append(rid)
    return [sorted(ids) for _,ids in sorted(grouped.items())]
