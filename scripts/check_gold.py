#!/usr/bin/env python3
"""Standalone stdlib validator for the public Chinese Gold mapping."""
import argparse,csv,collections,hashlib,pathlib,sys
p=argparse.ArgumentParser();p.add_argument('csv');p.add_argument('--expected-news',type=int,default=1283);p.add_argument('--expected-events',type=int,default=185);a=p.parse_args()
f=pathlib.Path(a.csv);rows=list(csv.DictReader(f.open(encoding='utf-8-sig',newline='')))
if not rows:sys.exit('FAIL: no rows')
for name in ('report_id','gold_v2_event_id'):
 if name not in rows[0]:sys.exit(f'FAIL: missing column {name}')
ids=[x['report_id'].strip() for x in rows]; eids=[x['gold_v2_event_id'].strip() for x in rows]
if any(not x for x in ids+eids):sys.exit('FAIL: empty ID')
if len(ids)!=len(set(ids)):sys.exit('FAIL: duplicate report_id')
if len(rows)!=a.expected_news or len(set(eids))!=a.expected_events:sys.exit(f'FAIL: news={len(rows)}, events={len(set(eids))}')
if sorted(set(eids))!=[f'zh_gold_{i:04d}' for i in range(1,a.expected_events+1)]:sys.exit('FAIL: non-contiguous public event IDs')
sizes=collections.Counter(eids)
print('PASS','news',len(rows),'events',len(sizes),'singletons',sum(v==1 for v in sizes.values()),'max_size',max(sizes.values()))
print('SHA256',hashlib.sha256(f.read_bytes()).hexdigest())
