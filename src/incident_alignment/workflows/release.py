"""Fail-closed, read-only release manifest (freeze metadata without modifying research assets)."""
from pathlib import Path
from datetime import datetime,timezone
from .io import hash_file,write_json_atomic


def freeze_manifest(assets:dict[str,str], *, out_file, version:str, metadata=None):
    if not assets or not version.strip():raise ValueError('assets and version required')
    files={}
    for label,filename in sorted(assets.items()):
        p=Path(filename).resolve(strict=True)
        if not p.is_file():raise ValueError(f'asset is not a file: {p}')
        files[label]={'path':str(p),'sha256':hash_file(p),'bytes':p.stat().st_size}
    doc={'version':version,'created_utc':datetime.now(timezone.utc).isoformat(),
         'assets':files,'metadata':metadata or {}}
    write_json_atomic(out_file,doc)
    return doc

def verify_manifest(path):
    from .io import read_json
    doc=read_json(path)
    for key,record in doc['assets'].items():
        if hash_file(record['path'])!=record['sha256']:raise ValueError(f'hash mismatch for {key}')
    return True
