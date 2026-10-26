"""Controlled legacy five-fold training hand-off; preserves original sampling and checkpoint selection.

This is an adapter, not a replacement for the original train_fold implementation.
"""
from pathlib import Path
from .io import hash_file,write_json_atomic

def prepare_legacy_training(script, split_root, *, out_file, interpreter='python', args=()):
    script=Path(script).resolve(strict=True)
    splits=Path(split_root).resolve(strict=True)
    if not script.is_file() or not splits.is_dir():raise ValueError('invalid legacy training assets')
    if any(str(a).startswith('--test') or str(a).startswith('--chinese') for a in args):
        raise ValueError('test/Chinese Gold arguments are not part of training')
    payload={'status':'requires_explicit_execution','adapter':'legacy_five_fold',
             'script':str(script),'script_sha256':hash_file(script),
             'split_root':str(splits),'argv':[str(interpreter),str(script),*map(str,args)],
             'notes':'Execute legacy training with original sampling/dev checkpoint semantics; no assets modified here.'}
    write_json_atomic(out_file,payload)
    return payload
