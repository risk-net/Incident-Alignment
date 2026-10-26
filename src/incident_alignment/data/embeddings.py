"""Read-only aligned embedding store for precomputed numpy files."""
from pathlib import Path
import numpy as np

def validate_embedding_assets(root: str | Path, expected_ids: list[str], fields: list[str]) -> dict:
    root=Path(root)
    ids=[line.strip() for line in (root/'case_ids.txt').read_text(encoding='utf-8').splitlines() if line.strip()]
    if ids!=[str(v) for v in expected_ids]:raise ValueError('embedding case_ids ordering differs from expected IDs')
    if len(ids)!=len(set(ids)):raise ValueError('duplicated embedding IDs')
    shape={}
    for field in fields:
        arr=np.load(root/f'emb_{field}.npy',mmap_mode='r',allow_pickle=False)
        mask=np.load(root/f'valid_mask_{field}.npy',mmap_mode='r',allow_pickle=False)
        if arr.ndim!=2 or arr.shape[0]!=len(ids) or mask.shape[0]!=len(ids):
            raise ValueError(f'bad embedding/mask shapes for {field}: {arr.shape}, {mask.shape}')
        if mask.ndim not in (1,2) or (mask.ndim==2 and mask.shape[1]!=1):
            raise ValueError(f'bad mask dimensions for {field}')
        shape[field]=tuple(arr.shape)
    return shape

class MemmapEmbeddings:
    def __init__(self,root: str | Path, fields: list[str],expected_ids: list[str]):
        self.root=Path(root)
        self.shapes=validate_embedding_assets(root,expected_ids,fields)
        self.indices={str(x):i for i,x in enumerate(expected_ids)}
        self.embeddings={field:np.load(self.root/f'emb_{field}.npy',mmap_mode='r',allow_pickle=False) for field in fields}
        self.masks={field:np.load(self.root/f'valid_mask_{field}.npy',mmap_mode='r',allow_pickle=False) for field in fields}
    def get(self,case_id: str,field: str):
        i=self.indices[str(case_id)]
        return self.embeddings[field][i],bool(np.asarray(self.masks[field][i]).item())
