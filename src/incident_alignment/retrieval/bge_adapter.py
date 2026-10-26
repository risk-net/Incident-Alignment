"""Explicit BGE-M3 adapter. No automatic downloads, instantiation or preprocessing."""
from typing import Sequence
import numpy as np
class BGEM3Adapter:
    def __init__(self,model): self.model=model
    def encode(self,texts:Sequence[str])->np.ndarray:
        result=self.model.encode(list(texts))
        if isinstance(result,dict):
            if 'dense_vecs' not in result:raise ValueError('BGE-M3 response lacks dense_vecs')
            result=result['dense_vecs']
        result=np.asarray(result,dtype=np.float32)
        if result.ndim!=2 or result.shape[0]!=len(texts):raise ValueError('unexpected BGE-M3 embedding shape')
        return result
