"""Adapter around caller-supplied FAISS index; no index creation or data mutation."""
import numpy as np
class FaissIndexAdapter:
    def __init__(self,index): self.index=index
    def search(self,queries:np.ndarray,k:int):
        if k<=0:raise ValueError('k must be positive')
        queries=np.asarray(queries,dtype=np.float32)
        if queries.ndim!=2:raise ValueError('queries must be 2D')
        return self.index.search(np.ascontiguousarray(queries),k)
