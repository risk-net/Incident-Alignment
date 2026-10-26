"""Minimal ports; heavy dependencies are entirely optional."""
from typing import Protocol, Sequence
import numpy as np
class TextEmbedder(Protocol):
    def encode(self,texts:Sequence[str])->np.ndarray: ...
class VectorIndex(Protocol):
    def search(self,queries:np.ndarray,k:int)->tuple[np.ndarray,np.ndarray]: ...
