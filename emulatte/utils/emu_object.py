
#%%
from __future__ import annotations

from abc import ABCMeta, abstractmethod
import numpy as np
from numpy.typing import NDArray, ArrayLike

class Model(metaclass=ABCMeta):
    @abstractmethod
    def set_params():
        pass

class Source(metaclass=ABCMeta):
    @abstractmethod
    def _compute_hankel_transform_dlf():
        pass

class DataArray(np.ndarray):
    def __new__(cls, data : ArrayLike, dtype=None, meta=None):
        self = np.asarray(data, dtype=dtype).view(cls)
        self.meta = meta
        return self