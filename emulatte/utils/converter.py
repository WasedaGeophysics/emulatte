import numpy as np
import sys

def kroneckers_delta(ii, jj):
    if ii == jj:
        return 1
    else:
        return 0

def array(array_like):
    array_types = [list, tuple, np.ndarray]
    if type(array_like) in types:
        array = np.array(array_like, dtype=np.float64)
    elif type(array_like) in [int, float]:
        array = np.array([array_like], dtype=np.float64)
    elif type(array_like) in [complex]:
        array = np.array([array_like], dtype=np.complex64)
    else:
        raise Exception('invalid parameter type')
    return array