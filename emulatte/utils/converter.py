from msilib.schema import Component
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

def check_tx_type(ontime):
    if ontime is None:
        tx_type = "f"
    else:
        ontime = array(ontime)
        if len(ontime) > 1:
            tx_type = "a"
        elif len(ontime) == 1:
            if ontime[0] > 0:
                tx_type = "step-on"
            elif ontime[0] == 0:
                tx_type = "impulse"
            else:
                tx_type = "step-off"
        else:
            raise Exception(ValueError)
    return tx_type