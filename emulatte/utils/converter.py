import numpy as np

def array(array_like):
    array_types = [list, tuple, np.ndarray]
    if type(array_like) in array_types:
        array = np.array(array_like, dtype=np.float64)
    elif type(array_like) in [int, float]:
        array = np.array([array_like], dtype=np.float64)
    elif type(array_like) in [complex]:
        array = np.array([array_like], dtype=np.complex64)
    else:
        raise Exception('invalid parameter type')
    return array

def check_waveform(ontime):
    if ontime is None:
        tx_type = "f"
    else:
        ontime = array(ontime)
        if len(ontime) > 1:
            tx_type = "awave"
        elif len(ontime) == 1:
            if ontime[0] > 0:
                tx_type = "stepon"
            elif ontime[0] == 0:
                tx_type = "impulse"
            else:
                tx_type = "stepoff"
        else:
            raise Exception(ValueError)
    return tx_type