import numpy as np
import numbers

def is_scalar(obj : object) -> bool:
    result : bool = isinstance(obj, numbers.Number)
    return result

def check_waveform(magnitude, ontime, frequency):
    size = len(magnitude)
    if ontime is None:
        domain = "frequency"
        if size == 1:
            signal = "constant"
            magnitude = complex(magnitude[0])
        elif size == len(frequency):
            signal = "specific"
            magnitude = np.array(magnitude, dtype=complex)
            frequency = np.array(frequency, dtype=float)
        else:
            raise Exception(ValueError)

    elif frequency is None:
        domain = "time"
        if size == 1:
            signal = "default"
            magnitude = magnitude.astype(float)
            if ontime == 1:
                signal = "stepon"
            elif ontime == 0:
                signal = "impulse"
            elif ontime == -1:
                signal = "stepoff"
            else:
                raise Exception(ValueError)
        elif (size == len(ontime)) \
                & (magnitude[0] == magnitude[-1] == 0) \
                & len(ontime) > 2:
            signal = "arbitrary"
            magnitude = magnitude.astype(float)
            ontime = np.array(ontime, dtype=float)
        else:
            raise Exception(ValueError)
    else:
        raise Exception(("cannot recognize domain. Please make sure source's"
                         "ontime or frequency is None"))
    return domain, signal, magnitude, ontime, frequency

def split_time(time):
    nnegative = sum(time < 0)
    time_neg = time[:nnegative]
    time_pos = time[nnegative:]
    return