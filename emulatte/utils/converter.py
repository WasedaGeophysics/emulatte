import numpy as np
import numbers

def is_scalar(obj : object) -> bool:
    result : bool = isinstance(obj, numbers.Number)
    return result

def check_waveform(magnitude, ontime, frequency):
    # sourceの入力処理
    if ontime is None:
        domain = "frequency"
        if is_scalar(magnitude):
            signal = "constant"
            magnitude = complex(magnitude)
        elif len(magnitude) == len(frequency):
            signal = "specific"
            magnitude = np.array(magnitude, dtype=complex)
            frequency = np.array(frequency, dtype=float)
        else:
            raise Exception(ValueError)

    elif frequency is None:
        domain = "time"
        if type(magnitude) in {float, int}:
            signal = "default"
            magnitude = float(magnitude)
            if ontime == 1:
                signal = "stepon"
            elif ontime == 0:
                signal = "impulse"
            elif ontime == -1:
                signal = "stepoff"
        elif (len(magnitude) == len(ontime)) \
                & (magnitude[0] == magnitude[-1] == 0) \
                & len(ontime) > 2:
            signal = "arbitrary"
            magnitude = np.array(magnitude, dtype=float)
            ontime = np.array(ontime, dtype=float)
        else:
            raise Exception(ValueError)
    else:
        raise Exception("cannot recognize domain. Please make sure source's \
                         ontime or frequency is None")
    return domain, signal, magnitude, ontime, frequency