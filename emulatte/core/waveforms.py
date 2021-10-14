import numpy as np
from emulatte.waveform_files import walktem_hm, walktem_lm


def load_waveform(waveform_name, moment_type):
    # TODO 探査システムの追加
    if waveform_name == "walktem":
        if moment_type == "hm":
            off_time, waveform_time, waveform_current = load_walktem_hm()
        elif moment_type == "lm":
            off_time, waveform_time, waveform_current = load_walktem_lm()
        else:
            raise NameError('invalid moment type')
    else:
        raise NameError('invalid waveform name')

    # return off_time, waveform_time, waveform_current
    return np.array(off_time), np.array(waveform_time), np.array(waveform_current)


def load_walktem_hm():
    off_time = walktem_hm.off_time
    waveform_time = walktem_hm.waveform_time
    waveform_current = walktem_hm.waveform_current
    return off_time, waveform_time, waveform_current


def load_walktem_lm():
    off_time = walktem_lm.off_time
    waveform_time = walktem_lm.waveform_time
    waveform_current = walktem_lm.waveform_current
    return off_time, waveform_time, waveform_current
