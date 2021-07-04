from emulatte.scripts_forward import modelw
from emulatte.scripts_forward.transmitter import *

def model(thicks, **kwargs):
    mdl = modelw.Subsurface1D(thicks, **kwargs)
    return mdl

def transmitter(name, freqtime, **kwargs):
    cls = globals()[name]
    tmr = cls(freqtime, **kwargs)
    return tmr
