from emulatte.forwardscr import emgmodel
from emulatte.forwardscr.transmitter import *

def model(thicks, **kwargs):
    mdl = emgmodel.Subsurface1D(thicks, **kwargs)
    return mdl

def transmitter(name, freqtime, **kwargs):
    cls = globals()[name]
    tmr = cls(freqtime, **kwargs)
    return tmr