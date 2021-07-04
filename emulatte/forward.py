from emulatte.scripts_forward import modelw
from emulatte.scripts_forward.transmitter import *

def model(thicks, **kwargs):
    mdl = modelw.Subsurface1D(thicks, **kwargs)
    return mdl

def transceiver(name, freqtime, **kwargs):
    cls = globals()[name]
    transmitter = cls(freqtime, **kwargs)
    return 
