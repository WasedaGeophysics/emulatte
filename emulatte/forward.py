from emulatte.scripts_forward import modelw
from emulatte.scripts_forward.transceiver import *

def model(thicks, **kwargs):
    mdl = modelw.Subsurface1D(thicks, **kwargs)
    return mdl

def transceiver(name, freqtime, **kwargs):
    cls = globals()[name]
    tcv = cls(freqtime, **kwargs)
    return tcv
