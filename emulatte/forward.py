
# Implementation has been based on Ward and Hohmann (1987) and Zhanhui Li,et.al.(2018)
# Developed MATLAB and Python version by I.Saito since 2019
# Redesigned by T.Nishino in 2021

from emulatte.core import emlayers
from emulatte.core.emsource import *

def model(thicks, **kwargs):
    mdl = emlayers.Subsurface1D(thicks, **kwargs)
    return mdl

def transmitter(name, freqtime, **kwargs):
    cls = globals()[name]
    tmr = cls(freqtime, **kwargs)
    return tmr