# Copyright 2021 Waseda Geophysics Laboratory
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .core import emlayers
from .core.emsource import *

def model(thicks):
    """
    Parameters
    ----------
    thicks : array-like \\
        List of layer thickness (m) \\
        e.g.)
            Air      ∞ m
            ------------
            L1     100 m
            ------------
            L2      50 m      =>    thicks = [100, 50, 200]
            ------------
            L3     200 m
            ------------
            L4       ∞ m

    """
    mdl = emlayers.Subsurface1D(thicks)
    return mdl

def transmitter(name, freqtime, **kwargs):
    """
    Parameters
    ----------

    name : string
        "VMD"               Vertical Magnetic Dipole

        "HMDx"              x-directed Horizontal Magnetic Dipole

        "HMDy"              y-directed Horizontal Magnetic Dipole

        "VED"               Vertical Electric Dipole

        "HEDx"              x-directed Horizontal Electric Dipole

        "HEDy"              y-directed Horizontal Electric Dipole

        "CircularLoop"      Circular Loop

        "CoincidentLoop"    Coincident Loop

        "GroundedWire"      Grounded Wire
    

    freqtime : number or list
        Frequencies (Hz) in FD transmittion
         or
        Sampling Time Gates (s) in TD transmittion

    **kwargs : see below

    Keyword Arguments
    -----------------

    - VMD, HMDx, HMDy
        moment : number


    - VED, HEDx, HEDy
        ds : number
            Finite length of bipole as a substitute for
            infinitesimal length of dipole
        
        current : number
            Amplitude of frequency-domain harmonic electrical current (A)
            or that of time-domain step-off current (A).
            
    - CircularLoop
        current : number
            Amplitude of frequency-domain harmonic electrical current (A)
            or that of time-domain step-off current (A).

        radius : number
            radius of the loop (m)

        turns : int
            number of loop turns


    - CoincidentLoop
        current : number
            Amplitude of frequency-domain harmonic electrical current (A)
            or that of time-domain step-off current (A).

        radius : number
            radius of the loop (m)

        turns : int
            number of loop turns

    - Grounded Wire
        current : number
            Amplitude of frequency-domain harmonic electrical current (A)
            or that of time-domain step-off current (A).

        split : int
            Splits the specified number of wires into current strands


    Returns
    -------
    ans : dictionary of received EM field
        keys:
            "e_x", "e_y", "e_z",
            "h_x", "h_y", "h_z"
    """
    cls = globals()[name]
    tmr = cls(freqtime, **kwargs)
    return tmr