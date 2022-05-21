# Copyright 2022 Waseda Geophysics Laboratory
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

# -*- coding: utf-8 -*-

import numpy as np
from numpy.typing import NDArray
from ..utils.converter import check_waveform
from ..utils.emu_object import Source
from .kernel.edipole import *
from .kernel.mdipole import (
    compute_kernel_loop_e_r,
    compute_kernel_loop_h_r,
    compute_kernel_loop_h_z
)


class CircularLoop(Source):
    def __init__(self, current, radius, turns = 1, ontime = None, frequency = None):
        current = np.array(current, ndmin=1, dtype=complex)
        self.current = current
        self.radius = radius
        self.turns = turns
        self.area = np.pi * self.radius ** 2

        self.kernel_te_up_sign = 1
        self.kernel_te_down_sign = 1
        self.kernel_tm_up_sign = 0
        self.kernel_tm_down_sign = 0

        magnitude = current

        domain, signal, magnitude, ontime, frequency = \
            check_waveform(magnitude, ontime, frequency)

        self.domain = domain
        self.signal = signal
        self.ontime = ontime
        self.frequency = frequency
        if domain == "frequency":
            self.magnitude_f = magnitude
        else:
            self.magnitude_f = 1
            self.magnitude_t = magnitude

    def _compute_hankel_transform_dlf(
            self, model, direction, bessel_j0, bessel_j1, magnetic : bool):
        si = model.si[0]
        zs = model.zs[0]
        ri = model.ri
        z = model.z

        rho = model.rho[0]
        lambda_ = model.lambda_[0]

        us = model.u[0,:,si]
        ur = model.u[0,:,ri]

        impedivity_s = model.impedivity[:, si]
        impedivity_r = model.impedivity[:, ri]
        
        u_te = model.u_te[0]
        d_te = model.d_te[0]
        e_up = model.e_up[0]
        e_down = model.e_down[0]

        sin_phi = model.sin_phi[0]
        cos_phi = model.cos_phi[0]

        nfreq = model.nfreq
        rad = self.radius

        ans = []
        
        if not magnetic:
            kernel_er = None
            if "x" in direction:
                factor = impedivity_s * self.radius * sin_phi / 2
                kernel_e_r = compute_kernel_loop_e_r(
                    u_te, d_te, e_up, e_down, si, ri, us, zs, z, lambda_, rho)
                e_x = factor * (kernel_e_r @ bessel_j1) / rad
                ans.append(e_x)
            if "y" in direction:
                if kernel_e_r is None:
                    factor = impedivity_s * self.radius * -cos_phi / 2
                    kernel_e_r = compute_kernel_loop_e_r(
                                        u_te, d_te, e_up, e_down, 
                                        si, ri, us, zs, z, lambda_, rho)
                    e_y = factor * (kernel_e_r @ bessel_j1) / rad
                else:
                    e_y = e_x * -cos_phi / sin_phi
                ans.append(e_y)
            if "z" in direction:
                e_z = np.zeros(nfreq)
                ans.append(e_z)
        
        else:
            kernel_h_r = None
            if "x" in direction:
                factor = self.radius * -cos_phi / 2
                factor = factor * impedivity_s / impedivity_r
                kernel_h_r = compute_kernel_loop_h_r(
                                        u_te, d_te, e_up, e_down,
                                        si, ri, us, ur, zs, z, lambda_, rho)
                h_x = factor * (kernel_h_r @ bessel_j1) / rad
                ans.append(h_x)
            if "y" in direction:
                if kernel_h_r is None:
                    factor = self.radius * -sin_phi / 2
                    factor = factor * impedivity_s / impedivity_r
                    kernel_h_r = compute_kernel_loop_h_r(
                                        u_te, d_te, e_up, e_down,
                                        si, ri, us, ur, zs, z, lambda_, rho)
                    h_y = factor * (kernel_h_r @ bessel_j1) / rad
                else:
                    h_y = h_x * sin_phi / cos_phi
                ans.append(h_y)
            if "z" in direction:
                factor = self.radius / 2 * impedivity_s / impedivity_r
                kernel_h_z = compute_kernel_loop_h_z(
                                        u_te, d_te, e_up, e_down, 
                                        si, ri, us, zs, z, lambda_, rho)
                h_z = factor * (kernel_h_z @ bessel_j1) / rad
                ans.append(h_z)

        ans = np.array(ans) * self.turns
        return ans