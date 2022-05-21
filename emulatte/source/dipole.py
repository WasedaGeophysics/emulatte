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
    compute_kernel_vmd_e_r,
    compute_kernel_vmd_h_r,
    compute_kernel_vmd_h_z
)


class VMD(Source):
    def __init__(self, moment, ontime = None, frequency=None) -> None:
        moment = np.array(moment, ndmin=1, dtype=complex)
        self.moment = moment

        self.kernel_te_up_sign = 1
        self.kernel_te_down_sign = 1
        self.kernel_tm_up_sign = 0
        self.kernel_tm_down_sign = 0

        magnitude = moment
        
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
            self, model, direction, bessel_j0, bessel_j1, magnetic : bool
            ) -> NDArray:
        # kernel components
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

        ans = []

        # Electric field E
        if not magnetic:
            kernel_e_r = None
            if "x" in direction:
                factor = impedivity_s / (4 * np.pi) * sin_phi
                kernel_e_r = compute_kernel_vmd_e_r(
                    u_te, d_te, e_up, e_down, si, ri, us, zs, z, lambda_)
                e_x = factor * (kernel_e_r @ bessel_j1) / rho
                ans.append(e_x)
            if "y" in direction:
                if kernel_e_r is None:
                    factor = impedivity_s / (4 * np.pi) * -cos_phi
                    kernel_e_r = compute_kernel_vmd_e_r(
                        u_te, d_te, e_up, e_down, si, ri, us, zs, z, lambda_)
                    e_y = factor * (kernel_e_r @ bessel_j1) / rho
                else:
                    # e_xの結果を再利用
                    e_y = e_x * -cos_phi / sin_phi
                ans.append(e_y)
            if "z" in direction:
                e_z = np.zeros(nfreq)
                ans.append(e_z)
        # Magnetic field H
        else:
            kernel_h_r = None
            if "x" in direction:
                factor = cos_phi / (4 * np.pi) * impedivity_s / impedivity_r
                kernel_h_r = compute_kernel_vmd_h_r(
                                        u_te, d_te, e_up, e_down,
                                        si, ri, us, ur, zs, z, lambda_)
                h_x = factor * (kernel_h_r @ bessel_j1) / rho
                ans.append(h_x)
            if "y" in direction:
                if kernel_h_r is None:
                    factor = sin_phi / (4 * np.pi) * impedivity_s / impedivity_r
                    kernel_h_r = compute_kernel_vmd_h_r(
                                        u_te, d_te, e_up, e_down, 
                                        si, ri, us, ur, zs, z, lambda_)
                    h_y = factor * (kernel_h_r @ bessel_j1) / rho
                else:
                    h_y = h_x * sin_phi / cos_phi
                ans.append(h_y)
            if "z" in direction:
                factor = 1 / (4 * np.pi) * impedivity_s / impedivity_r
                kernel_h_z = compute_kernel_vmd_h_z(
                                        u_te, d_te, e_up, e_down, 
                                        si, ri, us, zs, z, lambda_)
                h_z = factor * (kernel_h_z @ bessel_j0) / rho
                ans.append(h_z)

        ans = np.array(ans)

        return ans