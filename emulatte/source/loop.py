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
from ..utils.interpret import check_waveform
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
            self, model, direction, magnetic : bool):
        model._compute_kernel_components()
        si = model.si
        zs = model.zs
        ri = model.ri
        z = model.z

        rho = model.rho
        lambda_ = model.lambda_
        bessel_j1 = model.bessel_j1

        us = model.u[:,si]
        ur = model.u[:,ri]

        impz_s = model.impedivity[:, si]
        impz_r = model.impedivity[:, ri]
        
        u_te = model.u_te
        d_te = model.d_te
        e_up = model.e_up
        e_down = model.e_down

        sin_phi = model.sin_phi
        cos_phi = model.cos_phi

        nfreq = model.nfreq
        rad = self.radius

        ans = []
        
        if not magnetic:
            kernel_er = None
            if "x" in direction:
                factor = impz_s * self.radius * sin_phi / 2
                kernel_e_r = compute_kernel_loop_e_r(
                    u_te, d_te, e_up, e_down, si, ri, us, zs, z, lambda_, rho)
                e_x = factor * (kernel_e_r @ bessel_j1) / rad
                ans.append(e_x)
            if "y" in direction:
                if kernel_e_r is None:
                    factor = impz_s * self.radius * -cos_phi / 2
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
                factor = factor * impz_s / impz_r
                kernel_h_r = compute_kernel_loop_h_r(
                                        u_te, d_te, e_up, e_down,
                                        si, ri, us, ur, zs, z, lambda_, rho)
                h_x = factor * (kernel_h_r @ bessel_j1) / rad
                ans.append(h_x)
            if "y" in direction:
                if kernel_h_r is None:
                    factor = self.radius * -sin_phi / 2
                    factor = factor * impz_s / impz_r
                    kernel_h_r = compute_kernel_loop_h_r(
                                        u_te, d_te, e_up, e_down,
                                        si, ri, us, ur, zs, z, lambda_, rho)
                    h_y = factor * (kernel_h_r @ bessel_j1) / rad
                else:
                    h_y = h_x * sin_phi / cos_phi
                ans.append(h_y)
            if "z" in direction:
                factor = self.radius / 2 * impz_s / impz_r
                kernel_h_z = compute_kernel_loop_h_z(
                                        u_te, d_te, e_up, e_down, 
                                        si, ri, us, zs, z, lambda_, rho)
                h_z = factor * (kernel_h_z @ bessel_j1) / rad
                ans.append(h_z)

        ans = np.array(ans) * self.turns
        return ans

class PolygonalLoop(Source):
    def __init__(self, current, turns=1, split=None, ontime = None, frequency = None):
        current = np.array(current, ndmin=1, dtype=complex)
        self.current = current
        self.turns = turns
        self.split = split
        self.kernel_te_up_sign = 1
        self.kernel_te_down_sign = 1
        self.kernel_tm_up_sign = -1
        self.kernel_tm_down_sign = 1

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
            self, model, direction, magnetic : bool):
        model._compute_kernel_components_multi()
        si = model.si
        zs = model.zs
        ri = model.ri
        z = model.z

        lambda_ = model.lambda_
        bessel_j0 = model.bessel_j0
        bessel_j1 = model.bessel_j1

        us = model.u[:,:,si]
        ur = model.u[:,:,ri]

        impz_s = model.impedivity[:, si]
        impz_r = model.impedivity[:, ri]
        
        u_te = model.u_te
        d_te = model.d_te
        e_up = model.e_up
        e_down = model.e_down

        cos_theta = model.cos_theta
        sin_theta = model.sin_theta

        nvertex = model.nvertex
        split = model.split
        ds = model.ds
        y_ys_pole = model.dist_y_pole
        r = model.rho_list
        n = model.index_slice

        jj = model.ndims[0]
        kk = model.ndims[1]
        mm = model.ndims[3]

        ans = []

        lambda3d = np.zeros((kk, jj, mm), dtype=complex)
        lambda3d[:] = lambda_
        lambda3d = lambda3d.transpose((1,0,2))

        nfreq = model.nfreq

        if not magnetic:
            if ("x" in direction) or ("y" in direction):
                kernel_e_r_te = compute_kernel_hed_e_r_te(
                    u_te, d_te, e_up, e_down, si, ri, us, ur, zs, z, lambda3d
                )
                e_x = 0
                e_y = 0
                for i in range(nvertex):
                    te_ex_line = (kernel_e_r_te[0][n[i]:n[i+1]] @ bessel_j0).T / r[i]
                    te_ex_line = te_ex_line @ np.ones(split[i])
                    ex_line = - impz_s / (4 * np.pi) * ds[i] * te_ex_line

                    if "x" in direction:
                        e_x = cos_theta[i] * ex_line + e_x
                    if "y" in direction:
                        e_y = sin_theta[i] * ex_line + e_y

                if "x" in direction:
                    ans.append(e_x)
                if "y" in direction:
                    ans.append(e_y)

            if "z" in direction:
                e_z = np.zeros(nfreq, dtype=complex)
                ans.append(e_z)

            ans = np.array(ans, dtype=complex)

        # Magnetic field H
        else:
            kernel_exist = False
            if ("x" in direction) or ("y" in direction):
                kernel_h_r_te = compute_kernel_hed_h_r_te(
                    u_te, d_te, e_up, e_down, si, ri, us, ur, zs, z, lambda3d
                )
                h_x = 0
                h_y = 0
                for i in range(nvertex):
                    te_hy_line = np.dot(kernel_h_r_te[0][n[i]:n[i+1]], bessel_j0).T / r[i]
                    te_hy_line = te_hy_line @ np.ones(split[i])
                    hy_line = 1 / (4 * np.pi) * impz_s / impz_r * te_hy_line * ds[i]

                    if "x" in direction:
                        h_x = -sin_theta[i] * hy_line + h_x
                    if "y" in direction:
                        h_y = cos_theta[i] * hy_line + h_y
                        
                if "x" in direction:
                    ans.append(h_x)
                if "y" in direction:
                    ans.append(h_y)

            if "z" in direction:
                kernel_h_z_te = compute_kernel_hed_h_z_te(
                     u_te, d_te, e_up, e_down, si, ri, us, ur, zs, z, lambda3d
                )
                h_z = 0
                for i in range(nvertex):
                    te_hz_line = (kernel_h_z_te[n[i]:n[i+1]] @ bessel_j1).T / r[i]
                    te_hz_line = te_hz_line @ (y_ys_pole[i] / r[i])
                    hz_line = - 1 / (4 * np.pi) * impz_s / impz_r * te_hz_line * ds[i]

                    h_z = hz_line + h_z
                ans.append(h_z)
        
        ans = np.array(ans) * self.turns
        return ans