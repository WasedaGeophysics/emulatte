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
import scipy.constants as const

from ..utils.converter import array
from ..function.filter import load_hankel_filter, load_fft_filter
from .element import *

class EM1D:
    r"""Summary

    """
    def __init__(self, thickness):
        r"""
        Parameters
        ----------
        thick : array_like
            layer thickness (m), except for the first (on the ground) & last layers
        """
        thickness = array(thickness)
        # LAYERS BOUNDARY DEPTH
        self.depth = np.array([0, *np.cumsum(thickness)])
        # THICKNESS
        self.thick = np.array([np.inf, *thickness, np.inf])
        # NUMBER OF LAYERS
        self.N = len(thickness) + 1
        # LENGTH OF LAYER PROPERTY ARRAYS
        self.L = len(thickness) + 2
        # STATE
        self.state = "quasistatic"

    def set_params(self, resistivity, epsilon_r=None, mu_r=None):
        r"""
        Parameters
        ----------
        res : array_like
            Resistivity :math:`\rho` (Ωm)

        epsilon_r : array-like
            Relative Electric Permittivity (-)
            default -> `epsilon_r` = [1, 1, ... ]

        mu_r : array-like
            Relative Magnetic Permeability (-)
            default -> `mu_r` = [1, 1, ... ]
        """
        ### define conductivity ###
        sigma = 1 / array(resistivity)
        if not len(sigma) == self.N:
            raise Exception
        self.sigma = np.append(0, sigma)
        
        ### define electric permittivity ###
        if epsilon_r is None:
            epsilon = np.ones(self.N) * const.epsilon_0
        else:
            if not (len(epsilon_r) == self.N):
                raise Exception
            else:
                epsilon = array(epsilon_r) * const.epsilon_0
        self.epsilon = np.append(const.epsilon_0, epsilon)

        ### define magnetic permeability ###
        if mu_r is None:
            mu = np.ones(self.N) * const.mu_0
        else:
            if not (len(mu_r) == self.N):
                raise Exception
            else:
                mu = array(mu_r) * const.mu_0
        self.mu = np.append(const.mu_0, mu)
        self.params_applied = True
        
    def set_source(self, source, loc):
        self.source = source
        self.sc = array(loc)
        self.source_type = source.__class__.__name__
        self.source_applied = True
        
    def set_filter(self, hankel_filter, ftdt_filter=None):
        self.hankel_filter = hankel_filter
        self.ftdt_filter = ftdt_filter
        self.filter_applied = True
        
    def field(self, which, direction, loc, freqtime, time_diff=False):
        self._check_input()
        direction = [char for char in direction]
        self.rc = array(loc)
        self.time_diff = time_diff
        self._geometric_basics()
        signal = self.source.signal
        if signal == "f":
            omega = 2 * np.pi * array(freqtime)
            if which == "E":
                ans = self._electric_field_f(direction, omega)
            elif which == "H":
                ans = self._magnetic_field_f(direction, omega)
            elif which == "D":
                eperm = self.epsilon[self.ri]
                ans = eperm * self._electric_field_f(direction, omega)
            elif which == "B":
                mperm = self.mu[self.ri]
                ans = mperm * self._magnetic_field_f(direction, omega)
            elif which == "J":
                sigma = self.sigma[self.ri]
                ans = sigma * self._electric_field_f(direction, omega)
            else:
                raise Exception
        else:
            time = array(freqtime)
            if which == "E":
                ans = self._electric_field_t(direction, time)
            elif which == "H":
                ans = self._magnetic_field_t(direction, time)
            elif which == "D":
                eperm = self.epsilon[self.ri]
                ans = eperm * self._electric_field_t(direction, time)
            elif which == "B":
                mperm = self.mu[self.ri]
                ans = mperm * self._magnetic_field_t(direction, time)
            elif which == "J":
                sigma = self.sigma[self.ri]
                ans = sigma * self._electric_field_t(direction, time)
            else:
                raise Exception
        return ans

    def receive(self, receiver, rc):
        # 受信機器固有の測定量（コインシデント、電圧ダイポールなど）
        pass

    def _check_input(self):
        if not self.params_applied:
            print('set parameter')
            raise Exception
        elif not self.source_applied:
            print('set source')
            raise Exception
        elif not self.filter_applied:
            print('set filter')
        else:
            pass

    def _geometric_basics(self):
        def in_which_layer(z, depth):
            layer_index = self.N
            for i in range(self.L-1):
                if z <= depth[i]:
                    layer_index = i
                    break
                else:
                    continue
            return layer_index

        point_source = ["VMD", "HMD", "HED", "Loop"]
        if self.source_type in point_source:
            sx, sy, sz = self.sc
            rx, ry, rz = self.rc
            r = ((sx-rx) ** 2 + (sy-ry) ** 2 + (sz-rz) ** 2) ** 0.5
            if r == 0:
                r = 1e-8

            # ANGLE
            cos_phi = (rx - sx) / r
            sin_phi = (ry - sy) / r

            # 特異点の回避
            if self.hankel_filter == "anderson801":
                delta_z = 1e-4
            else:
                delta_z = 1e-8
            # TODO WHY?
            if sz in self.depth:
                sz = sz - delta_z
            # to avoid singularity of source potential
            if sz == rz:
                sz = sz - delta_z

            # 送受信点が含まれる層の特定
            si = in_which_layer(sz, self.depth)
            ri = in_which_layer(rz, self.depth)

            # return to self
            self.sx, self.sy, self.sz = sx, sy, sz
            self.rx, self.ry, self.rz = rx, ry, rz
            self.si = si
            self.ri = ri
            self.r = r
            self.cos_phi = cos_phi
            self.sin_phi = sin_phi

        elif self.source_type == "Line":
            pass
            
    def _compute_admittivity(self, omega):
        self.admz = 1j * omega.reshape(-1,1) * self.mu.reshape(1,-1)
        self.admy = (np.ones((self.K, self.L)) * self.sigma.reshape(1,-1)).astype(complex)
        self.admy[:, 0] = 1j * omega * self.epsilon[0]
        return None

    def _compute_kernel_integrants(self, omega, y_base, r):
        """
        Parameters
        ----------
        omega : ndarray(ndim=1, hankel_length)
        lambda_ : ndarray(ndmi=1, length=hankel_length)
        """
        lambda_ = y_base / r
        self.lambda_ = lambda_
        self.K = len(omega)
        self.M = len(lambda_)
        self._compute_admittivity(omega)
        self.k = np.sqrt(-self.admy*self.admz)
        # calculate 3d tensor u = lambda**2 - k**2
        lambda_v = np.zeros((self.K, self.L, self.M), dtype=complex)
        lambda_v[:,:] = lambda_
        k_v = np.zeros((self.M, self.L, self.K), dtype=complex)
        k_v[:] = self.k.T
        k_v = k_v.transpose((2,1,0))
        self.u = np.sqrt(lambda_v ** 2 - k_v ** 2)
        # compute damping coefficients
        # TODO separate TE, TM, and both
        u_te, d_te, u_tm, d_tm, e_up, e_down = tetm_mode_damping(self)
        self.u_te, self.d_te = u_te, d_te
        self.u_tm, self.d_tm = u_tm, d_tm
        self.e_up, self.e_down = e_up, e_down
        # HACK for debug
        self.dr_te, self.dr_tm, self.ur_te, self.ur_tm, self.yuc, self.zuc = \
            reflection(
                self.si, self.K, self.L, self.M, self.u,
                self.thick, self.admy, self.admz
                )
        return lambda_

    def _electric_field_f(self, direction, omega):
        # 方向ごとの繰り返し計算不要
        y_base, wt0, wt1 = load_hankel_filter(self.hankel_filter)
        ans = self.source._hankel_transform_e(self, direction, omega, y_base, wt0, wt1)
        if len(direction) == 1:
            ans = ans[0]
        return ans

    def _magnetic_field_f(self, direction, omega):
        # 方向ごとの繰り返し計算不要
        y_base, wt0, wt1 = load_hankel_filter(self.hankel_filter)
        ans = self.source._hankel_transform_h(self, direction, omega, y_base, wt0, wt1)
        if len(direction) == 1:
            ans = ans[0]
        return ans

    def _electric_field_t(self, xyz_comp):
        ans = np.array([])
        return ans