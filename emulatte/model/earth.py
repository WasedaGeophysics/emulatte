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
from ..dlf.loader import load_hankel_filter
from ..tdem.ftdt import lagged_convolution, interp_fft
from .element import *

class DynamicEM1D:
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
        self.state = "dynamic"
        # DEFAULT HANKEL FILTER
        self.hankel_filter = "key201"
        # DEFAULT FTDT METHOD
        self.ftdt_method = "dlag"

        self.set_params_done = False
        self.set_source_done = False
        self.set_filter_done = False


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
        self.set_params_done = True
        
    def set_source(self, source, loc):
        r"""
        Parameters
        ----------
        source : emulatte Source class

        loc : array-like
            coordinate of specified source (x, y, z)
        """
        self.source = source
        self.sc = array(loc)
        self.source_type = source.__class__.__name__
        self.set_source_done = True
        
    def set_filter(self, hankel_filter, ftdt_config=None):
        r"""
        Parameters
        ----------
        hankel_filter : str or array-like
            name of linear filter of Hankel transformation
            - collection -
            anderson801 / kong241 / mizunaga90 / werthmuller201 / key201


        ftdt_config : dict
            select a method for frequency-time domain transformation
            - collection -
            dlag

            default = dlag

        """

        if type(hankel_filter) is str:
            self.hankel_filter = hankel_filter
        #TODO hankel filter direct input
        else:
            raise Exception

        if ftdt_config is None:
            ftdt_config = {
                "method" : "dlag"
            }

        self.ftdt_method = ftdt_config["method"]
        if ftdt_config["method"] == "dlag":
            pass
        elif ftdt_config["method"] == "fft":
            self.fft_filter = ftdt_config["filter"]
        else:
            raise Exception

        self.set_filter_done = True
        
    def field(self, which, direction, loc, freqtime, time_derivative=False):
        r"""
        Parameters
        ----------
        which : str
            type of electromagnetic vector field
            'E' for electric field
            'H' for magnetic field
            'D' for electric flux density
            'B' for magnetic flux density
            'J' for electrical current

        direction : str
            measurement direction, x, y, and z

        loc : array-like
            coordinate of measurement point (x, y, z)

        freqtime : array-like
            measurement frequency or time of the receiver

        time-derivative : bool
            if True is input, the returned values are time derivatives

        Returns
        -------
        ans : ndarray
            in FD, dtype = complex
            in TD, dtype = real 
        """
        self._check_input()
        direction = [char for char in direction]
        self.rc = array(loc)
        self.time_derivative = time_derivative
        self._calc_geometric_basics()
        signal = self.source.signal
        if signal == "f":
            omega = 2 * np.pi * array(freqtime)
            if which == "E":
                ans = self._em_field_f('e', direction, omega)
            elif which == "H":
                ans = self._em_field_f('m', direction, omega)
            elif which == "D":
                eperm = self.epsilon[self.ri]
                ans = eperm * self._em_field_f('e', direction, omega)
            elif which == "B":
                mperm = self.mu[self.ri]
                ans = mperm * self._em_field_f('m', direction, omega)
            elif which == "J":
                sigma = self.sigma[self.ri]
                ans = sigma * self._em_field_f('e', direction, omega)
            else:
                raise Exception
        else:
            time = array(freqtime)
            if which == "E":
                ans = self._em_field_t('e', direction, time)
            elif which == "H":
                ans = self._em_field_t('m', direction, time)
            elif which == "D":
                eperm = self.epsilon[self.ri]
                ans = eperm * self._em_field_t('e', direction, time)
            elif which == "B":
                mperm = self.mu[self.ri]
                ans = mperm * self._em_field_t('m', direction, time)
            elif which == "J":
                sigma = self.sigma[self.ri]
                ans = sigma * self._em_field_t('e', direction, time)
            else:
                raise Exception
        return ans

    def receive(self, receiver, rc):
        # 受信機器固有の測定量（コインシデント、電圧ダイポールなど）
        # そのうち実装する予定
        pass

    def _check_input(self):
        if not self.set_params_done:
            print("set parameter")
            raise Exception
        elif not self.set_source_done:
            print("set source")
            raise Exception
        elif not self.set_filter_done:
            self.set_filter("key201", {"method" : "dlag"})
        else: 
            pass

    def _calc_geometric_basics(self):
        def _get_layer_index(z, depth):
            is_above = list(depth <= z)
            layer_index = is_above.count(True)
            return layer_index

        point_source = ["VMD", "HMD", "HED", "HCL"]
        if self.source_type in point_source:
            sx, sy, sz = self.sc
            rx, ry, rz = self.rc
            rh = ((sx-rx) ** 2 + (sy-ry) ** 2) ** 0.5
            if rh == 0:
                rh = 1e-8
            # ANGLE
            cos_phi = (rx - sx) / rh
            sin_phi = (ry - sy) / rh
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
            si = _get_layer_index(sz, self.depth)
            ri = _get_layer_index(rz, self.depth)
            # return to self
            self.sx, self.sy, self.sz = sx, sy, sz
            self.rx, self.ry, self.rz = rx, ry, rz
            self.si, self.ri = si, ri
            self.rh, self.cos_phi, self.sin_phi = rh, cos_phi, sin_phi

        elif self.source_type == "HGW":
            sx, sy, sz = self.sc.T
            rx, ry, rz = self.rc
            length = np.sqrt((sx[1] - sx[0]) ** 2 + (sy[1] - sy[0]) ** 2)
            cos_theta = (sx[1] - sx[0]) / length
            sin_theta = (sy[1] - sy[0]) / length

            if sz[0] != sz[1]:
                raise Exception('Z-coordinates of the wire ends must be the same value.')

            # 計算できない送受信座標が入力された場合の処理
            # 特異点の回避
            if self.hankel_filter == "anderson801":
                delta_z = 1e-4
            else:
                delta_z = 1e-8
            if sz[0] in self.depth:
                sz = sz - delta_z
            if sz[0] == rz:
                sz = sz - delta_z

            if self.source.split is None:
                ssx = sx[1] - sx[0]
                ssy = sy[1] - sy[0]
                srx = rx - sx[0]
                sry = ry - sy[0]
                srz = rz - sz[0]
                u_vec = np.array([ssx, ssy, 0])
                v_vec = np.array([srx, sry, srz])
                uv = u_vec @ v_vec
                u2 = u_vec @ u_vec
                t = uv/u2
                if t > 1:
                    d_vec = v_vec - u_vec
                    dist = np.sqrt(float(d_vec @ d_vec))
                elif t < 0:
                    dist = np.sqrt(float(v_vec @ v_vec))
                else:
                    d_vec = v_vec - t*u_vec
                    dist = np.sqrt(float(d_vec @ d_vec))
                sup = dist / 5
                split = int(length/sup) + 1
            else:
                split = self.source.split

            # 節点
            sx_node = np.linspace(sx[0], sx[1], split + 1)
            sy_node = np.linspace(sy[0], sy[1], split + 1)
            sz_node = np.linspace(sz[0], sz[1], split + 1)

            dx = sx[1] - sx[0] / split
            dy = sy[1] - sy[0] / split
            dz = sz[1] - sz[0] / split
            ds = length / split

            sx_dipole = sx_node[:-1] + dx / 2
            sy_dipole = sy_node[:-1] + dy / 2
            sz_dipole = sz_node[:-1] + dz / 2

            rot_matrix = np.array([
                [cos_theta, sin_theta, 0],
                [-sin_theta, cos_theta, 0],
                [0, 0, 1]
                ], dtype=float)
            rot_sc = np.dot(rot_matrix, np.array([sx_dipole, sy_dipole, sz_dipole]).reshape(3,-1)).T
            rot_rc = np.dot(rot_matrix, self.rc)

            xx = rot_rc[0] - rot_sc[:, 0]
            yy = rot_rc[1] - rot_sc[:, 1]
            squared = (xx ** 2 + yy ** 2).astype(float)
            rn = np.sqrt(squared)

            si = _get_layer_index(sz[0], self.depth)
            ri = _get_layer_index(rz, self.depth)

            self.sx, self.sy, self.sz = sx, sy, sz[0]
            self.rx, self.ry, self.rz = rx, ry, rz
            self.xx, self.yy, self.rn = xx, yy, rn
            self.ds = ds
            self.si = si
            self.ri = ri
            self.split = split
            self.cos_theta = cos_theta
            self.sin_theta = sin_theta
        else:
            raise Exception
            
    def _calc_em_admittion(self, omega):
        self.omega = omega
        self.K = len(omega)
        self.admz = 1j * omega.reshape(-1,1) @ self.mu.reshape(1,-1)
        self.admy = self.sigma.reshape(1,-1) + 1j * omega.reshape(-1,1) @ self.epsilon.reshape(1,-1)
        self.k = np.sqrt(-self.admy*self.admz)
        return None

    def _calc_u_4d(self, lambda_):
        # for multiple dipole
        # calculate 4d tensor u = lambda**2 - k**2
        lambda_v = np.zeros((self.K, self.L, self.J, self.M), dtype=complex)
        lambda_v[:,:] = lambda_
        lambda_v = lambda_v.transpose((2,0,1,3))
        k_v = np.zeros((self.J, self.M, self.L, self.K), dtype=complex)
        k_v[:,:] = self.k.T
        k_v = k_v.transpose((0,3,2,1))
        self.u = np.sqrt(lambda_v ** 2 - k_v ** 2)
        return None

    def _calc_kernel_components(self, y_base, rn):
        if np.isscalar(rn):
            rn = np.array([rn])
        self.J = len(rn)
        lambda_ = y_base.reshape(1,-1) / rn.reshape(-1,1)
        self.lambda_ = lambda_
        self._calc_u_4d(lambda_)

        # TODO separate TE, TM, and both
        u_te, d_te, u_tm, d_tm, e_up, e_down = tetm_mode_damping(self)
        self.u_te, self.d_te = u_te, d_te
        self.u_tm, self.d_tm = u_tm, d_tm
        self.e_up, self.e_down = e_up, e_down

        # HACK for debug
        self.dr_te, self.dr_tm, self.ur_te, self.ur_tm, self.yuc, self.zuc = \
            reflection(
                self.si, self.J, self.K, self.L, self.M, self.u,
                self.thick, self.admy, self.admz
                )
        return None

    def _em_field_f(self, which, direction, omega):
        # 方向ごとの繰り返し計算不要
        y_base, wt0, wt1 = load_hankel_filter(self.hankel_filter)
        self.M = len(y_base)
        if which == 'e':
            ans = self.source._hankel_transform_e(self, direction, omega, y_base, wt0, wt1)
        elif which == 'm':
            ans = self.source._hankel_transform_h(self, direction, omega, y_base, wt0, wt1)
        if len(direction) == 1:
            ans = ans[0]
        return ans

    def _em_field_t(self, which, direction, time):
        signal = self.source.signal
        if signal == 'awave':
            self.source.moment = 1
            #arbitrary waveform
            raise Exception
        else:
            if self.ftdt_method == "dlag":
                ans = lagged_convolution(self, which, direction, time, signal)
            elif self.ftdt_method == "fft":
                ans = interp_fft(self, which, direction, time, signal)
            else:
                raise Exception
        return ans

class QuasiStaticEM1D(DynamicEM1D):
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
        # DEFAULT HANKEL FILTER
        self.hankel_filter = "key201"
        # DEFAULT FTDT METHOD
        self.ftdt_method = "dlag"
        self.set_params_done = False
        self.set_source_done = False
        self.set_filter_done = False

    def _calc_em_admittion(self, omega):
        self.omega = omega
        self.K = len(omega)
        self.admz = 1j * omega.reshape(-1,1) * self.mu.reshape(1,-1)
        admy = self.sigma.reshape(1,-1).astype(complex)
        self.admy = (np.ones((self.K, self.L)) * admy).astype(complex)
        self.admy[:, 0] = 1e-30
        self.k = np.sqrt(-self.admy*self.admz)
        return None

class PeltonIPEM1D(DynamicEM1D):
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
        self.state = "IP"
        # DEFAULT HANKEL FILTER
        self.hankel_filter = "key201"
        # DEFAULT FTDT METHOD
        self.ftdt_method = "dlag"
        self.set_params_done = False
        self.set_source_done = False
        self.set_filter_done = False

    def set_params(self, resistivity, chargeability, relax_time, exponent, mu_r=None):
        r"""
        Parameters
        ----------
        resisitivity : array_like
            Resistivity :math:`\rho` (Ωm)

        chargeability : array_like
            Chargeability :math:`m` (V/V)

        relax_time : array_like
            Relaxation time constant :math:`\tau` (s)

        exponent : array_like
            Frequency-dependent exponent :math:`c` (-)

        mu_r : array-like
            Relative Magnetic Permeability (-)
            default -> `mu_r` = [1, 1, ... ]
        """
        ### define conductivity ###
        res_0 = array(resistivity)
        if not len(res_0) == self.N:
            raise Exception
        self.res_0 = np.append(1e30, res_0)
        self.cond_0 = 1 / self.res_0

        m = array(chargeability)
        if not len(m) == self.N:
            raise Exception
        self.m = np.append(0, m)

        tau = array(relax_time)
        if not len(tau) == self.N:
            raise Exception
        self.tau = np.append(0, tau)

        c = array(exponent)
        if not len(c) == self.N:
            raise Exception
        self.c = np.append(0, c)

        ### define magnetic permeability ###
        if mu_r is None:
            mu = np.ones(self.N) * const.mu_0
        else:
            if not (len(mu_r) == self.N):
                raise Exception
            else:
                mu = array(mu_r) * const.mu_0
        self.mu = np.append(const.mu_0, mu)
        self.set_params_done = True

    def _calc_em_admittion(self, omega):
        self.omega = omega
        self.K = len(omega)
        self.admz = 1j * omega.reshape(-1,1) * self.mu.reshape(1,-1)
        im = 1 + (1j * omega.reshape(-1,1) * self.tau.reshape(1,-1)) ** self.c.reshape(1,-1)
        zeta = 1 - self.m.reshape(1,-1) * (1 - 1 / im)
        self.admy = 1 / zeta * self.cond_0.reshape(1,-1)
        self.admy[:, 0] = 1j * omega * const.epsilon_0
        self.zeta = 1 / self.admy
        self.k = np.sqrt(-self.admy*self.admz)
        return None

    def field(self, which, direction, loc, freqtime, time_derivative=False):
        r"""
        Parameters
        ----------
        which : str
            type of electromagnetic vector field
            'E' for electric field
            'H' for magnetic field
            'D' for electric flux density
            'B' for magnetic flux density
            'J' for electrical current
            'JIP' for capacitive current density
            'JEM' for conduction current density

        direction : str
            measurement direction, x, y, and z

        loc : array-like
            coordinate of measurement point (x, y, z)

        freqtime : array-like
            measurement frequency or time of the receiver

        time-derivative : bool
            if True is input, the returned values are time derivatives

        Returns
        -------
        ans : ndarray
            in FD, dtype = complex
            in TD, dtype = real 
        """
        self._check_input()
        direction = [char for char in direction]
        self.rc = array(loc)
        self.time_derivative = time_derivative
        self._calc_geometric_basics()
        signal = self.source.signal
        if signal == "f":
            omega = 2 * np.pi * array(freqtime)
            if which == "E":
                ans = self._em_field_f('e', direction, omega)
            elif which == "H":
                ans = self._em_field_f('m', direction, omega)
            elif which == "D":
                eperm = self.epsilon[self.ri]
                ans = eperm * self._em_field_f('e', direction, omega)
            elif which == "B":
                mperm = self.mu[self.ri]
                ans = mperm * self._em_field_f('m', direction, omega)
            elif which == "J":
                admy = self.admy[:, self.ri]
                ans = admy * self._em_field_f('e', direction, omega)
            elif which == "JIP":
                ans = self._em_field_f('ipr', direction, omega)
            elif which == "JEM":
                ans = self._em_field_f('ipc', direction, omega)
            else:
                raise Exception
        else:
            time = array(freqtime)
            if which == "E":
                ans = self._em_field_t('e', direction, time)
            elif which == "H":
                ans = self._em_field_t('m', direction, time)
            elif which == "D":
                eperm = self.epsilon[self.ri]
                ans = eperm * self._em_field_t('e', direction, time)
            elif which == "B":
                mperm = self.mu[self.ri]
                ans = mperm * self._em_field_t('m', direction, time)
            elif which == "J":
                ans = self._em_field_t('ipt', direction, time)
            elif which == "JIP":
                ans = self._em_field_t('ipr', direction, time)
            elif which == "JEM":
                ans = self._em_field_t('ipc', direction, time)
            else:
                raise Exception
        return ans

    def _em_field_f(self, which, direction, omega):
        self._calc_em_admittion(omega)
        # 方向ごとの繰り返し計算不要
        y_base, wt0, wt1 = load_hankel_filter(self.hankel_filter)
        self.M = len(y_base)
        if which == 'e':
            ans = self.source._hankel_transform_e(self, direction, omega, y_base, wt0, wt1)
        elif which == 'm':
            ans = self.source._hankel_transform_h(self, direction, omega, y_base, wt0, wt1)
        elif which == 'ipt':
            ri = self.ri
            admy = self.admy[:,ri]
            ans = admy * self.source._hankel_transform_e(self, direction, omega, y_base, wt0, wt1)
        elif which == 'ipr':
            ri = self.ri
            m = self.m[ri]
            tau = self.tau[ri]
            c = self.c[ri]
            admy = self.admy[:,ri]
            nume = m * (1j * omega * tau) ** c
            deno = 1 - m  + nume
            ratio = nume / deno
            ans = ratio * admy * self.source._hankel_transform_e(self, direction, omega, y_base, wt0, wt1)
        elif which == 'ipc':
            ri = self.ri
            m = self.m[ri]
            tau = self.tau[ri]
            c = self.c[ri]
            admy = self.admy[:,ri]
            nume = m * (1j * omega * tau) ** c
            deno = 1 - m  + nume
            ratio = (1 - m) / deno
            ans = ratio * admy * self.source._hankel_transform_e(self, direction, omega, y_base, wt0, wt1)
        if len(direction) == 1:
            ans = ans[0]
        return ans
