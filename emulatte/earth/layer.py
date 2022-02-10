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

class EM1D:
    r"""Summary

    """
    def __init__(self, thicks):
        r"""
        Parameters
        ----------
        thicks : array_like
            layer thickness (m), except for the first (on the ground) & last layers
        """
        thicks = array(thicks)
        # THICKNESS
        self.thicks = thicks
        # LAYERS BOUNDARY DEPTH
        self.depth = np.array([0, *np.cumsum(thicks)])
        # NUMBER OF LAYERS
        self.N = len(thicks) + 2
        #
        self.state = "quasistatic"

    def set_params(self, res, epsilon_r=None, mu_r=None):
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
        self.sigma = 1 / array(res)
        if not (len(self.sigma) == self.N):
            emtxt = "The length of input array 'res' is " + \
                    "{0}, but it must be {1}".format(len(res), self.N
                    )
            raise Exception(emtxt)
        
        ### define electric permittivity ###
        if epsilon_r is None:
            self.epsilon = np.ones(self.N) * const.epsilon_0
        else:
            epsilon_r = array(epsilon_r)
            if not (len(epsilon_r) == self.N):
                emtxt = "The length of input array 'epsilon_r' is " + \
                        " {0}, but it must be {1}".format(
                            len(epsilon_r), self.N
                        )
                raise Exception(emtxt)
            else:
                self.epsilon = epsilon_r * const.epsilon_0

        ### define magnetic permeability ###
        if mu_r is None:
            self.mu = np.ones(self.N) * const.mu_0
        else:
            mu_r = array(mu_r)
            if not (len(mu_r) == self.N):
                emtxt = "The length of input array 'mu_r' is " + \
                    "{0}, but it must be {1}".format(len(mu_r), self.N)
                raise Exception(emtxt)
            else:
                self.mu = array(mu_r) * const.mu_0
        
    def locate(self, source, sc):
        pass

    def dlf_config(self):
        pass

    def damping_factor(self, omega):
        ztilde = np.ones(self.num_layer, dtype=complex)
        ytilde = np.ones(self.num_layer, dtype=complex)
        k = np.zeros(self.num_layer, dtype=complex)
        u = np.ones((self.num_layer, self.filter_length), dtype=complex)
        Y = np.ones((self.num_layer, self.filter_length), dtype=complex)
        Z = np.ones((self.num_layer, self.filter_length), dtype=complex)
        tanhuh = np.ones((self.num_layer, self.filter_length), dtype=complex)

        # インピーダンス＆アドミタンス
        ztilde[:] = 1j * omega * self.mu[:]

        # w1dem.pyでは何か変なことになってる
        if self.ignore_displacement_current:
            ytilde[:] = self.sigma[:]
            ytilde[0] = 1e-13
            k[:] = (- 1.j * omega * self.mu[:] * self.sigma[:]) ** 0.5
            k[0] = 0  # !!!
        else:
            ytilde[:] = self.sigma[:] + 1.j * omega * self.epsln[:]
            k[:] = (omega ** 2.0 * self.mu[:] * self.epsln[:]
                    - 1.j * omega * self.mu[:] * self.sigma[:]) ** 0.5

        # u = (kx^2 + ky^2 - km^2)^0.5
        for i in range(self.num_layer):
            u[i] = (self.lambda_ ** 2 - k[i] ** 2) ** 0.5

        # tanh
        for i in range(1, self.num_layer - 1):
            tanhuh[i] = np.tanh(u[i] * self.thicks[i - 1])

        for i in range(self.num_layer):
            Y[i] = u[i] / ztilde[i]
            Z[i] = u[i] / ytilde[i]

        # return to self
        self.ztilde = ztilde
        self.ytilde = ytilde
        self.k = k
        self.u = u

        # TE/TM mode 境界係数
        r_te = np.ones((self.num_layer, self.filter_length), dtype=complex)
        r_tm = np.ones((self.num_layer, self.filter_length), dtype=complex)
        R_te = np.ones((self.num_layer, self.filter_length), dtype=complex)
        R_tm = np.ones((self.num_layer, self.filter_length), dtype=complex)

        # 送受信層index+1　for コード短縮
        si = self.slayer
        ri = self.rlayer

        ### DOWN ADMITTANCE & IMPEDANCE ###
        Ytilde = np.zeros((self.num_layer, self.filter_length), dtype=complex)
        Ztilde = np.zeros((self.num_layer, self.filter_length), dtype=complex)

        Ytilde[-1] = Y[-1]
        Ztilde[-1] = Z[-1]

        r_te[-1] = 0
        r_tm[-1] = 0

        for ii in range(self.num_layer - 1, si, -1):
            numerator_Y = Ytilde[ii] + Y[ii - 1] * tanhuh[ii - 1]
            denominator_Y = Y[ii - 1] + Ytilde[ii] * tanhuh[ii - 1]
            Ytilde[ii - 1] = Y[ii - 1] * numerator_Y / denominator_Y

            numerator_Z = Ztilde[ii] + Z[ii - 1] * tanhuh[ii - 1]
            denominator_Z = Z[ii - 1] + Ztilde[ii] * tanhuh[ii - 1]
            Ztilde[ii - 1] = Z[ii - 1] * numerator_Z / denominator_Z

            r_te[ii - 1] = (Y[ii - 1] - Ytilde[ii]) / (Y[ii - 1] + Ytilde[ii])
            r_tm[ii - 1] = (Z[ii - 1] - Ztilde[ii]) / (Z[ii - 1] + Ztilde[ii])

        if si != self.num_layer:
            r_te[si - 1] = (Y[si - 1] - Ytilde[si]) / (Y[si - 1] + Ytilde[si])
            r_tm[si - 1] = (Z[si - 1] - Ztilde[si]) / (Z[si - 1] + Ztilde[si])

        ### UP ADMITTANCE & IMPEDANCE ###
        Yhat = np.ones((self.num_layer, self.filter_length), dtype=complex)
        Zhat = np.ones((self.num_layer, self.filter_length), dtype=complex)

        Yhat[0] = Y[0]
        Zhat[0] = Z[0]

        R_te[0] = 0
        R_tm[0] = 0

        for ii in range(2, si):
            numerator_Y = Yhat[ii - 2] + Y[ii - 1] * tanhuh[ii - 1]
            denominator_Y = Y[ii - 1] + Yhat[ii - 2] * tanhuh[ii - 1]
            Yhat[ii - 1] = Y[ii - 1] * numerator_Y / denominator_Y
            # (2)Yhat{2,3,\,si-2,si-1}

            numerator_Z = Zhat[ii - 2] + Z[ii - 1] * tanhuh[ii - 1]
            denominator_Z = Z[ii - 1] + Zhat[ii - 2] * tanhuh[ii - 1]
            Zhat[ii - 1] = Z[ii - 1] * numerator_Z / denominator_Z

            R_te[ii - 1] = (Y[ii - 1] - Yhat[ii - 2]) / (Y[ii - 1] + Yhat[ii - 2])
            R_tm[ii - 1] = (Z[ii - 1] - Zhat[ii - 2]) / (Z[ii - 1] + Zhat[ii - 2])
        if si != 1:
            R_te[si - 1] = (Y[si - 1] - Yhat[si - 2]) / (Y[si - 1] + Yhat[si - 2])
            R_tm[si - 1] = (Z[si - 1] - Zhat[si - 2]) / (Z[si - 1] + Zhat[si - 2])

        U_te = np.ones((self.num_layer, self.filter_length), dtype=complex)
        U_tm = np.ones((self.num_layer, self.filter_length), dtype=complex)
        D_te = np.ones((self.num_layer, self.filter_length), dtype=complex)
        D_tm = np.ones((self.num_layer, self.filter_length), dtype=complex)

        # In the layer containing the source (slayer)
        if si == 1:
            U_te[si - 1] = 0
            U_tm[si - 1] = 0
            D_te[si - 1] = self.src.kernel_te_down_sign * r_te[si - 1] \
                * np.exp(-u[si - 1] * (self.depth[si - 1] - self.sz))
            D_tm[si - 1] = self.src.kernel_tm_down_sign * r_tm[si - 1] \
                * np.exp(-u[si - 1] * (self.depth[si - 1] - self.sz))
        elif si == self.num_layer:
            U_te[si - 1] = self.src.kernel_te_up_sign * R_te[si - 1] \
                * np.exp(u[si - 1] * (self.depth[si - 2] - self.sz))
            U_tm[si - 1] = self.src.kernel_tm_up_sign * R_tm[si - 1] \
                * np.exp(u[si - 1] * (self.depth[si - 2] - self.sz))
            D_te[si - 1] = 0
            D_tm[si - 1] = 0
        else:
            exp_term1 = np.exp(-2 * u[si - 1]
                               * (self.depth[si - 1] - self.depth[si - 2]))
            exp_term2u = np.exp(u[si - 1] * (self.depth[si - 2] - 2 * self.depth[si - 1] + self.sz))
            exp_term2d = np.exp(-u[si - 1] * (self.depth[si - 1] - 2 * self.depth[si - 2] + self.sz))
            exp_term3u = np.exp(u[si - 1] * (self.depth[si - 2] - self.sz))
            exp_term3d = np.exp(-u[si - 1] * (self.depth[si - 1] - self.sz))

            U_te[si - 1] = \
                1 / (1 - R_te[si - 1] * r_te[si - 1] * exp_term1) \
                * R_te[si - 1] \
                * (self.src.kernel_te_down_sign * r_te[si - 1] * exp_term2u
                    + self.src.kernel_te_up_sign * exp_term3u)

            U_tm[si - 1] = \
                1 / (1 - R_tm[si - 1] * r_tm[si - 1] * exp_term1) \
                * R_tm[si - 1] \
                * (self.src.kernel_tm_down_sign * r_tm[si - 1] * exp_term2u
                    + self.src.kernel_tm_up_sign * exp_term3u)

            D_te[si - 1] = \
                1 / (1 - R_te[si - 1] * r_te[si - 1] * exp_term1) \
                * r_te[si - 1] \
                * (self.src.kernel_te_up_sign * R_te[si - 1] * exp_term2d
                    + self.src.kernel_te_down_sign * exp_term3d)

            D_tm[si - 1] = \
                1 / (1 - R_tm[si - 1] * r_tm[si - 1] * exp_term1) \
                * r_tm[si - 1] \
                * (self.src.kernel_tm_up_sign * R_tm[si - 1] * exp_term2d
                    + self.src.kernel_tm_down_sign * exp_term3d)

        # for the layers above the slayer
        if ri < si:
            if si == self.num_layer:
                exp_term = np.exp(-u[si - 1] * (self.sz - self.depth[si - 2]))

                D_te[si - 2] = \
                    (Y[si - 2] * (1 + R_te[si - 1]) + Y[si - 1] * (1 - R_te[si - 1])) \
                    / (2 * Y[si - 2]) * self.src.kernel_te_up_sign * exp_term

                D_tm[si - 2] = \
                    (Z[si - 2] * (1 + R_tm[si - 1]) + Z[si - 1] * (1 - R_tm[si - 1])) \
                    / (2 * Z[si - 2]) * self.src.kernel_tm_up_sign * exp_term

            elif si != 1 and si != self.num_layer:
                exp_term = np.exp(-u[si - 1] * (self.sz - self.depth[si - 2]))
                exp_termii = np.exp(-u[si - 1] * (self.depth[si - 1] - self.depth[si - 2]))

                D_te[si - 2] = \
                    (Y[si - 2] * (1 + R_te[si - 1]) + Y[si - 1] * (1 - R_te[si - 1])) \
                    / (2 * Y[si - 2]) * (D_te[si - 1] * exp_termii + self.src.kernel_te_up_sign * exp_term)

                D_tm[si - 2] = \
                    (Z[si - 2] * (1 + R_tm[si - 1]) + Z[si - 1] * (1 - R_tm[si - 1])) \
                    / (2 * Z[si - 2]) * (D_tm[si - 1] * exp_termii + self.src.kernel_tm_up_sign * exp_term)

            for jj in range(si - 2, 0, -1):
                exp_termjj = np.exp(-u[jj]
                                    * (self.depth[jj] - self.depth[jj - 1]))
                D_te[jj - 1] = \
                    (Y[jj - 1] * (1 + R_te[jj]) + Y[jj] * (1 - R_te[jj])) \
                    / (2 * Y[jj - 1]) * D_te[jj] * exp_termjj
                D_tm[jj - 1] = \
                    (Z[jj - 1] * (1 + R_tm[jj]) + Z[jj] * (1 - R_tm[jj])) \
                    / (2 * Z[jj - 1]) * D_tm[jj] * exp_termjj

            for jj in range(si - 1, 1, -1):
                exp_termjj = np.exp(u[jj - 1] * (self.depth[jj - 2] - self.depth[jj - 1]))
                U_te[jj - 1] = D_te[jj - 1] * exp_termjj * R_te[jj - 1]
                U_tm[jj - 1] = D_tm[jj - 1] * exp_termjj * R_tm[jj - 1]
            U_te[0] = 0
            U_tm[0] = 0

        # for the layers below the slayer
        if ri > si:
            if si == 1:
                exp_term = np.exp(-u[si - 1] * (self.depth[si - 1] - self.sz))
                U_te[si] = (Y[si] * (1 + r_te[si - 1])
                            + Y[si - 1] * (1 - r_te[si - 1])) \
                    / (2 * Y[si]) \
                    * self.src.kernel_te_down_sign * exp_term
                U_tm[si] = (Z[si] * (1 + r_tm[si - 1])
                            + Z[si - 1] * (1 - r_tm[si - 1])) \
                    / (2 * Z[si]) \
                    * self.src.kernel_tm_down_sign * exp_term

            elif si != 1 and si != self.num_layer:
                exp_termi = np.exp(-u[si - 1]
                                   * (self.depth[si - 1] - self.depth[si - 2]))
                exp_termii = np.exp(-u[si - 1]
                                    * (self.depth[si - 1] - self.sz))
                U_te[si] = (Y[si] * (1 + r_te[si - 1])
                            + Y[si - 1] * (1 - r_te[si - 1])) \
                    / (2 * Y[si]) \
                    * (U_te[si - 1] * exp_termi
                       + self.src.kernel_te_down_sign * exp_termii)
                U_tm[si] = (Z[si] * (1 + r_tm[si - 1]) + Z[si - 1]
                            * (1 - r_tm[si - 1])) \
                    / (2 * Z[si]) \
                    * (U_tm[si - 1] * exp_termi
                       + self.src.kernel_tm_down_sign * exp_termii)

            for jj in range(si + 2, self.num_layer + 1):
                exp_term = np.exp(-u[jj - 2]
                                  * (self.depth[jj - 2] - self.depth[jj - 3]))
                U_te[jj - 1] = (Y[jj - 1] * (1 + r_te[jj - 2])
                                + Y[jj - 2] * (1 - r_te[jj - 2])) \
                    / (2 * Y[jj - 1]) * U_te[jj - 2] * exp_term
                U_tm[jj - 1] = (Z[jj - 1] * (1 + r_tm[jj - 2])
                                + Z[jj - 2] * (1 - r_tm[jj - 2])) \
                    / (2 * Z[jj - 1]) * U_tm[jj - 2] * exp_term

            for jj in range(si + 1, self.num_layer):
                D_te[jj - 1] = U_te[jj - 1] * np.exp(-u[jj - 1]
                                                     * (self.depth[jj - 1] - self.depth[jj - 2])) \
                    * r_te[jj - 1]
                D_tm[jj - 1] = U_tm[jj - 1] * np.exp(-u[jj - 1]
                                                     * (self.depth[jj - 1] - self.depth[jj - 2])) \
                    * r_tm[jj - 1]
            D_te[self.num_layer - 1] = 0
            D_tm[self.num_layer - 1] = 0

        # compute Damping coefficient
        if ri == 1:
            e_up = np.zeros(self.filter_length, dtype=np.complex)
            e_down = np.exp(u[ri - 1] * (self.rz - self.depth[ri - 1]))
        elif ri == self.num_layer:
            e_up = np.exp(-u[ri - 1] * (self.rz - self.depth[ri - 2]))
            e_down = np.zeros(self.filter_length, dtype=np.complex)
        else:
            e_up = np.exp(-u[ri - 1] * (self.rz - self.depth[ri - 2]))
            e_down = np.exp(u[ri - 1] * (self.rz - self.depth[ri - 1]))

        #self.r_te = r_te
        #self.r_tm = r_tm
        #self.R_te = R_te
        #self.R_tm = R_tm
        #self.U_te = U_te
        #self.U_tm = U_tm
        #self.D_te = D_te
        #self.D_tm = D_tm
        #self.e_up = e_up
        #self.e_down = e_down
        return U_te, U_tm, D_te, D_tm, e_up, e_down