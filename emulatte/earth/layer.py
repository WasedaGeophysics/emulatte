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

from ..utils.converter import array, mode
from ..core.factor import compute_coefficients

class EM1D:
    r"""Summary

    """
    def __init__(self, thickness):
        r"""
        Parameters
        ----------
        thicks : array_like
            layer thickness (m), except for the first (on the ground) & last layers
        """
        thickness = array(thickness)
        # LAYERS BOUNDARY DEPTH
        self.depth = np.array([0, *np.cumsum(thickness)])
        # THICKNESS
        self.thicks = np.array([np.inf, *thickness, np.inf])
        # NUMBER OF LAYERS
        self.n = len(thickness) + 2
        #
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
        self.conductivity = 1 / array(resistivity)
        if not (len(self.sigma) == self.N):
            error_message = "The length of input array 'res' is " + \
                    "{0}, but it must be {1}".format(
                        len(resistivity), self.N
                    )
            raise Exception(error_message)
        
        ### define electric permittivity ###
        if epsilon_r is None:
            self.epsilon = np.ones(self.N) * const.epsilon_0
        else:
            epsilon_r = array(epsilon_r)
            if not (len(epsilon_r) == self.N):
                error_message = "The length of input array 'epsilon_r' is " + \
                        " {0}, but it must be {1}".format(
                            len(epsilon_r), self.N
                        )
                raise Exception(error_message)
            else:
                self.epsilon = epsilon_r * const.epsilon_0

        ### define magnetic permeability ###
        if mu_r is None:
            self.mu = np.ones(self.N) * const.mu_0
        else:
            mu_r = array(mu_r)
            if not (len(mu_r) == self.N):
                error_message = "The length of input array 'mu_r' is " + \
                    "{0}, but it must be {1}".format(len(mu_r), self.N)
                raise Exception(error_message)
            else:
                self.mu = array(mu_r) * const.mu_0
        
    def locate_source(self, source, loc):
        self.source = source
        

    def dlf_config(self):
        pass

    def _compute_admittivity(self, omega):
        self.admittivity_m = 1j * omega * self.mu
        self.admittivity_e = self.sigma
        return None

    def _compute_coefficients(self, omega, lambda_):
        self._compute_admittivity(omega)
        self.k = np.sqrt(-self.admittivity_e*self.admittivity_m)
        self.u = np.sqrt(lambda_ ** 2 - self.k ** 2)
        self._compute_coefficients()
        return None

    def _compute_e_f(self, xyz_comp):
        ans = np.array([])
        for direction in xyz_comp:
            e_f = self.source.hankel_transform_e(self, direction)
            np.append(ans, e_f)
        return ans

    def _compute_e_t(self, xyz_comp):
        ans = np.array([])
        for direction in xyz_comp:
            e_f = self._compute_e_f(direction)[0]
            e_t = self.call_fdtd_transform(self.source, e_f)
            np.append(ans, e_t)
        return ans
        
    def E(self, direction, loc, freqtime, time_diff=False):
        direction = [char for char in direction]
        tx_type = self.source.tx_type
        self.rc = loc
        if tx_type == "f":
            ans = self._compute_e_f(direction, freqtime)
        else:
            ans = self._compute_e_t(direction, freqtime, tx_type)
        return ans

    def H(self, rc):
        pass

    def B(self, rc):
        pass

    def J(self, rc):
        pass

    def receive(self, receiver, rc):
        # 受信機器固有の測定量（コインシデント、電圧ダイポールなど）
        pass