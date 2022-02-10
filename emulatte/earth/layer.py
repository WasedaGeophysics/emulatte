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
from ...utils.converter import array

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
            raise Exception(
                "The length of input array 'res' is {0}, \\
                but it must be {1}".format(len(res), self.N)
            )
        
        ### define electric permittivity ###
        if not epsilon_r:
            self.epsilon = np.ones(self.N) * const.epsilon_0
        else:
            epsilon = array(epsilon)
            if not (len(epsilon) == self.N):
                raise Exception(
                    "The length of input array 'epsilon_r' is {0}, \\
                    but it must be {1}".format(len(self.epsilon), self.N)
                )
            else:
                self.epsilon = array(epsilon_r) * const.epsilon_0

        ### define magnetic permeability ###
        if not mu_r:
            self.mu = np.ones(self.N) * const.mu_0
        else:
            mu = array(mu)
            if not (len(mu) == self.N):
                raise Exception(
                    "The length of input array 'mu_r' is {0}, \\
                    but it must be {1}".format(len(self.mu), self.N)
                )
            else:
                self.epsln = array(mu_r) * const.mu_0

        

class IPEM1D:
