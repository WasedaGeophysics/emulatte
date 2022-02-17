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
from ..utils.converter import array, check_waveform
from .kernel.dipole_kernel import *


class VMD:
    def __init__(self, moment, ontime = None):
        self.moment = moment
        self.num_dipole = 1
        self.kernel_te_up_sign = 1
        self.kernel_te_down_sign = 1
        self.kernel_tm_up_sign = 0
        self.kernel_tm_down_sign = 0
        self.signal = check_waveform(ontime)
        self.mode = "TE"

    def _hankel_transform_e(self, model, direction, omega, y_base, wt0, wt1):
        lambda_ = model._compute_kernel_integrants(omega, y_base, model.r)
        z = model.admz[:, model.ri]
        r = model.r
        ans = []
        if "x" in direction:
            factor = - z / (4 * np.pi * r) * model.sin_phi
            kernel = call_kernel_vmd_e(model, lambda_)
            e_x = self.moment * factor * (kernel @ wt1)
            ans.append(e_x)
        if "y" in direction:
            factor = - z / (4 * np.pi * r) * model.cos_phi
            kernel = call_kernel_vmd_e(model, lambda_)
            e_y = self.moment * factor * (kernel @ wt1)
            ans.append(e_y)
        if "z" in direction:
            e_z = np.zeros(model.K)
            ans.append(e_z)
        ans = np.array(ans)
        if model.time_diff:
            ans = ans * omega * 1j
        return ans
        
    def _hankel_transform_h(self, model, direction, omega, y_base, wt0, wt1):
        lambda_ = model._compute_kernel_integrants(omega, y_base, model.r)
        zr = model.admz[:, model.ri]
        zs = model.admz[:, model.si]
        r = model.r
        ans = []
        if "x" in direction:
            factor = 1 / (4 * np.pi * r) * model.cos_phi
            kernel = call_kernel_vmd_h_r(model, lambda_)
            h_x = self.moment * factor * (kernel @ wt1)
            ans.append(h_x)
        if "y" in direction:
            factor = 1 / (4 * np.pi * r) * model.sin_phi
            kernel = call_kernel_vmd_h_r(model, lambda_)
            h_y = self.moment * factor * (kernel @ wt1)
            ans.append(h_y)
        if "z" in direction:
            factor = 1 / (4 * np.pi * r) * zs / zr
            kernel = call_kernel_vmd_h_z(model, lambda_)
            model.kernel = kernel
            model.factor = factor
            h_z = self.moment * factor * (kernel @ wt0)
            ans.append(h_z)
        ans = np.array(ans)
        if model.time_diff:
            ans = ans * omega * 1j
        return ans

class HED:
    def __init__(self, moment, ontime = None):
        self.moment = moment
        self.num_dipole = 1
        self.kernel_te_up_sign = 1
        self.kernel_te_down_sign = 1
        self.kernel_tm_up_sign = 0
        self.kernel_tm_down_sign = 0
        self.signal = check_waveform(ontime)
        self.mode = "TE"

    def _hankel_transform_e(self, model, direction, omega, y_base, wt0, wt1):
        lambda_ = model._compute_kernel_integrants(omega, y_base, model.r)
        z = model.admz[:, model.ri]
        r = model.r
        ans = []
        if "x" in direction:
            factor = - z / (4 * np.pi * r) * model.sin_phi
            kernel = call_kernel_vmd_e(model, lambda_)
            e_x = self.moment * factor * (kernel @ wt1)
            ans.append(e_x)
        if "y" in direction:
            factor = - z / (4 * np.pi * r) * model.cos_phi
            kernel = call_kernel_vmd_e(model, lambda_)
            e_y = self.moment * factor * (kernel @ wt1)
            ans.append(e_y)
        if "z" in direction:
            e_z = np.zeros(model.K)
            ans.append(e_z)
        ans = np.array(ans)
        if model.time_diff:
            ans = ans * omega * 1j
        return ans
        
    def _hankel_transform_h(self, model, direction, omega, y_base, wt0, wt1):
        lambda_ = model._compute_kernel_integrants(omega, y_base, model.r)
        zr = model.admz[:, model.ri]
        zs = model.admz[:, model.si]
        r = model.r
        ans = []
        if "x" in direction:
            factor = 1 / (4 * np.pi * r) * model.cos_phi
            kernel = call_kernel_vmd_h_r(model, lambda_)
            h_x = self.moment * factor * (kernel @ wt1)
            ans.append(h_x)
        if "y" in direction:
            factor = 1 / (4 * np.pi * r) * model.sin_phi
            kernel = call_kernel_vmd_h_r(model, lambda_)
            h_y = self.moment * factor * (kernel @ wt1)
            ans.append(h_y)
        if "z" in direction:
            factor = 1 / (4 * np.pi * r) * zs / zr
            kernel = call_kernel_vmd_h_z(model, lambda_)
            model.kernel = kernel
            model.factor = factor
            h_z = self.moment * factor * (kernel @ wt0)
            ans.append(h_z)
        ans = np.array(ans)
        if model.time_diff:
            ans = ans * omega * 1j
        return ans
