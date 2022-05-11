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
import scipy
from ..utils.converter import array, check_waveform


class HCL:
    def __init__(self, current, radius, turns = 1, ontime = None):
        current = array(current)
        moment = current * turns
        self.radius = radius
        self.turns = turns
        self.kernel_te_up_sign = 1
        self.kernel_te_down_sign = 1
        self.kernel_tm_up_sign = 0
        self.kernel_tm_down_sign = 0

        self.signal = check_waveform(ontime)
        self.mode = "TE"

        if len(current) == 1:
            self.moment = moment[0]
        else:
            self.moment_time = moment

    def _hankel_transform_e(self, model, direction, omega, y_base, wt0, wt1):
        model._calc_em_admittion(omega)
        model._calc_kernel_components(y_base, self.radius)
        lambda_ = model.lambda_
        zs = model.admz[:, model.si]
        ans = []
        factor = zs / 2
        kernel_fetched = False
        if "x" in direction:
            kernel = self._calc_kernel_loop_e(model, lambda_)
            kernel_fetched = True
            e_x = self.moment * factor * (kernel[0] @ wt1) * model.sin_phi
            ans.append(e_x)
        if "y" in direction:
            if not kernel_fetched:
                kernel = self._calc_kernel_loop_e(model, lambda_)
            e_y = self.moment * factor * (kernel[0] @ wt1) * - model.cos_phi
            ans.append(e_y)
        if "z" in direction:
            e_z = np.zeros(model.K)
            ans.append(e_z)
        ans = np.array(ans)
        if model.time_derivative:
            ans = ans * omega * 1j
        return ans
        
    def _hankel_transform_h(self, model, direction, omega, y_base, wt0, wt1):
        model._calc_em_admittion(omega)
        model._calc_kernel_components(y_base, self.radius)
        lambda_ = model.lambda_
        zr = model.admz[:, model.ri]
        zs = model.admz[:, model.si]
        ans = []
        factor =  zs / zr / 2
        if "x" in direction:
            kernel = self._calc_kernel_loop_h_r(model, lambda_)
            h_x = self.moment * factor * (kernel[0] @ wt1) * - model.cos_phi
            ans.append(h_x)
        if "y" in direction:
            kernel = self._calc_kernel_loop_h_r(model, lambda_)
            h_y = self.moment * factor * (kernel[0] @ wt1) * - model.sin_phi
            ans.append(h_y)
        if "z" in direction:
            kernel = self._calc_kernel_loop_h_z(model, lambda_)
            h_z = self.moment * factor * (kernel[0] @ wt1)
            ans.append(h_z)
        ans = np.array(ans)
        if model.time_derivative:
            ans = ans * omega * 1j
        return ans

    def _calc_kernel_loop_e(self, model, lambda_):
        u_te = model.u_te
        d_te = model.d_te
        e_up = model.e_up
        e_down = model.e_down
        si = model.si
        ri = model.ri
        su = model.u[:,:,si]
        sz = model.sz
        rz = model.rz
        kernel = u_te * e_up + d_te * e_down
        kernel_add = int(si==ri) * np.exp(- su * abs(sz - rz))
        kernel = kernel + kernel_add
        bessel = scipy.special.jn(1, lambda_ * model.rh)
        kernel = kernel * lambda_ * lambda_ / su * bessel
        return kernel

    def _calc_kernel_loop_h_r(self, model, lambda_):
        u_te = model.u_te
        d_te = model.d_te
        e_up = model.e_up
        e_down = model.e_down
        si = model.si
        ri = model.ri
        su = model.u[:,:,si]
        ru = model.u[:,:,ri]
        sz = model.sz
        rz = model.rz
        kernel = -u_te * e_up + d_te * e_down
        kernel_add = - int(si==ri) * np.exp(- su * abs(sz - rz))
        kernel_add = kernel_add * np.sign(rz - sz)
        kernel = kernel + kernel_add
        bessel = scipy.special.jn(1, lambda_ * model.rh)
        kernel = kernel * lambda_ * ru / su * bessel
        return kernel

    def _calc_kernel_loop_h_z(self, model, lambda_):
        u_te = model.u_te
        d_te = model.d_te
        e_up = model.e_up
        e_down = model.e_down
        si = model.si
        ri = model.ri
        su = model.u[:,:,si]
        sz = model.sz
        rz = model.rz
        kernel = u_te * e_up + d_te * e_down
        kernel_add = int(si==ri) * np.exp(- su * abs(sz - rz))
        kernel = kernel + kernel_add
        bessel = scipy.special.jn(0, lambda_ * model.rh)
        kernel = kernel * lambda_ * lambda_ / su * bessel
        return kernel

