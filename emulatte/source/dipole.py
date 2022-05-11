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


class VMD:
    def __init__(self, moment, ontime = None):
        moment = array(moment)
        self.kernel_te_up_sign = 1
        self.kernel_te_down_sign = 1
        self.kernel_tm_up_sign = 0
        self.kernel_tm_down_sign = 0
        self.signal = check_waveform(ontime)

        if len(moment) == 1:
            self.moment = moment[0]
        else:
            self.moment_array = moment

    def _hankel_transform_e(self, model, direction, omega, y_base, wt0, wt1):
        model._calc_em_admittion(omega)
        model._calc_kernel_components(y_base, model.rh)
        lambda_ = model.lambda_
        zs = model.admz[:, model.si]
        rh = model.rh
        ans = []
        fetched = False
        factor = zs / (4 * np.pi * rh)
        if "x" in direction:
            kernel = self._calc_kernel_vmd_e(model, lambda_)
            fetched = True
            e_x = self.moment * factor * (kernel[0] @ wt1) * model.sin_phi
            ans.append(e_x)
        if "y" in direction:
            if not fetched:
                kernel = self._calc_kernel_vmd_e(model, lambda_)
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
        model._calc_kernel_components(y_base, model.rh)
        lambda_ = model.lambda_
        zr = model.admz[:, model.ri]
        zs = model.admz[:, model.si]
        rh = model.rh
        ans = []
        fetched = False
        factor = 1 / (4 * np.pi * rh) * zs / zr
        if "x" in direction:
            kernel = self._calc_kernel_vmd_hr(model, lambda_)
            fetched = True
            h_x = self.moment * factor * (kernel[0] @ wt1) * model.cos_phi
            ans.append(h_x)
        if "y" in direction:
            if not fetched:
                kernel = self._calc_kernel_vmd_hr(model, lambda_)
            h_y = self.moment * factor * (kernel[0] @ wt1) * model.sin_phi
            ans.append(h_y)
        if "z" in direction:
            kernel = self._calc_kernel_vmd_hz(model, lambda_)
            h_z = self.moment * factor * (kernel[0] @ wt0)
            ans.append(h_z)
        ans = np.array(ans)
        if model.time_derivative:
            ans = ans * omega * 1j
        return ans

    def _calc_kernel_vmd_e(self, model, lambda_):
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
        kernel_add = int(si==ri) * np.exp(- su * abs(rz - sz))
        kernel = kernel + kernel_add
        kernel = kernel * lambda_ ** 2 / su
        return kernel

    def _calc_kernel_vmd_hr(self, model, lambda_):
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
        kernel = u_te * e_up - d_te * e_down
        kernel_add = int(si==ri) * np.exp(- su * abs(rz - sz))
        kernel_add = kernel_add * np.sign(rz - sz)
        kernel = kernel + kernel_add
        kernel = kernel * lambda_ ** 2 * ru / su
        return kernel

    def _calc_kernel_vmd_hz(self, model, lambda_):
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
        kernel_add = int(si==ri) * np.exp(- su * abs(rz - sz))
        kernel = kernel + kernel_add
        kernel = kernel * lambda_ ** 3 / su
        return kernel

class HMD:
    def __init__(self, moment, ontime = None):
        moment = array(moment)
        self.kernel_te_up_sign = 1
        self.kernel_te_down_sign = 1
        self.kernel_tm_up_sign = 0
        self.kernel_tm_down_sign = 0
        self.signal = check_waveform(ontime)

        if len(moment) == 1:
            self.moment = moment[0]
        else:
            self.moment_array = moment

    def _hankel_transform_e(self, model, direction, omega, y_base, wt0, wt1):
        model._calc_em_admittion(omega)
        model._calc_kernel_components(y_base, model.rh)
        lambda_ = model.lambda_
        zs = model.admz[:, model.si]
        rh = model.rh
        ans = []
        fetched = False
        factor = zs / (4 * np.pi * rh)
        if "x" in direction:
            kernel = self._calc_kernel_vmd_e(model, lambda_)
            fetched = True
            e_x = self.moment * factor * (kernel[0] @ wt1) * model.sin_phi
            ans.append(e_x)
        if "y" in direction:
            if not fetched:
                kernel = self._calc_kernel_vmd_e(model, lambda_)
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
        model._calc_kernel_components(y_base, model.rh)
        lambda_ = model.lambda_
        zr = model.admz[:, model.ri]
        zs = model.admz[:, model.si]
        rh = model.rh
        ans = []
        fetched = False
        factor = 1 / (4 * np.pi * rh) * zs / zr
        if "x" in direction:
            kernel = self._calc_kernel_vmd_h_r(model, lambda_)
            fetched = True
            h_x = self.moment * factor * (kernel[0] @ wt1) * model.cos_phi
            ans.append(h_x)
        if "y" in direction:
            if not fetched:
                kernel = self._calc_kernel_vmd_h_r(model, lambda_)
            h_y = self.moment * factor * (kernel[0] @ wt1) * model.sin_phi
            ans.append(h_y)
        if "z" in direction:
            kernel = self._calc_kernel_vmd_h_z(model, lambda_)
            h_z = self.moment * factor * (kernel[0] @ wt0)
            ans.append(h_z)
        ans = np.array(ans)
        if model.time_derivative:
            ans = ans * omega * 1j
        return ans

    def _calc_kernel_hmd_tm_er(self, model):
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
        kernel_add = -np.sign(rz - sz) * int(si==ri) * np.exp(- su * abs(rz - sz))
        kernel = kernel + kernel_add
        kernel = kernel * ru / su
        return kernel

    def _calc_kernel_hmd_te_er(self, model):
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
        kernel_add = np.sign(rz - sz) * int(si==ri) * np.exp(- su * abs(rz - sz))
        kernel = kernel + kernel_add
        return kernel

    def _calc_kernel_hmd_tm_ez(self, model):
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
        kernel_add = int(si==ri) * np.exp(- su * abs(rz - sz))
        kernel = kernel + kernel_add
        kernel = kernel / su
        return kernel

    def _calc_kernel_hmd_tm_hr(self, model):
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
        kernel_add = int(si==ri) * np.exp(- su * abs(rz - sz))
        kernel = kernel + kernel_add
        kernel = kernel / su
        return kernel

    def _calc_kernel_hmd_te_hr(self, model):
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
        kernel_add = -int(si==ri) * np.exp(- su * abs(rz - sz))
        kernel = kernel + kernel_add
        kernel = kernel * ru
        return kernel

    def _calc_kernel_hmd_te_hz(self, model):
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
        kernel_add = np.sign(rz - sz) * int(si==ri) * np.exp(- su * abs(rz - sz))
        kernel = kernel + kernel_add
        return kernel