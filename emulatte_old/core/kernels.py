# Copyright 2021 Waseda Geophysics Laboratory
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
from scipy.special import erf, erfc, jn
from ..utils.function import kroneckers_delta

def compute_kernel_vmd(model, omega):
    """
    
    """
    U_te, U_tm, D_te, D_tm, e_up, e_down = model.compute_coefficients(omega)
    kernel_te = U_te[model.rlayer - 1] * e_up \
                    + D_te[model.rlayer - 1] * e_down \
                    + kroneckers_delta(model.rlayer, model.slayer) \
                    * np.exp(-model.u[model.slayer - 1] \
                        * np.abs(model.rz - model.sz))
    kernel_te_hr = U_te[model.rlayer - 1] * e_up \
                    - D_te[model.rlayer - 1] * e_down \
                    +  kroneckers_delta(model.rlayer, model.slayer) \
                    * (model.rz - model.sz) / np.abs(model.rz - model.sz) \
                    * np.exp(-model.u[model.slayer - 1] \
                        * np.abs(model.rz - model.sz))
    kernel_e_phi = kernel_te * model.lambda_ ** 2 \
                    / model.u[model.slayer - 1]
    kernel_h_r = kernel_te_hr * model.lambda_ ** 2 \
                    * model.u[model.rlayer - 1] \
                    / model.u[model.slayer - 1]
    kernel_h_z = kernel_e_phi * model.lambda_
    kernel = [kernel_e_phi, kernel_h_r, kernel_h_z]
    kernel = np.array(kernel)
    model.kernel = kernel
    return kernel

def compute_kernel_hmd(model, omega):
    """
    
    """
    U_te, U_tm, D_te, D_tm, e_up, e_down = model.compute_coefficients(omega)
    kernel_tm_er = (-U_tm[model.rlayer - 1] * e_up \
                        + D_tm[model.rlayer - 1] * e_down \
                        - np.sign(model.rz - model.sz) \
                        * kroneckers_delta(model.rlayer, model.slayer) \
                        * np.exp(-model.u[model.slayer - 1] \
                                * np.abs(model.rz -model.sz))) \
                    * model.u[model.rlayer - 1] \
                    / model.u[model.slayer - 1]
    kernel_te_er = U_te[model.rlayer - 1] * e_up \
                    + D_te[model.rlayer - 1] * e_down \
                    + np.sign(model.rz - model.sz) \
                    * kroneckers_delta(model.rlayer, model.slayer) \
                    * np.exp(-model.u[model.slayer - 1] \
                            * np.abs(model.rz - model.sz))
    kernel_tm_ez = (U_tm[model.rlayer - 1] * e_up \
                        + D_tm[model.rlayer - 1] * e_down \
                        + kroneckers_delta(model.rlayer, model.slayer) \
                        * np.exp(-model.u[model.slayer - 1] \
                                * np.abs(model.rz -model.sz))) \
                    / model.u[model.slayer - 1]
    kernel_tm_hr = (U_tm[model.rlayer - 1] * e_up \
                        + D_tm[model.rlayer - 1] * e_down \
                        + kroneckers_delta(model.rlayer, model.slayer) \
                        * np.exp(-model.u[model.slayer - 1] \
                                * np.abs(model.rz -model.sz))) \
                    / model.u[model.slayer - 1]
    kernel_te_hr = (-U_te[model.rlayer - 1] * e_up \
                        + D_te[model.rlayer - 1] * e_down \
                        - kroneckers_delta(model.rlayer, model.slayer) \
                        * np.exp(-model.u[model.slayer - 1] \
                                * np.abs(model.rz - model.sz))) \
                    * model.u[model.rlayer - 1]
    kernel_te_hz = U_te[model.rlayer - 1] * e_up \
                    + D_te[model.rlayer - 1] * e_down \
                    + np.sign(model.rz - model.sz) \
                    * kroneckers_delta(model.rlayer, model.slayer) \
                    * np.exp(-model.u[model.slayer - 1] \
                            * np.abs(model.rz - model.sz))
    kernel = [kernel_tm_er , kernel_te_er, kernel_tm_ez,
                   kernel_tm_hr, kernel_te_hr, kernel_te_hz]
    kernel = np.array(kernel)
    return kernel

def compute_kernel_ved(model, omega):
    """

    """
    U_te, U_tm, D_te, D_tm, e_up, e_down = model.compute_coefficients(omega)
    kernel_tm = U_tm[model.rlayer - 1] * e_up \
                    + D_tm[model.rlayer - 1] * e_down \
                    + kroneckers_delta(model.rlayer, model.slayer) \
                    * np.exp(-model.u[model.slayer - 1] \
                            * np.abs(model.rz - model.sz))
    kernel_tm_er = -U_tm[model.rlayer - 1] * e_up \
                    + D_tm[model.rlayer - 1] * e_down \
                    - (model.rz - model.sz) / np.abs(model.rz - model.sz) \
                    * kroneckers_delta(model.rlayer, model.slayer)  \
                    * np.exp(-model.u[model.slayer - 1] \
                            * np.abs(model.rz - model.sz))
    kernel_e_phi = kernel_tm_er * model.u[model.rlayer - 1] \
                    / model.u[model.slayer - 1]
    kernel_e_z = kernel_tm / model.u[model.slayer - 1]
    kernel_h_r = kernel_tm / model.u[model.slayer - 1]
    kernel = np.array([kernel_e_phi, kernel_e_z ,kernel_h_r])
    return kernel

def compute_kernel_hed(model, omega):
    """

    """
    U_te, U_tm, D_te, D_tm, e_up, e_down = model.compute_coefficients(omega)
    kernel_tm_er = (-U_tm[model.rlayer - 1] * e_up \
                        + D_tm[model.rlayer - 1] * e_down \
                        - kroneckers_delta(model.rlayer, model.slayer) \
                        * np.exp(-model.u[model.slayer - 1] \
                                * np.abs(model.rz - model.sz))) \
                    * model.u[model.rlayer - 1]
    kernel_te_er = (U_te[model.rlayer - 1] * e_up \
                        + D_te[model.rlayer - 1] * e_down \
                        + kroneckers_delta(model.rlayer, model.slayer) \
                        * np.exp(-model.u[model.slayer - 1] \
                            * np.abs(model.rz - model.sz))) \
                    / model.u[model.slayer - 1]
    kernel_tm_ez = U_tm[model.rlayer - 1] * e_up \
                    + D_tm[model.rlayer - 1] * e_down \
                    + (1-kroneckers_delta(model.rz - 1e-2, model.sz)) \
                    * np.sign(model.rz - model.sz) \
                    * kroneckers_delta(model.rlayer, model.slayer) \
                    * np.exp(-model.u[model.slayer - 1] \
                            * np.abs(model.rz - model.sz))
    kernel_tm_hr = U_tm[model.rlayer - 1] * e_up \
                    + D_tm[model.rlayer - 1] * e_down \
                    + np.sign(model.rz - model.sz) \
                    * kroneckers_delta(model.rlayer, model.slayer) \
                    * np.exp(-model.u[model.slayer - 1] \
                            * np.abs(model.rz - model.sz))
    kernel_te_hr = (-U_te[model.rlayer - 1] * e_up \
                        + D_te[model.rlayer - 1] * e_down \
                        - np.sign(model.rz - model.sz) \
                        * kroneckers_delta(model.rlayer, model.slayer) \
                        * np.exp(-model.u[model.slayer - 1] \
                                * np.abs(model.rz - model.sz))) \
                    * model.u[model.rlayer - 1] \
                    / model.u[model.slayer - 1]
    kernel_te_hz = kernel_te_er
    kernel = np.array([kernel_tm_er , kernel_te_er, kernel_tm_ez,
                    kernel_tm_hr, kernel_te_hr, kernel_te_hz])
    return kernel

def compute_kernel_circular(model, omega):
    """

    """
    U_te, U_tm, D_te, D_tm, e_up, e_down = model.compute_coefficients(omega)
    kernel_te = U_te[model.rlayer - 1] * e_up \
                    + D_te[model.rlayer - 1] * e_down \
                    + kroneckers_delta(model.rlayer, model.slayer) \
                    * np.exp(-model.u[model.slayer - 1] \
                            * np.abs(model.rz - model.sz))
    kernel_te_hr = -U_te[model.rlayer - 1] * e_up \
                    + D_te[model.rlayer - 1] * e_down \
                    - kroneckers_delta(model.rlayer, model.slayer) \
                    * (model.rz - model.sz) / np.abs(model.rz - model.sz) \
                    * np.exp(-model.u[model.slayer - 1] \
                            * np.abs(model.rz - model.sz))
    besk1 = jn(1, model.lambda_ * model.r)
    besk0 = jn(0, model.lambda_ * model.r)

    kernel_e_phi = kernel_te * model.lambda_ * besk1 \
                    / model.u[model.slayer - 1]
    kernel_h_r = kernel_te_hr * model.lambda_ * besk1 \
                    * model.u[model.rlayer - 1] \
                    / model.u[model.slayer - 1]
    kernel_h_z = kernel_te * model.lambda_ ** 2 * besk0 \
                    / model.u[model.slayer - 1]
    kernel = [kernel_e_phi, kernel_h_r, kernel_h_z]
    kernel = np.array(kernel)
    return kernel

def compute_kernel_coincident(model, omega):
    """

    """
    U_te, U_tm, D_te, D_tm, e_up, e_down = model.compute_coefficients(omega)
    kernel_te = U_te[model.rlayer - 1] * e_up \
                    + D_te[model.rlayer - 1] * e_down \
                    - kroneckers_delta(model.rlayer, model.slayer) \
                    * np.exp(-model.u[model.slayer - 1] \
                            * np.abs(model.rz - model.sz))
    besk1rad = jn(1, model.lambda_ * model.src.radius)
    kernel_h_z = kernel_te * model.lambda_ * besk1rad \
                    / model.u[model.slayer - 1]
    kernel = np.array(kernel_h_z)
    return kernel
