import numpy as np
from scipy.special import erf, erfc, jn
from emulatte.scripts_forward.utils import kroneckers_delta

def compute_kernel_vmd(model, omega):
    U_te, U_tm, D_te, D_tm, e_up, e_down = model.compute_coefficients(omega)
    kernel_te = U_te[model.rcv_layer - 1] * e_up \
                    + D_te[model.rcv_layer - 1] * e_down \
                    + kroneckers_delta(model.rcv_layer, model.tmt_layer) \
                    * np.exp(-model.u[model.tmt_layer - 1] \
                        * np.abs(model.rz - model.tz))
    kernel_te_hr = U_te[model.rcv_layer - 1] * e_up \
                    - D_te[model.rcv_layer - 1] * e_down \
                    +  kroneckers_delta(model.rcv_layer, model.tmt_layer) \
                    * (model.rz - model.tz) / np.abs(model.rz - model.tz) \
                    * np.exp(-model.u[model.tmt_layer - 1] \
                        * np.abs(model.rz - model.tz))
    kernel_e_phi = kernel_te * model.lambda_ ** 2 \
                    / model.u[model.tmt_layer - 1]
    kernel_h_r = kernel_te_hr * model.lambda_ ** 2 \
                    * model.u[model.rcv_layer - 1] \
                    / model.u[model.tmt_layer - 1]
    kernel_h_z = kernel_e_phi * model.lambda_
    kernel = []
    kernel.extend((kernel_e_phi, kernel_h_r, kernel_h_z))
    kernel = np.array(kernel)
    return kernel

def compute_kernel_hmd(model, omega):
    U_te, U_tm, D_te, D_tm, e_up, e_down = model.compute_coefficients(omega)
    kernel_tm_er = (-U_tm[model.rcv_layer - 1] * e_up \
                        + D_tm[model.rcv_layer - 1] * e_down \
                        - np.sign(model.rz - model.tz) \
                        * kroneckers_delta(model.rcv_layer, model.tmt_layer) \
                        * np.exp(-model.u[model.tmt_layer - 1] \
                                * np.abs(model.rz -model.tz))) \
                    * model.u[model.rcv_layer - 1] \
                    / model.u[model.tmt_layer - 1]
    kernel_te_er = U_te[model.rcv_layer - 1] * e_up \
                    + D_te[model.rcv_layer - 1] * e_down \
                    + np.sign(model.rz - model.tz) \
                    * kroneckers_delta(model.rcv_layer, model.tmt_layer) \
                    * np.exp(-model.u[model.tmt_layer - 1] \
                            * np.abs(model.rz - model.tz))
    kernel_tm_ez = (U_tm[model.rcv_layer - 1] * e_up \
                        + D_tm[model.rcv_layer - 1] * e_down \
                        + kroneckers_delta(model.rcv_layer, model.tmt_layer) \
                        * np.exp(-model.u[model.tmt_layer - 1] \
                                * np.abs(model.rz -model.tz))) \
                    / model.u[model.tmt_layer - 1]
    kernel_tm_hr = (U_tm[model.rcv_layer - 1] * e_up \
                        + D_tm[model.rcv_layer - 1] * e_down \
                        + kroneckers_delta(model.rcv_layer, model.tmt_layer) \
                        * np.exp(-model.u[model.tmt_layer - 1] \
                                * np.abs(model.rz -model.tz))) \
                    / model.u[model.tmt_layer - 1]
    kernel_te_hr = (-U_te[model.rcv_layer - 1] * e_up \
                        + D_te[model.rcv_layer - 1] * e_down \
                        - kroneckers_delta(model.rcv_layer, model.tmt_layer) \
                        * np.exp(-model.u[model.tmt_layer - 1] \
                                * np.abs(model.rz - model.tz))) \
                    * model.u[model.rcv_layer - 1]
    kernel_te_hz = U_te[model.rcv_layer - 1] * e_up \
                    + D_te[model.rcv_layer - 1] * e_down \
                    + np.sign(model.rz - model.tz) \
                    * kroneckers_delta(model.rcv_layer, model.tmt_layer) \
                    * np.exp(-model.u[model.tmt_layer - 1] \
                            * np.abs(model.rz - model.tz))
    kernel = []
    kernel.extend((kernel_tm_er , kernel_te_er, kernel_tm_ez,
                   kernel_tm_hr, kernel_te_hr, kernel_te_hz))
    kernel = np.array(kernel)
    return kernel

def compute_kernel_ved(model, omega):
    U_te, U_tm, D_te, D_tm, e_up, e_down = model.compute_coefficients(omega)
    kernel_tm = U_tm[model.rcv_layer - 1] * e_up \
                    + D_tm[model.rcv_layer - 1] * e_down \
                    + kroneckers_delta(model.rcv_layer, model.tmt_layer) \
                    * np.exp(-model.u[model.tmt_layer - 1] \
                            * np.abs(model.rz - model.tz))
    kernel_tm_er = -U_tm[model.rcv_layer - 1] * e_up \
                    + D_tm[model.rcv_layer - 1] * e_down \
                    - (model.rz - model.tz) / np.abs(model.rz - model.tz) \
                    * kroneckers_delta(model.rcv_layer, model.tmt_layer)  \
                    * np.exp(-model.u[model.tmt_layer - 1] \
                            * np.abs(model.rz - model.tz))
    kernel_e_phai = kernel_tm_er * model.u[model.rcv_layer - 1] \
                    / model.u[model.tmt_layer - 1]
    kernel_e_z = kernel_tm / model.u[model.tmt_layer - 1]
    kernel_h_r = kernel_tm / model.u[model.tmt_layer - 1]
    kernel = []
    kernel.extend((kernel_e_phai, kernel_e_z ,kernel_h_r))
    kernel = np.array(kernel)
    return kernel

def compute_kernel_hed(model, omega):
    U_te, U_tm, D_te, D_tm, e_up, e_down = model.compute_coefficients(omega)
    kernel_tm_er = (-U_tm[model.rcv_layer - 1] * e_up \
                        + D_tm[model.rcv_layer - 1] * e_down \
                        - kroneckers_delta(model.rcv_layer, model.tmt_layer) \
                        * np.exp(-model.u[model.tmt_layer - 1] \
                                * np.abs(model.rz - model.tz))) \
                    * model.u[model.rcv_layer - 1]
    kernel_te_er = (U_te[model.rcv_layer - 1] * e_up \
                        + D_te[model.rcv_layer - 1] * e_down \
                        + kroneckers_delta(model.rcv_layer, model.tmt_layer) \
                        * np.exp(-model.u[model.tmt_layer - 1] \
                            * np.abs(model.rz - model.tz))) \
                    / model.u[model.tmt_layer - 1]
    kernel_tm_ez = U_tm[model.rcv_layer - 1] * e_up \
                    + D_tm[model.rcv_layer - 1] * e_down \
                    + (1-kroneckers_delta(model.rz - 1e-2, model.tz)) \
                    * np.sign(model.rz - model.tz) \
                    * kroneckers_delta(model.rcv_layer, model.tmt_layer) \
                    * np.exp(-model.u[model.tmt_layer - 1] \
                            * np.abs(model.rz - model.tz))
    kernel_tm_hr = U_tm[model.rcv_layer - 1] * e_up \
                    + D_tm[model.rcv_layer - 1] * e_down \
                    + np.sign(model.rz - model.tz) \
                    * kroneckers_delta(model.rcv_layer, model.tmt_layer) \
                    * np.exp(-model.u[model.tmt_layer - 1] \
                            * np.abs(model.rz - model.tz))
    kernel_te_hr = (-U_te[model.rcv_layer - 1] * e_up \
                        + D_te[model.rcv_layer - 1] * e_down \
                        - np.sign(model.rz - model.tz) \
                        * kroneckers_delta(model.rcv_layer, model.tmt_layer) \
                        * np.exp(-model.u[model.tmt_layer - 1] \
                                * np.abs(model.rz - model.tz))) \
                    * model.u[model.rcv_layer - 1] \
                    / model.u[model.tmt_layer - 1]
    kernel_te_hz = kernel_te_er
    kernel = []
    kernel.extend((kernel_tm_er , kernel_te_er, kernel_tm_ez,
                    kernel_tm_hr, kernel_te_hr, kernel_te_hz))
    kernel = np.array(kernel)
    return kernel

def compute_kernel_circular(model, omega):
    U_te, U_tm, D_te, D_tm, e_up, e_down = model.compute_coefficients(omega)
    kernel_te = U_te[model.rcv_layer - 1] * e_up \
                    + D_te[model.rcv_layer - 1] * e_down \
                    + kroneckers_delta(model.rcv_layer, model.tmt_layer) \
                    * np.exp(-model.u[model.tmt_layer - 1] \
                            * np.abs(model.rz - model.tz))
    kernel_te_hr = -U_te[model.rcv_layer - 1] * e_up \
                    + D_te[model.rcv_layer - 1] * e_down \
                    - kroneckers_delta(model.rcv_layer, model.tmt_layer) \
                    * (model.rz - model.tz) / np.abs(model.rz - model.tz) \
                    * np.exp(-model.u[model.tmt_layer - 1] \
                            * np.abs(model.rz - model.tz))
    kernel = []
    besk1 = scipy.special.jn(1, model.lambda_ * model.r)
    besk0 = scipy.special.jn(0, model.lambda_ * model.r)

    kernel_e_phai = kernel_te * model.lambda_ * besk1 \
                    / model.u[model.tmt_layer - 1]
    kernel_h_r = kernel_te_hr * model.lambda_ * besk1 \
                    * model.u[model.rcv_layer - 1] \
                    / model.u[model.tmt_layer - 1]
    kernel_h_z = kernel_te * model.lambda_ ** 2 * besk0 \
                    / model.u[model.tmt_layer - 1]
    kernel.extend((kernel_e_phai, kernel_h_r, kernel_h_z))
    kernel = np.array(kernel)
    return kernel

def compute_kernel_coincident(model, omega):
    U_te, U_tm, D_te, D_tm, e_up, e_down = model.compute_coefficients(omega)
    kernel_te = U_te[model.rcv_layer - 1] * e_up \
                    + D_te[model.rcv_layer - 1] * e_down \
                    - kroneckers_delta(model.rcv_layer, model.tmt_layer) \
                    * np.exp(-model.u[model.tmt_layer - 1] \
                            * np.abs(model.rz - model.tz))
    kernel = []
    besk1rad = jn(1, model.lambda_ * model.transmitter.radius)
    kernel_h_z = kernel_te * model.lambda_ * besk1rad \
                    / model.u[model.tmt_layer - 1]
    kernel.append(kernel_h_z)
    kernel = np.array(kernel)
    return kernel
