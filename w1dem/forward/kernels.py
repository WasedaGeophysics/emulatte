import numpy as np
from scipy.special import erf, erfc
from w1dem.forward.utils import kroneckers_delta

def compute_kernel_vmd(mdl):
    U_te, U_tm, D_te, D_tm, e_up, e_down = mdl.compute_coefficients()
    kernel_te = U_te[mdl.rcv_layer - 1] * e_up + D_te[mdl.rcv_layer - 1] * e_down   \
                    + kroneckers_delta(mdl.rcv_layer, mdl.src_layer) * np.exp(-mdl.u[mdl.src_layer - 1] * np.abs(mdl.rz - mdl.sz))
    kernel_te_hr = U_te[mdl.rcv_layer - 1] * e_up - D_te[mdl.rcv_layer - 1] * e_down \
                    +  kroneckers_delta(mdl.rcv_layer, mdl.src_layer) \
                    * (mdl.rz - mdl.sz) / np.abs(mdl.rz - mdl.sz) \
                    * np.exp(-mdl.u[mdl.src_layer - 1] * np.abs(mdl.rz - mdl.sz))
    kernel = []
    kernel_e_phi = kernel_te * mdl.lambda_ ** 2 / mdl.u[mdl.src_layer - 1]
    kernel_h_r = kernel_te_hr * mdl.lambda_ ** 2 * mdl.u[mdl.rcv_layer - 1] / mdl.u[mdl.src_layer - 1]
    kernel_h_z = kernel_e_phi * mdl.lambda_
    kernel.extend((kernel_e_phi, kernel_h_r, kernel_h_z))
    kernel = np.array(kernel)

    #for test
    mdl.kernel_te = kernel_te
    mdl.kernel_h_z
    mdl.kernel = kernel
    return kernel