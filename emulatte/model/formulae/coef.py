import numpy as np
import scipy.constants as const
from numba import jit

def _compute_wavenumber(res, rep, rmp, omega, qss):
    # 2D array (nfreq, nlayer)
    cond = 1 / res
    eperm = rep * const.epsilon_0
    mperm = rmp * const.mu_0
    nfreq = len(omega)

    impedivity = 1j * omega.reshape(-1,1) @ mperm.reshape(1,-1)

    if qss:
        admittivity = np.ones((nfreq, 1)) @ cond.reshape(1,-1)
        admittivity = admittivity.astype(complex)
        admittivity[:, 0] = 1e-31
    else:
        admittivity = cond.reshape(1,-1) \
                        + 1j * omega.reshape(-1,1) @ eperm.reshape(1,-1)

    k = np.sqrt(-admittivity * impedivity)

    return admittivity, impedivity, k, nfreq

def _compute_lambda(ybase_phase, rho):
    r"""
    
    """
    lambda_ = ybase_phase.reshape(1,-1) / rho.reshape(-1,1)
    return lambda_

def _compute_u(lambda_, k, size):
    # for multiple dipole
    # calculate 4d tensor u = lambda**2 - k**2
    # lambda_ : (ndipole, nphase)
    # k       : (nfreq, nlayer)
    jj, kk, ll, mm = size
    lambda_tensor = np.zeros((kk, ll, jj, mm), dtype=complex)
    lambda_tensor[:,:] = lambda_
    lambda_tensor = lambda_tensor.transpose((2,0,1,3))
    k_tensor = np.zeros((jj, mm, kk, ll), dtype=complex)
    k_tensor[:,:] = k
    k_tensor = k_tensor.transpose((0,2,3,1))
    u = np.sqrt(lambda_tensor ** 2 - k_tensor ** 2)
    return u

def _compute_up_down_damping(
        thick_all, depth, zs_array, z, si_array, ri, size, u, 
        admittivity, impedivity, te_dsign, tm_dsign, te_usign, tm_usign
        ):
    
    jj = size[0] # number of dipole
    kk = size[1] # number of frequency
    ll = size[2] # number of layer
    mm = size[3] # number of phase (lambda*rho)
    admy = admittivity
    admz = impedivity

    si_previous = -1

    tf_array = zs_array / zs[0] != np.oneslike(zs)
    horizontal_order = not np.any(tf_array)

    if horizontal_order:
        si = si_array[0]
        zs = zs_array[0]

        dr_te, dr_tm, ur_te, ur_tm, yuc, zuc = _compute_reflection(
                si, jj, kk, ll, mm, u, thick_all, admy, admz
                )

        u_te, d_te, u_tm, d_tm = _compute_damping_coef(
                jj, kk, ll, mm, si, ri, zs, depth, u,
                dr_te, dr_tm, ur_te, ur_tm, yuc, zuc,
                te_dsign, tm_dsign, te_usign, tm_usign
                )

        e_up, e_down = _compute_exp_up_down(jj, kk, ll, mm, z, depth, ri, u)
    else:
        # 3次元任意方向のマルチダイポールソース（Embeded Wire, Polygonal Loop)
        pass

    return u_te[:,:,ri], d_te[:,:,ri], u_tm[:,:,ri], d_tm[:,:,ri], e_up, e_down

# for multi dipole
@jit(("Tuple(c16[:, :, :, :], c16[:, :, :, :], c16[:, :, :, :], "
            "c16[:, :, :, :], c16[:, :, :, :], c16[:, :, :, :])"
     "(i8, i8, i8, i8, i8, c16[:, :, :, :], f8[:], c16[:, :], c16[:, :])"), 
     nopython = True)
def _compute_reflection(si, jj, kk, ll, mm, u, thick_all, admy, impz):
    # yuc := yuc, zuc := zuc (uc : uppercase)
    tensor_thick = np.zeros((jj, kk, mm, ll), dtype=complex)
    tensor_thick[:,:,:,1:-1] = thick_all[1:-1]
    tensor_thick = tensor_thick.transpose((0,1,3,2))

    tanhuh = np.tanh(u * tensor_thick)

    tensor_admy = np.zeros((jj, mm, ll, kk), dtype=complex)
    tensor_impz = np.zeros((jj, mm, ll, kk), dtype=complex)
    tensor_admy[:,:] = admy.T
    tensor_impz[:,:] = impz.T
    tensor_admy = tensor_admy.transpose((0,3,2,1))
    tensor_impz = tensor_impz.transpose((0,3,2,1))

    yuc = u / tensor_impz
    zuc = u / tensor_admy
    
    yuc_down = np.ones((jj, kk, ll, mm), dtype=complex)
    zuc_down = np.ones((jj, kk, ll, mm), dtype=complex)
    # rlc := r, ruc := R (lc : lowercase)
    dr_te = np.ones((jj, kk, ll, mm), dtype=complex)
    dr_tm = np.ones((jj, kk, ll, mm), dtype=complex)
    ur_te = np.ones((jj, kk, ll, mm), dtype=complex)
    ur_tm = np.ones((jj, kk, ll, mm), dtype=complex)

    yuc_down[:,:,-1] = yuc[:,:,-1]
    zuc_down[:,:,-1] = zuc[:,:,-1]

    dr_te[:,:,-1] = 0
    dr_tm[:,:,-1] = 0

    for i in range(ll - 2, si, -1):
        numerator_y = yuc_down[:,:, i+1] + yuc[:,:, i] * tanhuh[:,:, i]
        denominator_y = yuc[:,:, i] + yuc_down[:,:, i+1] * tanhuh[:,:, i]
        yuc_down[:,:, i] = yuc[:,:, i] * numerator_y / denominator_y

        numerator_z = zuc_down[:,:, i+1] + zuc[:,:, i] * tanhuh[:,:, i]
        denominator_z = zuc[:,:, i] + zuc_down[:,:, i+1] * tanhuh[:,:, i]
        zuc_down[:,:, i] = zuc[:,:, i] * numerator_z / denominator_z

        dr_te[:,:, i] = (yuc[:,:, i] - yuc_down[:,:, i+1]) \
                                    / (yuc[:,:, i] + yuc_down[:,:, i+1])
        dr_tm[:,:, i] = (zuc[:,:, i] - zuc_down[:,:, i+1]) \
                                    / (zuc[:,:, i] + zuc_down[:,:, i+1])

    if si != ll-1:
        dr_te[:,:, si] = (yuc[:,:, si] - yuc_down[:,:, si+1]) \
                                    / (yuc[:,:, si] + yuc_down[:,:, si+1])
        dr_tm[:,:, si] = (zuc[:,:, si] - zuc_down[:,:, si+1]) \
                                    / (zuc[:,:, si] + zuc_down[:,:, si+1])

    ### UP ADMITTANCE & IMPEDANCE ###
    yuc_up = np.ones((jj, kk, ll, mm), dtype=complex)
    zuc_up = np.ones((jj, kk, ll, mm), dtype=complex)

    yuc_up[:,:,0] = yuc[:,:,0]
    zuc_up[:,:,0] = zuc[:,:,0]

    ur_te[:,:,0] = 0
    ur_tm[:,:,0] = 0

    for i in range(2, si+1):
        numerator_y = yuc_up[:,:, i-2] + yuc[:,:, i-1] * tanhuh[:,:, i-1]
        denominator_y = yuc[:,:, i-1] + yuc_up[:,:, i-2] * tanhuh[:,:, i-1]
        yuc_up[:,:, i-1] = yuc[:,:, i-1] * numerator_y / denominator_y
        # (2)yuc_up{2,3,\,si-2,si}

        numerator_z = zuc_up[:,:, i-2] + zuc[:,:, i-1] * tanhuh[:,:, i-1]
        denominator_z = zuc[:,:, i-1] + zuc_up[:,:, i-2] * tanhuh[:,:, i-1]
        zuc_up[:,:, i-1] = zuc[:,:, i-1] * numerator_z / denominator_z

        ur_te[:,:, i-1] = (yuc[:,:, i-1] - yuc_up[:,:, i-2]) \
                                    / (yuc[:,:, i-1] + yuc_up[:,:, i-2])
        ur_tm[:,:, i-1] = (zuc[:,:, i-1] - zuc_up[:,:, i-2]) \
                                    / (zuc[:,:, i-1] + zuc_up[:,:, i-2])

    if si != 0:
        ur_te[:,:, si] = (yuc[:,:, si] - yuc_up[:,:, si-1]) \
                                    / (yuc[:,:, si] + yuc_up[:,:, si-1])
        ur_tm[:,:, si] = (zuc[:,:, si] - zuc_up[:,:, si-1]) \
                                    / (zuc[:,:, si] + zuc_up[:,:, si-1])

    return dr_te, dr_tm, ur_te, ur_tm, yuc, zuc

@jit((
    "Tuple("
        "c16[:, :, :, :], c16[:, :, :, :], c16[:, :, :, :], c16[:, :, :, :],"
        "("
            "i8, i8, i8, i8, i8, i8, f8, f8[:], c16[:, :, :, :]"
            "c16[:, :, :, :], c16[:, :, :, :], c16[:, :, :, :], "
            "c16[:, :, :, :], c16[:, :, :, :], c16[:, :, :, :], "
            "i8, i8, i8, i8"
        ")"
    ), nopython=True)
def _compute_damping_coef(
        jj, kk, ll, mm, si, ri, zs, depth, u,
        dr_te, dr_tm, ur_te, ur_tm, yuc, zuc,
        te_dsign, tm_dsign, te_usign, tm_usign
        ):
    u_te = np.ones((jj, kk, ll, mm), dtype=complex)
    u_tm = np.ones((jj, kk, ll, mm), dtype=complex)
    d_te = np.ones((jj, kk, ll, mm), dtype=complex)
    d_tm = np.ones((jj, kk, ll, mm), dtype=complex)

    if si == 0:
        u_te[:,:, si] = 0
        u_tm[:,:, si] = 0
        d_te[:,:, si] = te_dsign * dr_te[:,:, si] \
            * np.exp(-u[:,:, si] * (depth[si] - zs))
        d_tm[:,:, si] = tm_dsign * dr_tm[:,:, si] \
            * np.exp(-u[:,:, si] * (depth[si] - zs))
    elif si == ll-1:
        u_te[:,:, si] = te_usign * ur_te[:,:, si] \
            * np.exp(u[:,:, si] * (depth[si-1] - zs))
        u_tm[:,:, si] = tm_usign * ur_tm[:,:, si] \
            * np.exp(u[:,:, si] * (depth[si-1] - zs))
        d_te[:,:, si] = 0
        d_tm[:,:, si] = 0
    else:
        e_1 = np.exp(-2 * u[:,:, si]
                            * (depth[si] - depth[si-1]))
        e_2u = np.exp(u[:,:, si] * (depth[si-1] - 2 * depth[si] + zs))
        e_2d = np.exp(-u[:,:, si] * (depth[si] - 2 * depth[si-1] + zs))
        e_3u = np.exp(u[:,:, si] * (depth[si-1] - zs))
        e_3d = np.exp(-u[:,:, si] * (depth[si] - zs))

        u_te[:,:, si] = \
            1 / (1 - ur_te[:,:, si] * dr_te[:,:, si] * e_1) \
            * ur_te[:,:, si] \
            * (te_dsign * dr_te[:,:, si] * e_2u + te_usign * e_3u)

        u_tm[:,:, si] = \
            1 / (1 - ur_tm[:,:, si] * dr_tm[:,:, si] * e_1) \
            * ur_tm[:,:, si] \
            * (tm_dsign * dr_tm[:,:, si] * e_2u + tm_usign * e_3u)

        d_te[:,:, si] = \
            1 / (1 - ur_te[:,:, si] * dr_te[:,:, si] * e_1) \
            * dr_te[:,:, si] \
            * (te_usign * ur_te[:,:, si] * e_2d + te_dsign * e_3d)

        d_tm[:,:, si] = \
            1 / (1 - ur_tm[:,:, si] * dr_tm[:,:, si] * e_1) \
            * dr_tm[:,:, si] \
            * (tm_usign * ur_tm[:,:, si] * e_2d + tm_dsign * e_3d)

    # for the layers above the slayer
    if ri < si:
        if si == ll-1:
            e_u = np.exp(-u[:,:, si] * (zs - depth[si-1]))

            d_te[:,:, si-1] = \
                (yuc[:,:, si-1] * (1 + ur_te[:,:, si]) + yuc[:,:, si] * (1 - ur_te[:,:, si])) \
                / (2 * yuc[:,:, si-1]) * te_usign * e_u

            d_tm[:,:, si-1] = \
                (zuc[:,:, si-1] * (1 + ur_tm[:,:, si]) + zuc[:,:, si] * (1 - ur_tm[:,:, si])) \
                / (2 * zuc[:,:, si-1]) * tm_usign * e_u

        elif si != 0 and si != ll-1:
            e_u = np.exp(-u[:,:, si] * (zs - depth[si-1]))
            exp_termii = np.exp(-u[:,:, si] * (depth[si] - depth[si-1]))

            d_te[:,:, si-1] = \
                (yuc[:,:, si-1] * (1 + ur_te[:,:, si]) + yuc[:,:, si] * (1 - ur_te[:,:, si])) \
                / (2 * yuc[:,:, si-1]) \
                * (d_te[:,:, si] * exp_termii + te_usign * e_u)

            d_tm[:,:, si-1] = \
                (zuc[:,:, si-1] * (1 + ur_tm[:,:, si]) + zuc[:,:, si] * (1 - ur_tm[:,:, si])) \
                / (2 * zuc[:,:, si-1]) \
                * (d_tm[:,:, si] * exp_termii + tm_usign * e_u)
        else:
            pass

        for jj in range(si-1, 0, -1):
            exp_termjj = np.exp(-u[:,:, jj] * (depth[jj] - depth[jj-1]))
            d_te[:,:, jj-1] = \
                (yuc[:,:, jj-1] * (1 + ur_te[:,:, jj]) + yuc[:,:, jj] * (1 - ur_te[:,:, jj])) \
                / (2 * yuc[:,:, jj-1]) * d_te[:,:, jj] * exp_termjj
            d_tm[:,:, jj-1] = \
                (zuc[:,:, jj-1] * (1 + ur_tm[:,:, jj]) + zuc[:,:, jj] * (1 - ur_tm[:,:, jj])) \
                / (2 * zuc[:,:, jj-1]) * d_tm[:,:, jj] * exp_termjj

        for jj in range(si, 1, -1):
            exp_termjj = np.exp(u[:,:, jj-1] * (depth[jj-2] - depth[jj-1]))
            u_te[:,:, jj-1] = d_te[:,:, jj-1] * exp_termjj * ur_te[:,:, jj-1]
            u_tm[:,:, jj-1] = d_tm[:,:, jj-1] * exp_termjj * ur_tm[:,:, jj-1]
        u_te[:,:, 0] = 0
        u_tm[:,:, 0] = 0

    # for the layers below the slayer
    if ri > si:
        if si == 0:
            e_u = np.exp(-u[:,:, si] * (depth[si] - zs))
            u_te[:,:, si+1] = (yuc[:,:, si+1] * (1 + dr_te[:,:, si]) \
                                            + yuc[:,:, si] * (1 - dr_te[:,:, si])) \
                        / (2 * yuc[:,:, si+1]) * te_dsign * e_u
            u_tm[:,:, si+1] = (zuc[:,:, si+1] * (1 + dr_tm[:,:, si]) \
                                            + zuc[:,:, si] * (1 - dr_tm[:,:, si])) \
                        / (2 * zuc[:,:, si+1]) * tm_dsign * e_u

        elif si != 0 and si != ll-1:
            exp_termi = np.exp(-u[:,:, si] * (depth[si] - depth[si-1]))
            exp_termii = np.exp(-u[:,:, si] * (depth[si] - zs))
            u_te[:,:, si+1] = (yuc[:,:, si+1] * (1 + dr_te[:,:, si]) \
                                            + yuc[:,:, si] * (1 - dr_te[:,:, si])) \
                        / (2 * yuc[:,:, si+1]) \
                        * (u_te[:,:, si] * exp_termi + te_dsign * exp_termii)
            u_tm[:,:, si+1] = (zuc[:,:, si+1] * (1 + dr_tm[:,:, si]) \
                                            + zuc[:,:, si] * (1 - dr_tm[:,:, si])) \
                        / (2 * zuc[:,:, si+1]) \
                        * (u_tm[:,:, si] * exp_termi + tm_dsign * exp_termii)
        else:
            pass

        for jj in range(si + 3, ll + 1):
            e_u = np.exp(-u[:,:, jj-2] * (depth[jj-2] - depth[jj - 3]))
            u_te[:,:, jj-1] = (yuc[:,:, jj-1] * (1 + dr_te[:,:, jj-2])
                            + yuc[:,:, jj-2] * (1 - dr_te[:,:, jj-2])) \
                        / (2 * yuc[:,:, jj-1]) * u_te[:,:, jj-2] * e_u
            u_tm[:,:, jj-1] = (zuc[:,:, jj-1] * (1 + dr_tm[:,:, jj-2])
                            + zuc[:,:, jj-2] * (1 - dr_tm[:,:, jj-2])) \
                        / (2 * zuc[:,:, jj-1]) * u_tm[:,:, jj-2] * e_u

        for jj in range(si + 2, ll):
            d_te[:,:, jj-1] = u_te[:,:, jj-1] \
                        * np.exp(-u[:,:, jj-1] * (depth[jj-1] - depth[jj-2])) \
                        * dr_te[:,:, jj-1]
            d_tm[:,:, jj-1] = u_tm[:,:, jj-1] \
                        * np.exp(-u[:,:, jj-1] * (depth[jj-1] - depth[jj-2])) \
                        * dr_tm[:,:, jj-1]
        d_te[:,:, ll - 1] = 0
        d_tm[:,:, ll - 1] = 0

    return u_te, d_te, u_tm, d_tm


def _compute_exp_up_down(jj, kk, ll, mm, z, depth, ri, u):
    if ri == 0:
        e_up = np.zeros((jj, kk, mm), dtype=complex)
        e_down = np.exp(u[:, :, ri] * (z - depth[ri]))
    elif ri == ll-1:
        e_up = np.exp(-u[:, :, ri] * (z - depth[ri-1]))
        e_down = np.zeros((jj, kk, mm), dtype=complex)
    else:
        e_up = np.exp(-u[:, :, ri] * (z - depth[ri-1]))
        e_down = np.exp(u[:, :, ri] * (z - depth[ri]))
    return e_up, e_down