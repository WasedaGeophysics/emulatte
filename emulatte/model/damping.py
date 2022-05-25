import numpy as np
from numba import jit

def compute_coefficients(
        thick_all, depth, zs, z, si, ri, ndims, u, 
        admittivity, impedivity, te_dsign, tm_dsign, te_usign, tm_usign
        ):

    kk = ndims[0] # number of frequency
    ll = ndims[1] # number of layer
    mm = ndims[2] # number of phase (lambda*rho)
    admy = admittivity
    admz = impedivity

    dr_te, dr_tm, ur_te, ur_tm, yuc, zuc = _compute_reflection(
            kk, ll, mm, si, u, thick_all, admy, admz
            )

    u_te, d_te, u_tm, d_tm = _compute_damping_coef(
            kk, ll, mm, si, ri, zs, depth, u,
            dr_te, dr_tm, ur_te, ur_tm, yuc, zuc,
            te_dsign, tm_dsign, te_usign, tm_usign
            )

    e_up, e_down = _compute_exp_up_down(kk, ll, mm, z, depth, ri, u)

    return u_te[:,ri], d_te[:,ri], u_tm[:,ri], d_tm[:,ri], e_up, e_down
    

def compute_coefficients_multi(
        thick_all, depth, zs, z, si, ri, ndims, u, 
        admittivity, impedivity, te_dsign, tm_dsign, te_usign, tm_usign
        ):
    
    jj = ndims[0] # number of dipole
    kk = ndims[1] # number of frequency
    ll = ndims[2] # number of layer
    mm = ndims[3] # number of phase (lambda*rho)
    admy = admittivity
    admz = impedivity

    u_te = np.ones((jj, kk, ll, mm), dtype=complex)
    u_tm = np.ones((jj, kk, ll, mm), dtype=complex)
    d_te = np.ones((jj, kk, ll, mm), dtype=complex)
    d_tm = np.ones((jj, kk, ll, mm), dtype=complex)
    e_up = np.ones((jj, kk, mm), dtype=complex)
    e_down = np.ones((jj, kk, mm), dtype=complex)

    # 並列化可能だが、高速化はそこまで期待できない
    for j in range(jj):
        uj = u[j]
        dr_te, dr_tm, ur_te, ur_tm, yuc, zuc = _compute_reflection(
                kk, ll, mm, si, uj, thick_all, admy, admz
                )

        u_te[j], d_te[j], u_tm[j], d_tm[j] = _compute_damping_coef(
                kk, ll, mm, si, ri, zs, depth, uj,
                dr_te, dr_tm, ur_te, ur_tm, yuc, zuc,
                te_dsign, tm_dsign, te_usign, tm_usign
                )

        e_up[j], e_down[j] = _compute_exp_up_down(kk, ll, mm, z, depth, ri, uj)

    return u_te[:,:,ri], d_te[:,:,ri], u_tm[:,:,ri], d_tm[:,:,ri], e_up, e_down

# TODO numba
#@jit(("Tuple(c16[:, :, :, :], c16[:, :, :, :], c16[:, :, :, :], "
#            "c16[:, :, :, :], c16[:, :, :, :], c16[:, :, :, :])"
#     "(i8, i8, i8, i8, i8, c16[:, :, :, :], f8[:], c16[:, :], c16[:, :])"), 
#     nopython = True)
def _compute_reflection(kk, ll, mm, si, u, thick_all, admy, impz):
    # yuc := yuc, zuc := zuc (uc : uppercase)
    tensor_thick = np.zeros((kk, mm, ll), dtype=complex)
    tensor_thick[:,:,1:-1] = thick_all[1:-1]
    tensor_thick = tensor_thick.transpose((0,2,1))

    tanhuh = np.tanh(u * tensor_thick)

    tensor_admy = np.zeros((mm, ll, kk), dtype=complex)
    tensor_impz = np.zeros((mm, ll, kk), dtype=complex)
    tensor_admy[:,:] = admy.T
    tensor_impz[:,:] = impz.T
    tensor_admy = tensor_admy.transpose((2,1,0))
    tensor_impz = tensor_impz.transpose((2,1,0))

    yuc = u / tensor_impz
    zuc = u / tensor_admy
    
    yuc_down = np.ones((kk, ll, mm), dtype=complex)
    zuc_down = np.ones((kk, ll, mm), dtype=complex)
    # xlc := x, ruc := X (lc : lowercase, uc : uppercase)
    dr_te = np.ones((kk, ll, mm), dtype=complex)
    dr_tm = np.ones((kk, ll, mm), dtype=complex)
    ur_te = np.ones((kk, ll, mm), dtype=complex)
    ur_tm = np.ones((kk, ll, mm), dtype=complex)

    yuc_down[:,-1] = yuc[:,-1]
    zuc_down[:,-1] = zuc[:,-1]

    dr_te[:,-1] = 0
    dr_tm[:,-1] = 0

    for i in range(ll - 2, si, -1):
        numerator_y = yuc_down[:, i+1] + yuc[:, i] * tanhuh[:, i]
        denominator_y = yuc[:, i] + yuc_down[:, i+1] * tanhuh[:, i]
        yuc_down[:, i] = yuc[:, i] * numerator_y / denominator_y

        numerator_z = zuc_down[:, i+1] + zuc[:, i] * tanhuh[:, i]
        denominator_z = zuc[:, i] + zuc_down[:, i+1] * tanhuh[:, i]
        zuc_down[:, i] = zuc[:, i] * numerator_z / denominator_z

        dr_te[:, i] = (yuc[:, i] - yuc_down[:, i+1]) \
                                    / (yuc[:, i] + yuc_down[:, i+1])
        dr_tm[:, i] = (zuc[:, i] - zuc_down[:, i+1]) \
                                    / (zuc[:, i] + zuc_down[:, i+1])

    if si != ll-1:
        dr_te[:, si] = (yuc[:, si] - yuc_down[:, si+1]) \
                                    / (yuc[:, si] + yuc_down[:, si+1])
        dr_tm[:, si] = (zuc[:, si] - zuc_down[:, si+1]) \
                                    / (zuc[:, si] + zuc_down[:, si+1])

    ### UP ADMITTANCE & IMPEDANCE ###
    yuc_up = np.ones((kk, ll, mm), dtype=complex)
    zuc_up = np.ones((kk, ll, mm), dtype=complex)

    yuc_up[:,0] = yuc[:,0]
    zuc_up[:,0] = zuc[:,0]

    ur_te[:,0] = 0
    ur_tm[:,0] = 0

    for i in range(2, si+1):
        numerator_y = yuc_up[:, i-2] + yuc[:, i-1] * tanhuh[:, i-1]
        denominator_y = yuc[:, i-1] + yuc_up[:, i-2] * tanhuh[:, i-1]
        yuc_up[:, i-1] = yuc[:, i-1] * numerator_y / denominator_y
        # (2)yuc_up{2,3,\,si-2,si}

        numerator_z = zuc_up[:, i-2] + zuc[:, i-1] * tanhuh[:, i-1]
        denominator_z = zuc[:, i-1] + zuc_up[:, i-2] * tanhuh[:, i-1]
        zuc_up[:, i-1] = zuc[:, i-1] * numerator_z / denominator_z

        ur_te[:, i-1] = (yuc[:, i-1] - yuc_up[:, i-2]) \
                                    / (yuc[:, i-1] + yuc_up[:, i-2])
        ur_tm[:, i-1] = (zuc[:, i-1] - zuc_up[:, i-2]) \
                                    / (zuc[:, i-1] + zuc_up[:, i-2])

    if si != 0:
        ur_te[:, si] = (yuc[:, si] - yuc_up[:, si-1]) \
                                    / (yuc[:, si] + yuc_up[:, si-1])
        ur_tm[:, si] = (zuc[:, si] - zuc_up[:, si-1]) \
                                    / (zuc[:, si] + zuc_up[:, si-1])

    return dr_te, dr_tm, ur_te, ur_tm, yuc, zuc

#TODO let's numba !
#@jit((
#    "Tuple("
#        "c16[:, :, :, :], c16[:, :, :, :], c16[:, :, :, :], c16[:, :, :, :],"
#        "("
#            "i8, i8, i8, i8, i8, i8, f8, f8[:], c16[:, :, :, :]"
#            "c16[:, :, :, :], c16[:, :, :, :], c16[:, :, :, :], "
#            "c16[:, :, :, :], c16[:, :, :, :], c16[:, :, :, :], "
#            "i8, i8, i8, i8"
#        ")"
#    ), nopython=True)
def _compute_damping_coef(
        kk, ll, mm, si, ri, zs, depth, u,
        dr_te, dr_tm, ur_te, ur_tm, yuc, zuc,
        te_dsign, tm_dsign, te_usign, tm_usign
        ):
    u_te = np.ones((kk, ll, mm), dtype=complex)
    u_tm = np.ones((kk, ll, mm), dtype=complex)
    d_te = np.ones((kk, ll, mm), dtype=complex)
    d_tm = np.ones((kk, ll, mm), dtype=complex)

    if si == 0:
        u_te[:, si] = 0
        u_tm[:, si] = 0
        d_te[:, si] = te_dsign * dr_te[:, si] \
            * np.exp(-u[:, si] * (depth[si] - zs))
        d_tm[:, si] = tm_dsign * dr_tm[:, si] \
            * np.exp(-u[:, si] * (depth[si] - zs))
    elif si == ll-1:
        u_te[:, si] = te_usign * ur_te[:, si] \
            * np.exp(u[:, si] * (depth[si-1] - zs))
        u_tm[:, si] = tm_usign * ur_tm[:, si] \
            * np.exp(u[:, si] * (depth[si-1] - zs))
        d_te[:, si] = 0
        d_tm[:, si] = 0
    else:
        e_double = np.exp(-2 * u[:, si] * (depth[si] - depth[si-1]))
        e_ud = np.exp(u[:, si] * (depth[si-1] - 2 * depth[si] + zs))
        e_du = np.exp(-u[:, si] * (depth[si] - 2 * depth[si-1] + zs))
        e_uu = np.exp(u[:, si] * (depth[si-1] - zs))
        e_dd = np.exp(-u[:, si] * (depth[si] - zs))

        u_te[:, si] = \
            1 / (1 - ur_te[:, si] * dr_te[:, si] * e_double) \
            * ur_te[:, si] \
            * (te_dsign * dr_te[:, si] * e_ud + te_usign * e_uu)

        u_tm[:, si] = \
            1 / (1 - ur_tm[:, si] * dr_tm[:, si] * e_double) \
            * ur_tm[:, si] \
            * (tm_dsign * dr_tm[:, si] * e_ud + tm_usign * e_uu)

        d_te[:, si] = \
            1 / (1 - ur_te[:, si] * dr_te[:, si] * e_double) \
            * dr_te[:, si] \
            * (te_usign * ur_te[:, si] * e_du + te_dsign * e_dd)

        d_tm[:, si] = \
            1 / (1 - ur_tm[:, si] * dr_tm[:, si] * e_double) \
            * dr_tm[:, si] \
            * (tm_usign * ur_tm[:, si] * e_du + tm_dsign * e_dd)

    # for the layers above the slayer
    if ri < si:
        if si == ll-1:
            exp_damp_s = np.exp(-u[:, si] * (zs - depth[si-1]))

            d_te[:, si-1] = \
                (yuc[:, si-1] * (1 + ur_te[:, si]) \
                    + yuc[:, si] * (1 - ur_te[:, si])) \
                / (2 * yuc[:, si-1]) * te_usign * exp_damp_s

            d_tm[:, si-1] = \
                (zuc[:, si-1] * (1 + ur_tm[:, si]) \
                    + zuc[:, si] * (1 - ur_tm[:, si])) \
                / (2 * zuc[:, si-1]) * tm_usign * exp_damp_s

        elif si != 0 and si != ll-1:
            exp_damp_s = np.exp(-u[:, si] * (zs - depth[si-1]))
            exp_damp = np.exp(-u[:, si] * (depth[si] - depth[si-1]))

            d_te[:, si-1] = \
                (yuc[:, si-1] * (1 + ur_te[:, si]) \
                    + yuc[:, si] * (1 - ur_te[:, si])) \
                / (2 * yuc[:, si-1]) \
                * (d_te[:, si] * exp_damp + te_usign * exp_damp_s)

            d_tm[:, si-1] = \
                (zuc[:, si-1] * (1 + ur_tm[:, si]) \
                    + zuc[:, si] * (1 - ur_tm[:, si])) \
                / (2 * zuc[:, si-1]) \
                * (d_tm[:, si] * exp_damp + tm_usign * exp_damp_s)
        else:
            pass

        for i in range(si-2, -1, -1):
            exp_damp = np.exp(-u[:, i+1] * (depth[i+1] - depth[i]))
            d_te[:, i] = \
                (yuc[:, i] * (1 + ur_te[:, i+1]) \
                    + yuc[:, i+1] * (1 - ur_te[:, i+1])) \
                / (2 * yuc[:, i]) * d_te[:, i+1] * exp_damp
            d_tm[:, i] = \
                (zuc[:, i] * (1 + ur_tm[:, i+1]) \
                    + zuc[:, i+1] * (1 - ur_tm[:, i+1])) \
                / (2 * zuc[:, i]) * d_tm[:, i+1] * exp_damp

        for i in range(si-1, 0, -1):
            exp_damp = np.exp(-u[:, i] * (depth[i] - depth[i-1]))
            u_te[:, i] = d_te[:, i] * exp_damp * ur_te[:, i]
            u_tm[:, i] = d_tm[:, i] * exp_damp * ur_tm[:, i]
        u_te[:, 0] = 0
        u_tm[:, 0] = 0

    # for the layers below the slayer
    if ri > si:
        if si == 0:
            exp_damp_s = np.exp(-u[:, si] * (depth[si] - zs))
            u_te[:, si+1] = \
                (yuc[:, si+1] * (1 + dr_te[:, si]) \
                    + yuc[:, si] * (1 - dr_te[:, si])) \
                / (2 * yuc[:, si+1]) * te_dsign * exp_damp_s
            u_tm[:, si+1] = \
                (zuc[:, si+1] * (1 + dr_tm[:, si]) \
                    + zuc[:, si] * (1 - dr_tm[:, si])) \
                / (2 * zuc[:, si+1]) * tm_dsign * exp_damp_s

        elif si != 0 and si != ll-1:
            exp_damp = np.exp(-u[:, si] * (depth[si] - depth[si-1]))
            exp_damp_s = np.exp(-u[:, si] * (depth[si] - zs))
            u_te[:, si+1] = \
                (yuc[:, si+1] * (1 + dr_te[:, si]) \
                    + yuc[:, si] * (1 - dr_te[:, si])) \
                / (2 * yuc[:, si+1]) \
                * (u_te[:, si] * exp_damp + te_dsign * exp_damp_s)
            u_tm[:, si+1] = \
                (zuc[:, si+1] * (1 + dr_tm[:, si]) \
                    + zuc[:, si] * (1 - dr_tm[:, si])) \
                / (2 * zuc[:, si+1]) \
                * (u_tm[:, si] * exp_damp + tm_dsign * exp_damp_s)
        else:
            pass

        for i in range(si + 2, ll):
            exp_damp = np.exp(-u[:, i-1] * (depth[i-1] - depth[i-2]))
            u_te[:, i] = \
                (yuc[:, i] * (1 + dr_te[:, i-1])
                    + yuc[:, i-1] * (1 - dr_te[:, i-1])) \
                / (2 * yuc[:, i]) * u_te[:, i-1] * exp_damp
            u_tm[:, i] = \
                (zuc[:, i] * (1 + dr_tm[:, i-1])
                    + zuc[:, i-1] * (1 - dr_tm[:, i-1])) \
                / (2 * zuc[:, i]) * u_tm[:, i-1] * exp_damp

        for i in range(si + 1, ll - 1):
            exp_damp = np.exp(-u[:, i] * (depth[i] - depth[i-1]))
            d_te[:, i] = u_te[:, i] * exp_damp * dr_te[:, i]
            d_tm[:, i] = u_tm[:, i] * exp_damp * dr_tm[:, i]
        d_te[:, ll - 1] = 0
        d_tm[:, ll - 1] = 0

    return u_te, d_te, u_tm, d_tm


def _compute_exp_up_down(kk, ll, mm, z, depth, ri, u):
    if ri == 0:
        e_up = np.zeros((kk, mm), dtype=complex)
        e_down = np.exp(-u[:, ri] * (depth[ri] - z))
    elif ri == ll-1:
        e_up = np.exp(-u[:, ri] * (z - depth[ri-1]))
        e_down = np.zeros((kk, mm), dtype=complex)
    else:
        e_up = np.exp(-u[:, ri] * (z - depth[ri-1]))
        e_down = np.exp(-u[:, ri] * (depth[ri] - z))
    return e_up, e_down