import numpy as np
def reflection(si, K, L, M, u, thick, admy, admz):
    # yuc := yuc, zuc := zuc (uc : uppercase)
    tensor_thick = np.zeros((K, M, L), dtype=complex)
    tensor_thick[:,:,1:-1] = thick[1:-1]
    tensor_thick = tensor_thick.transpose((0,2,1))
    tanhuh = np.tanh(u * tensor_thick)
    tensor_admy = np.zeros((M, L, K), dtype=complex)
    tensor_admz = np.zeros((M, L, K), dtype=complex)
    tensor_admy[:] = admy.T
    tensor_admz[:] = admz.T
    tensor_admy = tensor_admy.transpose((2,1,0))
    tensor_admz = tensor_admz.transpose((2,1,0))
    yuc = u / tensor_admz
    zuc = u / tensor_admy
    
    yuc_down = np.ones((K, L, M), dtype=complex)
    zuc_down = np.ones((K, L, M), dtype=complex)
    # rlc := r, ruc := R (lc : lowercase)
    dr_te = np.ones((K, L, M), dtype=complex)
    dr_tm = np.ones((K, L, M), dtype=complex)
    ur_te = np.ones((K, L, M), dtype=complex)
    ur_tm = np.ones((K, L, M), dtype=complex)

    yuc_down[:,-1] = yuc[:,-1]
    zuc_down[:,-1] = zuc[:,-1]

    dr_te[:,-1] = 0
    dr_tm[:,-1] = 0

    for i in range(L - 2, si, -1):
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

    if si != L-1:
        dr_te[:, si] = (yuc[:, si] - yuc_down[:, si+1]) \
                                    / (yuc[:, si] + yuc_down[:, si+1])
        dr_tm[:, si] = (zuc[:, si] - zuc_down[:, si+1]) \
                                    / (zuc[:, si] + zuc_down[:, si+1])

    ### UP ADMITTANCE & IMPEDANCE ###
    yuc_up = np.ones((K, L, M), dtype=complex)
    zuc_up = np.ones((K, L, M), dtype=complex)

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

def tetm_mode_damping(model):
    # TODO 引数にインスタンスを使わないようにしたい
    si, ri = model.si, model.ri
    sz, rz = model.sz, model.rz
    K = model.K
    L = model.L
    M = model.M
    u = model.u
    thick = model.thick
    depth = model.depth
    admy = model.admy
    admz = model.admz
    te_dsign = model.source.kernel_te_down_sign
    tm_dsign = model.source.kernel_tm_down_sign
    te_usign = model.source.kernel_te_up_sign
    tm_usign = model.source.kernel_tm_up_sign

    dr_te, dr_tm, ur_te, ur_tm, yuc, zuc= reflection(
        si, K, L, M, u, thick, admy, admz
        )

    u_te = np.ones((K, L, M), dtype=complex)
    u_tm = np.ones((K, L, M), dtype=complex)
    d_te = np.ones((K, L, M), dtype=complex)
    d_tm = np.ones((K, L, M), dtype=complex)

    # In the layer containing the source (slayer)
    if si == 0:
        u_te[:, si] = 0
        u_tm[:, si] = 0
        d_te[:, si] = te_dsign * dr_te[:, si] \
            * np.exp(-u[:, si] * (depth[si] - sz))
        d_tm[:, si] = tm_dsign * dr_tm[:, si] \
            * np.exp(-u[:, si] * (depth[si] - sz))
    elif si == L-1:
        u_te[:, si] = te_usign * ur_te[:, si] \
            * np.exp(u[:, si] * (depth[si-1] - sz))
        u_tm[:, si] = tm_usign * ur_tm[:, si] \
            * np.exp(u[:, si] * (depth[si-1] - sz))
        d_te[:, si] = 0
        d_tm[:, si] = 0
    else:
        e_1 = np.exp(-2 * u[:, si]
                            * (depth[si] - depth[si-1]))
        e_2u = np.exp(u[:, si] * (depth[si-1] - 2 * depth[si] + sz))
        e_2d = np.exp(-u[:, si] * (depth[si] - 2 * depth[si-1] + sz))
        e_3u = np.exp(u[:, si] * (depth[si-1] - sz))
        e_3d = np.exp(-u[:, si] * (depth[si] - sz))

        u_te[:, si] = \
            1 / (1 - ur_te[:, si] * dr_te[:, si] * e_1) \
            * ur_te[:, si] \
            * (te_dsign * dr_te[:, si] * e_2u + te_usign * e_3u)

        u_tm[:, si] = \
            1 / (1 - ur_tm[:, si] * dr_tm[:, si] * e_1) \
            * ur_tm[:, si] \
            * (tm_dsign * dr_tm[:, si] * e_2u + tm_usign * e_3u)

        d_te[:, si] = \
            1 / (1 - ur_te[:, si] * dr_te[:, si] * e_1) \
            * dr_te[:, si] \
            * (te_usign * ur_te[:, si] * e_2d + te_dsign * e_3d)

        d_tm[:, si] = \
            1 / (1 - ur_tm[:, si] * dr_tm[:, si] * e_1) \
            * dr_tm[:, si] \
            * (tm_usign * ur_tm[:, si] * e_2d + tm_dsign * e_3d)

    # for the layers above the slayer
    if ri < si:
        if si == L-1:
            e_u = np.exp(-u[:, si] * (sz - depth[si-1]))

            d_te[:, si-1] = \
                (yuc[:, si-1] * (1 + ur_te[:, si]) + yuc[:, si] * (1 - ur_te[:, si])) \
                / (2 * yuc[:, si-1]) * te_usign * e_u

            d_tm[:, si-1] = \
                (zuc[:, si-1] * (1 + ur_tm[:, si]) + zuc[:, si] * (1 - ur_tm[:, si])) \
                / (2 * zuc[:, si-1]) * tm_usign * e_u

        elif si != 0 and si != L-1:
            e_u = np.exp(-u[:, si] * (sz - depth[si-1]))
            exp_termii = np.exp(-u[:, si] * (depth[si] - depth[si-1]))

            d_te[:, si-1] = \
                (yuc[:, si-1] * (1 + ur_te[:, si]) + yuc[:, si] * (1 - ur_te[:, si])) \
                / (2 * yuc[:, si-1]) \
                * (d_te[:, si] * exp_termii + te_usign * e_u)

            d_tm[:, si-1] = \
                (zuc[:, si-1] * (1 + ur_tm[:, si]) + zuc[:, si] * (1 - ur_tm[:, si])) \
                / (2 * zuc[:, si-1]) \
                * (d_tm[:, si] * exp_termii + tm_usign * e_u)
        else:
            pass

        for jj in range(si-1, 0, -1):
            exp_termjj = np.exp(-u[:, jj] * (depth[jj] - depth[jj-1]))
            d_te[:, jj-1] = \
                (yuc[:, jj-1] * (1 + ur_te[:, jj]) + yuc[:, jj] * (1 - ur_te[:, jj])) \
                / (2 * yuc[:, jj-1]) * d_te[:, jj] * exp_termjj
            d_tm[:, jj-1] = \
                (zuc[:, jj-1] * (1 + ur_tm[:, jj]) + zuc[:, jj] * (1 - ur_tm[:, jj])) \
                / (2 * zuc[:, jj-1]) * d_tm[:, jj] * exp_termjj

        for jj in range(si, 1, -1):
            exp_termjj = np.exp(u[:, jj-1] * (depth[jj-2] - depth[jj-1]))
            u_te[:, jj-1] = d_te[:, jj-1] * exp_termjj * ur_te[:, jj-1]
            u_tm[:, jj-1] = d_tm[:, jj-1] * exp_termjj * ur_tm[:, jj-1]
        u_te[:, 0] = 0
        u_tm[:, 0] = 0

    # for the layers below the slayer
    if ri > si:
        if si == 0:
            e_u = np.exp(-u[:, si] * (depth[si] - sz))
            u_te[:, si+1] = (yuc[:, si+1] * (1 + dr_te[:, si]) \
                                            + yuc[:, si] * (1 - dr_te[:, si])) \
                        / (2 * yuc[:, si+1]) * te_dsign * e_u
            u_tm[:, si+1] = (zuc[:, si+1] * (1 + dr_tm[:, si]) \
                                            + zuc[:, si] * (1 - dr_tm[:, si])) \
                        / (2 * zuc[:, si+1]) * tm_dsign * e_u

        elif si != 0 and si != L-1:
            exp_termi = np.exp(-u[:, si] * (depth[si] - depth[si-1]))
            exp_termii = np.exp(-u[:, si] * (depth[si] - sz))
            u_te[:, si+1] = (yuc[:, si+1] * (1 + dr_te[:, si]) \
                                            + yuc[:, si] * (1 - dr_te[:, si])) \
                        / (2 * yuc[:, si+1]) \
                        * (u_te[:, si] * exp_termi + te_dsign * exp_termii)
            u_tm[:, si+1] = (zuc[:, si+1] * (1 + dr_tm[:, si]) \
                                            + zuc[:, si] * (1 - dr_tm[:, si])) \
                        / (2 * zuc[:, si+1]) \
                        * (u_tm[:, si] * exp_termi + tm_dsign * exp_termii)
        else:
            pass

        for jj in range(si + 3, L + 1):
            e_u = np.exp(-u[:, jj-2] * (depth[jj-2] - depth[jj - 3]))
            u_te[:, jj-1] = (yuc[:, jj-1] * (1 + dr_te[:, jj-2])
                            + yuc[:, jj-2] * (1 - dr_te[:, jj-2])) \
                        / (2 * yuc[:, jj-1]) * u_te[:, jj-2] * e_u
            u_tm[:, jj-1] = (zuc[:, jj-1] * (1 + dr_tm[:, jj-2])
                            + zuc[jj-2] * (1 - dr_tm[:, jj-2])) \
                        / (2 * zuc[:, jj-1]) * u_tm[:, jj-2] * e_u

        for jj in range(si + 2, L):
            d_te[:, jj-1] = u_te[:, jj-1] \
                        * np.exp(-u[:, jj-1] * (depth[jj-1] - depth[jj-2])) \
                        * dr_te[:, jj-1]
            d_tm[:, jj-1] = u_tm[:, jj-1] \
                        * np.exp(-u[:, jj-1] * (depth[jj-1] - depth[jj-2])) \
                        * dr_tm[:, jj-1]
        d_te[:, L - 1] = 0
        d_tm[:, L - 1] = 0

    # compute Damping coefficient
    if ri == 0:
        e_up = np.zeros((K, M), dtype=np.complex)
        e_down = np.exp(u[:, ri] * (rz - depth[ri]))
    elif ri == L-1:
        e_up = np.exp(-u[:, ri] * (rz - depth[ri-1]))
        e_down = np.zeros((K, M), dtype=np.complex)
    else:
        e_up = np.exp(-u[:, ri] * (rz - depth[ri-1]))
        e_down = np.exp(u[:, ri] * (rz - depth[ri]))
    
    return u_te[:,ri], d_te[:,ri], u_tm[:,ri], d_tm[:,ri], e_up, e_down