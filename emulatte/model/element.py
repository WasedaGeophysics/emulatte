import numpy as np
def reflection(si, K, L, M, u, thicks, admy, admz):
    tanhuh = np.zeros((K, L, M), dtype=complex)
    # yuc := yuc, zuc := zuc (uc : uppercase)
    yuc = np.ones((K, L, M), dtype=complex)
    zuc = np.ones((K, L, M), dtype=complex)
    for i in range(L):
        tanhuh[:,i] = np.tanhuh(u[:,i] * thicks[i])
    for i in range(M):
        yuc[:,:,i] = u[:,:,i] / admz
        zuc[:,:,i] = u[:,:,i] / admy
    
    yuc_down = np.ones((K, L, M), dtype=complex)
    zuc_down = np.ones((K, L, M), dtype=complex)
    # rlc := r, ruc := R (lc : lowercase)
    rlc_te = np.ones((K, L, M), dtype=complex)
    rlc_tm = np.ones((K, L, M), dtype=complex)
    ruc_te = np.ones((K, L, M), dtype=complex)
    ruc_tm = np.ones((K, L, M), dtype=complex)

    yuc_down[:,-1] = yuc[:,-1]
    zuc_down[:,-1] = zuc[:,-1]

    rlc_te[:,-1] = 0
    rlc_tm[:,-1] = 0

    for i in range(L - 1, si + 1, -1):
        numerator_Y = yuc_down[:, i] + yuc[:, i-1] * tanhuh[:, i-1]
        denominator_Y = yuc[:, i-1] + yuc_down[:, i] * tanhuh[:, i-1]
        yuc_down[:, i-1] = yuc[:, i-1] * numerator_Y / denominator_Y

        numerator_Z = zuc_down[:, i] + zuc[:, i-1] * tanhuh[:, i-1]
        denominator_Z = zuc[:, i-1] + zuc_down[:, i] * tanhuh[:, i-1]
        zuc_down[:, i-1] = zuc[:, i-1] * numerator_Z / denominator_Z

        rlc_te[:, i-1] = (yuc[:, i-1] - yuc_down[:, i]) \
                                    / (yuc[:, i-1] + yuc_down[:, i])
        rlc_tm[:, i-1] = (zuc[:, i-1] - zuc_down[:, i]) \
                                    / (zuc[:, i-1] + zuc_down[:, i])

    if si != L-1:
        rlc_te[:, si] = (yuc[:, si] - yuc_down[si+1]) \
                                    / (yuc[:, si] + yuc_down[si+1])
        rlc_tm[:, si] = (zuc[:, si] - zuc_down[si+1]) \
                                    / (zuc[:, si] + zuc_down[si+1])

    ### UP ADMITTANCE & IMPEDANCE ###
    yuc_up = np.ones((K, L, M), dtype=complex)
    zuc_up = np.ones((K, L, M), dtype=complex)

    yuc_up[0] = yuc[0]
    zuc_up[0] = zuc[0]

    ruc_te[0] = 0
    ruc_tm[0] = 0

    for i in range(2, si+1):
        numerator_Y = yuc_up[:, i-2] + yuc[:, i-1] * tanhuh[:, i-1]
        denominator_Y = yuc[:, i-1] + yuc_up[:, i-2] * tanhuh[:, i-1]
        yuc_up[:, i-1] = yuc[:, i-1] * numerator_Y / denominator_Y
        # (2)yuc_up{2,3,\,si-2,si}

        numerator_Z = zuc_up[:, i-2] + zuc[:, i-1] * tanhuh[:, i-1]
        denominator_Z = zuc[:, i-1] + zuc_up[:, i-2] * tanhuh[:, i-1]
        zuc_up[:, i-1] = zuc[:, i-1] * numerator_Z / denominator_Z

        ruc_te[:, i-1] = (yuc[:, i-1] - yuc_up[:, i-2]) \
                                    / (yuc[:, i-1] + yuc_up[:, i-2])
        ruc_tm[:, i-1] = (zuc[:, i-1] - zuc_up[:, i-2]) \
                                    / (zuc[:, i-1] + zuc_up[:, i-2])

    if si != 0:
        ruc_te[:, si] = (yuc[:, si] - yuc_up[:, si-1]) \
                                    / (yuc[:, si] + yuc_up[:, si-1])
        ruc_tm[:, si] = (zuc[:, si] - zuc_up[:, si-1]) \
                                    / (zuc[:, si] + zuc_up[:, si-1])

    return rlc_te, rlc_tm, ruc_te, ruc_tm, yuc, zuc
    
def te_mode_damping(model):
    si = model.si
    ri = model.ri
    sz = model.sz
    rz = model.rz
    K = model.K
    L = model.L
    M = model.M
    u = model.u
    thicks = model.thicks
    depth = model.depth
    admy = model.admy
    admz = model.admz
    teds = model.source.kernel_te_down_sign
    tmds = model.source.kernel_tm_down_sign
    teus = model.source.kernel_te_up_sign
    tmus = model.source.kernel_tm_up_sign

    rlc_te, rlc_tm, ruc_te, ruc_tm, yuc, zuc= reflection(
        si, K, L, M, u, thicks, admy, admz
        )

    u_te = np.ones((K, L, M), dtype=complex)
    u_tm = np.ones((K, L, M), dtype=complex)
    d_te = np.ones((K, L, M), dtype=complex)
    d_tm = np.ones((K, L, M), dtype=complex)

    # In the layer containing the source (slayer)
    if si == 0:
        u_te[:, si] = 0
        d_te[:, si] = teds * rlc_te[:, si] \
            * np.exp(-u[:, si] * (depth[si] - sz))
    elif si == L-1:
        u_te[:, si] = teus * ruc_te[:, si] \
            * np.exp(u[:, si] * (depth[si-1] - sz))
        d_te[:, si] = 0
    else:
        exp_term1 = np.exp(-2 * u[:, si]
                            * (depth[si] - depth[si-1]))
        exp_term2u = np.exp(u[:, si] * (depth[si-1] - 2 * depth[si] + sz))
        exp_term2d = np.exp(-u[:, si] * (depth[si] - 2 * depth[si-1] + sz))
        exp_term3u = np.exp(u[:, si] * (depth[si-1] - sz))
        exp_term3d = np.exp(-u[:, si] * (depth[si] - sz))

        u_te[:, si] = \
            1 / (1 - ruc_te[:, si] * rlc_te[:, si] * exp_term1) \
            * ruc_te[:, si] \
            * (teds * rlc_te[:, si] * exp_term2u + teus * exp_term3u)

        d_te[:, si] = \
            1 / (1 - ruc_te[:, si] * rlc_te[:, si] * exp_term1) \
            * rlc_te[:, si] \
            * (teus * ruc_te[:, si] * exp_term2d + teds * exp_term3d)

    # for the layers above the slayer
    if ri < si:
        if si == L-1:
            exp_term = np.exp(-u[:, si] * (sz - depth[si-1]))

            d_te[:, si-1] = \
                (yuc[:, si-1] * (1 + ruc_te[:, si]) + yuc[:, si] * (1 - ruc_te[:, si])) \
                / (2 * yuc[:, si-1]) * teus * exp_term

        elif si != 0 and si != L-1:
            exp_term = np.exp(-u[:, si] * (sz - depth[si-1]))
            exp_termii = np.exp(-u[:, si] * (depth[si] - depth[si-1]))

            d_te[:, si-1] = \
                (yuc[:, si-1] * (1 + ruc_te[:, si]) + yuc[:, si] * (1 - ruc_te[:, si])) \
                / (2 * yuc[:, si-1]) \
                * (d_te[:, si] * exp_termii + teus * exp_term)
        else:
            pass

        for jj in range(si-1, 0, -1):
            exp_termjj = np.exp(-u[:, jj] * (depth[jj] - depth[jj-1]))
            d_te[:, jj-1] = \
                (yuc[:, jj-1] * (1 + ruc_te[:, jj]) + yuc[:, jj] * (1 - ruc_te[:, jj])) \
                / (2 * yuc[:, jj-1]) * d_te[:, jj] * exp_termjj

        for jj in range(si, 1, -1):
            exp_termjj = np.exp(u[:, jj-1] * (depth[jj-2] - depth[jj-1]))
            u_te[:, jj-1] = d_te[:, jj-1] * exp_termjj * ruc_te[:, jj-1]
        u_te[:, 0] = 0

    # for the layers below the slayer
    if ri > si:
        if si == 0:
            exp_term = np.exp(-u[:, si] * (depth[si] - sz))
            u_te[:, si+1] = (yuc[:, si+1] * (1 + rlc_te[:, si]) \
                                            + yuc[:, si] * (1 - rlc_te[:, si])) \
                        / (2 * yuc[:, si+1]) * teds * exp_term

        elif si != 0 and si != L-1:
            exp_termi = np.exp(-u[:, si] * (depth[si] - depth[si-1]))
            exp_termii = np.exp(-u[:, si] * (depth[si] - sz))
            u_te[:, si+1] = (yuc[:, si+1] * (1 + rlc_te[:, si]) \
                                            + yuc[:, si] * (1 - rlc_te[:, si])) \
                        / (2 * yuc[:, si+1]) \
                        * (u_te[:, si] * exp_termi + teds * exp_termii)
        else:
            pass

        for jj in range(si + 1, L + 1):
            exp_term = np.exp(-u[:, jj-2] * (depth[jj-2] - depth[jj - 3]))
            u_te[:, jj-1] = (yuc[:, jj-1] * (1 + rlc_te[:, jj-2])
                            + yuc[:, jj-2] * (1 - rlc_te[:, jj-2])) \
                        / (2 * yuc[:, jj-1]) * u_te[:, jj-2] * exp_term

        for jj in range(si, L):
            d_te[:, jj-1] = u_te[:, jj-1] \
                        * np.exp(-u[:, jj-1] * (depth[jj-1] - depth[jj-2])) \
                        * rlc_te[:, jj-1]
        d_te[:, L - 1] = 0

    # compute Damping coefficient
    if ri == 0:
        exp_up = np.zeros((K, M), dtype=np.complex)
        exp_down = np.exp(u[:, ri] * (rz - depth[ri]))
    elif ri == L-1:
        exp_up = np.exp(-u[:, ri] * (rz - depth[ri-1]))
        exp_down = np.zeros((K, M), dtype=np.complex)
    else:
        exp_up = np.exp(-u[:, ri] * (rz - depth[ri-1]))
        exp_down = np.exp(u[:, ri] * (rz - depth[ri]))

    damp_up = u_te[:, si]
    damp_down = d_te[:, si]

    return damp_up, damp_down, exp_up, exp_down

def tm_mode_damping(model):
    si = model.si
    ri = model.ri
    sz = model.sz
    rz = model.rz
    K = model.K
    L = model.L
    M = model.M
    u = model.u
    thicks = model.thicks
    depth = model.depth
    admy = model.admy
    admz = model.admz
    teds = model.source.kernel_te_down_sign
    tmds = model.source.kernel_tm_down_sign
    teus = model.source.kernel_te_up_sign
    tmus = model.source.kernel_tm_up_sign

    rlc_te, rlc_tm, ruc_te, ruc_tm, yuc, zuc= reflection(
        si, K, L, M, u, thicks, admy, admz
        )

    u_tm = np.ones((K, L, M), dtype=complex)
    d_tm = np.ones((K, L, M), dtype=complex)

    # In the layer containing the source (slayer)
    if si == 0:
        u_tm[:, si] = 0
        d_tm[:, si] = tmds * rlc_tm[:, si] \
            * np.exp(-u[:, si] * (depth[si] - sz))
    elif si == L-1:
        u_tm[:, si] = tmus * ruc_tm[:, si] \
            * np.exp(u[:, si] * (depth[si-1] - sz))
        d_tm[:, si] = 0
    else:
        exp_term1 = np.exp(-2 * u[:, si]
                            * (depth[si] - depth[si-1]))
        exp_term2u = np.exp(u[:, si] * (depth[si-1] - 2 * depth[si] + sz))
        exp_term2d = np.exp(-u[:, si] * (depth[si] - 2 * depth[si-1] + sz))
        exp_term3u = np.exp(u[:, si] * (depth[si-1] - sz))
        exp_term3d = np.exp(-u[:, si] * (depth[si] - sz))

        u_tm[:, si] = \
            1 / (1 - ruc_tm[:, si] * rlc_tm[:, si] * exp_term1) \
            * ruc_tm[:, si] \
            * (tmds * rlc_tm[:, si] * exp_term2u + tmus * exp_term3u)

        d_tm[:, si] = \
            1 / (1 - ruc_tm[:, si] * rlc_tm[:, si] * exp_term1) \
            * rlc_tm[:, si] \
            * (tmus * ruc_tm[:, si] * exp_term2d + tmds * exp_term3d)

    # for the layers above the slayer
    if ri < si:
        if si == L-1:
            exp_term = np.exp(-u[:, si] * (sz - depth[si-1]))

            d_tm[:, si-1] = \
                (zuc[:, si-1] * (1 + ruc_tm[:, si]) + zuc[:, si] * (1 - ruc_tm[:, si])) \
                / (2 * zuc[:, si-1]) * tmus * exp_term

        elif si != 0 and si != L-1:
            exp_term = np.exp(-u[:, si] * (sz - depth[si-1]))
            exp_termii = np.exp(-u[:, si] * (depth[si] - depth[si-1]))

            d_tm[:, si-1] = \
                (zuc[:, si-1] * (1 + ruc_tm[:, si]) + zuc[:, si] * (1 - ruc_tm[:, si])) \
                / (2 * zuc[:, si-1]) \
                * (d_tm[:, si] * exp_termii + tmus * exp_term)
        else:
            pass

        for jj in range(si-1, 0, -1):
            exp_termjj = np.exp(-u[:, jj] * (depth[jj] - depth[jj-1]))
            d_tm[:, jj-1] = \
                (zuc[:, jj-1] * (1 + ruc_tm[:, jj]) + zuc[:, jj] * (1 - ruc_tm[:, jj])) \
                / (2 * zuc[:, jj-1]) * d_tm[:, jj] * exp_termjj

        for jj in range(si, 1, -1):
            exp_termjj = np.exp(u[:, jj-1] * (depth[jj-2] - depth[jj-1]))
            u_tm[:, jj-1] = d_tm[:, jj-1] * exp_termjj * ruc_tm[:, jj-1]
        u_tm[:, 0] = 0

    # for the layers below the slayer
    if ri > si:
        if si == 0:
            exp_term = np.exp(-u[:, si] * (depth[si] - sz))
            u_tm[:, si+1] = (zuc[:, si+1] * (1 + rlc_tm[:, si]) \
                                            + zuc[:, si] * (1 - rlc_tm[:, si])) \
                        / (2 * zuc[:, si+1]) * tmds * exp_term

        elif si != 0 and si != L-1:
            exp_termi = np.exp(-u[:, si] * (depth[si] - depth[si-1]))
            exp_termii = np.exp(-u[:, si] * (depth[si] - sz))
            u_tm[:, si+1] = (zuc[:, si+1] * (1 + rlc_tm[:, si]) \
                                            + zuc[:, si] * (1 - rlc_tm[:, si])) \
                        / (2 * zuc[:, si+1]) \
                        * (u_tm[:, si] * exp_termi + tmds * exp_termii)
        else:
            pass

        for jj in range(si + 1, L + 1):
            exp_term = np.exp(-u[:, jj-2] * (depth[jj-2] - depth[jj - 3]))
            u_tm[:, jj-1] = (zuc[:, jj-1] * (1 + rlc_tm[:, jj-2])
                            + zuc[jj-2] * (1 - rlc_tm[:, jj-2])) \
                        / (2 * zuc[:, jj-1]) * u_tm[:, jj-2] * exp_term

        for jj in range(si, L):
            d_tm[:, jj-1] = u_tm[:, jj-1] \
                        * np.exp(-u[:, jj-1] * (depth[jj-1] - depth[jj-2])) \
                        * rlc_tm[:, jj-1]
        d_tm[:, L - 1] = 0

    # compute Damping coefficient
    if ri == 0:
        exp_up = np.zeros((K, M), dtype=np.complex)
        exp_down = np.exp(u[:, ri] * (rz - depth[ri]))
    elif ri == L-1:
        exp_up = np.exp(-u[:, ri] * (rz - depth[ri-1]))
        exp_down = np.zeros((K, M), dtype=np.complex)
    else:
        exp_up = np.exp(-u[:, ri] * (rz - depth[ri-1]))
        exp_down = np.exp(u[:, ri] * (rz - depth[ri]))

    damp_up = u_tm[:, si]
    damp_down = d_tm[:, si]
    
    return damp_up, damp_down, exp_up, exp_down

def tetm_mode_damping(model):
    si = model.si
    ri = model.ri
    sz = model.sz
    rz = model.rz
    K = model.K
    L = model.L
    M = model.M
    u = model.u
    thicks = model.thicks
    depth = model.depth
    admy = model.admy
    admz = model.admz
    teds = model.source.kernel_te_down_sign
    tmds = model.source.kernel_tm_down_sign
    teus = model.source.kernel_te_up_sign
    tmus = model.source.kernel_tm_up_sign

    rlc_te, rlc_tm, ruc_te, ruc_tm, yuc, zuc= reflection(
        si, K, L, M, u, thicks, admy, admz
        )

    u_te = np.ones((K, L, M), dtype=complex)
    u_tm = np.ones((K, L, M), dtype=complex)
    d_te = np.ones((K, L, M), dtype=complex)
    d_tm = np.ones((K, L, M), dtype=complex)

    # In the layer containing the source (slayer)
    if si == 0:
        u_te[:, si] = 0
        u_tm[:, si] = 0
        d_te[:, si] = teds * rlc_te[:, si] \
            * np.exp(-u[:, si] * (depth[si] - sz))
        d_tm[:, si] = tmds * rlc_tm[:, si] \
            * np.exp(-u[:, si] * (depth[si] - sz))
    elif si == L-1:
        u_te[:, si] = teus * ruc_te[:, si] \
            * np.exp(u[:, si] * (depth[si-1] - sz))
        u_tm[:, si] = tmus * ruc_tm[:, si] \
            * np.exp(u[:, si] * (depth[si-1] - sz))
        d_te[:, si] = 0
        d_tm[:, si] = 0
    else:
        exp_term1 = np.exp(-2 * u[:, si]
                            * (depth[si] - depth[si-1]))
        exp_term2u = np.exp(u[:, si] * (depth[si-1] - 2 * depth[si] + sz))
        exp_term2d = np.exp(-u[:, si] * (depth[si] - 2 * depth[si-1] + sz))
        exp_term3u = np.exp(u[:, si] * (depth[si-1] - sz))
        exp_term3d = np.exp(-u[:, si] * (depth[si] - sz))

        u_te[:, si] = \
            1 / (1 - ruc_te[:, si] * rlc_te[:, si] * exp_term1) \
            * ruc_te[:, si] \
            * (teds * rlc_te[:, si] * exp_term2u + teus * exp_term3u)

        u_tm[:, si] = \
            1 / (1 - ruc_tm[:, si] * rlc_tm[:, si] * exp_term1) \
            * ruc_tm[:, si] \
            * (tmds * rlc_tm[:, si] * exp_term2u + tmus * exp_term3u)

        d_te[:, si] = \
            1 / (1 - ruc_te[:, si] * rlc_te[:, si] * exp_term1) \
            * rlc_te[:, si] \
            * (teus * ruc_te[:, si] * exp_term2d + teds * exp_term3d)

        d_tm[:, si] = \
            1 / (1 - ruc_tm[:, si] * rlc_tm[:, si] * exp_term1) \
            * rlc_tm[:, si] \
            * (tmus * ruc_tm[:, si] * exp_term2d + tmds * exp_term3d)

    # for the layers above the slayer
    if ri < si:
        if si == L-1:
            exp_term = np.exp(-u[:, si] * (sz - depth[si-1]))

            d_te[:, si-1] = \
                (yuc[:, si-1] * (1 + ruc_te[:, si]) + yuc[:, si] * (1 - ruc_te[:, si])) \
                / (2 * yuc[:, si-1]) * teus * exp_term

            d_tm[:, si-1] = \
                (zuc[:, si-1] * (1 + ruc_tm[:, si]) + zuc[:, si] * (1 - ruc_tm[:, si])) \
                / (2 * zuc[:, si-1]) * tmus * exp_term

        elif si != 0 and si != L-1:
            exp_term = np.exp(-u[:, si] * (sz - depth[si-1]))
            exp_termii = np.exp(-u[:, si] * (depth[si] - depth[si-1]))

            d_te[:, si-1] = \
                (yuc[:, si-1] * (1 + ruc_te[:, si]) + yuc[:, si] * (1 - ruc_te[:, si])) \
                / (2 * yuc[:, si-1]) \
                * (d_te[:, si] * exp_termii + teus * exp_term)

            d_tm[:, si-1] = \
                (zuc[:, si-1] * (1 + ruc_tm[:, si]) + zuc[:, si] * (1 - ruc_tm[:, si])) \
                / (2 * zuc[:, si-1]) \
                * (d_tm[:, si] * exp_termii + tmus * exp_term)
        else:
            pass

        for jj in range(si-1, 0, -1):
            exp_termjj = np.exp(-u[:, jj] * (depth[jj] - depth[jj-1]))
            d_te[:, jj-1] = \
                (yuc[:, jj-1] * (1 + ruc_te[:, jj]) + yuc[:, jj] * (1 - ruc_te[:, jj])) \
                / (2 * yuc[:, jj-1]) * d_te[:, jj] * exp_termjj
            d_tm[:, jj-1] = \
                (zuc[:, jj-1] * (1 + ruc_tm[:, jj]) + zuc[:, jj] * (1 - ruc_tm[:, jj])) \
                / (2 * zuc[:, jj-1]) * d_tm[:, jj] * exp_termjj

        for jj in range(si, 1, -1):
            exp_termjj = np.exp(u[:, jj-1] * (depth[jj-2] - depth[jj-1]))
            u_te[:, jj-1] = d_te[:, jj-1] * exp_termjj * ruc_te[:, jj-1]
            u_tm[:, jj-1] = d_tm[:, jj-1] * exp_termjj * ruc_tm[:, jj-1]
        u_te[:, 0] = 0
        u_tm[:, 0] = 0

    # for the layers below the slayer
    if ri > si:
        if si == 0:
            exp_term = np.exp(-u[:, si] * (depth[si] - sz))
            u_te[:, si+1] = (yuc[:, si+1] * (1 + rlc_te[:, si]) \
                                            + yuc[:, si] * (1 - rlc_te[:, si])) \
                        / (2 * yuc[:, si+1]) * teds * exp_term
            u_tm[:, si+1] = (zuc[:, si+1] * (1 + rlc_tm[:, si]) \
                                            + zuc[:, si] * (1 - rlc_tm[:, si])) \
                        / (2 * zuc[:, si+1]) * tmds * exp_term

        elif si != 0 and si != L-1:
            exp_termi = np.exp(-u[:, si] * (depth[si] - depth[si-1]))
            exp_termii = np.exp(-u[:, si] * (depth[si] - sz))
            u_te[:, si+1] = (yuc[:, si+1] * (1 + rlc_te[:, si]) \
                                            + yuc[:, si] * (1 - rlc_te[:, si])) \
                        / (2 * yuc[:, si+1]) \
                        * (u_te[:, si] * exp_termi + teds * exp_termii)
            u_tm[:, si+1] = (zuc[:, si+1] * (1 + rlc_tm[:, si]) \
                                            + zuc[:, si] * (1 - rlc_tm[:, si])) \
                        / (2 * zuc[:, si+1]) \
                        * (u_tm[:, si] * exp_termi + tmds * exp_termii)
        else:
            pass

        for jj in range(si + 1, L + 1):
            exp_term = np.exp(-u[:, jj-2] * (depth[jj-2] - depth[jj - 3]))
            u_te[:, jj-1] = (yuc[:, jj-1] * (1 + rlc_te[:, jj-2])
                            + yuc[:, jj-2] * (1 - rlc_te[:, jj-2])) \
                        / (2 * yuc[:, jj-1]) * u_te[:, jj-2] * exp_term
            u_tm[:, jj-1] = (zuc[:, jj-1] * (1 + rlc_tm[:, jj-2])
                            + zuc[jj-2] * (1 - rlc_tm[:, jj-2])) \
                        / (2 * zuc[:, jj-1]) * u_tm[:, jj-2] * exp_term

        for jj in range(si, L):
            d_te[:, jj-1] = u_te[:, jj-1] \
                        * np.exp(-u[:, jj-1] * (depth[jj-1] - depth[jj-2])) \
                        * rlc_te[:, jj-1]
            d_tm[:, jj-1] = u_tm[:, jj-1] \
                        * np.exp(-u[:, jj-1] * (depth[jj-1] - depth[jj-2])) \
                        * rlc_tm[:, jj-1]
        d_te[:, L - 1] = 0
        d_tm[:, L - 1] = 0

    # compute Damping coefficient
    if ri == 0:
        exp_up = np.zeros((K, M), dtype=np.complex)
        exp_down = np.exp(u[:, ri] * (rz - depth[ri]))
    elif ri == L-1:
        exp_up = np.exp(-u[:, ri] * (rz - depth[ri-1]))
        exp_down = np.zeros((K, M), dtype=np.complex)
    else:
        exp_up = np.exp(-u[:, ri] * (rz - depth[ri-1]))
        exp_down = np.exp(u[:, ri] * (rz - depth[ri]))

    damp_up = np.array([u_te[:, si], u_tm[:, si]])
    damp_down = np.array([d_te[:, si], d_tm[:, si]])    
    return damp_up, damp_down, exp_up, exp_down