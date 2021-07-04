import numpy as np
from emulatte.scripts_forward.utils import ndarray_filter

class Subsurface1D:
    #== CONSTRUCTOR ======================================#
    def __init__(self, thicks, displacement_current=False):
        thicks = ndarray_filter(thicks, 'thicks')
        self.thicks = thicks
        self.depth = np.array([0, *np.cumsum(thicks)]) #層境界深度
        self.num_layer = len(thicks) + 2 #空気（水）層と最下層を含める
        self.displacement_current = displacement_current
        # PERMITTIVITY OF VACUUM
        epsln0 = 8.85418782 * 1e-12
        self.epsln0 = epsln0
        self.epsln = np.ones(self.num_layer) * epsln0
        # PERMEABILITY OF VACUUM
        mu0 = 4. * np.pi * 1e-7
        self.mu0 = mu0
        self.mu = np.ones(self.num_layer) * mu0
        # Spectre IP
        self.sip_mode = False
    
    #== CHARACTERIZING LAYERS ============================#
    def add_conductivity(self, sigma):
        self.sigma = ndarray_filter(sigma, 'sigma')

    def add_resistivity(self, res):
        self.res = ndarray_filter(res, 'res')
        self.sigma = 1 / res
    
    # SIP
    def add_colecole_params(self, dres, charg, tconst, fconst):
        self.dres = ndarray_filter(dres, 'dres')
        self.charg = ndarray_filter(charg, 'charg')
        self.tconst = ndarray_filter(tconst, 'tconst')
        self.fconst = ndarray_filter(fconst, 'fconst')
        self.sip_mode = True
    
    # 実装予定
    def add_permittivity(self, epsln, relative=False):
        epsln = ndarray_filter(epsln, 'epsln')
        if relative:
            self.epsln *= epsln
        else:
            self.epsln = epsln

    def add_permeability(self, mu, relative=False):
        mu = ndarray_filter(mu, 'mu')
        if relative:
            self.mu *= mu
        else:
            self.mu = mu

    #== SET UP ===========================================#
    def locate(self, transceiver, tc, rc):
        """
        計算に必要な送受信機の情報をモデルに引き渡す
        """
        self.tcv = transceiver
        self.num_dipole = transceiver.num_dipole
        self.kernel_te_up_sign = transceiver.kernel_te_up_sign
        self.kernel_te_down_sign = transceiver.kernel_te_down_sign
        self.kernel_tm_up_sign = transceiver.kernel_tm_up_sign
        self.kernel_tm_down_sign = transceiver.kernel_tm_down_sign
        tc = ndarray_filter(tc, 'tc')
        rc = ndarray_filter(rc, 'rc')

        tx, ty, tz = np.array([tc]).T
        rx, ry, rz = np.array([rc]).T
        r = np.sqrt((rx - tx) ** 2 + (ry - ty) ** 2)
        cos_phi = (rx - tx) / r
        sin_phi = (ry - ty) / r

        # 計算できない送受信座標が入力された場合の処理
        delta_z = 1e-8      #filterがanderson801の時は1e-4?

        if r == 0:
            r -= 1e-8
        if tz in self.depth:
            tz = tz - delta_z
        if tz == rz:
            tz -= delta_z

        # 送受信点が含まれる層の特定
        tmt_layer = self.in_which_layer(tz)
        rcv_layer = self.in_which_layer(rz)

        # return to self
        self.tx, self.ty ,self.tz = tx, ty, tz
        self.rx, self.ry ,self.rz = rx, ry, rz
        self.tmt_layer = tmt_layer 
        self.rcv_layer = rcv_layer
        self.r = r
        self.cos_phi = cos_phi
        self.sin_phi = sin_phi

    #== MAIN EXECUTOR ====================================#
    def emulate(self, hankel_filter, time_diff=False, td_transform=None):

        if not td_transform:
            self.domain = 'Freq'
        else:
            self.domain = 'Time'
        self.hankel_filter = hankel_filter
        self.time_diff = time_diff

        if hankel_filter == 'anderson801':
            delta_z = 1e-4 - 1e-8
            if self.tz in self.depth:
                self.tz -= delta_z
            if self.tz == self.rz:
                self.tz -= delta_z

        ans, freqtime = self.tcv.get_result(
                        self, time_diff=time_diff, td_transform=td_transform)
        
        return ans, freqtime

    #== COMPUTE COEFFICIENTS ===============================================#
    def compute_coefficients(self, omega):
        ztilde = np.ones((1, self.num_layer, 1), dtype=complex)
        ytilde = np.ones((1, self.num_layer, 1), dtype=complex)
        k = np.zeros(self.num_layer, dtype=np.complex)
        u = np.ones(
                (self.num_layer, self.filter_length, self.num_dipole),
                dtype=complex
            )
        tanhuh = np.zeros(
                (self.num_layer - 1, self.filter_length, self.num_dipole),
                dtype=complex
            )
        Y = np.ones(
                (self.num_layer, self.filter_length, self.num_dipole),
                dtype=complex
            )
        Z = np.ones(
                (self.num_layer, self.filter_length, self.num_dipole),
                dtype=complex
            )

        # Cole-Cole の複素比抵抗モデル
        if self.sip_mode == True:
            im = 1 - (1j * omega * self.tconst) ** self.fconst
            res = self.dres * (1 - self.charg * (1 - 1 / im))
            self.sigma = 1 / res
        
        # インピーダンス＆アドミタンス
        ztilde[0, 0, 0] = 1j * omega * self.mu[0]
        ztilde[0, 1:self.num_layer, 0] = 1j * omega \
                                            * self.mu[1:self.num_layer]
        if self.displacement_current:
            ytilde[0, 0, 0] = 1j * omega * self.epsln[0]
        else:
            ytilde[0, 0, 0] = 1e-13
        ytilde[0, 1:self.num_layer, 0] = self.sigma[0:self.num_layer - 1] + 1j * omega * self.epsln0


        k[0] = (omega ** 2.0 * self.mu[0] * self.epsln[0]) ** 0.5
        k[1:] = (omega ** 2.0 * self.mu[1:] * self.epsln[1:] - 1j * omega * self.mu[1:] * self.sigma) ** 0.5
        # 誘電率を無視する近似
        #k[0] = 0
        #k[1:self.num_layer] = (- 1j * omega * self.mu[1:self.num_layer] \
        #                        * self.sigma) ** 0.5

        self.k = k


        # 層に係る量
        u[0] = self.lambda_
        for ii in range(1, self.num_layer):
            u[ii] = (self.lambda_ ** 2 - k[ii] ** 2) ** 0.5

        # tanh
        for ii in range(1, self.num_layer - 1):
            tanhuh[ii] = (1 - np.exp(-2*u[ii]* self.thicks[ii - 1])) \
                            / (1 + np.exp(-2*u[ii] * self.thicks[ii - 1]))

        for ii in range(0, self.num_layer):
            Y[ii] = u[ii] / ztilde[0, ii, 0]
            Z[ii] = u[ii] / ytilde[0, ii, 0]

        #return to self
        self.ztilde = ztilde
        self.ytilde = ytilde
        self.u = u

        #TE/TM mode 境界係数
        r_te = np.ones(
                    (self.num_layer, self.filter_length, self.num_dipole),
                    dtype=np.complex
                )
        r_tm = np.ones(
                    (self.num_layer, self.filter_length, self.num_dipole), 
                    dtype=np.complex
                )
        R_te = np.ones(
                    (self.num_layer, self.filter_length, self.num_dipole),
                    dtype=np.complex
                )
        R_tm = np.ones(
                    (self.num_layer, self.filter_length, self.num_dipole),
                    dtype=np.complex
                )

        #送受信層index+1　for コード短縮
        ti = self.tmt_layer
        ri = self.rcv_layer

        Ytilde = np.zeros(
                    (self.num_layer, self.filter_length, self.num_dipole),
                    dtype=np.complex
                )
        Ztilde = np.zeros(
                    (self.num_layer, self.filter_length, self.num_dipole),
                    dtype=np.complex
                )
        Ytilde[-1] = Y[-1]  # (1) Ytilde{self.num_layer}
        Ztilde[-1] = Z[-1]

        r_te[-1] = 0
        r_tm[-1] = 0

        for ii in range(self.num_layer - 1, ti, -1):  
            # (2) Ytilde{self.num_layer-1,self.num_layer-2,\,ti}
            numerator_Y = Ytilde[ii] + Y[ii - 1] * tanhuh[ii - 1]
            denominator_Y = Y[ii - 1] + Ytilde[ii] * tanhuh[ii - 1]
            Ytilde[ii - 1] = Y[ii - 1] * numerator_Y / denominator_Y

            numerator_Z = Ztilde[ii] + Z[ii - 1] * tanhuh[ii - 1]
            denominator_Z = Z[ii - 1] + Ztilde[ii] * tanhuh[ii - 1]
            Ztilde[ii - 1] = Z[ii - 1] * numerator_Z / denominator_Z

            r_te[ii - 1] = (Y[ii - 1] - Ytilde[ii]) / (Y[ii - 1] + Ytilde[ii])
            r_tm[ii - 1] = (Z[ii - 1] - Ztilde[ii]) / (Z[ii - 1] + Ztilde[ii])
        if ti != self.num_layer:
            r_te[ti - 1] = (Y[ti - 1] - Ytilde[ti]) / (Y[ti - 1] + Ytilde[ti])
            r_tm[ti - 1] = (Z[ti - 1] - Ztilde[ti]) / (Z[ti - 1] + Ztilde[ti])

        Yhat = np.ones(
                    (self.num_layer, self.filter_length, self.num_dipole),
                    dtype=complex
                )
        Zhat = np.ones(
                    (self.num_layer, self.filter_length, self.num_dipole),
                    dtype=complex
                )
        Yhat[0] = Y[0]  # (1)Y{0}
        Zhat[0] = Z[0]

        R_te[0] = 0
        R_tm[0] = 0

        for ii in range(2, ti):
            numerator_Y = Yhat[ii - 2] + Y[ii - 1] * tanhuh[ii - 1]
            denominator_Y = Y[ii - 1] + Yhat[ii - 2] * tanhuh[ii - 1]
            Yhat[ii - 1] = Y[ii - 1] * numerator_Y / denominator_Y  
            # (2)Yhat{2,3,\,ti-2,ti-1}

            numerator_Z = Zhat[ii - 2] + Z[ii - 1] * tanhuh[ii - 1]
            denominator_Z = Z[ii - 1] + Zhat[ii - 2] * tanhuh[ii - 1]
            Zhat[ii - 1] = Z[ii - 1] * numerator_Z / denominator_Z

            R_te[ii - 1] = (Y[ii - 1] - Yhat[ii - 2]) \
                                / (Y[ii - 1] + Yhat[ii - 2])
            R_tm[ii - 1] = (Z[ii - 1] - Zhat[ii - 2]) \
                                / (Z[ii - 1] + Zhat[ii - 2])
        if ti != 1 :
            R_te[ti - 1] = (Y[ti - 1] - Yhat[ti - 2]) \
                                / (Y[ti - 1] + Yhat[ti - 2])
            R_tm[ti - 1] = (Z[ti - 1] - Zhat[ti - 2]) \
                                / (Z[ti - 1] + Zhat[ti - 2])

        U_te = np.ones(
                    (self.num_layer, self.filter_length, self.num_dipole),
                    dtype=np.complex
                )
        U_tm = np.ones(
                    (self.num_layer, self.filter_length, self.num_dipole),
                    dtype=np.complex
                )
        D_te = np.ones(
                    (self.num_layer, self.filter_length, self.num_dipole),
                    dtype=np.complex
                )
        D_tm = np.ones(
                    (self.num_layer, self.filter_length, self.num_dipole),
                    dtype=np.complex
                )

        # In the layer containing the source (tmt_layer)
        if ti == 1:
            U_te[ti - 1] = 0
            U_tm[ti - 1] = 0
            D_te[ti - 1] = self.kernel_te_down_sign * r_te[ti - 1] \
                            * np.exp(-u[ti - 1] \
                            * (self.depth[ti - 1] - self.tz))
            D_tm[ti - 1] = self.kernel_tm_down_sign * r_tm[ti - 1] \
                            * np.exp(-u[ti - 1] \
                            * (self.depth[ti - 1] - self.tz))
        elif ti == self.num_layer:
            U_te[ti - 1] = self.kernel_te_up_sign * R_te[ti - 1] \
                            * np.exp(u[ti - 1] 
                                    * (self.depth[ti - 2] - self.tz))
            U_tm[ti - 1] = self.kernel_tm_up_sign * R_tm[ti - 1] \
                            * np.exp(u[ti - 1] 
                                    * (self.depth[ti - 2] - self.tz))
            D_te[ti - 1] = 0
            D_tm[ti - 1] = 0
        else:
            exp_term1 = np.exp(-2 * u[ti - 1]
                                * (self.depth[ti - 1] - self.depth[ti - 2]))
            exp_term2u = np.exp( u[ti - 1] 
                                * (self.depth[ti - 2] 
                                    - 2 * self.depth[ti - 1] + self.tz))
            exp_term2d = np.exp(-u[ti - 1] 
                                * (self.depth[ti - 1] 
                                    - 2 * self.depth[ti - 2] + self.tz))
            exp_term3u = np.exp( u[ti - 1] * (self.depth[ti - 2] - self.tz))
            exp_term3d = np.exp(-u[ti - 1] * (self.depth[ti - 1] - self.tz))

            U_te[ti - 1] = 1 / R_te[ti - 1] \
                            * (1 - R_te[ti - 1] * r_te[ti - 1] * exp_term1) \
                            * (self.kernel_te_down_sign \
                                * r_te[ti - 1] * exp_term2u \
                                + self.kernel_te_up_sign * exp_term3u)

            U_tm[ti - 1] = 1 / R_tm[ti - 1] \
                            * (1 - R_tm[ti - 1] * r_tm[ti - 1] * exp_term1) \
                            * (self.kernel_tm_down_sign \
                                * r_tm[ti - 1] * exp_term2u \
                                + self.kernel_tm_up_sign * exp_term3u)

            D_te[ti - 1] = 1 / r_te[ti - 1] \
                            * (1 - R_te[ti - 1] * r_te[ti - 1] * exp_term1) \
                            * (self.kernel_te_up_sign \
                                * R_te[ti - 1] * exp_term2d \
                                + self.kernel_te_down_sign * exp_term3d)

            D_tm[ti - 1] = 1 / r_tm[ti - 1] \
                            * (1 - R_tm[ti - 1] * r_tm[ti - 1] * exp_term1) \
                            * (self.kernel_tm_up_sign \
                                * R_tm[ti - 1] * exp_term2d \
                                + self.kernel_tm_down_sign * exp_term3d)

        # for the layers above the tmt_layer
        if ri < ti:
            if ti == self.num_layer:
                exp_term = np.exp(-u[ti - 1] * (self.tz - self.depth[ti - 2]))
                D_te[ti - 2] = (Y[ti - 2] * (1 + R_te[ti - 1]) 
                                    + Y[ti - 1] * (1 - R_te[ti - 1])) \
                                / (2 * Y[ti - 2]) \
                                * self.kernel_te_up_sign * exp_term
                D_tm[ti - 2] = (Z[ti - 2] * (1 + R_tm[ti - 1]) \
                                    + Z[ti - 1] * (1 - R_tm[ti - 1])) \
                                / (2 * Z[ti - 2]) \
                                * self.kernel_tm_up_sign * exp_term

            elif ti != 1 and ti != self.num_layer:
                exp_term = np.exp(-u[ti - 1] * (self.tz - self.depth[ti - 2]))
                exp_termii = np.exp(-u[ti - 1] \
                            * (self.depth[ti - 1] - self.depth[ti - 2]))
                D_te[ti - 2] = (Y[ti - 2] * (1 + R_te[ti - 1]) \
                                    + Y[ti - 1] * (1 - R_te[ti - 1])) \
                                / (2 * Y[ti - 2]) * (D_te[ti - 1] \
                                * exp_termii \
                                + self.kernel_te_up_sign * exp_term)
                D_tm[ti - 2] = (Z[ti - 2] * (1 + R_tm[ti - 1]) \
                                    + Z[ti - 1] * (1 - R_tm[ti - 1])) \
                                / (2 * Z[ti - 2]) * (D_tm[ti - 1] \
                                    * exp_termii \
                                    + self.kernel_tm_up_sign * exp_term)

            for jj in range(ti - 2, 0, -1):
                exp_termjj = np.exp(-u[jj] \
                                    * (self.depth[jj] - self.depth[jj - 1]))
                D_te[jj - 1] = (Y[jj - 1] * (1 + R_te[jj]) \
                                    + Y[jj] * (1 - R_te[jj])) \
                                / (2 * Y[jj - 1]) * D_te[jj] * exp_termjj
                D_tm[jj - 1] = (Z[jj - 1] * (1 + R_tm[jj]) \
                                    + Z[jj] * (1 - R_tm[jj])) \
                                / (2 * Z[jj - 1]) * D_tm[jj] * exp_termjj

            for jj in range(ti - 1, 1, -1):
                exp_termjj = np.exp(u[jj - 1] \
                                    * (self.depth[jj - 2] \
                                        - self.depth[jj - 1]))
                U_te[jj - 1] = D_te[jj - 1] * exp_termjj * R_te[jj - 1]
                U_tm[jj - 1] = D_tm[jj - 1] * exp_termjj * R_tm[jj - 1]
            U_te[0] = 0
            U_tm[0] = 0

        # for the layers below the tmt_layer
        if ri > ti:
            if ti == 1:
                exp_term = np.exp(-u[ti - 1] * (self.depth[ti - 1] - self.tz))
                U_te[ti] = (Y[ti] * (1 + r_te[ti - 1]) \
                                + Y[ti - 1] * (1 - r_te[ti - 1])) \
                            / (2 * Y[ti]) \
                            * self.kernel_te_down_sign * exp_term
                U_tm[ti] = (Z[ti] * (1 + r_tm[ti - 1]) \
                                + Z[ti - 1] * (1 - r_tm[ti - 1])) \
                            / (2 * Z[ti]) \
                            * self.kernel_tm_down_sign * exp_term

            elif ti != 1 and ti != self.num_layer:
                exp_termi = np.exp(-u[ti - 1] \
                                * (self.depth[ti - 1] - self.depth[ti - 2]))
                exp_termii = np.exp(-u[ti - 1] 
                                * (self.depth[ti - 1] - self.tz))
                U_te[ti] = (Y[ti] * (1 + r_te[ti - 1]) \
                                    + Y[ti - 1] * (1 - r_te[ti - 1])) \
                                / (2 * Y[ti]) \
                                * (U_te[ti - 1] * exp_termi \
                                    + self.kernel_te_down_sign * exp_termii)
                U_tm[ti] = (Z[ti] * (1 + r_tm[ti - 1]) + Z[ti - 1] \
                                    * (1 - r_tm[ti - 1])) \
                                / (2 * Z[ti]) \
                                * (U_tm[ti - 1] * exp_termi \
                                    + self.kernel_tm_down_sign * exp_termii)

            for jj in range(ti + 2, self.num_layer + 1):
                exp_term = np.exp(-u[jj - 2] \
                                * (self.depth[jj - 2] - self.depth[jj - 3]))
                U_te[jj - 1] = (Y[jj - 1] * (1 + r_te[jj - 2]) \
                                    + Y[jj - 2] * (1 - r_te[jj - 2])) \
                                / (2 * Y[jj - 1]) * U_te[jj - 2] * exp_term
                U_tm[jj - 1] = (Z[jj - 1] * (1 + r_tm[jj - 2]) \
                                    + Z[jj - 2] * (1 - r_tm[jj - 2])) \
                                / (2 * Z[jj - 1]) * U_tm[jj - 2] * exp_term
                                
            for jj in range(ti + 1, self.num_layer):
                D_te[jj - 1] = U_te[jj - 1] * np.exp(-u[jj - 1] \
                                * (self.depth[jj - 1] - self.depth[jj - 2])) \
                                * r_te[jj - 1]
                D_tm[jj - 1] = U_tm[jj - 1] * np.exp(-u[jj - 1] \
                                * (self.depth[jj - 1] - self.depth[jj - 2])) \
                                * r_tm[jj - 1]
            D_te[self.num_layer - 1] = 0
            D_tm[self.num_layer - 1] = 0

        # compute Damping coefficient
        if ri == 1:
            e_up = np.zeros(
                        (self.filter_length, self.num_dipole),
                        dtype=np.complex
                    )
            e_down = np.exp(u[ri - 1] * (self.rz - self.depth[ri - 1]))
        elif ri == self.num_layer:
            e_up = np.exp(-u[ri - 1] * (self.rz - self.depth[ri - 2]))
            e_down = np.zeros(
                        (self.filter_length, self.num_dipole),
                        dtype=np.complex
                    )
        else:
            e_up = np.exp(-u[ri - 1] * (self.rz - self.depth[ri - 2]))
            e_down = np.exp(u[ri - 1] * (self.rz - self.depth[ri - 1]))

        return U_te, U_tm, D_te, D_tm, e_up, e_down

    def in_which_layer(self, z):
        layer_id = 1
        for i in range(self.num_layer-1, 0, -1):
            if z >= self.depth[i-1]:
                layer_id = i + 1
                break
            else:
                continue
        return layer_id
