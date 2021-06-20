import numpy as np

class Subsurface1D:
    #== CONSTRUCTOR ======================================#
    def __init__(self, thicks, displacement_current=False):
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
    
    #== CLASS DESCRIPTION ================================#
    def __str__(self):
        description =(
            'Horizontal Multi-Layered Model \n'
            'ここに使い方の説明を書く'
        )
        return description
    
    #== CHARACTERIZING LAYERS ============================#
    def add_conductivity(self, sigma):
        self.sigma = sigma

    def add_resistivity(self, rho):
        self.sigma = 1/rho

    def add_colecole_params(self, freqs, rho0, m, tau, c):
        # 要検討
        omega = 2 * np.pi * freqs
        im = 1 - (1j * omega * tau) ** c
        res = rho0 * (1 - m * (1 - 1/im))
        self.sigma = 1/res
        #return res
    
    def add_permittivity(self, epsln, relative=False):
        if relative:
            self.epsln *= epsln
        else:
            self.epsln = epsln

    def add_permeability(self, mu, relative=False):
        if relative:
            self.mu *= mu
        else:
            self.mu = mu

    #== SET UP ===========================================#
    def locate(self, tcv, sc, rc):
        self.tcv = tcv
        self.num_dipole = tcv.num_dipole
        self.kernel_te_up_sign = tcv.kernel_te_up_sign
        self.kernel_te_down_sign = tcv.kernel_te_down_sign
        self.kernel_tm_up_sign = tcv.kernel_tm_up_sign
        self.kernel_tm_down_sign = tcv.kernel_tm_down_sign
        sx, sy, sz = np.array([sc]).T
        rx, ry, rz = np.array([rc]).T
        r = np.sqrt((rx - sx) ** 2 + (ry - sy) ** 2)
        cos_phi = (rx - sx) / r
        sin_phi = (ry - sy) / r

        # 計算できない送受信座標が入力された場合の処理
        delta_z = 1e-8      #filterがanderson801の時は1e-4?

        if r == 0:
            r -= 1e-8
        if sz in self.depth:
            sz = sz - delta_z
        if sz == rz:
            sz -= delta_z

        # 送受信点が含まれる層の特定
        src_layer = self.in_which_layer(sz)
        rcv_layer = self.in_which_layer(rz)

        # return to self
        self.sx, self.sy ,self.sz = sx, sy, sz
        self.rx, self.ry ,self.rz = rx, ry, rz
        self.src_layer = src_layer 
        self.rcv_layer = rcv_layer
        self.r = r
        self.cos_phi = cos_phi
        self.sin_phi = sin_phi
    
    #== MAIN EXECUTOR ====================================#
    def run(
            self, freqtime, hankel_filter, domain='fd', 
            time_diff=False, td_transform=None, interpolate=None):

        self.freqtime = freqtime
        self.domain = domain
        self.hankel_filter = hankel_filter
        self.omega = 2 * np.pi * self.freqtime
        self.ft_size = len(freqtime)

        ans = self.tcv.get_result(
            self, freqtime, hankel_filter, domain, 
            time_diff=time_diff, td_transform=td_transform, 
            interpolate=interpolate
            )
        
        return ans

    #== INNER FUNCTIONS ==========================================================================#
    def compute_coefficients(self):
        ztilde = np.ones((1, self.num_layer, 1), dtype=np.complex)
        ytilde = np.ones((1, self.num_layer, 1), dtype=np.complex)
        k = np.zeros((1, self.num_layer), dtype=np.complex)
        u = np.ones((self.num_layer, self.filter_length, self.num_dipole), dtype=complex)
        tanhuh = np.zeros((self.num_layer - 1, self.filter_length, self.num_dipole), dtype=complex)
        Y = np.ones((self.num_layer, self.filter_length, self.num_dipole), dtype=complex)
        Z = np.ones((self.num_layer, self.filter_length, self.num_dipole), dtype=complex)

        # インピーダンス＆アドミタンス
        ztilde[0, 0, 0] = 1j * self.omega * self.mu[0]
        ztilde[0, 1:self.num_layer, 0] = 1j * self.omega * self.mu[1:self.num_layer]
        if self.displacement_current:
            ytilde[0, 0, 0] = 1j * self.omega * self.epsln[0]
        else:
            ytilde[0, 0, 0] = 1e-13
        ytilde[0, 1:self.num_layer, 0] = self.sigma[0:self.num_layer - 1] #+ 1j * self.omega * self.epsln0


        # 波数
        #k[0, 0] = (self.omega ** 2.0 * self.mu[0] * self.epsln[0]) ** 0.5
        #k[0, 1:self.num_layer] = (self.omega ** 2.0 * self.mu[1:self.num_layer] * self.epsln[1:self.num_layer] \
        #                          - 1j * self.omega * self.mu[1:self.num_layer] * self.sigma) ** 0.5 #:self.num_layer不要では
        # 誘電率を無視する近似
        k[0, 0] = 0
        k[0, 1:self.num_layer] = (- 1j * self.omega * self.mu[1:self.num_layer] * self.sigma) ** 0.5
        self.k = k

        # 層に係る量
        u[0] = self.lambda_
        for ii in range(1, self.num_layer):
            u[ii] = (self.lambda_ ** 2 - k[0,ii] ** 2) ** 0.5
        self.u = u

        # tanh
        for ii in range(1, self.num_layer - 1):
            tanhuh[ii] = (1-np.exp(-2*u[ii]* self.thicks[ii - 1]))/((1+np.exp(-2*u[ii]*self.thicks[ii - 1])))

        for ii in range(0, self.num_layer):
            Y[ii] = u[ii] / ztilde[0, ii, 0]
            Z[ii] = u[ii] / ytilde[0, ii, 0]
        self.ztilde = ztilde
        self.ytilde = ytilde
        self.Y = Y
        self.Z = Z

        r_te = np.ones((self.num_layer, self.filter_length, self.num_dipole), dtype=np.complex)
        r_tm = np.ones((self.num_layer, self.filter_length, self.num_dipole), dtype=np.complex)
        R_te = np.ones((self.num_layer, self.filter_length, self.num_dipole), dtype=np.complex)
        R_tm = np.ones((self.num_layer, self.filter_length, self.num_dipole), dtype=np.complex)

        # te, tmモードおける、下側境界の境界係数の計算　
        Ytilde = np.zeros((self.num_layer, self.filter_length, self.num_dipole), dtype=np.complex)
        Ztilde = np.zeros((self.num_layer, self.filter_length, self.num_dipole), dtype=np.complex)
        Ytilde[self.num_layer - 1] = Y[self.num_layer - 1]  # (1) Ytilde{self.num_layer}
        Ztilde[self.num_layer - 1] = Z[self.num_layer - 1]

        r_te[self.num_layer - 1] = 0
        r_tm[self.num_layer - 1] = 0

        for ii in range(self.num_layer - 1, self.src_layer, -1):  # (2) Ytilde{self.num_layer-1,self.num_layer-2,\,self.src_layer}
            numerator_Y = Ytilde[ii] + Y[ii - 1] * tanhuh[ii - 1]
            denominator_Y = Y[ii - 1] + Ytilde[ii] * tanhuh[ii - 1]
            Ytilde[ii - 1] = Y[ii - 1] * numerator_Y / denominator_Y

            numerator_Z = Ztilde[ii] + Z[ii - 1] * tanhuh[ii - 1]
            denominator_Z = Z[ii - 1] + Ztilde[ii] * tanhuh[ii - 1]
            Ztilde[ii - 1] = Z[ii - 1] * numerator_Z / denominator_Z

            r_te[ii - 1] = (Y[ii - 1] - Ytilde[ii]) / (Y[ii - 1] + Ytilde[ii])
            r_tm[ii - 1] = (Z[ii - 1] - Ztilde[ii]) / (Z[ii - 1] + Ztilde[ii])
        if self.src_layer != self.num_layer:
            r_te[self.src_layer - 1] = (Y[self.src_layer - 1] - Ytilde[self.src_layer]) / (Y[self.src_layer - 1] + Ytilde[self.src_layer])
            r_tm[self.src_layer - 1] = (Z[self.src_layer - 1] - Ztilde[self.src_layer]) / (Z[self.src_layer - 1] + Ztilde[self.src_layer])

        #test
        self.Ytilde = Ytilde
        # te,tmモードおける、上側境界の境界係数の計算
        Yhat = np.ones((self.num_layer, self.filter_length, self.num_dipole), dtype=np.complex)
        Zhat = np.ones((self.num_layer, self.filter_length, self.num_dipole), dtype=np.complex)
        Yhat[0] = Y[0]  # (1)Y{0}
        Zhat[0] = Z[0]

        R_te[0] = 0
        R_tm[0] = 0

        for ii in range(2, self.src_layer):
            numerator_Y = Yhat[ii - 2] + Y[ii - 1] * tanhuh[ii - 1]
            denominator_Y = Y[ii - 1] + Yhat[ii - 2] * tanhuh[ii - 1]
            Yhat[ii - 1] = Y[ii - 1] * numerator_Y / denominator_Y  # (2)Yhat{2,3,\,self.src_layer-2,self.src_layer-1}

            numerator_Z = Zhat[ii - 2] + Z[ii - 1] * tanhuh[ii - 1]
            denominator_Z = Z[ii - 1] + Zhat[ii - 2] * tanhuh[ii - 1]
            Zhat[ii - 1] = Z[ii - 1] * numerator_Z / denominator_Z

            R_te[ii - 1] = (Y[ii - 1] - Yhat[ii - 2]) / (Y[ii - 1] + Yhat[ii - 2])
            R_tm[ii - 1] = (Z[ii - 1] - Zhat[ii - 2]) / (Z[ii - 1] + Zhat[ii - 2])
        if self.src_layer != 1 :
            R_te[self.src_layer - 1] = (Y[self.src_layer - 1] - Yhat[self.src_layer - 2]) / (Y[self.src_layer - 1] + Yhat[self.src_layer - 2])
            R_tm[self.src_layer - 1] = (Z[self.src_layer - 1] - Zhat[self.src_layer - 2]) / (Z[self.src_layer - 1] + Zhat[self.src_layer - 2])

        U_te = np.ones((self.num_layer, self.filter_length, self.num_dipole), dtype=np.complex)
        U_tm = np.ones((self.num_layer, self.filter_length, self.num_dipole), dtype=np.complex)
        D_te = np.ones((self.num_layer, self.filter_length, self.num_dipole), dtype=np.complex)
        D_tm = np.ones((self.num_layer, self.filter_length, self.num_dipole), dtype=np.complex)

        # In the layer containing the source (src_layer)
        if self.src_layer == 1:
            U_te[self.src_layer - 1] = 0
            U_tm[self.src_layer - 1] = 0
            D_te[self.src_layer - 1] = self.kernel_te_down_sign * r_te[self.src_layer - 1] \
                * np.exp(-u[self.src_layer - 1] * (self.depth[self.src_layer - 1] - self.sz))
            D_tm[self.src_layer - 1] = self.kernel_tm_down_sign * r_tm[self.src_layer - 1] * np.exp(
                -u[self.src_layer - 1] * (self.depth[self.src_layer - 1] - self.sz))
        elif self.src_layer == self.num_layer:
            U_te[self.src_layer - 1] = self.kernel_te_up_sign * R_te[self.src_layer - 1] * np.exp(
                u[self.src_layer - 1] * (self.depth[self.src_layer - 2] - self.sz))
            U_tm[self.src_layer - 1] = self.kernel_tm_up_sign * R_tm[self.src_layer - 1] * np.exp(
                u[self.src_layer - 1] * (self.depth[self.src_layer - 2] - self.sz))
            D_te[self.src_layer - 1] = 0
            D_tm[self.src_layer - 1] = 0
        else:
            U_te[self.src_layer - 1] = 1 / (1 - R_te[self.src_layer - 1] * r_te[self.src_layer - 1] * np.exp(
                -2 * u[self.src_layer - 1] * (self.depth[self.src_layer - 1] - self.depth[self.src_layer - 2]))) * \
                                    R_te[self.src_layer - 1] * (
                                            self.kernel_te_down_sign * r_te[self.src_layer - 1] * np.exp(
                                        u[self.src_layer - 1] * (self.depth[self.src_layer - 2] - 2 * self.depth[
                                            self.src_layer - 1] + self.sz[
                                                                  0])) + self.kernel_te_up_sign * np.exp(
                                        u[self.src_layer - 1] * (self.depth[self.src_layer - 2] - self.sz[0])))
            U_tm[self.src_layer - 1] = 1 / (1 - R_tm[self.src_layer - 1] * r_tm[self.src_layer - 1] * np.exp(
                -2 * u[self.src_layer - 1] * (self.depth[self.src_layer - 1] - self.depth[self.src_layer - 2]))) * \
                                    R_tm[self.src_layer - 1] * (
                                            self.kernel_tm_down_sign * r_tm[self.src_layer - 1] * np.exp(
                                        u[self.src_layer - 1] * (self.depth[self.src_layer - 2] - 2 * self.depth[
                                            self.src_layer - 1] + self.sz[
                                                                  0])) + self.kernel_tm_up_sign * np.exp(
                                        u[self.src_layer - 1] * (self.depth[self.src_layer - 2] - self.sz[0])))
            D_te[self.src_layer - 1] = 1 / (1 - R_te[self.src_layer - 1] * r_te[self.src_layer - 1] * np.exp(
                -2 * u[self.src_layer - 1] * (self.depth[self.src_layer - 1] - self.depth[self.src_layer - 2]))) * \
                                    r_te[self.src_layer - 1] * (
                                            self.kernel_te_up_sign * R_te[self.src_layer - 1] * np.exp(
                                        -u[self.src_layer - 1] * (self.depth[self.src_layer - 1] - 2 * self.depth[
                                            self.src_layer - 2] + self.sz[
                                                                   0])) + self.kernel_te_down_sign * np.exp(
                                        -u[self.src_layer - 1] * (self.depth[self.src_layer - 1] - self.sz[0])))
            D_tm[self.src_layer - 1] = 1 / (1 - R_tm[self.src_layer - 1] * r_tm[self.src_layer - 1] * np.exp(
                -2 * u[self.src_layer - 1] * (self.depth[self.src_layer - 1] - self.depth[self.src_layer - 2]))) * \
                                    r_tm[self.src_layer - 1] * (
                                            self.kernel_tm_up_sign * R_tm[self.src_layer - 1] * np.exp(
                                        -u[self.src_layer - 1] * (self.depth[self.src_layer - 1] - 2 * self.depth[
                                            self.src_layer - 2] + self.sz[0])) + self.kernel_tm_down_sign * np.exp(
                                        -u[self.src_layer - 1] * (self.depth[self.src_layer - 1] - self.sz[0])))

        # for the layers above the src_layer
        if self.rcv_layer < self.src_layer:
            if self.src_layer == self.num_layer:
                D_te[self.src_layer - 2] = (Y[self.src_layer - 2] * (1 + R_te[self.src_layer - 1]) + Y[
                    self.src_layer - 1] * (1 - R_te[self.src_layer - 1])) / (2 * Y[self.src_layer - 2]) \
                                        * self.kernel_te_up_sign * (np.exp(
                    -u[self.src_layer - 1] * (self.sz[0] - self.depth[self.src_layer - 2])))
                D_tm[self.src_layer - 2] = (Z[self.src_layer - 2] * (1 + R_tm[self.src_layer - 1]) + Z[
                    self.src_layer - 1] * (1 - R_tm[self.src_layer - 1])) / (2 * Z[self.src_layer - 2]) \
                                        * self.kernel_tm_up_sign * (np.exp(
                    -u[self.src_layer - 1] * (self.sz[0] - self.depth[self.src_layer - 2])))
            elif self.src_layer != 1 and self.src_layer != self.num_layer:
                D_te[self.src_layer - 2] = (Y[self.src_layer - 2] * (1 + R_te[self.src_layer - 1]) + Y[
                    self.src_layer - 1] * (1 - R_te[self.src_layer - 1])) / (2 * Y[self.src_layer - 2]) \
                                        * (D_te[self.src_layer - 1] * np.exp(
                    -u[self.src_layer - 1] * (self.depth[self.src_layer - 1] - self.depth[self.src_layer - 2])) \
                                           + self.kernel_te_up_sign * np.exp(
                            -u[self.src_layer - 1] * (self.sz[0] - self.depth[self.src_layer - 2])))
                D_tm[self.src_layer - 2] = (Z[self.src_layer - 2] * (1 + R_tm[self.src_layer - 1]) + Z[
                    self.src_layer - 1] * (1 - R_tm[self.src_layer - 1])) / (2 * Z[self.src_layer - 2]) \
                                        * (D_tm[self.src_layer - 1] * np.exp(
                    -u[self.src_layer - 1] * (self.depth[self.src_layer - 1] - self.depth[self.src_layer - 2])) \
                                           + self.kernel_tm_up_sign * np.exp(
                            -u[self.src_layer - 1] * (self.sz[0] - self.depth[self.src_layer - 2])))

            for jj in range(self.src_layer - 2, 0, -1):
                D_te[jj - 1] = (Y[jj - 1] * (1 + R_te[jj]) + Y[jj] * (1 - R_te[jj])) / (2 * Y[jj - 1]) * D_te[
                    jj] * np.exp(-u[jj] * (self.depth[jj] - self.depth[jj - 1]))
                D_tm[jj - 1] = (Z[jj - 1] * (1 + R_tm[jj]) + Z[jj] * (1 - R_tm[jj])) / (2 * Z[jj - 1]) * D_tm[
                    jj] * np.exp(-u[jj] * (self.depth[jj] - self.depth[jj - 1]))
            for jj in range(self.src_layer - 1, 1, -1):
                U_te[jj - 1] = D_te[jj - 1] * np.exp(u[jj - 1] * (self.depth[jj - 2] - self.depth[jj - 1])) * \
                               R_te[jj - 1]
                U_tm[jj - 1] = D_tm[jj - 1] * np.exp(u[jj - 1] * (self.depth[jj - 2] - self.depth[jj - 1])) * \
                               R_tm[jj - 1]
            U_te[0] = 0
            U_tm[0] = 0

        # for the layers below the src_layer
        if self.rcv_layer > self.src_layer:
            if self.src_layer == 1:
                U_te[self.src_layer] = (Y[self.src_layer] * (1 + r_te[self.src_layer - 1]) + Y[self.src_layer - 1] * (
                        1 - r_te[self.src_layer - 1])) / (2 * Y[self.src_layer]) \
                                    * self.kernel_te_down_sign * (
                                        np.exp(-u[self.src_layer - 1] * (self.depth[self.src_layer - 1] - self.sz[0])))
                U_tm[self.src_layer] = (Z[self.src_layer] * (1 + r_tm[self.src_layer - 1]) + Z[self.src_layer - 1] * (
                        1 - r_tm[self.src_layer - 1])) / (2 * Z[self.src_layer]) \
                                    * self.kernel_tm_down_sign * (
                                        np.exp(-u[self.src_layer - 1] * (self.depth[self.src_layer - 1] - self.sz[0])))
            elif self.src_layer != 1 and self.src_layer != self.num_layer:
                U_te[self.src_layer] = (Y[self.src_layer] * (1 + r_te[self.src_layer - 1]) + Y[self.src_layer - 1] * (
                        1 - r_te[self.src_layer - 1])) / (2 * Y[self.src_layer]) \
                                    * (U_te[self.src_layer - 1] * np.exp(
                    -u[self.src_layer - 1] * (self.depth[self.src_layer - 1] - self.depth[self.src_layer - 2])) \
                                       + self.kernel_te_down_sign * np.exp(
                            -u[self.src_layer - 1] * (self.depth[self.src_layer - 1] - self.sz[0])))
                U_tm[self.src_layer] = (Z[self.src_layer] * (1 + r_tm[self.src_layer - 1]) + Z[self.src_layer - 1] * (
                        1 - r_tm[self.src_layer - 1])) / (2 * Z[self.src_layer]) \
                                    * (U_tm[self.src_layer - 1] * np.exp(
                    -u[self.src_layer - 1] * (self.depth[self.src_layer - 1] - self.depth[self.src_layer - 2])) \
                                       + self.kernel_tm_down_sign * np.exp(
                            -u[self.src_layer - 1] * (self.depth[self.src_layer - 1] - self.sz[0])))
            for jj in range(self.src_layer + 2, self.num_layer + 1):
                U_te[jj - 1] = (Y[jj - 1] * (1 + r_te[jj - 2]) + Y[jj - 2] * (1 - r_te[jj - 2])) / (
                        2 * Y[jj - 1]) * U_te[jj - 2] * np.exp(
                    -u[jj - 2] * (self.depth[jj - 2] - self.depth[jj - 3]))
                U_tm[jj - 1] = (Z[jj - 1] * (1 + r_tm[jj - 2]) + Z[jj - 2] * (1 - r_tm[jj - 2])) / (
                        2 * Z[jj - 1]) * U_tm[jj - 2] * np.exp(
                    -u[jj - 2] * (self.depth[jj - 2] - self.depth[jj - 3]))
            for jj in range(self.src_layer + 1, self.num_layer):
                D_te[jj - 1] = U_te[jj - 1] * np.exp(-u[jj - 1] * (self.depth[jj - 1] - self.depth[jj - 2])) * \
                               r_te[jj - 1]
                D_tm[jj - 1] = U_tm[jj - 1] * np.exp(-u[jj - 1] * (self.depth[jj - 1] - self.depth[jj - 2])) * \
                               r_tm[jj - 1]
            D_te[self.num_layer - 1] = 0
            D_tm[self.num_layer - 1] = 0

        # compute Damping coefficient
        if self.rcv_layer == 1:
            e_up = np.zeros((self.filter_length, self.num_dipole), dtype=np.complex)
            e_down = np.exp(u[self.rcv_layer - 1] * (self.rz[0] - self.depth[self.rcv_layer - 1]))
        elif self.rcv_layer == self.num_layer:
            e_up = np.exp(-u[self.rcv_layer - 1] * (self.rz[0] - self.depth[self.rcv_layer - 2]))
            e_down = np.zeros((self.filter_length, self.num_dipole), dtype=np.complex)
        else:
            e_up = np.exp(-u[self.rcv_layer - 1] * (self.rz[0] - self.depth[self.rcv_layer - 2]))
            e_down = np.exp(u[self.rcv_layer - 1] * (self.rz[0] - self.depth[self.rcv_layer - 1]))

        self.r_te = r_te
        return U_te, U_tm, D_te, D_tm, e_up, e_down

    def in_which_layer(self, z):
        layer_id = self.num_layer
        for i in range(self.num_layer+1):
            if z <= self.depth[i]:
                layer_id = i + 1
                break
            else:
                continue
        return layer_id
