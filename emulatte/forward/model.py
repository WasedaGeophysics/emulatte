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

        if hankel_filter == 'anderson801':
            delta_z = 1e-4 - 1e-8
            if sz in self.depth:
                sz = sz - delta_z
            if sz == rz:
                sz -= delta_z

        ans = self.tcv.get_result(
            self, freqtime, hankel_filter, domain, 
            time_diff=time_diff, td_transform=td_transform, 
            interpolate=interpolate
            )
        
        return ans

    #== INNER FUNCTIONS ==========================================================================#
    def compute_coefficients(self):
        ztilde = np.ones((1, self.num_layer, 1), dtype=complex)
        ytilde = np.ones((1, self.num_layer, 1), dtype=complex)
        k = np.zeros(self.num_layer, dtype=np.complex)
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


        # 波数 ColeColeと一緒に考えたいところ
        #k[0] = (self.omega ** 2.0 * self.mu[0] * self.epsln[0]) ** 0.5
        #k[1:] = (self.omega ** 2.0 * self.mu[1:] * self.epsln[1:] - 1j * self.omega * self.mu[1:] * self.sigma) ** 0.5
        # 誘電率を無視する近似
        k[0] = 0
        k[1:self.num_layer] = (- 1j * self.omega * self.mu[1:self.num_layer] * self.sigma) ** 0.5

        # 層に係る量
        u[0] = self.lambda_
        for ii in range(1, self.num_layer):
            u[ii] = (self.lambda_ ** 2 - k[ii] ** 2) ** 0.5

        # tanh
        for ii in range(1, self.num_layer - 1):
            tanhuh[ii] = (1-np.exp(-2*u[ii]* self.thicks[ii - 1]))/((1+np.exp(-2*u[ii]*self.thicks[ii - 1])))

        for ii in range(0, self.num_layer):
            Y[ii] = u[ii] / ztilde[0, ii, 0]
            Z[ii] = u[ii] / ytilde[0, ii, 0]

        #return to self
        self.ztilde = ztilde
        self.ytilde = ytilde
        self.u = u

        #TE/TM mode 境界係数
        r_te = np.ones((self.num_layer, self.filter_length, self.num_dipole), dtype=np.complex)
        r_tm = np.ones((self.num_layer, self.filter_length, self.num_dipole), dtype=np.complex)
        R_te = np.ones((self.num_layer, self.filter_length, self.num_dipole), dtype=np.complex)
        R_tm = np.ones((self.num_layer, self.filter_length, self.num_dipole), dtype=np.complex)

        #送受信層index+1　for コード短縮
        si = self.src_layer
        ri = self.rcv_layer

        Ytilde = np.zeros((self.num_layer, self.filter_length, self.num_dipole), dtype=np.complex)
        Ztilde = np.zeros((self.num_layer, self.filter_length, self.num_dipole), dtype=np.complex)
        Ytilde[-1] = Y[-1]  # (1) Ytilde{self.num_layer}
        Ztilde[-1] = Z[-1]

        r_te[-1] = 0
        r_tm[-1] = 0

        for ii in range(self.num_layer - 1, si, -1):  # (2) Ytilde{self.num_layer-1,self.num_layer-2,\,si}
            numerator_Y = Ytilde[ii] + Y[ii - 1] * tanhuh[ii - 1]
            denominator_Y = Y[ii - 1] + Ytilde[ii] * tanhuh[ii - 1]
            Ytilde[ii - 1] = Y[ii - 1] * numerator_Y / denominator_Y

            numerator_Z = Ztilde[ii] + Z[ii - 1] * tanhuh[ii - 1]
            denominator_Z = Z[ii - 1] + Ztilde[ii] * tanhuh[ii - 1]
            Ztilde[ii - 1] = Z[ii - 1] * numerator_Z / denominator_Z

            r_te[ii - 1] = (Y[ii - 1] - Ytilde[ii]) / (Y[ii - 1] + Ytilde[ii])
            r_tm[ii - 1] = (Z[ii - 1] - Ztilde[ii]) / (Z[ii - 1] + Ztilde[ii])
        if si != self.num_layer:
            r_te[si - 1] = (Y[si - 1] - Ytilde[si]) / (Y[si - 1] + Ytilde[si])
            r_tm[si - 1] = (Z[si - 1] - Ztilde[si]) / (Z[si - 1] + Ztilde[si])

        Yhat = np.ones((self.num_layer, self.filter_length, self.num_dipole), dtype=complex)
        Zhat = np.ones((self.num_layer, self.filter_length, self.num_dipole), dtype=complex)
        Yhat[0] = Y[0]  # (1)Y{0}
        Zhat[0] = Z[0]

        R_te[0] = 0
        R_tm[0] = 0

        for ii in range(2, si):
            numerator_Y = Yhat[ii - 2] + Y[ii - 1] * tanhuh[ii - 1]
            denominator_Y = Y[ii - 1] + Yhat[ii - 2] * tanhuh[ii - 1]
            Yhat[ii - 1] = Y[ii - 1] * numerator_Y / denominator_Y  # (2)Yhat{2,3,\,si-2,si-1}

            numerator_Z = Zhat[ii - 2] + Z[ii - 1] * tanhuh[ii - 1]
            denominator_Z = Z[ii - 1] + Zhat[ii - 2] * tanhuh[ii - 1]
            Zhat[ii - 1] = Z[ii - 1] * numerator_Z / denominator_Z

            R_te[ii - 1] = (Y[ii - 1] - Yhat[ii - 2]) / (Y[ii - 1] + Yhat[ii - 2])
            R_tm[ii - 1] = (Z[ii - 1] - Zhat[ii - 2]) / (Z[ii - 1] + Zhat[ii - 2])
        if si != 1 :
            R_te[si - 1] = (Y[si - 1] - Yhat[si - 2]) / (Y[si - 1] + Yhat[si - 2])
            R_tm[si - 1] = (Z[si - 1] - Zhat[si - 2]) / (Z[si - 1] + Zhat[si - 2])

        U_te = np.ones((self.num_layer, self.filter_length, self.num_dipole), dtype=np.complex)
        U_tm = np.ones((self.num_layer, self.filter_length, self.num_dipole), dtype=np.complex)
        D_te = np.ones((self.num_layer, self.filter_length, self.num_dipole), dtype=np.complex)
        D_tm = np.ones((self.num_layer, self.filter_length, self.num_dipole), dtype=np.complex)

        # In the layer containing the source (src_layer)
        if si == 1:
            U_te[si - 1] = 0
            U_tm[si - 1] = 0
            D_te[si - 1] = self.kernel_te_down_sign * r_te[si - 1] * np.exp(-u[si - 1] * (self.depth[si - 1] - self.sz))
            D_tm[si - 1] = self.kernel_tm_down_sign * r_tm[si - 1] * np.exp(-u[si - 1] * (self.depth[si - 1] - self.sz))
        elif si == self.num_layer:
            U_te[si - 1] = self.kernel_te_up_sign * R_te[si - 1] * np.exp(u[si - 1] * (self.depth[si - 2] - self.sz))
            U_tm[si - 1] = self.kernel_tm_up_sign * R_tm[si - 1] * np.exp(u[si - 1] * (self.depth[si - 2] - self.sz))
            D_te[si - 1] = 0
            D_tm[si - 1] = 0
        else:
            exp_term1 = np.exp(-2 * u[si - 1] * (self.depth[si - 1] - self.depth[si - 2]))
            exp_term2u = np.exp( u[si - 1] * (self.depth[si - 2] - 2 * self.depth[si - 1] + self.sz))
            exp_term2d = np.exp(-u[si - 1] * (self.depth[si - 1] - 2 * self.depth[si - 2] + self.sz))
            exp_term3u = np.exp( u[si - 1] * (self.depth[si - 2] - self.sz))
            exp_term3d = np.exp(-u[si - 1] * (self.depth[si - 1] - self.sz))

            U_te[si - 1] = 1 / R_te[si - 1] * (1 - R_te[si - 1] * r_te[si - 1] * exp_term1) * \
                            (self.kernel_te_down_sign * r_te[si - 1] * exp_term2u + \
                             self.kernel_te_up_sign * exp_term3u)

            U_tm[si - 1] = 1 / R_tm[si - 1] * (1 - R_tm[si - 1] * r_tm[si - 1] * exp_term1) * \
                            (self.kernel_tm_down_sign * r_tm[si - 1] * exp_term2u + \
                             self.kernel_tm_up_sign * exp_term3u)

            D_te[si - 1] = 1 / r_te[si - 1] * (1 - R_te[si - 1] * r_te[si - 1] * exp_term1) * \
                            (self.kernel_te_up_sign * R_te[si - 1] * exp_term2d + \
                             self.kernel_te_down_sign * exp_term3d)

            D_tm[si - 1] = 1 / r_tm[si - 1] * (1 - R_tm[si - 1] * r_tm[si - 1] * exp_term1) * \
                            (self.kernel_tm_up_sign * R_tm[si - 1] * exp_term2d + \
                             self.kernel_tm_down_sign * exp_term3d)

        # for the layers above the src_layer
        if ri < si:
            exp_termi = np.exp(-u[si - 1] * (self.sz - self.depth[si - 2]))
            exp_termii = np.exp(-u[si - 1] * (self.depth[si - 1] - self.depth[si - 2]))

            if si == self.num_layer:
                D_te[si - 2] = (Y[si - 2] * (1 + R_te[si - 1]) + Y[si - 1] * (1 - R_te[si - 1])) \
                                / (2 * Y[si - 2]) * self.kernel_te_up_sign * exp_termi
                D_tm[si - 2] = (Z[si - 2] * (1 + R_tm[si - 1]) + Z[si - 1] * (1 - R_tm[si - 1])) \
                                / (2 * Z[si - 2]) * self.kernel_tm_up_sign * exp_termi

            elif si != 1 and si != self.num_layer:
                D_te[si - 2] = (Y[si - 2] * (1 + R_te[si - 1]) + Y[si - 1] * (1 - R_te[si - 1])) \
                                / (2 * Y[si - 2]) * (D_te[si - 1] * exp_termii + self.kernel_te_up_sign * exp_termi)
                D_tm[si - 2] = (Z[si - 2] * (1 + R_tm[si - 1]) + Z[si - 1] * (1 - R_tm[si - 1])) \
                                / (2 * Z[si - 2]) * (D_tm[si - 1] * exp_termii + self.kernel_tm_up_sign * exp_termi)

            for jj in range(si - 2, 0, -1):
                exp_termjj = np.exp(-u[jj] * (self.depth[jj] - self.depth[jj - 1]))
                D_te[jj - 1] = (Y[jj - 1] * (1 + R_te[jj]) + Y[jj] * (1 - R_te[jj])) \
                                / (2 * Y[jj - 1]) * D_te[jj] * exp_termjj
                D_tm[jj - 1] = (Z[jj - 1] * (1 + R_tm[jj]) + Z[jj] * (1 - R_tm[jj])) \
                                / (2 * Z[jj - 1]) * D_tm[jj] * exp_termjj

            for jj in range(si - 1, 1, -1):
                exp_termjj = np.exp(u[jj - 1] * (self.depth[jj - 2] - self.depth[jj - 1]))
                U_te[jj - 1] = D_te[jj - 1] * exp_termjj * R_te[jj - 1]
                U_tm[jj - 1] = D_tm[jj - 1] * exp_termjj * R_tm[jj - 1]
            U_te[0] = 0
            U_tm[0] = 0

        # for the layers below the src_layer
        if ri > si:
            if si == 1:
                exp_term = np.exp(-u[si - 1] * (self.depth[si - 1] - self.sz))
                U_te[si] = (Y[si] * (1 + r_te[si - 1]) + Y[si - 1] * (1 - r_te[si - 1])) \
                            / (2 * Y[si]) * self.kernel_te_down_sign * exp_term
                U_tm[si] = (Z[si] * (1 + r_tm[si - 1]) + Z[si - 1] * (1 - r_tm[si - 1])) \
                            / (2 * Z[si]) * self.kernel_tm_down_sign * exp_term

            elif si != 1 and si != self.num_layer:
                exp_termi = np.exp(-u[si - 1] * (self.depth[si - 1] - self.depth[si - 2]))
                exp_termii = np.exp(-u[si - 1] * (self.depth[si - 1] - self.sz))
                U_te[si] = (Y[si] * (1 + r_te[si - 1]) + Y[si - 1] * (1 - r_te[si - 1])) \
                            / (2 * Y[si]) * (U_te[si - 1] * exp_termi + self.kernel_te_down_sign * exp_termii)
                U_tm[si] = (Z[si] * (1 + r_tm[si - 1]) + Z[si - 1] * (1 - r_tm[si - 1])) \
                            / (2 * Z[si]) * (U_tm[si - 1] * exp_termi + self.kernel_tm_down_sign * exp_termii)

            for jj in range(si + 2, self.num_layer + 1):
                exp_term = np.exp(-u[jj - 2] * (self.depth[jj - 2] - self.depth[jj - 3]))
                U_te[jj - 1] = (Y[jj - 1] * (1 + r_te[jj - 2]) + Y[jj - 2] * (1 - r_te[jj - 2])) \
                                / (2 * Y[jj - 1]) * U_te[jj - 2] * exp_term
                U_tm[jj - 1] = (Z[jj - 1] * (1 + r_tm[jj - 2]) + Z[jj - 2] * (1 - r_tm[jj - 2])) \
                                / (2 * Z[jj - 1]) * U_tm[jj - 2] * exp_term
                                
            for jj in range(si + 1, self.num_layer):
                D_te[jj - 1] = U_te[jj - 1] * np.exp(-u[jj - 1] * (self.depth[jj - 1] - self.depth[jj - 2])) * \
                               r_te[jj - 1]
                D_tm[jj - 1] = U_tm[jj - 1] * np.exp(-u[jj - 1] * (self.depth[jj - 1] - self.depth[jj - 2])) * \
                               r_tm[jj - 1]
            D_te[self.num_layer - 1] = 0
            D_tm[self.num_layer - 1] = 0

        # compute Damping coefficient
        if ri == 1:
            e_up = np.zeros((self.filter_length, self.num_dipole), dtype=np.complex)
            e_down = np.exp(u[ri - 1] * (self.rz - self.depth[ri - 1]))
        elif ri == self.num_layer:
            e_up = np.exp(-u[ri - 1] * (self.rz - self.depth[ri - 2]))
            e_down = np.zeros((self.filter_length, self.num_dipole), dtype=np.complex)
        else:
            e_up = np.exp(-u[ri - 1] * (self.rz - self.depth[ri - 2]))
            e_down = np.exp(u[ri - 1] * (self.rz - self.depth[ri - 1]))

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
