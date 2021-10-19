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
"""
Electromagnetic ground model class group
"""
import numpy as np
import scipy.constants as const
from emulatte.utils.function import ndarray_converter

class Subsurface1D:
    #== CONSTRUCTOR ======================================#
    def __init__(self, thicks):
        """

        """
        ### STRUCTURE ###
        thicks = ndarray_converter(thicks, 'thicks')
        # THICKNESS
        self.thicks = thicks
        # LAYERS BOUNDARY DEPTH
        self.depth = np.array([0, *np.cumsum(thicks)])
        # NUMBER OF LAYERS
        self.num_layer = len(thicks) + 2

    
    #== CHARACTERIZING LAYERS (ONLY ISOTROPIC MODEL)============================#
    def set_properties(self, **props):
        kwds = set(props.keys())
        self.cxres = False

        ### ELECTRIC PERMITTIVITY ###
        if 'epsilon' in kwds:
            self.epsln = ndarray_converter(props['epsilon'], 'epsilon')
        elif 'epsilon_r' in kwds:
            epsilon_r = ndarray_converter(props['epsilon_r'], 'epsilon_r')
            self.epsln = epsilon_r * const.epsilon_0
        else:
            self.epsln = np.ones(self.num_layer) * const.epsilon_0

        ### MAGNETIC PERMEABILITY ###
        if 'mu' in kwds:
            self.mu = ndarray_converter(props['mu'], 'mu')
        elif 'mu_r' in kwds:
            mu_r = ndarray_converter(props['mu_r'], 'mu_r')
            self.mu = mu_r * const.mu_0
        else:
            self.mu = np.ones(self.num_layer) * const.mu_0

        ### REAL RESISTIVITY MODEL ###
        if 'res' in kwds:
            res = ndarray_converter(props['res'], 'res')
            self.sigma = 1 / res
        ### COMPLEX RESISTIVITY MODEL (Pelton et al. (1978)) ###
        elif ('res_0' in kwds) & ('m' in kwds):
            self.res_0 = ndarray_converter(props['res_0'], 'res_0')
            self.m = ndarray_converter(props['m'], 'm')
            self.tau = ndarray_converter(props['tau'], 'tau')
            self.c = ndarray_converter(props['c'], 'c')
            self.cxres = True
        else:
            raise Exception('Could not find the input for resistivity values.')


    #== SET UP ===========================================#
    def locate(self, emsrc, sc, rc, **kwargs):
        """
        Coordinates & Angles
        """
        self.src = emsrc
        sc = ndarray_converter(sc, 'sc')
        rc = ndarray_converter(rc, 'rc')
        DIPOLE = [
            'VMD', 'HMDx', 'HMDy',
            'VED', 'HEDx', 'HEDy',
            'CoincidentLoop', 'CircularLoop'
        ]

        if emsrc.__class__.__name__ in DIPOLE:
            sx, sy, sz = np.array([sc]).T
            rx, ry, rz = np.array([rc]).T
            r = np.sqrt((rx - sx) ** 2 + (ry - sy) ** 2)

            # 計算できない送受信座標が入力された場合の処理
            delta_z = 1e-8      #filterがanderson801の時は1e-4?

            if r == 0:
                r = 1e-8
            if sz in self.depth:
                sz = sz - delta_z
            if sz == rz:
                sz = sz - delta_z

            # Azimuth?
            cos_phi = (rx - sx) / r
            sin_phi = (ry - sy) / r

            # 送受信点が含まれる層の特定
            slayer = self.in_which_layer(sz)
            rlayer = self.in_which_layer(rz)

            # return to self
            self.sx, self.sy ,self.sz = sx, sy, sz
            self.rx, self.ry ,self.rz = rx, ry, rz
            self.slayer = slayer 
            self.rlayer = rlayer
            self.r = r
            self.cos_phi = cos_phi
            self.sin_phi = sin_phi

        elif emsrc.__class__.__name__ == 'GroundedWire':
            sx, sy, sz = np.array([sc]).T
            rx, ry, rz = np.array([rc]).T
            length = np.sqrt((sx[1] - sx[0]) ** 2 + (sy[1] - sy[0]) ** 2)
            cos_theta = (sx[1] - sx[0]) / length
            sin_theta = (sy[1] - sy[0]) / length

            if sz[0] != sz[1]:
                raise Exception('Z-coordinates of the wire ends must be the same value.')

            # 計算できない送受信座標が入力された場合の処理
            delta_z = 1e-8
            if sz[0] in self.depth:
                sz = sz - delta_z
            if sz[0] == rz[0]:
                sz = sz - delta_z

            nsplit = emsrc.nsplit
            # 節点
            sx_node = np.linspace(sx[0], sx[1], nsplit + 1)
            sy_node = np.linspace(sy[0], sy[1], nsplit + 1)
            sz_node = np.linspace(sz[0], sz[1], nsplit + 1)

            sx_dipole = np.array([(sx_node[i] + sx_node[i+1]) / 2 for i in range(nsplit)])
            sy_dipole = np.array([(sy_node[i] + sy_node[i+1]) / 2 for i in range(nsplit)])
            sz_dipole = np.array([(sz_node[i] + sz_node[i+1]) / 2 for i in range(nsplit)])

            ds = length / nsplit

            def rotate_coordinate(x, y, z, cos_theta, sin_theta):
                rot_matrix = np.array([[cos_theta, sin_theta, 0], [-sin_theta, cos_theta, 0], [0, 0, 1]], dtype=object)
                rot_coord = np.dot(rot_matrix, np.array([x, y, z])).reshape(3)
                return rot_coord

            rotsc = np.ones((nsplit, 3))
            #for i in range(self.num_dipole):
            #    rotsc[i, :] = rotate_coordinate(sx_node[i], sy_node[i], sz_node[i], cos_theta, sin_theta)
            for i in range(nsplit):
                rotsc[i, :] = rotate_coordinate(sx_dipole[i], sy_dipole[i], sz_dipole[i], cos_theta, sin_theta)

            rotrc = rotate_coordinate(rx, ry, rz, cos_theta, sin_theta)

            xx = rotrc[0] - rotsc[:, 0]
            yy = rotrc[1] - rotsc[:, 1]

            rn = np.ones(nsplit)
            for i in range(nsplit):
                rn[i] = np.sqrt(xx[i] ** 2 + yy[i] ** 2)

            slayer = self.in_which_layer(sz[0])
            rlayer = self.in_which_layer(rz)

            self.sx, self.sy , self.sz = sx, sy, sz[0]
            self.rx, self.ry , self.rz = rx, ry, rz
            self.xx, self.yy , self.rn = xx, yy, rn
            self.ds = ds
            self.slayer = slayer 
            self.rlayer = rlayer
            self.cos_theta = cos_theta
            self.sin_theta = sin_theta

    #== MAIN EXECUTOR ====================================#
    def emulate(self, hankel_filter, 
            ignore_displacement_current = False, 
            time_diff=False, td_transform=None):
        """

        """
        if not bool(td_transform):
            self.domain = 'Freq'
        else:
            self.domain = 'Time'

        self.hankel_filter = hankel_filter
        self.ignore_displacement_current = ignore_displacement_current
        self.time_diff = time_diff

        # WHY?
        if hankel_filter == 'anderson801':
            delta_z = 1e-4 - 1e-8
            if self.sz in self.depth:
                self.sz -= delta_z
            if self.sz == self.rz:
                self.sz -= delta_z

        ans, freqtime = self.src.get_result(
                        self, time_diff=time_diff, td_transform=td_transform)
        
        return ans, freqtime

    #== COMPUTE COEFFICIENTS (called by kernel function) ===============================================#
    def compute_coefficients(self, omega):
        ztilde = np.ones(self.num_layer, dtype=complex)
        ytilde = np.ones(self.num_layer, dtype=complex)
        k = np.zeros(self.num_layer, dtype=complex)
        u = np.ones((self.num_layer, self.filter_length), dtype=complex)
        Y = np.ones((self.num_layer, self.filter_length), dtype=complex)
        Z = np.ones((self.num_layer, self.filter_length), dtype=complex)
        tanhuh = np.ones((self.num_layer, self.filter_length), dtype=complex)

        # COMPLEX RESISTIVITY MODEL (Pelton et al. (1978))
        if self.cxres == True:
            im = 1 - (1j * omega * self.tau) ** self.c
            res = self.res_0 * (1 - self.m * (1 - 1 / im))
            self.sigma = 1 / res
        
        # インピーダンス＆アドミタンス
        ztilde[:] = 1j * omega * self.mu[:]

        # w1dem.pyでは何か変なことになってる
        if self.ignore_displacement_current:
            ytilde[:] = self.sigma[:]
            ytilde[0] = 1e-13
            k[:] = (- 1.j * omega * self.mu[:] * self.sigma[:]) ** 0.5
            k[0] = 0 # !!!
        else:
            ytilde[:] = self.sigma[:] + 1.j * omega * self.epsln[:]
            k[:] = (omega ** 2.0 * self.mu[:] * self.epsln[:] \
                    - 1.j * omega * self.mu[:] * self.sigma[:]) ** 0.5
        
        # u = (kx^2 + ky^2 - km^2)^0.5
        for i in range(self.num_layer):
            u[i] = (self.lambda_ ** 2 - k[i] ** 2) ** 0.5

        # tanh
        for i in range(1, self.num_layer - 1):
            tanhuh[i] = np.tanh(u[i] * self.thicks[i - 1])

        for i in range(self.num_layer):
            Y[i] = u[i] / ztilde[i]
            Z[i] = u[i] / ytilde[i]

        #return to self
        self.ztilde = ztilde
        self.ytilde = ytilde
        self.k = k
        self.u = u

        #TE/TM mode 境界係数
        r_te = np.ones((self.num_layer, self.filter_length), dtype=complex)
        r_tm = np.ones((self.num_layer, self.filter_length), dtype=complex)
        R_te = np.ones((self.num_layer, self.filter_length), dtype=complex)
        R_tm = np.ones((self.num_layer, self.filter_length), dtype=complex)

        #送受信層index+1　for コード短縮
        si = self.slayer
        ri = self.rlayer

        ### DOWN ADMITTANCE & IMPEDANCE ###
        Ytilde = np.zeros((self.num_layer, self.filter_length), dtype=complex)
        Ztilde = np.zeros((self.num_layer, self.filter_length), dtype=complex)

        Ytilde[-1] = Y[-1]
        Ztilde[-1] = Z[-1]

        r_te[-1] = 0
        r_tm[-1] = 0

        for ii in range(self.num_layer - 1, si, -1):
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

        ### UP ADMITTANCE & IMPEDANCE ###
        Yhat = np.ones((self.num_layer, self.filter_length), dtype=complex)
        Zhat = np.ones((self.num_layer, self.filter_length), dtype=complex)
        
        Yhat[0] = Y[0]
        Zhat[0] = Z[0]

        R_te[0] = 0
        R_tm[0] = 0

        for ii in range(2, si):
            numerator_Y = Yhat[ii - 2] + Y[ii - 1] * tanhuh[ii - 1]
            denominator_Y = Y[ii - 1] + Yhat[ii - 2] * tanhuh[ii - 1]
            Yhat[ii - 1] = Y[ii - 1] * numerator_Y / denominator_Y  
            # (2)Yhat{2,3,\,si-2,si-1}

            numerator_Z = Zhat[ii - 2] + Z[ii - 1] * tanhuh[ii - 1]
            denominator_Z = Z[ii - 1] + Zhat[ii - 2] * tanhuh[ii - 1]
            Zhat[ii - 1] = Z[ii - 1] * numerator_Z / denominator_Z

            R_te[ii - 1] = (Y[ii - 1] - Yhat[ii - 2]) / (Y[ii - 1] + Yhat[ii - 2])
            R_tm[ii - 1] = (Z[ii - 1] - Zhat[ii - 2]) / (Z[ii - 1] + Zhat[ii - 2])
        if si != 1 :
            R_te[si - 1] = (Y[si - 1] - Yhat[si - 2]) / (Y[si - 1] + Yhat[si - 2])
            R_tm[si - 1] = (Z[si - 1] - Zhat[si - 2]) / (Z[si - 1] + Zhat[si - 2])

        U_te = np.ones((self.num_layer, self.filter_length), dtype=complex)
        U_tm = np.ones((self.num_layer, self.filter_length), dtype=complex)
        D_te = np.ones((self.num_layer, self.filter_length), dtype=complex)
        D_tm = np.ones((self.num_layer, self.filter_length), dtype=complex)

        # In the layer containing the source (slayer)
        if si == 1:
            U_te[si - 1] = 0
            U_tm[si - 1] = 0
            D_te[si - 1] = self.src.kernel_te_down_sign * r_te[si - 1] \
                            * np.exp(-u[si - 1] * (self.depth[si - 1] - self.sz))
            D_tm[si - 1] = self.src.kernel_tm_down_sign * r_tm[si - 1] \
                            * np.exp(-u[si - 1] * (self.depth[si - 1] - self.sz))
        elif si == self.num_layer:
            U_te[si - 1] = self.src.kernel_te_up_sign * R_te[si - 1] \
                            * np.exp(u[si - 1] * (self.depth[si - 2] - self.sz))
            U_tm[si - 1] = self.src.kernel_tm_up_sign * R_tm[si - 1] \
                            * np.exp(u[si - 1] * (self.depth[si - 2] - self.sz))
            D_te[si - 1] = 0
            D_tm[si - 1] = 0
        else:
            exp_term1 = np.exp(-2 * u[si - 1]
                                * (self.depth[si - 1] - self.depth[si - 2]))
            exp_term2u = np.exp( u[si - 1] * (self.depth[si - 2] - 2 * self.depth[si - 1] + self.sz))
            exp_term2d = np.exp(-u[si - 1] * (self.depth[si - 1] - 2 * self.depth[si - 2] + self.sz))
            exp_term3u = np.exp( u[si - 1] * (self.depth[si - 2] - self.sz))
            exp_term3d = np.exp(-u[si - 1] * (self.depth[si - 1] - self.sz))

            U_te[si - 1] = \
                1 / (1 - R_te[si - 1] * r_te[si - 1] * exp_term1) \
                * R_te[si - 1] \
                * (self.src.kernel_te_down_sign * r_te[si - 1] * exp_term2u \
                    + self.src.kernel_te_up_sign * exp_term3u)

            U_tm[si - 1] = \
                1 / (1 - R_tm[si - 1] * r_tm[si - 1] * exp_term1) \
                * R_tm[si - 1] \
                * (self.src.kernel_tm_down_sign  * r_tm[si - 1] * exp_term2u \
                    + self.src.kernel_tm_up_sign * exp_term3u)

            D_te[si - 1] = \
                1 / (1 - R_te[si - 1] * r_te[si - 1] * exp_term1) \
                * r_te[si - 1] \
                * (self.src.kernel_te_up_sign * R_te[si - 1] * exp_term2d \
                    + self.src.kernel_te_down_sign * exp_term3d)

            D_tm[si - 1] = \
                1 / (1 - R_tm[si - 1] * r_tm[si - 1] * exp_term1) \
                * r_tm[si - 1] \
                * (self.src.kernel_tm_up_sign * R_tm[si - 1] * exp_term2d \
                    + self.src.kernel_tm_down_sign * exp_term3d)

        # for the layers above the slayer
        if ri < si:
            if si == self.num_layer:
                exp_term = np.exp(-u[si - 1] * (self.sz - self.depth[si - 2]))

                D_te[si - 2] = \
                    (Y[si - 2] * (1 + R_te[si - 1]) + Y[si - 1] * (1 - R_te[si - 1])) \
                    / (2 * Y[si - 2]) * self.src.kernel_te_up_sign * exp_term

                D_tm[si - 2] = \
                    (Z[si - 2] * (1 + R_tm[si - 1]) + Z[si - 1] * (1 - R_tm[si - 1])) \
                    / (2 * Z[si - 2]) * self.src.kernel_tm_up_sign * exp_term

            elif si != 1 and si != self.num_layer:
                exp_term = np.exp(-u[si - 1] * (self.sz - self.depth[si - 2]))
                exp_termii = np.exp(-u[si - 1] * (self.depth[si - 1] - self.depth[si - 2]))
                
                D_te[si - 2] = \
                    (Y[si - 2] * (1 + R_te[si - 1]) + Y[si - 1] * (1 - R_te[si - 1])) \
                    / (2 * Y[si - 2]) * (D_te[si - 1] * exp_termii + self.src.kernel_te_up_sign * exp_term)

                D_tm[si - 2] = \
                    (Z[si - 2] * (1 + R_tm[si - 1]) + Z[si - 1] * (1 - R_tm[si - 1])) \
                    / (2 * Z[si - 2]) * (D_tm[si - 1]  * exp_termii + self.src.kernel_tm_up_sign * exp_term)

            for jj in range(si - 2, 0, -1):
                exp_termjj = np.exp(-u[jj] \
                                    * (self.depth[jj] - self.depth[jj - 1]))
                D_te[jj - 1] = \
                    (Y[jj - 1] * (1 + R_te[jj]) + Y[jj] * (1 - R_te[jj])) \
                    / (2 * Y[jj - 1]) * D_te[jj] * exp_termjj
                D_tm[jj - 1] = \
                    (Z[jj - 1] * (1 + R_tm[jj]) + Z[jj] * (1 - R_tm[jj])) \
                    / (2 * Z[jj - 1]) * D_tm[jj] * exp_termjj

            for jj in range(si - 1, 1, -1):
                exp_termjj = np.exp(u[jj - 1] * (self.depth[jj - 2] - self.depth[jj - 1]))
                U_te[jj - 1] = D_te[jj - 1] * exp_termjj * R_te[jj - 1]
                U_tm[jj - 1] = D_tm[jj - 1] * exp_termjj * R_tm[jj - 1]
            U_te[0] = 0
            U_tm[0] = 0

        # for the layers below the slayer
        if ri > si:
            if si == 1:
                exp_term = np.exp(-u[si - 1] * (self.depth[si - 1] - self.sz))
                U_te[si] = (Y[si] * (1 + r_te[si - 1]) \
                                + Y[si - 1] * (1 - r_te[si - 1])) \
                            / (2 * Y[si]) \
                            * self.src.kernel_te_down_sign * exp_term
                U_tm[si] = (Z[si] * (1 + r_tm[si - 1]) \
                                + Z[si - 1] * (1 - r_tm[si - 1])) \
                            / (2 * Z[si]) \
                            * self.src.kernel_tm_down_sign * exp_term

            elif si != 1 and si != self.num_layer:
                exp_termi = np.exp(-u[si - 1] \
                                * (self.depth[si - 1] - self.depth[si - 2]))
                exp_termii = np.exp(-u[si - 1] 
                                * (self.depth[si - 1] - self.sz))
                U_te[si] = (Y[si] * (1 + r_te[si - 1]) \
                                    + Y[si - 1] * (1 - r_te[si - 1])) \
                                / (2 * Y[si]) \
                                * (U_te[si - 1] * exp_termi \
                                    + self.src.kernel_te_down_sign * exp_termii)
                U_tm[si] = (Z[si] * (1 + r_tm[si - 1]) + Z[si - 1] \
                                    * (1 - r_tm[si - 1])) \
                                / (2 * Z[si]) \
                                * (U_tm[si - 1] * exp_termi \
                                    + self.src.kernel_tm_down_sign * exp_termii)

            for jj in range(si + 2, self.num_layer + 1):
                exp_term = np.exp(-u[jj - 2] \
                                * (self.depth[jj - 2] - self.depth[jj - 3]))
                U_te[jj - 1] = (Y[jj - 1] * (1 + r_te[jj - 2]) \
                                    + Y[jj - 2] * (1 - r_te[jj - 2])) \
                                / (2 * Y[jj - 1]) * U_te[jj - 2] * exp_term
                U_tm[jj - 1] = (Z[jj - 1] * (1 + r_tm[jj - 2]) \
                                    + Z[jj - 2] * (1 - r_tm[jj - 2])) \
                                / (2 * Z[jj - 1]) * U_tm[jj - 2] * exp_term
                                
            for jj in range(si + 1, self.num_layer):
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
            e_up = np.zeros(self.filter_length, dtype=np.complex)
            e_down = np.exp(u[ri - 1] * (self.rz - self.depth[ri - 1]))
        elif ri == self.num_layer:
            e_up = np.exp(-u[ri - 1] * (self.rz - self.depth[ri - 2]))
            e_down = np.zeros(self.filter_length, dtype=np.complex)
        else:
            e_up = np.exp(-u[ri - 1] * (self.rz - self.depth[ri - 2]))
            e_down = np.exp(u[ri - 1] * (self.rz - self.depth[ri - 1]))

        #self.r_te = r_te
        #self.r_tm = r_tm
        #self.R_te = R_te
        #self.R_tm = R_tm
        #self.U_te = U_te
        #self.U_tm = U_tm
        #self.D_te = D_te
        #self.D_tm = D_tm
        #self.e_up = e_up
        #self.e_down = e_down
        return U_te, U_tm, D_te, D_tm, e_up, e_down

    def in_which_layer(self, z):
        """

        """
        layer_id = 1
        for i in range(self.num_layer-1, 0, -1):
            if z > self.depth[i-1]:
                layer_id = i + 1
                break
            else:
                continue
        return layer_id
