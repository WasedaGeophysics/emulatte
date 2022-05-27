# Copyright 2022 Waseda Geophysics Laboratory
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

import numpy as np
from numpy.typing import NDArray
from ..utils.interpret import check_waveform
from ..utils.emu_object import Source
from .kernel.edipole import (
    compute_kernel_ved_e_phi,
    compute_kernel_ved_e_z,
    compute_kernel_ved_h_r,
    compute_kernel_hed_e_r_te,
    compute_kernel_hed_e_r_tm,
    compute_kernel_hed_e_z_tm,
    compute_kernel_hed_h_r_te,
    compute_kernel_hed_h_r_tm,
    compute_kernel_hed_h_z_te,
)
from .kernel.mdipole import (
    compute_kernel_vmd_e_phi,
    compute_kernel_vmd_h_r,
    compute_kernel_vmd_h_z,
    compute_kernel_hmd_e_r_te,
    compute_kernel_hmd_e_r_tm,
    compute_kernel_hmd_e_z_tm,
    compute_kernel_hmd_h_r_te,
    compute_kernel_hmd_h_r_tm,
    compute_kernel_hmd_h_z_te
)


class VMD(Source):
    def __init__(self, moment, ontime = None, frequency=None) -> None:
        moment = np.array(moment, ndmin=1, dtype=complex)
        self.moment = moment

        self.kernel_te_up_sign = 1
        self.kernel_te_down_sign = 1
        self.kernel_tm_up_sign = 0
        self.kernel_tm_down_sign = 0

        magnitude = moment
        
        domain, signal, magnitude, ontime, frequency = \
            check_waveform(magnitude, ontime, frequency)

        self.domain = domain
        self.signal = signal
        self.ontime = ontime
        self.frequency = frequency
        if domain == "frequency":
            self.magnitude_f = magnitude
        else:
            self.magnitude_f = 1
            self.magnitude_t = magnitude

    def _compute_hankel_transform_dlf(
            self, model, direction, magnetic : bool
            ) -> NDArray:
        # kernel components
        model._compute_kernel_components()
        si = model.si
        zs = model.zs
        ri = model.ri
        z = model.z

        rho = model.rho
        lambda_ = model.lambda_
        bessel_j0 = model.bessel_j0
        bessel_j1 = model.bessel_j1

        us = model.u[:,si]
        ur = model.u[:,ri]

        impedivity_s = model.impedivity[:, si]
        impedivity_r = model.impedivity[:, ri]
        
        u_te = model.u_te
        d_te = model.d_te
        e_up = model.e_up
        e_down = model.e_down

        sin_phi = model.sin_phi
        cos_phi = model.cos_phi

        nfreq = model.nfreq

        ans = []

        # Electric field E
        if not magnetic:
            kernel_e_phi = None
            if "x" in direction:
                factor = impedivity_s / (4 * np.pi) * sin_phi
                kernel_e_phi = compute_kernel_vmd_e_phi(
                    u_te, d_te, e_up, e_down, si, ri, us, zs, z, lambda_)
                e_x = factor * np.dot(kernel_e_phi, bessel_j1) / rho
                ans.append(e_x)
            if "y" in direction:
                if kernel_e_phi is None:
                    factor = impedivity_s / (4 * np.pi) * -cos_phi
                    kernel_e_phi = compute_kernel_vmd_e_phi(
                        u_te, d_te, e_up, e_down, si, ri, us, zs, z, lambda_)
                    e_y = factor * np.dot(kernel_e_phi, bessel_j1) / rho
                else:
                    # e_xの結果を再利用
                    e_y = e_x * -cos_phi / sin_phi
                ans.append(e_y)
            if "z" in direction:
                e_z = np.zeros(nfreq)
                ans.append(e_z)
        # Magnetic field H
        else:
            kernel_h_r = None
            if "x" in direction:
                factor = cos_phi / (4 * np.pi) * impedivity_s / impedivity_r
                kernel_h_r = compute_kernel_vmd_h_r(
                                        u_te, d_te, e_up, e_down,
                                        si, ri, us, ur, zs, z, lambda_)
                h_x = factor * np.dot(kernel_h_r, bessel_j1) / rho
                ans.append(h_x)
            if "y" in direction:
                if kernel_h_r is None:
                    factor = sin_phi / (4 * np.pi) * impedivity_s / impedivity_r
                    kernel_h_r = compute_kernel_vmd_h_r(
                                        u_te, d_te, e_up, e_down, 
                                        si, ri, us, ur, zs, z, lambda_)
                    h_y = factor * np.dot(kernel_h_r, bessel_j1) / rho
                else:
                    h_y = h_x * sin_phi / cos_phi
                ans.append(h_y)
            if "z" in direction:
                factor = 1 / (4 * np.pi) * impedivity_s / impedivity_r
                kernel_h_z = compute_kernel_vmd_h_z(
                                        u_te, d_te, e_up, e_down, 
                                        si, ri, us, zs, z, lambda_)
                h_z = factor * np.dot(kernel_h_z, bessel_j0) / rho
                ans.append(h_z)

        ans = np.array(ans)

        return ans

class VED(Source):
    def __init__(self, current, length, ontime = None, frequency=None) -> None:
        current = np.array(current, ndmin=1, dtype=complex)
        self.current = current
        self.length = length
        self.kernel_te_up_sign = 0
        self.kernel_te_down_sign = 0
        self.kernel_tm_up_sign = 1
        self.kernel_tm_down_sign = 1

        magnitude = current
        
        domain, signal, magnitude, ontime, frequency = \
            check_waveform(magnitude, ontime, frequency)

        self.domain = domain
        self.signal = signal
        self.ontime = ontime
        self.frequency = frequency
        if domain == "frequency":
            self.magnitude_f = magnitude
        else:
            self.magnitude_f = 1
            self.magnitude_t = magnitude

    def _compute_hankel_transform_dlf(
            self, model, direction, magnetic : bool
            ) -> NDArray:
        # kernel components
        model._compute_kernel_components()
        si = model.si
        zs = model.zs
        ri = model.ri
        z = model.z

        rho = model.rho
        lambda_ = model.lambda_
        bessel_j0 = model.bessel_j0
        bessel_j1 = model.bessel_j1

        us = model.u[:,si]
        ur = model.u[:,ri]

        admittivity_r = model.admittivity[:, ri]
        
        u_tm = model.u_tm
        d_tm = model.d_tm
        e_up = model.e_up
        e_down = model.e_down

        sin_phi = model.sin_phi
        cos_phi = model.cos_phi

        nfreq = model.nfreq

        ans = []

        # Electric field E
        if not magnetic:
            kernel_e_phi = None
            if "x" in direction:
                factor = -cos_phi / (4 * np.pi * admittivity_r)
                kernel_e_phi = compute_kernel_ved_e_phi(
                    u_tm, d_tm, e_up, e_down, si, ri, us, ur, zs, z, lambda_)
                e_x = factor * np.dot(kernel_e_phi, bessel_j1) / rho
                ans.append(e_x)
            if "y" in direction:
                if kernel_e_phi is None:
                    factor = -sin_phi / (4 * np.pi * admittivity_r)
                    kernel_e_phi = compute_kernel_ved_e_phi(
                        u_tm, d_tm, e_up, e_down, si, ri, us, ur, zs, z, lambda_)
                    e_y = factor * np.dot(kernel_e_phi, bessel_j1) / rho
                else:
                    # e_xの結果を再利用
                    e_y = e_x * sin_phi / cos_phi
                ans.append(e_y)
            if "z" in direction:
                factor = 1 / (4 * np.pi) / admittivity_r
                kernel_e_z = compute_kernel_ved_e_z(
                                        u_tm, d_tm, e_up, e_down, 
                                        si, ri, us, ur, zs, z, lambda_)
                e_z = factor * np.dot(kernel_e_z, bessel_j0) / rho
                ans.append(e_z)

        # Magnetic field H
        else:
            kernel_h_r = None
            if "x" in direction:
                factor = -sin_phi / (4 * np.pi)
                kernel_h_r = compute_kernel_ved_h_r(
                    u_tm, d_tm, e_up, e_down, si, ri, us, ur, zs, z, lambda_)
                h_x = factor * np.dot(kernel_h_r, bessel_j1) / rho
                ans.append(h_x)
            if "y" in direction:
                if kernel_h_r is None:
                    factor = cos_phi / (4 * np.pi)
                    kernel_h_r = compute_kernel_ved_h_r(
                        u_tm, d_tm, e_up, e_down, si, ri, us, ur, zs, z, lambda_)
                    h_y = factor * np.dot(kernel_h_r, bessel_j1) / rho
                else:
                    h_y = h_x * cos_phi / -sin_phi
                ans.append(h_y)
            if "z" in direction:
                h_z = np.zeros(nfreq)
                ans.append(h_z)
        ans = np.array(ans)
        ans = ans * self.length
        return ans

class HMD(Source):
    def __init__(self, moment, azimuth, ontime = None, frequency=None) -> None:
        
        moment = np.array(moment, ndmin=1, dtype=complex)
        self.moment = moment
        self.azimuth = azimuth # degree
        self.kernel_te_up_sign = -1
        self.kernel_te_down_sign = 1
        self.kernel_tm_up_sign = 1
        self.kernel_tm_down_sign = 1

        azimuth_rad = azimuth / 180 * np.pi

        magnitude = moment
        
        domain, signal, magnitude, ontime, frequency = \
            check_waveform(magnitude, ontime, frequency)

        self.azmrad = azimuth_rad
        self.domain = domain
        self.signal = signal
        self.ontime = ontime
        self.frequency = frequency
        if domain == "frequency":
            self.magnitude_f = magnitude
        else:
            self.magnitude_f = 1
            self.magnitude_t = magnitude

    def _compute_hankel_transform_dlf(
            self, model, direction, magnetic : bool
            ) -> NDArray:
        # kernel components
        model._compute_kernel_components()
        si = model.si
        zs = model.zs
        ri = model.ri
        z = model.z

        rho = model.rho
        lambda_ = model.lambda_
        bessel_j0 = model.bessel_j0
        bessel_j1 = model.bessel_j1

        us = model.u[:,si]
        ur = model.u[:,ri]

        impz_s = model.impedivity[:, si]
        impz_r = model.impedivity[:, ri]
        admy_r = model.admittivity[:, ri]

        ks = model.k[:, si]
        
        u_te = model.u_te
        d_te = model.d_te
        u_tm = model.u_tm
        d_tm = model.d_tm
        e_up = model.e_up
        e_down = model.e_down

        sin_phi = model.sin_phi
        cos_phi = model.cos_phi

        ans = []

        # Electric field E
        if not magnetic:
            if ("x" in direction) or ("y" in direction):
                # e_x
                extm0 = ks ** 2 * cos_phi * sin_phi / (4 * np.pi * admy_r)
                extm1 = - extm0 * 2 / rho
                exte0 = - impz_s * cos_phi * sin_phi / (4 * np.pi)
                exte1 = - exte0 * 2 / rho

                kernel_e_r_te = compute_kernel_hmd_e_r_te(
                    u_te, d_te, e_up, e_down, si, ri, us, ur, zs, z, lambda_
                )
                kernel_e_r_tm = compute_kernel_hmd_e_r_tm(
                    u_tm, d_tm, e_up, e_down, si, ri, us, ur, zs, z, lambda_
                )
                e_x = extm0 * np.dot(kernel_e_r_tm[0], bessel_j0) \
                        + extm1 * np.dot(kernel_e_r_tm[1], bessel_j1) \
                        + exte0 * np.dot(kernel_e_r_te[0], bessel_j0) \
                        + exte1 * np.dot(kernel_e_r_te[1], bessel_j1)
                e_x = e_x / rho

                # e_y
                cos_2phi = (2 * cos_phi ** 2 - 1)
                eytm0 = ks ** 2 * sin_phi ** 2 / (4 * np.pi * admy_r)
                eytm1 = - ks ** 2 * -cos_2phi / (4 * np.pi * admy_r * rho)
                eyte0 = impz_s * cos_phi ** 2 / (4 * np.pi)
                eyte1 = - impz_s * cos_2phi / (4 * np.pi * rho)

                e_y = eytm0 * np.dot(kernel_e_r_tm[0], bessel_j0) \
                        + eytm1 * np.dot(kernel_e_r_tm[1], bessel_j1) \
                        + eyte0 * np.dot(kernel_e_r_te[0], bessel_j0) \
                        + eyte1 * np.dot(kernel_e_r_te[1], bessel_j1)
                e_y = e_y / rho

                e_xy = np.array([e_x, e_y])
                psi = self.azmrad
                rot_mat = np.array([
                        [np.cos(psi), -np.sin(psi)],
                        [np.sin(psi), np.cos(psi)]
                    ])

                e_xy_original = rot_mat @ e_xy
                if "x" in direction:
                    ans.append(e_xy_original[0])
                if "y" in direction:
                    ans.append(e_xy_original[1])

            if "z" in direction:
                factor = - ks ** 2 * sin_phi / (4 * np.pi * admy_r)
                kernel_e_z = compute_kernel_hmd_e_z_tm(
                    u_tm, d_tm, e_up, e_down, si, ri, us, ur, zs, z, lambda_
                )
                e_z = factor * np.dot(kernel_e_z, bessel_j1) / rho
                ans.append(e_z)

        # Magnetic field H
        else:
            kernel_exist = False
            if ("x" in direction) or ("y" in direction):
                kernel_h_r_te = compute_kernel_hmd_h_r_te(
                    u_te, d_te, e_up, e_down, si, ri, us, ur, zs, z, lambda_
                )
                kernel_h_r_tm = compute_kernel_hmd_h_r_tm(
                    u_tm, d_tm, e_up, e_down, si, ri, us, ur, zs, z, lambda_
                )
                
                # h_x
                cos_2phi = (2 * cos_phi ** 2 - 1)
                hxtm0 = ks ** 2 * sin_phi ** 2 / (4 * np.pi)
                hxtm1 = - ks ** 2 * -cos_2phi / (4 * np.pi * rho)
                hxte0 = impz_s / impz_r * cos_phi ** 2 / (4 * np.pi)
                hxte1 = - impz_s / impz_r * cos_2phi / (4 * np.pi * rho)

                h_x = hxtm0 * np.dot(kernel_h_r_tm[0], bessel_j0) \
                        + hxtm1 * np.dot(kernel_h_r_tm[1], bessel_j1) \
                        + hxte0 * np.dot(kernel_h_r_te[0], bessel_j0) \
                        + hxte1 * np.dot(kernel_h_r_te[1], bessel_j1)
                h_x = h_x / rho

                # h_y
                hytm0 = - ks ** 2 * cos_phi * sin_phi / (4 * np.pi)
                hytm1 = - hytm0 * 2 / rho
                hyte0 = impz_s / impz_r * cos_phi * sin_phi / (4 * np.pi)
                hyte1 = - hyte0 * 2 / rho

                h_y = hytm0 * np.dot(kernel_h_r_tm[0], bessel_j0) \
                        + hytm1 * np.dot(kernel_h_r_tm[1], bessel_j1) \
                        + hyte0 * np.dot(kernel_h_r_te[0], bessel_j0) \
                        + hyte1 * np.dot(kernel_h_r_te[1], bessel_j1)
                h_y = h_y / rho

                h_xy = np.array([h_x, h_y])
                psi = self.azmrad
                rot_mat = np.array([
                        [np.cos(psi), -np.sin(psi)],
                        [np.sin(psi), np.cos(psi)]
                    ])
                h_xy_original = rot_mat @ h_xy
                if "x" in direction:
                    ans.append(h_xy_original[0])
                if "y" in direction:
                    ans.append(h_xy_original[1])

            if "z" in direction:
                factor = impz_s / impz_r * cos_phi / (4 * np.pi)
                kernel_h_z = compute_kernel_hmd_h_z_te(
                    u_te, d_te, e_up, e_down, si, ri, us, ur, zs, z, lambda_
                )
                h_z = factor * np.dot(kernel_h_z, bessel_j1) / rho
                ans.append(h_z)
        # Magnetic field H
        ans = np.array(ans)
        return ans

class HED(Source):
    def __init__(self, current, length, azimuth, ontime = None, frequency=None
            ) -> None:
        current = np.array(current, ndmin=1, dtype=complex)
        self.current = current
        self.length = length
        self.azimuth = azimuth # degree
        self.kernel_te_up_sign = 1
        self.kernel_te_down_sign = 1
        self.kernel_tm_up_sign = -1
        self.kernel_tm_down_sign = 1

        azimuth_rad = azimuth / 180 * np.pi

        magnitude = current
        
        domain, signal, magnitude, ontime, frequency = \
            check_waveform(magnitude, ontime, frequency)

        self.azmrad = azimuth_rad
        self.domain = domain
        self.signal = signal
        self.ontime = ontime
        self.frequency = frequency
        if domain == "frequency":
            self.magnitude_f = magnitude
        else:
            self.magnitude_f = 1
            self.magnitude_t = magnitude

    def _compute_hankel_transform_dlf(
            self, model, direction, magnetic : bool
            ) -> NDArray:
        # kernel components
        model._compute_kernel_components()
        si = model.si
        zs = model.zs
        ri = model.ri
        z = model.z

        rho = model.rho
        lambda_ = model.lambda_
        bessel_j0 = model.bessel_j0
        bessel_j1 = model.bessel_j1

        us = model.u[:,si]
        ur = model.u[:,ri]

        impz_s = model.impedivity[:, si]
        impz_r = model.impedivity[:, ri]
        admy_r = model.admittivity[:, ri]
        
        u_te = model.u_te
        d_te = model.d_te
        u_tm = model.u_tm
        d_tm = model.d_tm
        e_up = model.e_up
        e_down = model.e_down

        sin_phi = model.sin_phi
        cos_phi = model.cos_phi

        ans = []

        # Electric field E
        if not magnetic:
            if ("x" in direction) or ("y" in direction):

                kernel_e_r_te = compute_kernel_hed_e_r_te(
                    u_te, d_te, e_up, e_down, si, ri, us, ur, zs, z, lambda_
                )
                kernel_e_r_tm = compute_kernel_hed_e_r_tm(
                    u_tm, d_tm, e_up, e_down, si, ri, us, ur, zs, z, lambda_
                )

                # e_x
                cos_2phi = (2 * cos_phi ** 2 - 1)
                extm0 = cos_phi ** 2 / (4 * np.pi * admy_r)
                extm1 = - cos_2phi / (4 * np.pi * admy_r * rho)
                exte0 = impz_s * cos_phi ** 2 / (4 * np.pi)
                exte1 = - impz_s * cos_2phi / (4 * np.pi * rho)
                exte_line = - impz_s / (4 * np.pi)

                e_x = extm0 * np.dot(kernel_e_r_tm[0], bessel_j0) \
                        + extm1 * np.dot(kernel_e_r_tm[1], bessel_j1) \
                        + exte0 * np.dot(kernel_e_r_te[0], bessel_j0) \
                        + exte1 * np.dot(kernel_e_r_te[1], bessel_j1) \
                        + exte_line * np.dot(kernel_e_r_te[0], bessel_j0)
                e_x = e_x / rho

                # e_y
                eytm0 = sin_phi * cos_phi / (4 * np.pi * admy_r)
                eytm1 = - eytm0 * 2 / rho
                eyte0 = impz_s * cos_phi * sin_phi / (4 * np.pi)
                eyte1 = - eyte0 * 2 / rho

                e_y = eytm0 * np.dot(kernel_e_r_tm[0], bessel_j0) \
                        + eytm1 * np.dot(kernel_e_r_tm[1], bessel_j1) \
                        + eyte0 * np.dot(kernel_e_r_te[0], bessel_j0) \
                        + eyte1 * np.dot(kernel_e_r_te[1], bessel_j1)

                e_y = e_y / rho

                e_xy = np.array([e_x, e_y])
                psi = self.azmrad
                rot_mat = np.array([
                        [np.cos(psi), -np.sin(psi)],
                        [np.sin(psi), np.cos(psi)]
                    ])
                e_xy_original = rot_mat @ e_xy
                if "x" in direction:
                    ans.append(e_xy_original[0])
                if "y" in direction:
                    ans.append(e_xy_original[1])

            if "z" in direction:
                factor = cos_phi / (4 * np.pi * admy_r)
                kernel_e_z = compute_kernel_hed_e_z_tm(
                    u_tm, d_tm, e_up, e_down, si, ri, us, ur, zs, z, lambda_
                )
                e_z = factor * np.dot(kernel_e_z, bessel_j1) / rho
                ans.append(e_z)

        # Magnetic field H
        else:
            kernel_exist = False
            if ("x" in direction) or ("y" in direction):
                kernel_h_r_te = compute_kernel_hed_h_r_te(
                    u_te, d_te, e_up, e_down, si, ri, us, ur, zs, z, lambda_
                )
                kernel_h_r_tm = compute_kernel_hed_h_r_tm(
                    u_tm, d_tm, e_up, e_down, si, ri, us, ur, zs, z, lambda_
                )
                
                # h_x

                hxtm0 = cos_phi * sin_phi / (4 * np.pi)
                hxtm1 = - hxtm0 * 2 / rho
                hxte0 = impz_s / impz_r * cos_phi * sin_phi / (4 * np.pi)
                hxte1 = - hxte0 * 2 / rho

                h_x = hxtm0 * np.dot(kernel_h_r_tm[0], bessel_j0) \
                        + hxtm1 * np.dot(kernel_h_r_tm[1], bessel_j1) \
                        + hxte0 * np.dot(kernel_h_r_te[0], bessel_j0) \
                        + hxte1 * np.dot(kernel_h_r_te[1], bessel_j1)
                h_x = h_x / rho

                # h_y
                cos_2phi = (2 * cos_phi ** 2 - 1)
                hytm0 = - cos_phi ** 2 / (4 * np.pi)
                hytm1 = cos_2phi / (4 * np.pi * rho)
                hyte0 = - impz_s / impz_r * cos_phi ** 2 / (4 * np.pi)
                hyte1 = impz_s / impz_r * cos_2phi / (4 * np.pi * rho)
                hyte_line = impz_s / impz_r / (4 * np.pi)

                h_y = hytm0 * np.dot(kernel_h_r_tm[0], bessel_j0) \
                        + hytm1 * np.dot(kernel_h_r_tm[1], bessel_j1) \
                        + hyte0 * np.dot(kernel_h_r_te[0], bessel_j0) \
                        + hyte1 * np.dot(kernel_h_r_te[1], bessel_j1) \
                        + hyte_line * np.dot(kernel_h_r_te[0], bessel_j0)
                h_y = h_y / rho

                h_xy = np.array([h_x, h_y])
                psi = self.azmrad
                rot_mat = np.array([
                        [np.cos(psi), -np.sin(psi)],
                        [np.sin(psi), np.cos(psi)]
                    ])
                h_xy_original = rot_mat @ h_xy
                if "x" in direction:
                    ans.append(h_xy_original[0])
                if "y" in direction:
                    ans.append(h_xy_original[1])

            if "z" in direction:
                factor = impz_s / impz_r * sin_phi / (4 * np.pi)
                kernel_h_z = compute_kernel_hed_h_z_te(
                    u_te, d_te, e_up, e_down, si, ri, us, ur, zs, z, lambda_
                )
                h_z = factor * np.dot(kernel_h_z, bessel_j1) / rho
                ans.append(h_z)
        # Magnetic field H
        ans = np.array(ans)
        ans = ans * self.length
        return ans

class AMD(Source):
    def __init__(self, moment, dip, azimuth, ontime = None, frequency=None) -> None:
        moment = np.array(moment, ndmin=1, dtype=complex)
        self.moment = moment

        self.dip = dip
        self.azimuth = azimuth

        self.diprad = dip / 180 * np.pi
        self.azmrad = azimuth / 180 * np.pi

        magnitude = moment
        
        domain, signal, magnitude, ontime, frequency = \
            check_waveform(magnitude, ontime, frequency)

        self.domain = domain
        self.signal = signal
        self.ontime = ontime
        self.frequency = frequency
        if domain == "frequency":
            self.magnitude_f = magnitude
        else:
            self.magnitude_f = 1
            self.magnitude_t = magnitude

    def _compute_hankel_transform_dlf(
            self, model, direction, magnetic : bool
            ) -> NDArray:

        # VMD
        self.kernel_te_up_sign = 1
        self.kernel_te_down_sign = 1
        self.kernel_tm_up_sign = 0
        self.kernel_tm_down_sign = 0
        model.cos_phi, model.sin_phi = model.cos_phi_v, model.sin_phi_v
        model._compute_kernel_components()
        vmd_field = VMD(1)._compute_hankel_transform_dlf(
                            model, direction, magnetic)
        # HMD
        self.kernel_te_up_sign = -1
        self.kernel_te_down_sign = 1
        self.kernel_tm_up_sign = 1
        self.kernel_tm_down_sign = 1
        model.cos_phi, model.sin_phi = model.cos_phi_h, model.sin_phi_h
        model._compute_kernel_components()
        hmd_field = HMD(1, self.azimuth)._compute_hankel_transform_dlf(
                            model, direction, magnetic)
        
        sin_dip = np.sin(self.diprad)
        cos_dip = np.cos(self.diprad)

        ans = vmd_field * sin_dip + hmd_field * cos_dip
        return ans

class AED(Source):
    def __init__(self, current, length, dip, azimuth, ontime = None, frequency=None) -> None:

        current = np.array(current, ndmin=1, dtype=complex)
        self.current = current
        self.length = length

        self.dip = dip
        self.azimuth = azimuth

        self.diprad = dip / 180 * np.pi
        self.azmrad = azimuth / 180 * np.pi

        azimuth_rad = azimuth / 180 * np.pi

        magnitude = current
        
        domain, signal, magnitude, ontime, frequency = \
            check_waveform(magnitude, ontime, frequency)

        self.azmrad = azimuth_rad
        self.domain = domain
        self.signal = signal
        self.ontime = ontime
        self.frequency = frequency
        if domain == "frequency":
            self.magnitude_f = magnitude
        else:
            self.magnitude_f = 1
            self.magnitude_t = magnitude

    def _compute_hankel_transform_dlf(
            self, model, direction, magnetic : bool
            ) -> NDArray:

        # VED
        self.kernel_te_up_sign = 0
        self.kernel_te_down_sign = 0
        self.kernel_tm_up_sign = 1
        self.kernel_tm_down_sign = 1
        model.cos_phi, model.sin_phi = model.cos_phi_v, model.sin_phi_v
        model._compute_kernel_components()
        vmd_field = VED(1, self.length)._compute_hankel_transform_dlf(
                            model, direction, magnetic)

        # HED
        self.kernel_te_up_sign = 1
        self.kernel_te_down_sign = 1
        self.kernel_tm_up_sign = -1
        self.kernel_tm_down_sign = 1
        model.cos_phi, model.sin_phi = model.cos_phi_h, model.sin_phi_h
        model._compute_kernel_components()
        hmd_field = \
            HED(1, self.length, self.azimuth)._compute_hankel_transform_dlf(
                            model, direction, magnetic)
        
        sin_dip = np.sin(self.diprad)
        cos_dip = np.cos(self.diprad)

        ans = vmd_field * sin_dip + hmd_field * cos_dip
        return ans