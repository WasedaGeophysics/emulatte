import numpy as np
from numpy.typing import NDArray
from ..utils.converter import check_waveform
from ..utils.emu_object import Source
from .kernel.edipole import (
    compute_kernel_hed_e_r_te,
    compute_kernel_hed_e_r_tm,
    compute_kernel_hed_e_z_tm,
    compute_kernel_hed_h_r_te,
    compute_kernel_hed_h_r_tm,
    compute_kernel_hed_h_z_te
)

class GroundedWire(Source):
    def __init__(self, current, split = None, ontime = None, frequency=None
            ) -> None:
        current = np.array(current, ndmin=1, dtype=complex)
        self.current = current
        self.split = split
        self.kernel_te_up_sign = 1
        self.kernel_te_down_sign = 1
        self.kernel_tm_up_sign = -1
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
            self, model, direction, bessel_j0, bessel_j1, magnetic : bool
            ) -> NDArray:
        # kernel components
        si = model.si[0]
        zs = model.zs[0]
        ri = model.ri
        z = model.z

        rho = model.rho
        lambda_ = model.lambda_

        us = model.u[0,:,si]
        ur = model.u[0,:,ri]

        impz_s = model.impedivity[:, si]
        impz_r = model.impedivity[:, ri]
        admy_s = model.admittivity[:, si]
        admy_r = model.admittivity[:, ri]
        
        u_te = model.u_te
        d_te = model.d_te
        u_tm = model.u_tm
        d_tm = model.d_tm
        e_up = model.e_up
        e_down = model.e_down

        cos_theta = model.cos_theta
        sin_theta = model.sin_theta

        split = model.split
        ds = model.ds

        xx = model.xx
        yy = model.yy

        jj = model.size4d[0]
        kk = model.size4d[1]
        mm = model.size4d[3]

        ans = []

        lambda3d = np.zeros((kk, jj, mm), dtype=complex)
        lambda3d[:] = lambda_
        lambda3d = lambda3d.transpose((1,0,2))

        # Electric field E
        if not magnetic:
            if ("x" in direction) or ("y" in direction):
                kernel_e_r_te = compute_kernel_hed_e_r_te(
                    u_te, d_te, e_up, e_down, si, ri, us, ur, zs, z, lambda3d
                )
                kernel_e_r_tm = compute_kernel_hed_e_r_tm(
                    u_tm, d_tm, e_up, e_down, si, ri, us, ur, zs, z, lambda3d
                )

                cos_1 = xx[0] / rho[0]
                sin_1 = yy[0] / rho[0]
                cos_2 = xx[-1] / rho[-1]
                sin_2 = yy[-1] / rho[-1]

                # line
                te_ex_line = (kernel_e_r_te[0] @ bessel_j0).T / rho
                te_ex_line = te_ex_line @ np.ones(split)
                    
                # begin
                tm_er_g_1 = (kernel_e_r_tm[1][0] @ bessel_j1) / rho[0]
                te_er_g_1 = (kernel_e_r_te[1][0] @ bessel_j1) / rho[0]

                # end
                tm_er_g_2 = (kernel_e_r_tm[1][-1] @ bessel_j1) / rho[-1]
                te_er_g_2 = (kernel_e_r_te[1][-1] @ bessel_j1) / rho[-1]

                amp_tm_ex_1 =  cos_1 / (4 * np.pi * admy_r)
                amp_tm_ex_2 = -cos_2 / (4 * np.pi * admy_r)
                amp_te_ex_1 =  cos_1 * impz_s / (4 * np.pi)
                amp_te_ex_2 = -cos_2 * impz_s / (4 * np.pi)
                amp_ex_line  = -impz_s / (4 * np.pi)

                e_x_rot = amp_tm_ex_1 * tm_er_g_1 \
                        + amp_tm_ex_2 * tm_er_g_2 \
                        + amp_te_ex_1 * te_er_g_1 \
                        + amp_te_ex_2 * te_er_g_2 \
                        + amp_ex_line * ds * te_ex_line

                amp_tm_ey_1 =  sin_1 / (4 * np.pi * admy_r)
                amp_tm_ey_2 = -sin_2 / (4 * np.pi * admy_r)
                amp_te_ey_1 =  sin_1 * impz_s / (4 * np.pi)
                amp_te_ey_2 = -sin_2 * impz_s / (4 * np.pi)
                
                e_y_rot = amp_tm_ey_1 * tm_er_g_1 \
                        + amp_tm_ey_2 * tm_er_g_2 \
                        + amp_te_ey_1 * te_er_g_1 \
                        + amp_te_ey_2 * te_er_g_2

                e_x_rot = e_x_rot
                e_y_rot = e_y_rot

                if "x" in direction:
                    e_x = cos_theta * e_x_rot - sin_theta * e_y_rot
                    ans.append(e_x)
                if "y" in direction:
                    e_y = sin_theta * e_x_rot + cos_theta * e_y_rot
                    ans.append(e_y)
            if "z" in direction:
                kernel_e_z_tm = compute_kernel_hed_e_z_tm(
                    u_te, d_te, e_up, e_down, si, ri, us, ur, zs, z, lambda3d)
                tm_ez_1 = (kernel_e_z_tm[0][0] @ bessel_j0) / rho[0]
                tm_ez_2 = (kernel_e_z_tm[0][-1] @ bessel_j0) / rho[-1]

                amp_tm_ez_1 = 1 / (4 * np.pi * admy_r)
                amp_tm_ez_2 = -1 / (4 * np.pi * admy_r)

                e_z = amp_tm_ez_1 * tm_ez_1 + amp_tm_ez_2 * tm_ez_2
                ans.append(e_z)

            ans = np.array(ans).astype(complex)

        # Magnetic field H
        else:
            kernel_exist = False
            if ("x" in direction) or ("y" in direction):
                kernel_h_r_te = compute_kernel_hed_h_r_te(
                    u_te, d_te, e_up, e_down, si, ri, us, ur, zs, z, lambda3d
                )
                kernel_h_r_tm = compute_kernel_hed_h_r_tm(
                    u_tm, d_tm, e_up, e_down, si, ri, us, ur, zs, z, lambda3d
                )
                
                cos_1 = xx[0] / rho[0]
                sin_1 = yy[0] / rho[0]
                cos_2 = xx[-1] / rho[-1]
                sin_2 = yy[-1] / rho[-1]
                # line
                te_hy_line = np.dot(kernel_h_r_te[0], bessel_j0).T / rho
                te_hy_line = te_hy_line @ np.ones(split)
                    
                # begin
                tm_hr_g_1 = (kernel_h_r_tm[1][0] @ bessel_j1) / rho[0]
                te_hr_g_1 = (kernel_h_r_te[1][0] @ bessel_j1) / rho[0]

                # end
                tm_hr_g_2 = (kernel_h_r_tm[1][-1] @ bessel_j1) / rho[-1]
                te_hr_g_2 = (kernel_h_r_te[1][-1] @ bessel_j1) / rho[-1]

                amp_tm_hx_1 =  sin_1 / (4 * np.pi)
                amp_tm_hx_2 = -sin_2 / (4 * np.pi)
                amp_te_hx_1 =  sin_1 / (4 * np.pi) * impz_s / impz_r
                amp_te_hx_2 = -sin_2 / (4 * np.pi) * impz_s / impz_r

                h_x_rot = amp_tm_hx_1 * tm_hr_g_1 \
                        + amp_tm_hx_2 * tm_hr_g_2 \
                        + amp_te_hx_1 * te_hr_g_1 \
                        + amp_te_hx_2 * te_hr_g_2

                amp_tm_hy_1 = -cos_1 / (4 * np.pi)
                amp_tm_hy_2 = cos_2 / (4 * np.pi)
                amp_te_hy_1 = -cos_1 / (4 * np.pi) * impz_s / impz_r
                amp_te_hy_2 = cos_2 / (4 * np.pi) * impz_s / impz_r
                amp_te_hy_line = 1 / (4 * np.pi) * impz_s / impz_r
                
                h_y_rot = amp_tm_hy_1 * tm_hr_g_1 \
                        + amp_tm_hy_2 * tm_hr_g_2 \
                        + amp_te_hy_1 * te_hr_g_1 \
                        + amp_te_hy_2 * te_hr_g_2 \
                        + amp_te_hy_line * ds * te_hy_line

                if "x" in direction:
                    h_x = cos_theta * h_x_rot - sin_theta * h_y_rot
                    ans.append(h_x)
                if "y" in direction:
                    h_y = sin_theta * h_x_rot + cos_theta * h_y_rot
                    ans.append(h_y)

            if "z" in direction:
                kernel_h_z_te = compute_kernel_hed_h_z_te(
                     u_te, d_te, e_up, e_down, si, ri, us, ur, zs, z, lambda3d
                )
                te_hz_line = (kernel_h_z_te @ bessel_j1).T / rho
                te_hz_line = (impz_s / impz_r * te_hz_line.T).T @ (yy / rho)
                h_z = ds / (4 * np.pi) * te_hz_line
                ans.append(h_z)

        ans = np.array(ans).astype(complex)
        return ans
