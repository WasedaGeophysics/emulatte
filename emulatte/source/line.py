import numpy as np
from numpy.typing import NDArray
from ..utils.interpret import check_waveform
from ..utils.emu_object import Source
from .kernel.edipole import (
    compute_kernel_hed_e_r_te,
    compute_kernel_hed_e_r_tm,
    compute_kernel_hed_e_z_tm,
    compute_kernel_hed_h_r_te,
    compute_kernel_hed_h_r_tm,
    compute_kernel_hed_h_z_te,
    compute_kernel_gw_e_z_tm
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
            self, model, direction, magnetic : bool
            ) -> NDArray:
        # kernel components
        model._compute_kernel_components_multi()
        si = model.si
        zs = model.zs
        ri = model.ri
        z = model.z

        rho = model.rho
        lambda_ = model.lambda_
        bessel_j0 = model.bessel_j0
        bessel_j1 = model.bessel_j1

        us = model.u[:,:,si]
        ur = model.u[:,:,ri]

        impz_s = model.impedivity[:, si]
        impz_r = model.impedivity[:, ri]
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

        y_ys_pole = model.dist_y_pole
        x_xs_node, y_ys_node = model.dist_x_node, model.dist_y_node

        jj = model.ndims[0]
        kk = model.ndims[1]
        mm = model.ndims[3]

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

                cos_1 = x_xs_node[0] / rho[0]
                sin_1 = y_ys_node[0] / rho[0]
                cos_2 = x_xs_node[-1] / rho[-1]
                sin_2 = y_ys_node[-1] / rho[-1]

                # line
                te_ex_line = (kernel_e_r_te[0][1:-1] @ bessel_j0).T / rho[1:-1]
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
                amp_ex_line = - impz_s / (4 * np.pi) * ds

                e_x_rot = amp_tm_ex_1 * tm_er_g_1 \
                        + amp_tm_ex_2 * tm_er_g_2 \
                        + amp_te_ex_1 * te_er_g_1 \
                        + amp_te_ex_2 * te_er_g_2 \
                        + amp_ex_line * te_ex_line

                amp_tm_ey_1 =  sin_1 / (4 * np.pi * admy_r)
                amp_tm_ey_2 = -sin_2 / (4 * np.pi * admy_r)
                amp_te_ey_1 =  sin_1 * impz_s / (4 * np.pi)
                amp_te_ey_2 = -sin_2 * impz_s / (4 * np.pi)
                
                e_y_rot = amp_tm_ey_1 * tm_er_g_1 \
                        + amp_tm_ey_2 * tm_er_g_2 \
                        + amp_te_ey_1 * te_er_g_1 \
                        + amp_te_ey_2 * te_er_g_2

                if "x" in direction:
                    e_x = cos_theta * e_x_rot - sin_theta * e_y_rot
                    ans.append(e_x)
                if "y" in direction:
                    e_y = sin_theta * e_x_rot + cos_theta * e_y_rot
                    ans.append(e_y)

            if "z" in direction:
                kernel_e_z_tm = compute_kernel_gw_e_z_tm(
                    u_tm, d_tm, e_up, e_down, si, ri, us, ur, zs, z, lambda3d)

                
                tm_ez_1 = (kernel_e_z_tm[0] @ bessel_j0) / rho[0]
                tm_ez_2 = (kernel_e_z_tm[-1] @ bessel_j0) / rho[-1]

                amp_tm_ez_1 = -1 / (4 * np.pi * admy_r)
                amp_tm_ez_2 = 1 / (4 * np.pi * admy_r)

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
                
                cos_1 = x_xs_node[0] / rho[0]
                sin_1 = y_ys_node[0] / rho[0]
                cos_2 = x_xs_node[-1] / rho[-1]
                sin_2 = y_ys_node[-1] / rho[-1]
                # line
                te_hy_line = np.dot(kernel_h_r_te[0][1:-1], bessel_j0).T / rho[1:-1]
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
                amp_te_hy_line = 1 / (4 * np.pi) * impz_s / impz_r * ds
                
                h_y_rot = amp_tm_hy_1 * tm_hr_g_1 \
                        + amp_tm_hy_2 * tm_hr_g_2 \
                        + amp_te_hy_1 * te_hr_g_1 \
                        + amp_te_hy_2 * te_hr_g_2 \
                        + amp_te_hy_line * te_hy_line

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
                te_hz_line = (kernel_h_z_te[1:-1] @ bessel_j1).T / rho[1:-1]
                te_hz_line = te_hz_line @ (y_ys_pole / rho[1:-1])
                h_z = 1 / (4 * np.pi) * impz_s / impz_r * te_hz_line * ds
                ans.append(h_z)
        ans = np.array(ans).astype(complex)
        return ans