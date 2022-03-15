import numpy as np
from ..utils.converter import array, check_waveform
from .kernel import _calc_kernel_hed_tm_er, _calc_kernel_hed_te_er, _calc_kernel_hed_tm_ez
from .kernel import _calc_kernel_hed_tm_hr, _calc_kernel_hed_te_hr, _calc_kernel_hed_te_hz

class HGW:
    def __init__(self, current, ontime = None):
        self.current = array(current)
        moment = array(current)
        self.kernel_te_up_sign = 1
        self.kernel_te_down_sign = 1
        self.kernel_tm_up_sign = -1
        self.kernel_tm_down_sign = 1

        self.signal = check_waveform(ontime)

        if len(moment) == 1:
            self.moment = moment[0]
        else:
            self.moment_array = moment

    def _hankel_transform_e(self, model, direction, omega, y_base, wt0, wt1):
        model._calc_em_admittion(omega)
        yr = model.admy[:, model.ri]
        zs = model.admz[:, model.si]
        ans = []
        if ("x" in direction) or ("y" in direction):
            cos_1 = model.xx[0] / model.rn[0]
            sin_1 = model.yy[0] / model.rn[0]
            cos_2 = model.xx[-1] / model.rn[-1]
            sin_2 = model.yy[-1] / model.rn[-1]
            # line
            model._calc_kernel_components(y_base, model.rn)
            lambda_tensor = np.zeros((model.K, model.J, model.M), dtype=complex)
            lambda_tensor[:] = model.lambda_
            lambda_tensor = lambda_tensor.transpose((1,0,2))
            kernel_line_te_er = _calc_kernel_hed_te_er(model)
            te_ex_line = (np.dot(kernel_line_te_er * lambda_tensor, wt0).T / model.rn).T
            te_ex_line = np.dot(np.ones(model.split), te_ex_line)
                
            # begin
            model._calc_kernel_components(y_base, model.rn[0])
            kernel = _calc_kernel_hed_tm_er(model)
            tm_er_g_1 = np.dot(kernel[0], wt1) / model.rn[0]
            kernel = kernel_line_te_er[0]
            te_er_g_1 = np.dot(kernel, wt1) / model.rn[0]

            # end
            model._calc_kernel_components(y_base, model.rn[-1])
            kernel = _calc_kernel_hed_tm_er(model)
            tm_er_g_2 = np.dot(kernel[0], wt1) / model.rn[-1]
            kernel = kernel_line_te_er[-1]
            te_er_g_2 = np.dot(kernel, wt1) / model.rn[-1]

            amp_tm_ex_1 =  cos_1 / (4 * np.pi * yr)
            amp_tm_ex_2 = -cos_2 / (4 * np.pi * yr)
            amp_te_ex_1 =  cos_1 * zs / (4 * np.pi)
            amp_te_ex_2 = -cos_2 * zs / (4 * np.pi)
            amp_ex_line  = -zs / (4 * np.pi)

            e_x_rot = amp_tm_ex_1 * tm_er_g_1 \
                    + amp_tm_ex_2 * tm_er_g_2 \
                    + amp_te_ex_1 * te_er_g_1 \
                    + amp_te_ex_2 * te_er_g_2 \
                    + amp_ex_line * model.ds * te_ex_line

            amp_tm_ey_1 =  sin_1 / (4 * np.pi * yr)
            amp_tm_ey_2 = -sin_2 / (4 * np.pi * yr)
            amp_te_ey_1 =  sin_1 * zs / (4 * np.pi)
            amp_te_ey_2 = -sin_2 * zs / (4 * np.pi)
            
            e_y_rot = amp_tm_ey_1 * tm_er_g_1 \
                    + amp_tm_ey_2 * tm_er_g_2 \
                    + amp_te_ey_1 * te_er_g_1 \
                    + amp_te_ey_2 * te_er_g_2

            e_x_rot = e_x_rot * self.moment
            e_y_rot = e_y_rot * self.moment


            if "x" in direction:
                e_x = model.cos_theta * e_x_rot - model.sin_theta * e_y_rot
                ans.append(e_x)
            if "y" in direction:
                e_y = model.sin_theta * e_x_rot + model.cos_theta * e_y_rot
                ans.append(e_y)
        if "z" in direction:
            model._calc_kernel_components(y_base, model.rn[0])
            kernel = _calc_kernel_hed_tm_ez(model)
            tm_ez_1 = np.dot(kernel[0] * model.lambda_, wt0) / model.rn[0]
            model._calc_kernel_components(y_base, model.rn[-1])
            kernel = _calc_kernel_hed_tm_ez(model)
            tm_ez_2 = np.dot(kernel[0] * model.lambda_, wt0) / model.rn[-1]

            amp_tm_ez_1 = 1 / (4 * np.pi * yr)
            amp_tm_ez_2 = -1 / (4 * np.pi * yr)
            e_z = amp_tm_ez_1 * tm_ez_1 + amp_tm_ez_2 * tm_ez_2
            e_z = self.moment * e_z
            ans.append(e_z)
        ans = np.array(ans).astype(complex)
        if model.time_derivative:
            ans = ans * omega * 1j
        return ans
    
    def _hankel_transform_h(self, model, direction, omega, y_base, wt0, wt1):
        model._calc_em_admittion(omega)
        zr = model.admz[:, model.ri]
        zs = model.admz[:, model.si]
        ans = []
        if ("x" in direction) or ("y" in direction):
            cos_1 = model.xx[0] / model.rn[0]
            sin_1 = model.yy[0] / model.rn[0]
            cos_2 = model.xx[-1] / model.rn[-1]
            sin_2 = model.yy[-1] / model.rn[-1]
            # line
            model._calc_kernel_components(y_base, model.rn)
            lambda_tensor = np.zeros((model.K, model.J, model.M), dtype=complex)
            lambda_tensor[:] = model.lambda_
            lambda_tensor = lambda_tensor.transpose((1,0,2))
            kernel_line_te_hr = _calc_kernel_hed_te_hr(model)
            te_hy_line = np.dot(kernel_line_te_hr * lambda_tensor, wt0) / model.rn.reshape(-1,1)
            te_hy_line = np.dot(np.ones(model.split), te_hy_line)
                
            # begin
            model._calc_kernel_components(y_base, model.rn[0])
            kernel = _calc_kernel_hed_tm_hr(model)
            tm_hr_g_1 = np.dot(kernel[0], wt1) / model.rn[0]

            kernel = kernel_line_te_hr[0]
            te_hr_g_1 = np.dot(kernel, wt1) / model.rn[0]

            # end
            model._calc_kernel_components(y_base, model.rn[-1])
            kernel = _calc_kernel_hed_tm_hr(model)
            tm_hr_g_2 = np.dot(kernel[0], wt1) / model.rn[-1]

            kernel = kernel_line_te_hr[-1]
            te_hr_g_2 = np.dot(kernel, wt1) / model.rn[-1]

            amp_tm_hx_1 =  sin_1 / (4 * np.pi)
            amp_tm_hx_2 = -sin_2 / (4 * np.pi)
            amp_te_hx_1 =  sin_1 / (4 * np.pi) * zs / zr
            amp_te_hx_2 = -sin_2 / (4 * np.pi) * zs / zr

            h_x_rot = amp_tm_hx_1 * tm_hr_g_1 \
                    + amp_tm_hx_2 * tm_hr_g_2 \
                    + amp_te_hx_1 * te_hr_g_1 \
                    + amp_te_hx_2 * te_hr_g_2

            amp_tm_hy_1 = -cos_1 / (4 * np.pi)
            amp_tm_hy_2 = cos_2 / (4 * np.pi)
            amp_te_hy_1 = -cos_1 / (4 * np.pi) * zs / zr
            amp_te_hy_2 = cos_2 / (4 * np.pi) * zs / zr
            amp_te_hy_line = 1 / (4 * np.pi) * zs / zr
            
            h_y_rot = amp_tm_hy_1 * tm_hr_g_1 \
                    + amp_tm_hy_2 * tm_hr_g_2 \
                    + amp_te_hy_1 * te_hr_g_1 \
                    + amp_te_hy_2 * te_hr_g_2 \
                    + amp_te_hy_line * model.ds * te_hy_line
            
            h_x_rot = h_x_rot * self.moment
            h_y_rot = h_y_rot * self.moment

            if "x" in direction:
                h_x = model.cos_theta * h_x_rot - model.sin_theta * h_y_rot
                ans.append(h_x)
            if "y" in direction:
                h_y = model.sin_theta * h_x_rot + model.cos_theta * h_y_rot
                ans.append(h_y)

        if "z" in direction:
            model._calc_kernel_components(y_base, model.rn)
            lambda_tensor = np.zeros((model.K, model.J, model.M), dtype=complex)
            lambda_tensor[:] = model.lambda_
            lambda_tensor = lambda_tensor.transpose((1,0,2))
            kernel_line_te_hz = _calc_kernel_hed_te_hz(model)
            te_hz_line = np.dot(kernel_line_te_hz * lambda_tensor ** 2, wt1) / model.rn.reshape(-1,1)
            te_hz_line = np.dot(model.yy / model.rn, zs / zr * te_hz_line)
            h_z = self.moment * model.ds / (4 * np.pi) * te_hz_line
            ans.append(h_z)

        ans = np.array(ans).astype(complex)
        if model.time_derivative:
            ans = ans * omega * 1j
        return ans

