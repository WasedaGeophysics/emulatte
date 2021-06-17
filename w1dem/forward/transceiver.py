import numpy as np
from w1dem.forward import transform

class VMD:
    def __init__(
            self, dipole_moment=1):
        #VMD固有設定値
        self.name = 'vmd'
        self.moment = dipole_moment
        self.num_dipole = 1
        self.kernel_te_up_sign = 1
        self.kernel_te_down_sign = 1
        self.kernel_tm_up_sign = 0
        self.kernel_tm_down_sign = 0

    def get_result(
            self, mdl, freqtime, hankel_filter, domain, 
            time_diff=False, td_transform=None, interpolate=None):
        """
        Docstring
        """
        ans = np.zeros((mdl.ft_size, 6), dtype=complex)
        for index, omega in enumerate(mdl.omega):
            mdl.omega = omega
            em_field = transform.HankelTransformFunctions.vmd(mdl, td_transform)
            # 電場の計算
            ans[index, 0] = em_field["e_x"]
            ans[index, 1] = em_field["e_y"]
            ans[index, 2] = em_field["e_z"]
            # 磁場の計算
            ans[index, 3] = em_field["h_x"]
            ans[index, 4] = em_field["h_y"]
            ans[index, 5] = em_field["h_z"]
            if time_diff:
                ans[index, :] = ans[index, :] * 1j * omega       
        ans = self.moment * ans
        ans = {
            "e_x": ans[:, 0], "e_y": ans[:, 1], "e_z": ans[:, 2],
            "h_x": ans[:, 3], "h_y": ans[:, 4], "h_z": ans[:, 5]
            }
        return ans







