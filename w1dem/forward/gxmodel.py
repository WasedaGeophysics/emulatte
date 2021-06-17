from transform import HankelTransform as hf

class Core1D:
    def __init__(self, struct, freqtime, domain, time_diff):
        self.struct = struct
        self.freqtime = freqtime
        self.domain = domain
        self.hankel_filter = hankel_filter
        self.time_diff = time_diff
        self.freqtime_size = len(freqtime)
        self.omega = 2 * np.pi * self.freqs
    
    def set_position(self, sc, rc):
        """
        sc : 2Dndarray(n,3) [[x1, y1, z1], [x2, y2, z2], ... ]
            Source Coordinates
        rc : 2Dndarray(n,3) [[x1, y1, z1], [x2, y2, z2], ... ]
            Resceiver Coordinates
        """
        sc = sc.T
        rc = rc.T
        sx, sy, sz = sc
        rx, ry, rz = rc

        r = np.sqrt((rx - sx) ** 2 + (ry - sy) ** 2)
        n = len(r) #送受信ペアの数
        cos_phai = (rx - sx) / r
        sin_phai = (ry - sy) / r

        # 計算できない送受信座標が入力された場合の処理
        for i in range(n):
            if r[i] == 0:
                r[i] = 1e-8 

        if self.hankel_filter == "anderson801":
            delta_z = 1e-4
            for i in range(n):
                if (sz[i] in self.struct.depth):
                    sz[i] -= delta_z
                if (sz[i] == rz[i]):
                    sz[i] -= delta_z
        else:
            delta_z = 1e-8
            for i in range(n):
                if (sz[i] in self.struct.depth):
                    sz[i] -= delta_z
                if (sz[i] == rz[i]):
                    sz[i] -= delta_z       


        # 送受信点が含まれる層の特定
        slayer = []
        rlayer = []
        for i in range(n):
            sloc = self.struct.in_which_layer(sz[i])
            rloc = self.struct.in_which_layer(rz[i])
            slayer.append(sloc)
            rlayer.append(rloc)

        # return to self
        self.sx, self.sy ,self.sz = sx, sy, sz
        self.rx, self.ry ,self.rz = rx, ry, rz
        self.slayer, self.rlayer = slayer, rlayer
        self.r = r
        self.cos_phai = cos_phai
        self.sin_phai = sin_phai

        return None


# 1DEM
class vmd(Core1D):
    def __init__(
            self, struct, freqtime, domain, hankel_filter, dipole_moment, 
            time_diff = False, displacement_current=None):
        super().__init__(struct, freqtime, domain, hankel_filter, time_diff)
        #VMD固有設定値
        self.name = 'vmd'
        self.moment = dipole_moment
        self.displacement_current = displacement_current
        self.kernel_te_up_sign = 1
        self.kernel_te_down_sign = 1
        self.kernel_tm_up_sign = 0
        self.kernel_tm_down_sign = 0

    def get_result(self):
        ans = np.zeros((self.freqtime_size, 6), dtype=complex)
        for index, omega in enumerate(self.omega):
            em_field = hf.vmd(self, omega)
            # 電場の計算
            ans[index, 0] = em_field["e_x"]
            ans[index, 1] = em_field["e_y"]
            ans[index, 2] = em_field["e_z"]
            # 磁場の計算
            ans[index, 3] = em_field["h_x"]
            ans[index, 4] = em_field["h_y"]
            ans[index, 5] = em_field["h_z"]
            if self.dbdt == 0:
                ans[index, :] = ans[index, :] * 1j * omega       
        ans = self.moment * ans
        ans = {"e_x": ans[:, 0], "e_y": ans[:, 1], "e_z": ans[:, 2], "h_x": ans[:, 3], "h_y": ans[:, 4], "h_z": ans[:, 5]}
        return ans







