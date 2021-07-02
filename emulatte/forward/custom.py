import numpy as np
from emulatte.forward import model, transceiver

class GroundedWire(model.Subsurface1D):
    # OverRide
    def __init__(self, thicks, freqtime, current, displacement_current=False):
        super().__init__(thicks, displacement_current=displacement_current)
        self.freqtime = freqtime # Common in FD and TD
        self.omegas = 2 * np.pi * freqtime # Only using in FD
        self.ft_size = len(freqtime)
        self.current = current

    # OverRide / 位置の決定とそれに伴う処理
    def locate(tc, rc):
        self.tx, self.ty, self.tz = tc.T
        self.rx, self.ry, self.rz = rc
        # 送信z座標が固定の時
        if len(np.unique(self.tz)) == 1:
            self.tz = np.append(self.tz, self.tz[0])
            self.d = np.sqrt((self.tx[1] - self.tx[0]) ** 2 + (self.ty[1] - self.ty[0]) ** 2)
            cos_theta = (self.tx[1] - self.tx[0]) / self.d
            sin_theta = (self.ty[1] - self.ty[0]) / self.d
            self.num_v  = len(self.tx) # v means vertex
            #num_v_end = self.num_v
            num_d = 2 * self.d

            tx_dipole = np.array([])
            ty_dipole = np.array([])
            tz_dipole = np.array([])

            tx_dipole = np.append(tx_dipole, np.linspace(self.tx[0], self.tx[1], num_d))
            ty_dipole = np.append(ty_dipole, np.linspace(self.ty[0], self.ty[1], num_d))
            tz_dipole = np.append(tz_dipole, np.linspace(self.tz[0], self.tz[1], num_d))

            diff_x = np.abs(self.tx[0] - self.tx[1]) / (np.trunc(num_d) - 1)
            diff_y = np.abs(self.ty[0] - self.ty[1]) / (np.trunc(num_d) - 1)
            diff_z = np.abs(self.tz[0] - self.tz[1]) / (np.trunc(num_d) - 1)
            tx_dipole = tx_dipole + diff_x / 2
            ty_dipole = ty_dipole + diff_y / 2
            tz_dipole = tz_dipole + diff_z / 2

            if self.tx[0] < self.tx[1]:
                self.tx_dipole = np.insert(tx_dipole, 0, tx_dipole[0] - diff_x)
            else:
                self.tx_dipole = np.append(tx_dipole, tx_dipole[-1] - diff_x)

            if self.ty[0] < self.ty[1]:
                self.ty_dipole = np.insert(ty_dipole, 0, ty_dipole[0] - diff_y)
            else:
                self.ty_dipole = np.append(ty_dipole, ty_dipole[-1] - diff_y)

            if self.tz[0] < self.tz[1]:
                self.tz_dipole = np.insert(tz_dipole, 0, tz_dipole[0] - diff_z)
            else:
                self.tz_dipole = np.append(tz_dipole, tz_dipole[-1] - diff_z)

            num_dipole = np.trunc(num_d)
            self.num_dipole = int(num_dipole)

            tx = np.ones((1, self.num_dipole))
            ty = np.ones((1, self.num_dipole))
            tz = np.ones((1, self.num_dipole))
            for ii in range(0, self.num_dipole):
                tx[0, ii] = 0.5 * (self.tx_dipole[ii + 1] + self.tx_dipole[ii])
                ty[0, ii] = 0.5 * (self.ty_dipole[ii + 1] + self.ty_dipole[ii])
                tz[0, ii] = 0.5 * (self.tz_dipole[ii + 1] + self.tz_dipole[ii])

            self.ds = self.d / self.num_dipole

            def change_coordinate(x, y, z, cos_theta, sin_theta):
                rot_theta = np.array([[cos_theta, sin_theta, 0], [-sin_theta, cos_theta, 0], [0, 0, 1]])
                new_coordinate = np.dot(rot_theta, np.array([x, y, z]))
                return new_coordinate

            new_transmitter = np.ones((self.num_dipole, 3))
            for ii in range(0, self.num_dipole):
                new_transmitter[ii, :] = change_coordinate(tx[0, ii], ty[0, ii], tz[0, ii], cos_theta, sin_theta)

            new_receiver = change_coordinate(self.x, self.y, self.z, cos_theta, sin_theta)
            self.xx = new_receiver[0] - new_transmitter[:,0]
            self.yy = new_receiver[1] - new_transmitter[:,1]
            self.z =  new_receiver[2]

            self.rn = np.ones((1, self.num_dipole))
            self.mix = False
            for ii in range(0, self.num_dipole):
                self.rn[0, ii] = np.sqrt(self.xx[ii] ** 2 + self.yy[ii] ** 2)
            return None
        else:
            num_v = 1
            self.d = np.sqrt((self.tx[1] - self.tx[0]) ** 2 + (self.ty[1] - self.ty[0]) ** 2 + (self.tz[1] - self.tz[0]) ** 2)
            cos_theta = (self.tx[1] - self.tx[0]) / np.sqrt((self.tx[1] - self.tx[0]) ** 2 + (self.ty[1] - self.ty[0]) ** 2)
            sin_theta = (self.ty[1] - self.ty[0]) / np.sqrt((self.tx[1] - self.tx[0]) ** 2 + (self.ty[1] - self.ty[0]) ** 2)
            cos_phi = (self.tz[1] - self.tz[0]) / self.d
            sin_phi = np.sqrt((self.tx[1] - self.tx[0]) ** 2 + (self.ty[1] - self.ty[0]) ** 2) / self.d
            theta = np.arccos(cos_theta)
            phi = np.arccos(cos_phi)

            #print("theta = " + str(np.rad2deg(theta)))
            #print("phai = " + str(np.rad2deg(phai)))

            num_d = 2 * self.d
            tx_dipole = np.array([])
            ty_dipole = np.array([])
            tz_dipole = np.array([])

            tx_dipole = np.append(tx_dipole, np.linspace(self.tx[0], self.tx[1], num_d))
            ty_dipole = np.append(ty_dipole, np.linspace(self.ty[0], self.ty[1], num_d))
            tz_dipole = np.append(tz_dipole, np.linspace(self.tz[0], self.tz[1], num_d))

            diff_x = np.abs(self.tx[0] - self.tx[1]) / (np.trunc(num_d) - 1)
            diff_y = np.abs(self.ty[0] - self.ty[1]) / (np.trunc(num_d) - 1)
            diff_z = np.abs(self.tz[0] - self.tz[1]) / (np.trunc(num_d) - 1)
            tx_dipole = tx_dipole + diff_x / 2
            ty_dipole = ty_dipole + diff_y / 2
            tz_dipole = tz_dipole + diff_z / 2

            if self.tx[0] < self.tx[1]:
                self.tx_dipole = np.insert(tx_dipole, 0, tx_dipole[0] - diff_x)
            else:
                self.tx_dipole = np.append(tx_dipole, tx_dipole[-1] - diff_x)

            if self.ty[0] < self.ty[1]:
                self.ty_dipole = np.insert(ty_dipole, 0, ty_dipole[0] - diff_y)
            else:
                self.ty_dipole = np.append(ty_dipole, ty_dipole[-1] - diff_y)

            if self.tz[0] < self.tz[1]:
                self.tz_dipole = np.insert(tz_dipole, 0, tz_dipole[0] - diff_z)
            else:
                self.tz_dipole = np.append(tz_dipole, tz_dipole[-1] - diff_z)

            num_dipole = np.trunc(num_d) + 1 - 1
            num_dipole = int(num_dipole)
            self.num_dipole = num_dipole
            self.ds = self.d / self.num_dipole
            self.mix = True
            return None

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

        if mix:
            ans = np.zeros((self.ft_size, 6), dtype=complex)
            ans_hedx = np.zeros((self.ft_size, 6), dtype=complex)
            ans_hedy = np.zeros((self.ft_size, 6), dtype=complex)
            ans_ved = np.zeros((self.ft_size, 6), dtype=complex)
            hedx = transceiver.HEDx(self.freqtime, self.ds, self.current)
            hedy = transceiver.HEDy(self.freqtime, self.ds, self.current)
            ved = transceiver.VED(self.freqtime, self.ds, self.current)

            for jj in range(self.num_dipole):
                self.tx[0] = 0.5 * (self.tx_dipole[jj + 1] + self.tx_dipole[jj])
                self.ty[0] = 0.5 * (self.ty_dipole[jj + 1] + self.ty_dipole[jj])
                self.tz[0] = 0.5 * (self.tz_dipole[jj + 1] + self.tz_dipole[jj])
                self.r = np.sqrt((self.rx - self.tx[0]) ** 2 + (self.ry - self.ty[0]) ** 2)

                self.tcv = hedx
                self.num_dipole = tcv.num_dipole
                self.kernel_te_up_sign = tcv.kernel_te_up_sign
                self.kernel_te_down_sign = tcv.kernel_te_down_sign
                self.kernel_tm_up_sign = tcv.kernel_tm_up_sign
                self.kernel_tm_down_sign = tcv.kernel_tm_down_sign
                em_field_hedx, freqtime = hedx.get_result(self, time_diff=time_diff, td_transform=td_transform)
                ans_hedx[:, 0] = em_field_hedx["e_x"] * self.ds
                ans_hedx[:, 1] = em_field_hedx["e_y"] * self.ds
                ans_hedx[:, 2] = em_field_hedx["e_z"] * self.ds
                ans_hedx[:, 3] = em_field_hedx["h_x"] * self.ds
                ans_hedx[:, 4] = em_field_hedx["h_y"] * self.ds
                ans_hedx[:, 5] = em_field_hedx["h_z"] * self.ds

                self.tcv = hedy
                em_field_hedy, freqtime = hedy.get_result(self, time_diff=time_diff, td_transform=td_transform)
                ans_hedy[:, 0] = em_field_hedy["e_x"] * self.ds
                ans_hedy[:, 1] = em_field_hedy["e_y"] * self.ds
                ans_hedy[:, 2] = em_field_hedy["e_z"] * self.ds
                ans_hedy[:, 3] = em_field_hedy["h_x"] * self.ds
                ans_hedy[:, 4] = em_field_hedy["h_y"] * self.ds
                ans_hedy[:, 5] = em_field_hedy["h_z"] * self.ds

                self.tcv = ved
                em_field_ved, freqtime = ved.get_result(self, time_diff=time_diff, td_transform=td_transform)
                ans_ved[:, 0] = em_field_ved["e_x"] * self.ds
                ans_ved[:, 1] = em_field_ved["e_y"] * self.ds
                ans_ved[:, 2] = em_field_ved["e_z"] * self.ds
                ans_ved[:, 3] = em_field_ved["h_x"] * self.ds
                ans_ved[:, 4] = em_field_ved["h_y"] * self.ds
                ans_ved[:, 5] = em_field_ved["h_z"] * self.ds

                ans = (sin_phai * (cos_theta * ans_hedx + sin_theta * ans_hedy) + cos_phai * ans_ved) + ans
            ans = {"e_x": ans[:, 0], "e_y": ans[:, 1], "e_z": ans[:, 2], "h_x": ans[:, 3], "h_y": ans[:, 4], "h_z": ans[:, 5]}
            return ans , freqtime
        else:
            gw = transceiver._GroundedWire(self.freqtime, self.current)
            self.tcv = gw
            em_field, freqtime = gw.get_result(self, time_diff=time_diff, td_transform=td_transform)
            ans = np.zeros((self.ft_size, 6), dtype=complex)
            ans[:, 0] = cos_theta * em_field["e_x"] - sin_theta * em_field["e_y"]
            ans[:, 1] = cos_theta * em_field["e_y"] + sin_theta * em_field["e_x"]
            ans[:, 2] = em_field["e_z"]
            ans[:, 3] = cos_theta * em_field["h_x"] - sin_theta * em_field["h_y"]
            ans[:, 4] = cos_theta * em_field["h_y"] + sin_theta * em_field["h_x"]
            ans[:, 5] = em_field["h_z"]

            ans = {
                "e_x": ans[:, 0], "e_y": ans[:, 1], "e_z": ans[:, 2],
                "h_x": ans[:, 3], "h_y": ans[:, 4], "h_z": ans[:, 5]
                }
            return ans, freqtime

class LoopSource(model.Subsurface1D):
    def __init__(self, thicks, freqtime, current, turns, displacement_current=False):
        super().__init__(thicks, displacement_current=displacement_current)
        self.freqtime = freqtime # Common in FD and TD
        self.omegas = 2 * np.pi * freqtime # Only using in FD
        self.ft_size = len(freqtime)
        self.current = current
        self.turns = turns
        self.moment = current
        self.kernel_te_up_sign = 1
        self.kernel_te_down_sign = 1
        self.kernel_tm_up_sign = -1
        self.kernel_tm_down_sign = 1

    def locate(self, tc, rc):
        self.tx, self.ty, self.tz = tc.T
        self.rx, self.ry, self.rz = rc
        if len(np.unique(self.tz)) == 1:
            num_v = len(self.tx)

            # 送信源固有のパラメータ設定
            self.tx.append(self.tx[0])
            self.ty.append(self.ty[0])
            self.tz.append(self.tz[0])


            ans = np.zeros((self.num_plot, 6), dtype=complex)
            for ii in range (num_v):
                self.d = np.sqrt((self.tx[ii+1] - self.tx[ii]) ** 2 + (self.ty[ii+1] - self.ty[ii]) ** 2)
                cos_theta = (self.tx[ii+1] - self.tx[ii]) / self.d
                sin_theta = (self.ty[ii+1] - self.ty[ii]) / self.d

                num_d = 2 * self.d

                tx_dipole = np.array([])
                ty_dipole = np.array([])
                tz_dipole = np.array([])
                tx_dipole = np.append(tx_dipole, np.linspace(self.tx[ii], self.tx[ii+1], num_d))
                ty_dipole = np.append(ty_dipole, np.linspace(self.ty[ii], self.ty[ii+1], num_d))
                tz_dipole = np.append(tz_dipole, np.linspace(self.tz[ii], self.tz[ii+1], num_d))

                diff_x = np.abs(self.tx[ii] - self.tx[ii+1]) / (np.trunc(num_d) - 1)
                diff_y = np.abs(self.ty[ii] - self.ty[ii+1]) / (np.trunc(num_d) - 1)
                diff_z = np.abs(self.tz[ii] - self.tz[ii+1]) / (np.trunc(num_d) - 1)
                tx_dipole = tx_dipole + diff_x / 2
                ty_dipole = ty_dipole + diff_y / 2
                tz_dipole = tz_dipole + diff_z / 2

                if self.tx[ii] < self.tx[ii+1]:
                    self.tx_dipole = np.insert(tx_dipole, 0, tx_dipole[0] - diff_x)
                else:
                    self.tx_dipole = np.append(tx_dipole, tx_dipole[-1] - diff_x)

                if self.ty[ii] < self.ty[ii+1]:
                    self.ty_dipole = np.insert(ty_dipole, 0, ty_dipole[0] - diff_y)
                else:
                    self.ty_dipole = np.append(ty_dipole, ty_dipole[-1] - diff_y)

                if self.tz[ii] < self.tz[ii+1]:
                    self.tz_dipole = np.insert(tz_dipole, 0, tz_dipole[0] - diff_z)
                else:
                    self.tz_dipole = np.append(tz_dipole, tz_dipole[-1] - diff_z)

                num_dipole = np.trunc(num_d) + 1 - 1
                self.num_dipole = int(num_dipole)

                tx = np.ones((1, self.num_dipole))
                ty = np.ones((1, self.num_dipole))
                tz = np.ones((1, self.num_dipole))
                for ii in range(0, self.num_dipole):
                    tx[0, ii] = 0.5 * (self.tx_dipole[ii + 1] + self.tx_dipole[ii])
                    ty[0, ii] = 0.5 * (self.ty_dipole[ii + 1] + self.ty_dipole[ii])
                    tz[0, ii] = 0.5 * (self.tz_dipole[ii + 1] + self.tz_dipole[ii])

                self.ds = self.d / self.num_dipole


                def change_coordinate(x,y,z,cos_theta,sin_theta):
                    rot_theta = np.array([[cos_theta,sin_theta,0], [-sin_theta, cos_theta,0],[0,0,1]])
                    new_coordinate = np.dot(rot_theta, np.array([x, y, z]))
                    return new_coordinate

                new_transmitter = np.ones((self.num_dipole,3))
                for ii in range(0, self.num_dipole):
                    new_transmitter[ii,:] = change_coordinate(tx[0, ii], ty[0, ii], tz[0, ii],cos_theta,sin_theta)

                new_receiver = change_coordinate(self.x, self.y, self.z,cos_theta,sin_theta)

                self.xx = new_receiver[0] - new_transmitter[:,0]
                self.yy = new_receiver[1] - new_transmitter[:,1]
                self.z =  new_receiver[2]

                self.rn = np.ones((1, self.num_dipole))
                for ii in range(0, self.num_dipole):
                    self.rn[0, ii] = np.sqrt(self.xx[ii] ** 2 + self.yy[ii] ** 2)
        else:
            num_v = len(self.tx)
            self.tx.append(self.tx[0])
            self.ty.append(self.ty[0])
            self.tz.append(self.tz[0])

            ans = 0
            for ii in range(num_v):
                self.d = np.sqrt((self.tx[ii + 1] - self.tx[ii]) ** 2 + (self.ty[ii + 1] - self.ty[ii]) ** 2 + (
                            self.tz[ii + 1] - self.tz[ii]) ** 2)
                cos_theta = (self.tx[ii + 1] - self.tx[ii]) / np.sqrt(
                    (self.tx[ii + 1] - self.tx[ii]) ** 2 + (self.ty[ii + 1] - self.ty[ii]) ** 2)
                sin_theta = (self.ty[ii + 1] - self.ty[ii]) / np.sqrt(
                    (self.tx[ii + 1] - self.tx[ii]) ** 2 + (self.ty[ii + 1] - self.ty[ii]) ** 2)
                # cos_phai = (self.tz[ii+1]-self.tz[ii]) / self.d
                sin_phai = np.sqrt((self.tx[ii + 1] - self.tx[ii]) ** 2 + (self.ty[ii + 1] - self.ty[ii]) ** 2) / self.d

                num_d = 2 * self.d
                tx_dipole = np.array([])
                ty_dipole = np.array([])
                tz_dipole = np.array([])

                tx_dipole = np.append(tx_dipole, np.linspace(self.tx[ii], self.tx[ii + 1], num_d))
                ty_dipole = np.append(ty_dipole, np.linspace(self.ty[ii], self.ty[ii + 1], num_d))
                tz_dipole = np.append(tz_dipole, np.linspace(self.tz[ii], self.tz[ii + 1], num_d))

                diff_x = np.abs(self.tx[ii] - self.tx[ii + 1]) / (np.trunc(num_d) - 1)
                diff_y = np.abs(self.ty[ii] - self.ty[ii + 1]) / (np.trunc(num_d) - 1)
                diff_z = np.abs(self.tz[ii] - self.tz[ii + 1]) / (np.trunc(num_d) - 1)
                tx_dipole = tx_dipole + diff_x / 2
                ty_dipole = ty_dipole + diff_y / 2
                tz_dipole = tz_dipole + diff_z / 2

                if self.tx[ii] < self.tx[ii + 1]:
                    self.tx_dipole = np.insert(tx_dipole, 0, tx_dipole[0] - diff_x)
                else:
                    self.tx_dipole = np.append(tx_dipole, tx_dipole[-1] - diff_x)

                if self.ty[ii] < self.ty[ii + 1]:
                    self.ty_dipole = np.insert(ty_dipole, 0, ty_dipole[0] - diff_y)
                else:
                    self.ty_dipole = np.append(ty_dipole, ty_dipole[-1] - diff_y)

                if self.tz[ii] < self.tz[ii + 1]:
                    self.tz_dipole = np.insert(tz_dipole, 0, tz_dipole[0] - diff_z)
                else:
                    self.tz_dipole = np.append(tz_dipole, tz_dipole[-1] - diff_z)

                num_dipole = np.trunc(num_d) + 1 - 1
                num_dipole = int(num_dipole)
                self.num = num_dipole

                self.ds = self.d / self.num

                ans_hedx = np.zeros((self.num_plot, 6), dtype=complex)
                ans_hedy = np.zeros((self.num_plot, 6), dtype=complex)
                ans_line = 0

                def x_line_source(self,current):
                    transmitter = sys._getframe().f_code.co_name
                    self.r = np.sqrt((self.x - self.tx[0]) ** 2 + (self.y - self.ty[0]) ** 2)
                    self.em1d_base()

                    self.lamda = self.y_base / self.r
                    self.moment = current
                    self.num_dipole = 1
                    self.kernel_te_up_sign = 1
                    self.kernel_te_down_sign = 1
                    self.kernel_tm_up_sign = -1
                    self.kernel_tm_down_sign = 1
                    ans, self.freqs = self.repeat_computation(transmitter)
                    return ans

                def y_line_source(self,current):
                    transmitter = sys._getframe().f_code.co_name
                    ans, self.freqs = self.repeat_computation(transmitter)
                    return ans

                for jj in range(num_dipole):
                    self.tx[0] = 0.5 * (self.tx_dipole[jj + 1] + self.tx_dipole[jj])
                    self.ty[0] = 0.5 * (self.ty_dipole[jj + 1] + self.ty_dipole[jj])
                    self.tz[0] = 0.5 * (self.tz_dipole[jj + 1] + self.tz_dipole[jj])

                    em_field_hedx = x_line_source(self,current)
                    ans_hedx[:, 0] = em_field_hedx["e_x"]
                    ans_hedx[:, 1] = em_field_hedx["e_y"]
                    ans_hedx[:, 2] = em_field_hedx["e_z"]
                    ans_hedx[:, 3] = em_field_hedx["h_x"]
                    ans_hedx[:, 4] = em_field_hedx["h_y"]
                    ans_hedx[:, 5] = em_field_hedx["h_z"]

                    em_field_hedy = y_line_source(self,current)
                    ans_hedy[:, 0] = em_field_hedy["e_x"]
                    ans_hedy[:, 1] = em_field_hedy["e_y"]
                    ans_hedy[:, 2] = em_field_hedy["e_z"]
                    ans_hedy[:, 3] = em_field_hedy["h_x"]
                    ans_hedy[:, 4] = em_field_hedy["h_y"]
                    ans_hedy[:, 5] = em_field_hedy["h_z"]

                    ans_line = sin_phai * (cos_theta * ans_hedx + sin_theta * ans_hedy) + ans_line
                ans = ans_line + ans
            ans = {"e_x": ans[:, 0] * turns, "e_y": ans[:, 1] * turns, "e_z": ans[:, 2]* turns, "h_x": ans[:, 3]* turns, "h_y": ans[:, 4]* turns,"h_z": ans[:, 5]* turns}
            return ans, self.freqs

