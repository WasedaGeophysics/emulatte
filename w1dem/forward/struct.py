import numpy as np

class Subsurface1D:
    #== CONSTRUCTOR ======================================#
    def __init__(self, thicks, hand_system = "left", displacement_current=False):
        self.thicks = thicks
        self.depth = np.array([0, *np.cumsum(thicks)]) #層境界深度
        self.num_layer = len(thicks) + 1 #空気（水）層と最下層を含める
        self.hand_system = hand_system
        self.displacement_current = displacement_current

        # PERMITTIVITY OF VACUUM
        epsln0 = 8.85418782 * 1e-12
        self.epsln = np.ones(num_layer) * epsln0
        # PERMEABILITY OF VACUUM
        mu0 = 4. * np.pi * 1e-7
        self.mu = np.ones(num_layer) * mu0
    
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
        sc = sc.T
        rc = rc.T
        sx, sy, sz = sc
        rx, ry, rz = rc
        r = np.sqrt((rx - sx) ** 2 + (ry - sy) ** 2)
        n = len(r) #送受信ペアの数
        cos_phai = (rx - sx) / r
        sin_phai = (ry - sy) / r

        # 計算できない送受信座標が入力された場合の処理
        delta_z = 1e-8      #filterがanderson801の時は1e-4?
        for i in range(n):
            if r[i] == 0:
                r[i] = 1e-8
            if (sz[i] in self.depth):
                sz[i] -= delta_z
            if (sz[i] == rz[i]):
                sz[i] -= delta_z

        # 送受信点が含まれる層の特定
        src_layer = []
        rcv_layer = []
        for i in range(n):
            src_loc = self.struct.in_which_layer(sz[i])
            rcv_loc = self.struct.in_which_layer(rz[i])
            src_layer.append(src_loc)
            rcv_layer.append(rcv_loc)

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
            self, freqtime, domain, hankel_filter, 
            time_diff=False, td_transform=None, interpolate=None):

        self.freqtime = freqtime
        self.domain = domain
        self.hankel_filter = hankel_filter
        self.omega = 2 * np.pi * self.freqs
        self.ft_size = len(freqtime)

        ans = self.tcv.get_result(
            self, freqtime, domain, hankel_filter, 
            time_diff=time_diff, td_transform=td_transform, 
            interpolate=interpolate
            )
        
        return ans

    def compute_coefficients(self):
        ztilde = np.ones((1, self.num_layer, 1), dtype=np.complex)
        ytilde = np.ones((1, self.num_layer, 1), dtype=np.complex)
        k = np.zeros((1, self.num_layer), dtype=np.complex)
        u = np.ones((self.num_layer, self.filter_length, self.num_dipole), dtype=np.complex)
        tanhuh = np.zeros((self.num_layer - 1, self.filter_length, self.num_dipole), dtype=np.complex)
        Y = np.ones((self.num_layer, self.filter_length, self.num_dipole), dtype=np.complex)
        Z = np.ones((self.num_layer, self.filter_length, self.num_dipole), dtype=np.complex)



    @classmethod
    def in_which_layer(cls, z):
        layer_id = cls.num_layer
        for i in cls.num_layer:
            if z <= cls.depth[i]:
                layer_id = i + 1
                break
            else:
                continue
        return layer_id
