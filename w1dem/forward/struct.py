import numpy as np

class Core1D:
    def __init__(self, thicks, hand_system = "left"):
        if type(thicks) != np.ndarray():
            thicks = np.array(thicks)
        
        self.thicks = thicks
        self.depth = np.array([0, *np.cumsum(thicks)]) #層境界深度
        self.num_layer = len(thicks) + 1
        self.hand_system = hand_system

    def in_which_layer(self, z):
        layer_id = self.num_layer
        for i in self.num_layer:
            if z <= self.depth[i]:
                layer_id = i + 1
                break
            else:
                continue
        return layer_id


class hmlayers(Core1D):
    def __init__(self, thicks, hand_system = "left"):
        #親クラスのコンストラクタの継承
        super().__init__(thicks)
        #真空の誘電率
        epsln_0 = 8.85418782 * 1e-12
        self.epsln = np.array([epsln_0 for i in range(num_layer)])
        #真空の透磁率
        mu_0 = 4. * np.pi * 1e-7
        self.mu = np.array([mu_0 for i in range(num_layer)])

    def __str__(self):
        description =(
            'Horizontal Multi-Layered Model \n'
            'ここに使い方の説明を書く'
        )
        return description
    
    ### CHARACTERIZING LAYERS ###
    def set_resistivity(self, res, complex_value = False):
        if type(thicks) != np.ndarray():
            res = np.array(res)
        self.sigma = 1/res

    def set_permittivity(self, epsln):
        return None
    def set_magnetic_permeability(self, mu):
        return None
    def set_cole_cole_model(self, dcres, m, tau, c, omega):
        return None
    def compute_coefficients(self):
        return None
