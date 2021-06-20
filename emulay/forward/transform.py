import numpy as np
from emulay.forward import kernels, filters

class HankelTransformFunctions:
    @staticmethod
    def vmd(mdl, td_transform):
        y_base, wt0, wt1 = filters.load_hankel_filter(mdl.hankel_filter)
        mdl.filter_length = len(y_base)
        mdl.lambda_ = y_base/mdl.r[0] #! これだと１対の送受信機にしか対応できない
        kernel = kernels.compute_kernel_vmd(mdl) #引数？
        ans = {}
        e_phi = np.dot(wt1.T, kernel[0]) / mdl.r
        h_r = np.dot(wt1.T, kernel[1]) / mdl.r
        if td_transform == 'eular':
            h_z = np.dot(wt0.T, kernel[2] * (mdl.omega**0.5)) / mdl.r
        else:
            h_z = np.dot(wt0.T, kernel[2]) / mdl.r
        ans["e_x"] = - 1 / (4 * np.pi) * mdl.ztilde[0, mdl.src_layer - 1] * - mdl.sin_phi * e_phi
        ans["e_y"] = - 1 / (4 * np.pi) * mdl.ztilde[0, mdl.src_layer - 1] * mdl.cos_phi * e_phi
        ans["e_z"] = 0
        ans["h_x"] = 1 / (4 * np.pi) * mdl.cos_phi * h_r
        ans["h_y"] = 1 / (4 * np.pi) * mdl.sin_phi * h_r
        ans["h_z"] = mdl.ztilde[0, mdl.src_layer - 1] / mdl.ztilde[0, mdl.rcv_layer - 1] / (4 * np.pi ) * h_z #送受信点が同じ層にいなければならない?
        return ans

    @staticmethod
    def hmdx(mdl):
        ans = {}
        return ans

    @staticmethod
    def hmdy(mdl):
        ans = {}
        return ans
    
    @staticmethod
    def ved(mdl):
        ans = {}
        return ans
    
    @staticmethod
    def hedx(mdl):
        ans = {}
        return ans
    
    @staticmethod
    def hedy(mdl):
        ans = {}
        return ans
    
    @staticmethod
    def circular_loop(mdl):
        ans = {}
        return ans
    
    @staticmethod
    def coincident_loop(mdl):
        ans = {}
        return ans
    
    @staticmethod
    def grounded_wire(mdl):
        ans = {}
        return ans
    
    @staticmethod
    def loop_source(mdl):
        ans = {}
        return ans

    @staticmethod
    def x_line_source(mdl):
        ans = {}
        return ans

    @staticmethod
    def y_line_source(mdl):
        ans = {}
        return ans