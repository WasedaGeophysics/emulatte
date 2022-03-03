import numpy as np

def _calc_kernel_hed_tm_er(model):
    u_tm = model.u_tm
    d_tm = model.d_tm
    e_up = model.e_up
    e_down = model.e_down
    si = model.si
    ri = model.ri
    su = model.u[:,:,si]
    ru = model.u[:,:,ri]
    sz = model.sz
    rz = model.rz
    kernel = -u_tm * e_up + d_tm * e_down
    kernel_add = - int(si==ri) * np.exp(- su * abs(rz - sz))
    kernel = kernel + kernel_add
    kernel = kernel * ru
    return kernel

def _calc_kernel_hed_te_er(model):
    u_te = model.u_te
    d_te = model.d_te
    e_up = model.e_up
    e_down = model.e_down
    si = model.si
    ri = model.ri
    su = model.u[:,:,si]
    sz = model.sz
    rz = model.rz
    kernel = u_te * e_up + d_te * e_down
    kernel_add = int(si==ri) * np.exp(- su * abs(rz - sz))
    kernel = kernel + kernel_add
    kernel = kernel / su
    return kernel

def _calc_kernel_hed_tm_ez(model):
    u_tm = model.u_tm
    d_tm = model.d_tm
    e_up = model.e_up
    e_down = model.e_down
    si = model.si
    ri = model.ri
    su = model.u[:,:,si]
    ru = model.u[:,:,ri]
    sz = model.sz
    rz = model.rz
    kernel = u_tm * e_up + d_tm * e_down
    kernel_add = (1 - int(sz - 1e-2 == rz)) * int(si==ri) * np.exp(- su * abs(rz - sz))
    kernel_add = kernel_add * np.sign(rz - sz)
    kernel = kernel + kernel_add
    return kernel

def _calc_kernel_hed_tm_hr(model):
    u_tm = model.u_tm
    d_tm = model.d_tm
    e_up = model.e_up
    e_down = model.e_down
    si = model.si
    ri = model.ri
    su = model.u[:,:,si]
    ru = model.u[:,:,ri]
    sz = model.sz
    rz = model.rz
    kernel = u_tm * e_up + d_tm * e_down
    kernel_add = int(si==ri) * np.exp(- su * abs(rz - sz)) * np.sign(rz - sz)
    kernel = kernel + kernel_add
    return kernel

def _calc_kernel_hed_te_hr(model):
    u_te = model.u_te
    d_te = model.d_te
    e_up = model.e_up
    e_down = model.e_down
    si = model.si
    ri = model.ri
    su = model.u[:,:,si]
    ru = model.u[:,:,ri]
    sz = model.sz
    rz = model.rz
    kernel = -u_te * e_up + d_te * e_down
    kernel_add = - int(si==ri) * np.exp(- su * abs(rz - sz)) * np.sign(rz - sz)
    kernel = kernel + kernel_add
    kernel = kernel * ru / su
    return kernel

def _calc_kernel_hed_te_hz(model):
    u_te = model.u_te
    d_te = model.d_te
    e_up = model.e_up
    e_down = model.e_down
    si = model.si
    ri = model.ri
    su = model.u[:,:,si]
    sz = model.sz
    rz = model.rz
    kernel = u_te * e_up + d_te * e_down
    kernel_add = int(si==ri) * np.exp(- su * abs(rz - sz))
    kernel = kernel + kernel_add
    kernel = kernel / su
    return kernel