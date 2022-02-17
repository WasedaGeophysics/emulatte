import numpy as np

def call_kernel_vmd_e(model, lambda_):
    u_te = model.u_te
    d_te = model.d_te
    e_up = model.e_up
    e_down = model.e_down
    si = model.si
    ri = model.ri
    su = model.u[:,si]
    sz = model.sz
    rz = model.rz
    kernel = u_te * e_up + d_te * e_down
    kernel_add = int(si==ri) * np.exp(- su * abs(sz - rz))
    kernel = kernel + kernel_add
    kernel = kernel * lambda_ ** 2 / su
    return kernel

def call_kernel_vmd_h_r(model, lambda_):
    u_te = model.u_te
    d_te = model.d_te
    e_up = model.e_up
    e_down = model.e_down
    si = model.si
    ri = model.ri
    su = model.u[:,si]
    ru = model.u[:,ri]
    sz = model.sz
    rz = model.rz
    kernel = u_te * e_up - d_te * e_down
    kernel_add = int(si==ri) * np.exp(- su * abs(sz - rz))
    kernel_add = kernel_add * np.sign(rz - sz)
    kernel = kernel + kernel_add
    kernel = kernel * lambda_ ** 2 * ru / su
    return kernel

def call_kernel_vmd_h_z(model, lambda_):
    u_te = model.u_te
    d_te = model.d_te
    e_up = model.e_up
    e_down = model.e_down
    si = model.si
    ri = model.ri
    su = model.u[:,si]
    sz = model.sz
    rz = model.rz
    kernel = u_te * e_up + d_te * e_down
    kernel_add = int(si==ri) * np.exp(- su * abs(sz - rz))
    kernel = kernel + kernel_add
    kernel = kernel * lambda_ ** 3 / su
    return kernel