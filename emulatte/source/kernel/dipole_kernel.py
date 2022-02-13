import numpy as np

def call_kernel_vmd_e(model, lambda_):
    damp_up = model.damp_up
    damp_down = model.damp_down
    exp_up = model.exp_up
    exp_down = model.exp_down
    si = model.si
    ri = model.ri
    su = model.u[:,si]
    sz = model.sz
    rz = model.rz
    kernel = damp_up * exp_up + damp_down * exp_down
    kernel_add = int(si==ri) * np.exp(- su * abs(sz - rz))
    kernel = kernel + kernel_add
    kernel = kernel * lambda_ ** 2 / su
    return kernel

def call_kernel_vmd_h_r(model, lambda_):
    damp_up = model.damp_up
    damp_down = model.damp_down
    exp_up = model.exp_up
    exp_down = model.exp_down
    si = model.si
    ri = model.ri
    su = model.u[:,si]
    ru = model.u[:,ri]
    sz = model.sz
    rz = model.rz
    kernel = damp_up * exp_up - damp_down * exp_down
    kernel_add = int(si==ri) * np.exp(- su * abs(sz - rz))
    kernel_add = kernel_add * np.sign(rz - sz)
    kernel = kernel + kernel_add
    kernel = kernel * lambda_ ** 2 * ru / su
    return kernel

def call_kernel_vmd_h_z(model, lambda_):
    damp_up = model.damp_up
    damp_down = model.damp_down
    exp_up = model.exp_up
    exp_down = model.exp_down
    si = model.si
    ri = model.ri
    su = model.u[:,si]
    sz = model.sz
    rz = model.rz
    kernel = damp_up * exp_up + damp_down * exp_down
    kernel_add = int(si==ri) * np.exp(- su * abs(sz - rz))
    kernel = kernel + kernel_add
    kernel = kernel * lambda_ ** 3 / su
    return kernel