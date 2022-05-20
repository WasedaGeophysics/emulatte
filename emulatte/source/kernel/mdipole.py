import numpy as np

def compute_kernel_vmd_er(
        u_te, d_te, e_up, e_down, si, ri, us, zs, z, lambda_):
    kernel = u_te * e_up + d_te * e_down
    kernel_add = int(si==ri) * np.exp(- us * abs(z - zs))
    kernel = kernel + kernel_add
    kernel = kernel * lambda_ ** 2 / us
    return kernel

def compute_kernel_vmd_hr(
        u_te, d_te, e_up, e_down, si, ri, us, ur, zs, z, lambda_):
    kernel = u_te * e_up - d_te * e_down
    kernel_add = int(si==ri) * np.sign(z - zs) * np.exp(- us * abs(z - zs))
    kernel = kernel + kernel_add
    kernel = kernel * lambda_ ** 2 * ur / us
    return kernel

def compute_kernel_vmd_hz(
        u_te, d_te, e_up, e_down, si, ri, us, zs, z, lambda_):
    kernel = u_te * e_up + d_te * e_down
    kernel_add = int(si==ri) * np.exp(- us * abs(z - zs))
    kernel = kernel + kernel_add
    kernel = kernel * lambda_ ** 3 / us
    return kernel