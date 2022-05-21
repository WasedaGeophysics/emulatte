import numpy as np

# VED ========================================================================\

def compute_kernel_ved_e_phi(
        u_tm, d_tm, e_up, e_down, si, ri, us, ur, zs, z, lambda_):
    kernel = - u_tm * e_up + d_tm * e_down
    kernel_add = - int(si==ri) * np.sign(z-zs) * np.exp(-us * abs(z - zs))
    kernel = kernel + kernel_add
    kernel = kernel * lambda_ ** 2 * ur / us
    return kernel

def compute_kernel_ved_e_z(
        u_tm, d_tm, e_up, e_down, si, ri, us, ur, zs, z, lambda_):
    kernel = u_tm * e_up + d_tm * e_down
    kernel_add = int(si==ri) * np.exp(-us * abs(z - zs))
    kernel = kernel + kernel_add
    kernel = kernel * lambda_ ** 3 / us
    return kernel

def compute_kernel_ved_h_r(
        u_tm, d_tm, e_up, e_down, si, ri, us, ur, zs, z, lambda_):
    kernel = u_tm * e_up + d_tm * e_down
    kernel_add = int(si==ri) * np.exp(-us * abs(z - zs))
    kernel = kernel + kernel_add
    kernel = kernel * lambda_ ** 2 / us
    return kernel

# HED ========================================================================\

def compute_kernel_hed_e_r_tm(
        u_tm, d_tm, e_up, e_down, si, ri, us, ur, zs, z, lambda_):
    kernel = -u_tm * e_up + d_tm * e_down
    kernel_add = - int(si==ri) * np.exp(- us * abs(z - zs))
    kernel = kernel + kernel_add
    kernel = kernel * ur
    kernel0 = kernel * lambda_
    kernel1 = kernel
    kernel = [kernel0, kernel1]
    return kernel

def compute_kernel_hed_e_r_te(
        u_te, d_te, e_up, e_down, si, ri, us, ur, zs, z, lambda_):
    kernel = u_te * e_up + d_te * e_down
    kernel_add = int(si==ri) * np.exp(- us * abs(z - zs))
    kernel = kernel + kernel_add
    kernel = kernel / us
    kernel0 = kernel * lambda_
    kernel1 = kernel
    kernel = [kernel0, kernel1]
    return kernel

def compute_kernel_hed_e_z_tm(
        u_tm, d_tm, e_up, e_down, si, ri, us, ur, zs, z, lambda_):
    kernel = u_tm * e_up + d_tm * e_down
    # TODO w1demの仕様をそのまま移植した場合（なぜ？）
    # kernel_add = (1 - int(zs - 1e-2 == z)) * int(si==ri) * np.exp(- us * abs(z - zs))
    # kernel_add = kernel_add * np.sign(z - zs)
    kernel_add = int(si==ri) * np.sign(z - zs) * np.exp(- us * abs(z - zs))
    kernel = kernel + kernel_add
    kernel = kernel * lambda_ ** 2
    return kernel

def compute_kernel_hed_h_r_tm(
        u_tm, d_tm, e_up, e_down, si, ri, us, ur, zs, z, lambda_):
    kernel = u_tm * e_up + d_tm * e_down
    kernel_add = int(si==ri) * np.exp(- us * abs(z - zs)) * np.sign(z - zs)
    kernel = kernel + kernel_add
    kernel0 = kernel * lambda_
    kernel1 = kernel
    kernel = [kernel0, kernel1]
    return kernel

def compute_kernel_hed_h_r_te(
        u_te, d_te, e_up, e_down, si, ri, us, ur, zs, z, lambda_):
    kernel = -u_te * e_up + d_te * e_down
    kernel_add = - int(si==ri) * np.exp(- us * abs(z - zs)) * np.sign(z - zs)
    kernel = kernel + kernel_add
    kernel = kernel * ur / us
    kernel0 = kernel * lambda_
    kernel1 = kernel
    kernel = [kernel0, kernel1]
    return kernel

def compute_kernel_hed_h_z_te(
        u_te, d_te, e_up, e_down, si, ri, us, ur, zs, z, lambda_):
    kernel = u_te * e_up + d_te * e_down
    kernel_add = int(si==ri) * np.exp(- us * abs(z - zs))
    kernel = kernel + kernel_add
    kernel = kernel * lambda_ ** 2 / us
    return kernel