import numpy as np
from scipy.special import j0, j1

def compute_kernel_vmd_e_r(
        u_te, d_te, e_up, e_down, si, ri, us, zs, z, lambda_):
    kernel = u_te * e_up + d_te * e_down
    kernel_add = int(si==ri) * np.exp(- us * abs(z - zs))
    kernel = kernel + kernel_add
    kernel = kernel * lambda_ ** 2 / us
    return kernel

def compute_kernel_vmd_h_r(
        u_te, d_te, e_up, e_down, si, ri, us, ur, zs, z, lambda_):
    kernel = u_te * e_up - d_te * e_down
    kernel_add = int(si==ri) * np.sign(z - zs) * np.exp(- us * abs(z - zs))
    kernel = kernel + kernel_add
    kernel = kernel * lambda_ ** 2 * ur / us
    return kernel

def compute_kernel_vmd_h_z(
        u_te, d_te, e_up, e_down, si, ri, us, zs, z, lambda_):
    kernel = u_te * e_up + d_te * e_down
    kernel_add = int(si==ri) * np.exp(- us * abs(z - zs))
    kernel = kernel + kernel_add
    kernel = kernel * lambda_ ** 3 / us
    return kernel

def compute_kernel_loop_e_r(
        u_te, d_te, e_up, e_down, si, ri, us, zs, z, lambda_, rho):
    kernel = u_te * e_up + d_te * e_down
    kernel_add = int(si==ri) * np.exp(- us * abs(z - zs))
    kernel = kernel + kernel_add
    bessel = j1(lambda_ * rho)
    kernel = kernel * lambda_ * lambda_ / us * bessel
    return kernel

def compute_kernel_loop_h_r(
        u_te, d_te, e_up, e_down, si, ri, us, ur, zs, z, lambda_, rho):
    kernel = -u_te * e_up + d_te * e_down
    kernel_add = - int(si==ri) * np.exp(- us * abs(z - zs))
    kernel_add = kernel_add * np.sign(z - zs)
    kernel = kernel + kernel_add
    bessel = j1(lambda_ * rho)
    kernel = kernel * lambda_ * ur / us * bessel
    return kernel

def compute_kernel_loop_h_z(
        u_te, d_te, e_up, e_down, si, ri, us, zs, z, lambda_, radius):
    kernel = u_te * e_up + d_te * e_down
    kernel_add = int(si==ri) * np.exp(- us * abs(z - zs))
    kernel = kernel + kernel_add
    bessel = j1(lambda_ * radius)
    kernel = kernel * lambda_ * lambda_ / us * bessel
    return kernel

