import numpy as np
import scipy
import scipy.signal as ss
from .filter import load_fft_filter
from ..utils.converter import array

def lagged_convolution(model, em, direction, time, signal, time_diff):
    base, cos, sin = load_fft_filter('anderson_sin_cos_filter_787')
    # etime
    XRE = 1.1051709180756477124
    #   = np.exp(np.float128(1.)) ** np.float128(0.1)
    #   ~ 1.10517091807564762 (anderson original)
    # cos filterの精度は小数第15位までなので少し桁が多い
    etime_size = int(10 * np.log(time[-1] / time[0])) + 2
    etime_init = np.ones(etime_size) * time[0]
    etime_base = np.ones(etime_size) * XRE
    etime_expo = np.arange(etime_size)
    etime = etime_init * etime_base ** etime_expo

    # phase = omega * t
    PHASE0 = 3.031594406899361e-19
    phase_size = len(cos)
    phase_init = np.ones(phase_size) * PHASE0
    phase_base = np.ones(phase_size) * XRE
    phase_expo = np.arange(phase_size)
    phase = phase_init * phase_base ** phase_expo

    omega_1 = phase[0] / etime
    omega_1 = omega_1[::-1]
    omega_2 = phase / etime[0]
    omega = np.concatenate([omega_1[:-1], omega_2])

    if em == 'e':
        fd_ans = model._electric_field_f(direction, omega)
    elif em == 'm':
        fd_ans = model._magnetic_field_f(direction, omega)
    else:
        raise Exception

    if signal == 'stepoff':
        kernel_cell = - 2 / np.pi * fd_ans.imag / omega
    elif signal == 'impulse':
        kernel_cell = - 2 / np.pi * fd_ans.real
    elif signal == 'stepon':
        kernel_cell = - 2 / np.pi * (fd_ans / (1j * omega)).imag
    else:
        raise Exception

    ndir = len(direction)

    freq = omega / 2 / np.pi
    # TODO cut_off depends on hankel filter
    cut_off = 5e5
    b, a = ss.iirfilter(1, cut_off, btype='lowpass', analog=True, ftype='butter')
    w, h = ss.freqs(b, a, freq)
    kernel_cell = kernel_cell * h

    kernel = np.zeros((ndir, etime_size, phase_size))
    for i in range(etime_size):
        cut = (etime_size - 1 - i, etime_size + phase_size - 1 - i)
        kernel[:, i] = kernel_cell[cut[0]:cut[1]]
    
    if signal in {'stepoff', 'impulse'}:
        eans = kernel @ cos / etime
    elif signal in {'stepon'}:
        eans = kernel @ sin / etime
    else:
        raise Exception

    # interpolation
    ans = []
    for i in range(ndir):
        get_field_along = scipy.interpolate.interp1d(np.log(etime), eans[i], kind='cubic')
        field = get_field_along(np.log(time))
        ans.append(field)
    ans = np.array(ans)
    if ndir == 1:
        ans = ans[0]
    return ans