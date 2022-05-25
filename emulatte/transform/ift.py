import numpy as np
import scipy
from scipy import interpolate
import scipy.signal as ss
from ..dlf.loader import load_sin_cos_filter

# dlf
def get_frequency_for_dlf(time_min, time_max, ft_config):
    sampling_method = ft_config["sampling"]
    if ft_config["_user_def"]:
        phase_base = ft_config["phase"]
    else:
        filter_name = ft_config["filter"]
        phase_base, _, _ = load_sin_cos_filter(filter_name)

    if sampling_method == "fdri":
        #TODO optimize this
        pts_per_decade = 30
        logfmin = int(np.log10(phase_base[0] / time_max)) - 2
        logfmax = int(np.log10(phase_base[-1] / time_min)) + 1
        freq_size = (logfmax - logfmin) * pts_per_decade
        omega = np.logspace(logfmin, logfmax, freq_size)
        freq = omega / (2 * np.pi)
        time_resample = None

    elif sampling_method == "lagged":
        # 10th root of eular number
        common = phase_base[1] / phase_base[0]

        # times from the start point to the end point passage
        etime_size = int(10 * np.log(time_max / time_min)) + 2
        etime_init = np.ones(etime_size) * time_min
        etime_base = np.ones(etime_size) * common
        etime_expo = np.arange(etime_size)
        etime = etime_init * etime_base ** etime_expo

        # phase = omega * t
        PHASE0 = phase_base[0]
        phase_size = len(phase_base)
        phase_init = np.ones(phase_size) * PHASE0
        phase_base = np.ones(phase_size) * common
        phase_expo = np.arange(phase_size)
        phase = phase_init * phase_base ** phase_expo

        omega_1 = phase[0] / etime
        omega_1 = omega_1[::-1]
        omega_2 = phase / etime[0]
        omega = np.concatenate([omega_1[:-1], omega_2])
        freq = omega / (2 * np.pi)
        time_resample = etime

    return freq, time_resample


def lagged_convolution(model, which, direction, time, signal):
    """Summary
    The common ratio of base array {ωt} must be e^0.1 in order to perform lagged convolution.
    """
    base, cos, sin = load_sin_cos_filter('anderson_time_787')
    # 10th root of eular number
    XRE = 1.1051709180756477124
    #   = np.exp(np.float128(1.)) ** np.float128(0.1)
    #   ~ 1.10517091807564762 (anderson original)

    # times from the start point to the end point passage
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

    fd_ans = model._em_field_f(which, direction, omega)

    if model.time_derivative:
        sin_mode = True
        if signal == 'stepoff':
            kernel_cell = 2 / np.pi * fd_ans.imag
        elif signal == 'impulse':
            kernel_cell = - 2 / np.pi * fd_ans.imag * omega
            sin_mode = False
        elif signal == 'stepon':
            kernel_cell = - 2 / np.pi * fd_ans.imag
        else:
            raise Exception
    else:
        sin_mode = False
        if signal == 'stepoff':
            kernel_cell = - 2 / np.pi * fd_ans.imag / omega
        elif signal == 'impulse':
            kernel_cell = 2 / np.pi * fd_ans.imag
            sin_mode = True
        elif signal == 'stepon':
            kernel_cell = 2 / np.pi * fd_ans.imag / omega
        else:
            raise Exception

    ndir = len(direction)

    freq = omega / 2 / np.pi
    # TODO cut_off depends on hankel filter
    cut_off = 1e5
    b, a = ss.iirfilter(1, cut_off, btype='lowpass', analog=True, ftype='butter')
    w, h = ss.freqs(b, a, freq)
    #kernel_cell = (kernel_cell * h).real
    kernel_cell = kernel_cell.real

    kernel = np.zeros((ndir, etime_size, phase_size), dtype=float)
    if ndir == 1:
        for i in range(etime_size):
            cut = (etime_size - 1 - i, etime_size + phase_size - 1 - i)
            kernel[:, i] = kernel_cell[cut[0]:cut[1]]
    else:
        for i in range(etime_size):
            cut = (etime_size - 1 - i, etime_size + phase_size - 1 - i)
            kernel[:, i] = kernel_cell[:, cut[0]:cut[1]]
    
    if not sin_mode:
        eans = kernel @ cos / etime
    else:
        eans = kernel @ sin / etime

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

def fdrift(model, which, direction, time, signal):

    fft_filter = model.fft_filter
    time_size = len(time)
    #TODO optimize this
    pts_per_decade = 30
    base, cos, sin = load_sin_cos_filter(model.fft_filter)
    logfmin = int(np.log10(base[0] / time[-1])) - 2
    logfmax = int(np.log10(base[-1] / time[0])) + 1
    freq_size = (logfmax - logfmin) * pts_per_decade
    freq = np.logspace(logfmin, logfmax, freq_size)

    omega = 2 * np.pi * freq
    fd_ans = model._em_field_f(which, direction, omega)
    interp_fd = interpolate.interp1d(omega, fd_ans.T, kind='cubic')

    ndir = len(direction)
    filter_size = len(base)
    kernel = np.zeros((ndir, time_size, filter_size), dtype=float)

    for i in range(time_size):
        omega_base = base / time[i]
        interp_fd_ans = interp_fd(omega_base)
        if model.time_derivative:
            sin_mode = True
            if signal == 'stepoff':
                kernel_cell = 2 / np.pi * interp_fd_ans.imag
            elif signal == 'impulse':
                kernel_cell = - 2 / np.pi * interp_fd_ans.imag * omega_base
            elif signal == 'stepon':
                kernel_cell = - 2 / np.pi * interp_fd_ans.imag
            else:
                raise Exception
        else:
            sin_mode = False
            if signal == 'stepoff':
                kernel_cell = - 2 / np.pi * interp_fd_ans.imag / omega_base
            elif signal == 'impulse':
                kernel_cell = 2 / np.pi * interp_fd_ans.imag
                sin_mode = True
            elif signal == 'stepon':
                kernel_cell = 2 / np.pi * interp_fd_ans.imag / omega_base
            else:
                raise Exception

        kernel[:, i] = kernel_cell[:]

    if not sin_mode:
        ans = kernel @ cos / time
    else:
        ans = kernel @ sin / time

    if ndir == 1:
        ans = ans[0]
    return ans