import numpy as np
import scipy
from scipy import interpolate
import scipy.signal as ss

# dlf
def get_freqtime_dlf(time, phase_base, boosting):
    time_max = time[-1]
    time_min = time[0]

    if boosting == "fdri":
        #TODO optimize this
        pts_per_decade = 30
        logfmin = int(np.log10(phase_base[0] / time_max)) - 2
        logfmax = int(np.log10(phase_base[-1] / time_min)) + 1
        freq_size = (logfmax - logfmin) * pts_per_decade
        omega = np.logspace(logfmin, logfmax, freq_size)
        freq = omega / (2 * np.pi)
        time_resampled = time

    elif boosting == "lag":
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
        time_resampled = etime

    return freq, time_resampled

def make_matrix_dlf(emf_fd, time, frequency, phase_base, boosting, ndirection):
    phase_size = len(phase_base)
    ntime = len(time)
    if boosting == "lag":
        kernel_matrix = np.zeros((ndirection, ntime, phase_size), dtype=float)
        tol = 1e-12
        for dd in range(ndirection):
            for i in range(ntime):
                slc = (ntime - 1 - i, ntime + phase_size - 1 - i)
                lagged_emf = emf_fd[dd, slc[0]:slc[1]]
                # adaptive convolution
                abs_max = abs(lagged_emf).max()
                cut_off_window = abs(lagged_emf) > abs_max * tol
                lagged_emf = lagged_emf * cut_off_window
                kernel_matrix[dd, i] = lagged_emf

    elif boosting == "fdri":
        omega = 2 * np.pi * frequency
        kernel_matrix = np.zeros((ndirection, ntime, phase_size), dtype=float)
        for dd in range(ndirection):
            interp_fd = interpolate.interp1d(omega, emf_fd[dd], kind='cubic')
            for i in range(ntime):
                omega_base = phase_base / time[i]
                interp_fd_ans = interp_fd(omega_base)
                kernel_matrix[dd, i] = interp_fd_ans[:]

    return kernel_matrix

def compute_fdwave(ontime, waveform, frequency):

    waveform_func = interpolate.interp1d(ontime, waveform, kind='linear')
    a = ontime[0]
    b = ontime[1]

    def integrant

