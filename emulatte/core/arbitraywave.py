import matplotlib.pyplot as plt
import numpy as np
from scipy.special import roots_legendre
from scipy.interpolate import InterpolatedUnivariateSpline as iuSpline  # スプライン補間
from scipy.interpolate import pchip  # 区分的3次エルミート補間

# for ver.0
import empymod
import emulatte.forward as fwd


class ArbitraryWave:
    @classmethod
    def walktem(cls, off_time, waveform_time, waveform_current, model):
        # get_time
        time = cls.get_time(off_time, waveform_time)
        # get_freq
        time, freq, ft, ftarg = empymod.utils.check_time(
            time=time,
            signal=1,
            ft='dlf',
            ftarg={'dlf': 'key_601_CosSin_2009'},
            verb=2
        )
        # emulatte
        tc = [0, 0, 0]  # Transmitter Coordinate (x, y, z)
        rc = [0, 0, 0]  # Receiver Coordinate (x, y, z)
        cl = fwd.transmitter('CircularLoop', freq, current=1, radius=40 / np.sqrt(np.pi), turns=1)
        model.locate(cl, tc, rc)  # 送受信機の設置

        # ===周波数領域===
        # TODO: ハンケルフィルターの考察
        em, _ = model.emulate(hankel_filter='werthmuller201')  # 実行
        hz = em['h_z']

        # ===計算結果をH->B, B->dB/dtに===
        dbdt = np.array(2j * np.pi * freq * hz * 4e-7 * np.pi)

        # ===Butterworth Filter===
        # TODO: ローパスフィルタの考察
        cutoff_freq = 4.5e5  # As stated in the WalkTEM manual
        h = cls.butterworth_type_filter(freq, cutoff_freq, order=1)
        dbdt_filter = dbdt * h

        # ===時間領域へ変換===
        dbdt_filter_td = empymod.model.tem(dbdt_filter[:, None], np.array([1]),
                                           freq, time, signal=1, ft=ft, ftarg=ftarg)

        # ===任意波形へ===
        walktem_em = cls.apply_waveform(time, dbdt_filter_td[0], off_time, waveform_time, waveform_current)
        return walktem_em

    @classmethod
    def get_time(cls, time, r_time):
        """Additional time for ramp.

            Because of the arbitrary waveform, we need to compute some times before and
            after the actually wanted times for interpolation of the waveform.

            Some implementation details: The actual times here don't really matter. We
            create a vector of time.size+2, so it is similar to the input times and
            accounts that it will require a bit earlier and a bit later times. Really
            important are only the minimum and maximum times. The Fourier DLF, with
            `pts_per_dec=-1`, computes times from minimum to at least the maximum,
            where the actual spacing is defined by the filter spacing. It subsequently
            interpolates to the wanted times. Afterwards, we interpolate those again to
            compute the actual waveform response.

            Note: We could first call `waveform`, and get the actually required times
                  from there. This would make this function obsolete. It would also
                  avoid the double interpolation, first in `empymod.model.time` for the
                  Fourier DLF with `pts_per_dec=-1`, and second in `waveform`. Doable.
                  Probably not or marginally faster. And the code would become much
                  less readable.

            Parameters
            ----------
            time : ndarray
                Desired times

            r_time : ndarray
                Waveform times

            Returns
            -------
            time_req : ndarray
                Required times
            """
        tmin = np.log10(max(time.min() - r_time.max(), 1e-10))
        tmax = np.log10(time.max() - r_time.min())
        return np.logspace(tmin, tmax, time.size + 2)

    @classmethod
    def apply_waveform(cls, times, resp, times_wanted, wave_time, wave_amp, nquad=3, method='spline'):
        """Apply a source waveform to the signal.

        Parameters
        ----------
        times : ndarray
            Times of computed input response; should start before and end after
            `times_wanted`.

        resp : ndarray
            EM-response corresponding to `times`.

        times_wanted : ndarray
            Wanted times.

        wave_time : ndarray
            Time steps of the wave.

        wave_amp : ndarray
            Amplitudes of the wave corresponding to `wave_time`, usually
            in the range of [0, 1].

        nquad : int
            Number of Gauss-Legendre points for the integration. Default is 3.

        method : str
            Method of complement response for Approximate forward calculation. Default is spline.

        Returns
        -------
        resp_wanted : ndarray
            EM field for `times_wanted`.

        """

        # Interpolate on log.
        if method == 'spline':
            PP = iuSpline(np.log10(times), resp)
        elif method == 'pchip':
            PP = pchip(np.log10(times), resp.reshape(-1))

        # Wave time steps.
        dt = np.diff(wave_time)
        dI = np.diff(wave_amp)
        dIdt = dI / dt

        # Gauss-Legendre Quadrature; 3 is generally good enough.
        # (Roots/weights could be cached.)
        g_x, g_w = roots_legendre(nquad)

        # Pre-allocate output.
        resp_wanted = np.zeros_like(times_wanted)

        # Loop over wave segments.
        for i, cdIdt in enumerate(dIdt):

            # We only have to consider segments with a change of current.
            if cdIdt == 0.0:
                continue

            # If wanted time is before a wave element, ignore it.
            ind_a = wave_time[i] < times_wanted
            if ind_a.sum() == 0:
                continue

            # If wanted time is within a wave element, we cut the element.
            ind_b = wave_time[i + 1] > times_wanted[ind_a]

            # Start and end for this wave-segment for all times.
            ta = times_wanted[ind_a] - wave_time[i]
            tb = times_wanted[ind_a] - wave_time[i + 1]
            tb[ind_b] = 0.0  # Cut elements

            # Gauss-Legendre for this wave segment. See
            # https://en.wikipedia.org/wiki/Gaussian_quadrature#Change_of_interval
            # for the change of interval, which makes this a bit more complex.
            logt = np.log10(np.outer((tb - ta) / 2, g_x) + (ta + tb)[:, None] / 2)
            fact = (tb - ta) / 2 * cdIdt
            resp_wanted[ind_a] += fact * np.sum(np.array(PP(logt) * g_w), axis=1)

        return resp_wanted

    # 引用元
    # https://github.com/simpeg/simpegEM1D/blob/master/simpegEM1D/Waveforms.py

    @classmethod
    def butterworth_type_filter(cls, frequency, highcut_frequency, order=2):
        """
        Butterworth low pass filter
        Parameters
        ----------
        highcut_frequency: float
            high-cut frequency for the low pass filter
        fs: float
            sampling rate, 1./ dt, (default = 1MHz)
        period:
            period of the signal (e.g. 25Hz base frequency, 0.04s)
        order: int
            The order of the butterworth filter
        Returns
        -------
        frequency, h: ndarray, ndarray
            Filter values (`h`) at frequencies (`frequency`) are provided.
        """

        # Nyquist frequency
        h = 1. / (1 + 1j * (frequency / highcut_frequency)) ** order
        highcut_frequency = 300 * 1e3
        h *= 1. / (1 + 1j * (frequency / highcut_frequency)) ** 1
        return np.array(h)
