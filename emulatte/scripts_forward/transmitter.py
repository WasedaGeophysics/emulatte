import numpy as np
from scipy import interpolate
from emulatte.scripts_forward import transform
class Core:
    def __init__(self, freqtime):
        self.name = self.__class__.__name__.lower()
        self.freqtime = freqtime # Common in FD and TD
        self.omegas = 2 * np.pi * freqtime # Only using in FD
        self.ft_size = len(freqtime)

    def get_result(
            self, model, time_diff=False, td_transform=None):
        """
        Docstring
        """
        #Frequancy Domain
        if model.domain == 'Freq':
            ans = np.zeros((self.ft_size, 6), dtype=complex)
            for index, omega in enumerate(self.omegas):
                em_field = self.hankel_transform(model, omega)
                # Electric fields
                ans[index, 0] = em_field["e_x"]
                ans[index, 1] = em_field["e_y"]
                ans[index, 2] = em_field["e_z"]
                # Magnetic fields
                ans[index, 3] = em_field["h_x"]
                ans[index, 4] = em_field["h_y"]
                ans[index, 5] = em_field["h_z"]
                if time_diff:
                    ans[index, :] = ans[index, :] * 1j * omega       
            ans = self.moment * ans
            ans = {
                "e_x": ans[:, 0], "e_y": ans[:, 1], "e_z": ans[:, 2],
                "h_x": ans[:, 3], "h_y": ans[:, 4], "h_z": ans[:, 5]
                }
            return ans, self.freqtime
        # Time Domain
        elif model.domain == 'Time':
            # Fast Fourier Transform
            if td_transform == 'FFT':
                ans = np.zeros((self.ft_size, 6),dtype=complex)
                nFreqsPerDecade = 1000
                if model.hankel_filter == 'werthmuller201':
                    freq = np.logspace(-6, 8, nFreqsPerDecade)
                elif model.hankel_filter == 'key201':
                    freq = np.logspace(-8, 12, nFreqsPerDecade)
                elif model.hankel_filter == 'anderson801':
                    freq = np.logspace(-21, 21, nFreqsPerDecade)
                else: # 他のフィルタは範囲不明
                    freq = np.logspace(-21, 21, nFreqsPerDecade)
                freq_ans = np.zeros((len(freq),6), dtype=complex)
                omegas = 2 * np.pi * freq
                for index, omega in enumerate(omegas):
                    hankel_result = self.hankel_transform(model, omega)
                    freq_ans[index,0] = hankel_result["e_x"]
                    freq_ans[index,1] = hankel_result["e_y"]
                    freq_ans[index,2] = hankel_result["e_z"]
                    freq_ans[index,3] = hankel_result["h_x"]
                    freq_ans[index,4] = hankel_result["h_y"]
                    freq_ans[index,5] = hankel_result["h_z"]

                f = interpolate.interp1d(
                        2*np.pi*freq, freq_ans.T,
                        kind='cubic', fill_value="extrapolate"
                    )

                for index, time in enumerate(self.freqtime):
                    time_ans = \
                        transform.FourierTransform.fast_fourier_transform(
                            model, f, time, time_diff
                        )
                    ans[index, 0] = time_ans[0]
                    ans[index, 1] = time_ans[1]
                    ans[index, 2] = time_ans[2]
                    ans[index, 3] = time_ans[3]
                    ans[index, 4] = time_ans[4]
                    ans[index, 5] = time_ans[5]
                ans = {
                    "e_x": ans[:, 0], "e_y": ans[:, 1], "e_z": ans[:, 2],
                    "h_x": ans[:, 3], "h_y": ans[:, 4], "h_z": ans[:, 5]
                    }
                return ans, self.freqtime
            # Adaptive Convolution
            elif td_transform == 'DLAG':
                nb = int(
                        np.fix(
                            10 * np.log(self.freqtime[-1] / self.freqtime[0])
                        ) + 1
                     )
                ans = np.zeros((nb, 6), dtype=complex)
                dans = {
                    "e_x": None, "e_y": None, "e_z": None,
                    "h_x": None, "h_y": None, "h_z": None
                    }
                emfield = list(dans.keys())
                if not time_diff:
                    for ii, emfield in enumerate(emfield):
                        time_ans, arg = transform.FourierTransform.dlagf0em(
                            model, nb, emfield
                        )
                        ans[:, ii] = time_ans
                else:
                    for ii, emfield in enumerate(emfield):
                        time_ans, arg = transform.FourierTransform.dlagf1em(
                            model, nb, emfield
                        )
                        ans[:, ii] = time_ans
                ans = - 2 / np.pi * self.moment * ans
                dans["e_x"] = ans[:, 0]
                dans["e_y"] = ans[:, 1]
                dans["e_z"] = ans[:, 2]
                dans["h_x"] = ans[:, 3]
                dans["h_y"] = ans[:, 4]
                dans["h_z"] = ans[:, 5]
                return dans, arg

class VMD(Core):
    """
    Vertical Magnetic Dipole
    Horizontal Co-Planar (HCP __ -> __ )
    """
    def __init__(self, freqtime, dipole_moment):
        super().__init__(freqtime)
        #VMD固有設定値
        self.moment = dipole_moment
        self.num_dipole = 1
        self.kernel_te_up_sign = 1
        self.kernel_te_down_sign = 1
        self.kernel_tm_up_sign = 0
        self.kernel_tm_down_sign = 0
        self.hankel_transform = transform.HankelTransform.vmd
        
class HMDx(Core):
    """
    Horizontal Magnetic Dipole x 
        Vertial Co-axial (VCA  | -> | )
    """
    def __init__(self, freqtime, dipole_moment):
        super().__init__(freqtime)
        self.moment = dipole_moment
        self.num_dipole = 1
        self.kernel_te_up_sign = -1
        self.kernel_te_down_sign = 1
        self.kernel_tm_up_sign = 1
        self.kernel_tm_down_sign = 1
        self.hankel_transform = transform.HankelTransform.hmdx

class HMDy(Core):
    """
    Horizontal Magnetic Dipole y 
        Vertial Co-planar (VCP  o -> o)
    """
    def __init__(self, freqtime, dipole_moment):
        super().__init__(freqtime)
        self.moment = dipole_moment
        self.num_dipole = 1
        self.kernel_te_up_sign = -1
        self.kernel_te_down_sign = 1
        self.kernel_tm_up_sign = 1
        self.kernel_tm_down_sign = 1
        self.hankel_transform = transform.HankelTransform.hmdy

class VED(Core):
    def __init__(self, freqtime, ds, current):
        super().__init__(freqtime)
        self.moment = ds * current
        self.num_dipole = 1
        self.kernel_te_up_sign = 0
        self.kernel_te_down_sign = 0
        self.kernel_tm_up_sign = 1
        self.kernel_tm_down_sign = 1
        self.hankel_transform = transform.HankelTransform.ved

class HEDx(Core):
    def __init__(self, freqtime, ds, current):
        super().__init__(freqtime)
        self.moment = ds * current
        self.num_dipole = 1
        self.kernel_te_up_sign = 1
        self.kernel_te_down_sign = 1
        self.kernel_tm_up_sign = -1
        self.kernel_tm_down_sign = 1
        self.hankel_transform = transform.HankelTransform.hedx
class HEDy(Core):
    def __init__(self, freqtime, ds, current):
        super().__init__(freqtime)
        self.moment = ds * current
        self.num_dipole = 1
        self.kernel_te_up_sign = 1
        self.kernel_te_down_sign = 1
        self.kernel_tm_up_sign = -1
        self.kernel_tm_down_sign = 1
        self.hankel_transform = transform.HankelTransform.hedy

class CircularLoop(Core):
    def __init__(self, freqtime, current, radius, turns):
        super().__init__(freqtime)
        self.current = current
        self.radius = radius
        self.moment = current * turrns
        self.num_dipole = 1
        self.kernel_te_up_sign = 1
        self.kernel_te_down_sign = 1
        self.kernel_tm_up_sign = 0
        self.kernel_tm_down_sign = 0
        self.hankel_transform = transform.HankelTransform.circular_loop

class CoincidentLoop(Core):
    def __init__(self, freqtime, current, radius, turns):
        super().__init__(freqtime)
        self.current = current
        self.radius = radius
        self.moment = current * turns ** 2
        self.num_dipole = 1
        self.kernel_te_up_sign = 1
        self.kernel_te_down_sign = 1
        self.kernel_tm_up_sign = 0
        self.kernel_tm_down_sign = 0
        self.hankel_transform = transform.HankelTransform.coincident_loop

class _GroundedWire(Core):
    def __init__(self, freqtime, current):
        super().__init__(freqtime)
        self.current = current
        self.moment = current
        self.kernel_te_up_sign = 1
        self.kernel_te_down_sign = 1
        self.kernel_tm_up_sign = -1
        self.kernel_tm_down_sign = 1
        self.hankel_transform = transform.HankelTransform.grounded_wire

class _LoopSource(Core):
    def __init__(self, freqtime, current, turns):
        super().__init__(freqtime)
        self.current = current
        self.turns = turns
        self.moment = current
        self.kernel_te_up_sign = 1
        self.kernel_te_down_sign = 1
        self.kernel_tm_up_sign = -1
        self.kernel_tm_down_sign = 1
        self.hankel_transform = transform.HankelTransform.loop_source








