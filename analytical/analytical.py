import numpy as np
from scipy.special import erf
from scipy.constants import mu_0


class VMD:
    def __init__(self):
        pass

    def fd_hz(self, res, r, freq, m=1.):
        omega = freq * 2 * np.pi
        k = np.sqrt(-1j * mu_0 * (1 / res) * omega)
        analytical_ans = m / (2 * np.pi * k**2 * r**5) \
            * (9 - (9 + 9 * 1j * k * r - 4 * k**2 * r**2 - 1j * k**3 * r**3)
               * np.exp(-1j * k * r))
        return analytical_ans

    def td_hz(self, res, r, time, m=1.):
        r"""Return equation 4.69a, Ward and Hohmann, 1988.

        Switch-off response (i.e., Hz(t)) of a homogeneous isotropic half-space,
        where the vertical magnetic source and receiver are at the interface.

        Parameters
        ----------
        time : array
            Times (t)
        res : float
            Halfspace resistivity (Ohm.m)
        r : float
            Offset (m)
        m : float, optional
            Magnetic moment, default is 1.

        Returns
        -------
        hz : array
            Vertical magnetic field (A/m)

        """
        theta = np.sqrt(mu_0 / (4 * res * time))
        theta_r = theta * r

        analytical_ans = -(9 / theta_r + 4 * theta_r) * np.exp(-theta_r**2) / np.sqrt(np.pi)
        analytical_ans += erf(theta_r) * (9 / (2 * theta_r**2) - 1)
        analytical_ans *= m / (4 * np.pi * r**3)

        return analytical_ans

    def td_dhzdt(self, res, r, time, m=1.):
        r"""Return equation 4.70, Ward and Hohmann, 1988.

        Impulse response (i.e., dHz(t)/dt) of a homogeneous isotropic half-space,
        where the vertical magnetic source and receiver are at the interface.

        Parameters
        ----------
        time : array
            Times (t)
        r : float
            Offset (m)
        res : float
            Halfspace resistivity (Ohm.m)
        m : float, optional
            Magnetic moment, default is 1.

        Returns
        -------
        dhz : array
            Time-derivative of the vertical magnetic field (A/m/s)

        """
        theta = np.sqrt(mu_0 / (4 * res * time))
        theta_r = theta * r

        analytical_ans = (9 + 6 * theta_r**2 + 4 * theta_r**4) * np.exp(-theta_r**2)
        analytical_ans *= -2 * theta_r / np.sqrt(np.pi)
        analytical_ans += 9 * erf(theta_r)
        analytical_ans *= -(m * res) / (2 * np.pi * mu_0 * r**5)

        return analytical_ans
