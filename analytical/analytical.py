import numpy as np
from scipy.special import erf
from scipy.constants import mu_0


class VMD:
    def __init__(self, res, r):
        self.res = res
        self.r = r

    def fd_hz(self, freq, m=1.):
        omega = freq * 2 * np.pi
        k = np.sqrt(-1j * mu_0 * (1 / self.res) * omega)
        Hz = m / (2 * np.pi * k**2 * self.r**5)
        Hz *= (9 - (9 + 9 * 1j * k * self.r - 4 * k**2 * self.r**2 - 1j * k**3 * self.r**3) * np.exp(-1j * k * self.r))
        return Hz

    def td_hz(self, time, m=1.):
        r"""Return equation 4.69a, Ward and Hohmann, 1988.

        Switch-off response (i.e., Hz(t)) of a homogeneous isotropic half-space,
        where the vertical magnetic source and receiver are at the interface.

        Parameters
        ----------
        time : array
            Times (t)
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
        theta = np.sqrt(mu_0 / (4 * self.res * time))
        theta_r = theta * self.r

        hz = -(9 / theta_r + 4 * theta_r) * np.exp(-theta_r**2) / np.sqrt(np.pi)
        hz += erf(theta_r) * (9 / (2 * theta_r**2) - 1)
        hz *= m / (4 * np.pi * self.r**3)
        return hz

    def td_dhzdt(self, time, m=1.):
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
        theta = np.sqrt(mu_0 / (4 * self.res * time))
        theta_r = theta * self.r

        dhzdt = (9 + 6 * theta_r**2 + 4 * theta_r**4) * np.exp(-theta_r**2)
        dhzdt *= -2 * theta_r / np.sqrt(np.pi)
        dhzdt += 9 * erf(theta_r)
        dhzdt *= -(m * self.res) / (2 * np.pi * mu_0 * self.r**5)
        return dhzdt


class CL:
    def __init__(self, res, radius):
        self.res = res
        self.radius = radius

    def fd_hz(self, freq, current=1.):
        omega = freq * 2 * np.pi
        k = np.sqrt(-1j * mu_0 * (1 / self.res) * omega)
        Hz = - current / (k**2 * self.radius**3)
        Hz *= (3 - (3 + 3j * k * self.radius - k**2 * self.radius**2) * np.exp(-1j * k * self.radius))
        return Hz

    def td_hz(self, time, current=1.):
        theta = np.sqrt(mu_0 / (4 * self.res * time))
        theta_r = theta * self.radius

        hz = current / (2 * self.radius)
        hz *= (3 / (np.sqrt(np.pi) * theta_r) * np.exp(-theta_r**2) + (1 - 3 / (2 * theta_r**2)) * erf(theta_r))
        return hz

    def td_dhzdt(self, time, current=1.):
        theta = np.sqrt(mu_0 / (4 * self.res * time))
        theta_r = theta * self.radius

        dhzdt = - current / (mu_0 * self.res**(-1) * self.radius**3)
        dhzdt *= (3 * erf(theta_r) - 2 / np.sqrt(np.pi) * theta_r * (3 + 2 * theta_r**2) * np.exp(-theta_r**2))
        return dhzdt
