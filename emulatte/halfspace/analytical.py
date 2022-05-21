import numpy as np
from scipy.special import erf, iv, kv
from scipy.constants import mu_0


class SurfaceVMD:
    def __init__(self, res, xy):
        self.res = res
        self.x = xy[0]
        self.y = xy[1]
        self.r = np.sqrt(xy[0]**2 + xy[1]**2)
        self.cos_phi = self.x / self.r
        self.sin_phi = self.y / self.r

    def fdem_ex(self, freq, m=1.):
        r = self.r
        sigma = 1 / self.res
        omega = freq * 2 * np.pi
        k = np.sqrt(-1j * mu_0 * sigma * omega)
        e_phi = - m  / (2 * np.pi * sigma * r ** 4)
        e_phi *= (3 - (3 + 3.j * k * r - (k * r) ** 2) * np.exp(-1.j * k * r))
        e_x = e_phi * -self.sin_phi
        return e_x

    def fdem_ey(self, freq, m=1.):
        r = self.r
        sigma = 1 / self.res
        omega = freq * 2 * np.pi
        k = np.sqrt(-1j * mu_0 * sigma * omega)
        e_phi = - m  / (2 * np.pi * sigma * r ** 4)
        e_phi *= (3 - (3 + 3.j * k * r - (k * r) ** 2) * np.exp(-1.j * k * r))
        e_y = e_phi * self.cos_phi
        return e_y

    def fdem_hx(self, freq, m=1.):
        r = self.r
        sigma = 1 / self.res
        omega = freq * 2 * np.pi
        k = np.sqrt(-1j * mu_0 * sigma * omega)
        arg = 1.j * k * r / 2
        h_r = - m * k * k / (4 * np.pi * r)
        h_r *=  (iv(1, arg) * kv(1, arg) - iv(2, arg) * kv(2, arg))
        h_x = h_r * self.cos_phi
        return h_x

    def fdem_hy(self, freq, m=1.):
        r = self.r
        sigma = 1 / self.res
        omega = freq * 2 * np.pi
        k = np.sqrt(-1j * mu_0 * sigma * omega)
        arg = 1.j * k * r / 2
        h_r = - m * k * k / (4 * np.pi * r)
        h_r *=  (iv(1, arg) * kv(1, arg) - iv(2, arg) * kv(2, arg))
        h_y = h_r * self.sin_phi
        return h_y

    def fdem_hz(self, freq, m=1.):
        r = self.r
        sigma = 1 / self.res
        omega = freq * 2 * np.pi
        k = np.sqrt(-1j * mu_0 * sigma * omega)
        h_z = m / (2 * np.pi * k**2 * r**5)
        h_z *= (9 - (9 + 9 * 1j * k * r - 4 * (k * r) **2 - 1j * (k * r) **3) \
                * np.exp(-1j * k * r))
        return h_z

    def tdem_hz(self, time, m=1.):
        r"""Return equation 4.69a, Ward and Hohmann, 1988.

        Switch-off response (i.e., Hz(t)) of a homogeneous isotropic half-space,
        where the vertical magnetic source and receiver are at the interface.

        Parameters
        ----------
        time : array
            Times (t)
            Halfspace resistivity (Ohm.m)
        m : float, optional
            Magnetic moment, default is 1.

        Returns
        -------
        hz : array
            Vertical magnetic field (A/m)

        """
        theta = np.sqrt(mu_0 / (4 * self.res * time))
        theta_r = theta * self.r

        hz = -(9 / theta_r + 4 * theta_r) * np.exp(-theta_r ** 2) \
                                                            / np.sqrt(np.pi)
        hz += erf(theta_r) * (9 / (2 * theta_r**2) - 1)
        hz *= m / (4 * np.pi * self.r**3)
        return hz

    def tdem_dhzdt(self, time, m=1.):
        r"""Return equation 4.70, Ward and Hohmann, 1988.

        Impulse response (i.e., dHz(t)/dt) of a homogeneous isotropic half-space,
        where the vertical magnetic source and receiver are at the interface.

        Parameters
        ----------
        time : array
            Times (t)
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


class Surface:
    def __init__(self, res, xy):
        self.res = res
        self.x = xy[0]
        self.y = xy[1]
        self.r = np.sqrt(xy[0]**2, xy[1]**2)
        self.cos_phi = self.x / self.r
        self.sin_phi = self.y / self.r

    


class SurfaceCircularLoop:
    def __init__(self, res, radius):
        self.res = res
        self.radius = radius

    def fdem_central_hz(self, freq, current=1.):
        radius = self.radius
        omega = freq * 2 * np.pi
        k = np.sqrt(-1j * mu_0 * (1 / self.res) * omega)
        Hz = - current / (k**2 * radius**3)
        Hz *= (3 - (3 + 3j * k * radius - k**2 * radius**2) \
                * np.exp(-1j * k * radius))
        return Hz

    def tdem_central_hz(self, time, current=1.):
        radius = self.radius
        theta = np.sqrt(mu_0 / (4 * self.res * time))
        theta_r = theta * radius

        hz = current / (2 * radius)
        hz *= (3 / (np.sqrt(np.pi) * theta_r) * np.exp(-theta_r**2) \
                + (1 - 3 / (2 * theta_r**2)) * erf(theta_r))
        return hz

    def td__central_dhzdt(self, time, current=1.):
        theta = np.sqrt(mu_0 / (4 * self.res * time))
        theta_r = theta * self.radius

        dhzdt = - current / (mu_0 * self.res**(-1) * self.radius**3)
        dhzdt *= (3 * erf(theta_r) - 2 / np.sqrt(np.pi) * theta_r * (3 + 2 * theta_r**2) * np.exp(-theta_r**2))
        # ?：なぜか符号が逆転してしまう
        return -dhzdt