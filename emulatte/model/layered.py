"""Horizontal multi-layered earth model for EM forward simulations

>

Example
-------
    >

Notes
-----
    >


"""

# Copyright 2021-2022 Waseda Geophysics Laboratory
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# from Python Standard Libraries
from __future__ import annotations

# from Third Party Libraries
import numpy as np
from numpy.typing import ArrayLike, DTypeLike, NDArray
import scipy.constants as const

# from Internal Packages
from . import geometrics, damping
from ..utils.emu_object import Model, Source, DataArray
from ..dlf import loader as dlf

class Earth1DEM(Model):
    r"""Main UI class holding model parameter and results

    """
    def __init__(self, depth : ArrayLike, qss : bool = True) -> None:
        r"""
        Parameters
        ----------
        depth : array-like
            層境界深度 (m) 
        state : :obj:`str`, optional
            準定常状態をデフォルトとする
        """
        self.depth = np.array(depth, dtype=float)
        self.thick = np.diff(self.depth)
        self.thick_all = np.array([np.inf, *self.thick, np.inf])
        # number of underground layer
        self.nstrata : int = len(self.depth)
        self.nlayer : int = self.nstrata + 1
        self.qss : bool  = qss

        # default setting for Hankel transform
        self.ht_config = {
            "method" : "dlf",
            # dlf settings
            "filter" : "key_201",
            "_user_def" : False,
            "phase" : None,
            "j0" : None,
            "j1" : None,
            # qwe settings
        }

        # default setting for Fourier transform
        self.ft_config = {
            "method" : "dlf",
            # dlf settings
            "sampling" : "lagged_convolution",
            "filter" : "key_time_201",
            "window" : None,
            "sincos" : False,
            "_user_def" : False,
            "phase" : None,
            "sin" : None,
            "cos" : None
        }

    ### UI METHODS ###
    def set_params(
            self,
            res : ArrayLike,
            rep : ArrayLike = None,
            rmp : ArrayLike = None
        ) -> None:
        r"""
        Parameters
        ----------
        res
            Resistivity (Ωm) or conductivity (S/m)

        rep
            Relative Electric Permittivity (-)
            default -> `epsilon_r` = [1, 1, ... ]

        rmp
            Relative Magnetic Permeability (-)
            default -> `mu_r` = [1, 1, ... ]
        """
        res = np.array(res, dtype=float)

        rep = np.ones(self.nstrata) if rep is None else \
              np.array(rep, dtype=float)
        rmp = np.ones(self.nstrata) if rmp is None else \
              np.array(rmp, dtype=float)

        res = np.append(1e9, res)
        rep = np.append(1, rep)
        rmp = np.append(1, rmp)
        
        self.resistivity = res
        self.rel_e_permittivity = rep
        self.rel_m_peameability = rmp
        self.params_specified = True

    def set_params_air(self, res : float, rep : float, rmp : float) -> None:
        r"""
        Parameters
        ----------
        res
            Resistivity (Ωm) or conductivity (S/m)

        rep
            Relative Electric Permittivity (-)
            default -> `epsilon_r` = [1, 1, ... ]

        rmp
            Relative Magnetic Permeability (-)
            default -> `mu_r` = [1, 1, ... ]
        """
        if self.params_specified:
            self.resistivity[0] = res
            self.rel_e_permittivity[0] = rep
            self.rel_m_peameability[0] = rmp
        else:
            raise Exception("Set parameters beforehand.")
    
    def set_source(self, source : Source, place : ArrayLike) -> None:
        r"""
        Parameters
        ----------
        source : Source

        loc : array-like
            coordinate of specified source (x, y, z)
        """
        self.source = source
        self.source_place = np.array(place, ndmin=2, dtype=float)
        self.source_name = source.__class__.__name__
        self.source_installed = True


    def read_config(self, **kwargs : dict) -> None:
        r"""
        Parameters
        ----------
        kwargs : dict
        """
        self.ht_config = kwargs["ht"]
        self.ft_config = kwargs["ft"]

    def change_dlf_filter(
            self,
            hankel_dlf : str | dict = "key_201", 
            fourier_sin_cos_dlf : str | dict = "key_time_201"
            ) -> None:
        self.ht_config["filter"] = hankel_dlf
        self.ft_config["filter"] = fourier_sin_cos_dlf

    def fdem(
            self,
            fieldtype : str,
            direction : str,
            coordinate : ArrayLike, 
            frequency : ArrayLike | float,
            time_derivative : bool = False,
            normalize : bool = False
        ) -> DataArray:
        r"""
        Parameters
        ----------
        fieldtype : str
            type of electromagnetic vector field
            'E' for electric field
            'H' for magnetic field
            'D' for electric flux density
            'B' for magnetic flux density
            'J' for electrical current

        direction : str
            measurement direction, x, y, and z

        coordinate : array-like
            coordinate of measurement point (x, y, z)

        frequency : array-like
            measurement frequency or time of the receiver

        time-derivative : bool
            if True is input, the returned values are time derivatives

        Returns
        -------
        ans : DataArray
            Subclass of numpy.ndarray
            
            dtype = complex
        """
        if fieldtype in {"E", "D", "J"}:
            magnetic = False
        elif fieldtype in {"H", "B"}:
            magnetic = True
        else:
            raise NameError

        self.coordinate = np.array(coordinate)
        # unpack direction strings
        direction = [char for char in direction]
        # calculate angular frequency
        frequency = np.array(frequency, ndmin=1, dtype=float)
        omega = 2 * np.pi * frequency
        self.omega = omega
        self.nfreq = len(omega)

        # normalize source vector magnitude in Fourier domain
        if normalize:
            self.source.magnitude_f = np.ones(self.nfreq)

        # compute normalized data in FD
        if self.ht_config["method"] == "dlf":
            geometrics.fetch_source_dependent_params_dlf(self)
            data = self._compute_fdem_responce_dlf(direction, magnetic)
            if fieldtype == "D":
                data = self.eperm[self.ri] * data
            elif fieldtype == "J":
                data = self.conductivity[self.ri] * data
            elif fieldtype == "B":
                data = self.mperm[self.ri] * data
        else:
            raise Exception

        # apply source magnitude
        data = data * self.source.magnitude_f

        # differentiate with respect to time
        if time_derivative:
            data = data * 1.j * omega

        # data = DataArray(data)

        if data.shape[0] == 1:
            data = data[0]
        return data

    def tdem(
            self,
            fieldtype : str,
            direction : str,
            coordinate : ArrayLike, 
            time : ArrayLike | float,
            time_derivative : bool = False
        ) -> DataArray:
        r"""
        Parameters
        ----------
        fieldtype : str
            type of electromagnetic vector field
            'E' for electric field
            'H' for magnetic field
            'D' for electric flux density
            'B' for magnetic flux density
            'J' for electrical current

        direction : str
            measurement direction, x, y, and z

        coordinate : array-like
            coordinate of measurement point (x, y, z)

        frequency : array-like
            measurement frequency or time of the receiver

        time-derivative : bool
            if True is input, the returned values are time derivatives

        Returns
        -------
        data : DataArray
            Subclass of numpy.ndarray
            
            dtype = complex
        """
        # 
        signal = self.source.signal
        if self.ft_config["method"] == "dlf":
            # 計算に必要な周波数を全て持ってくる
            # コンボリューション
            pass

        elif self.ft_config["method"] == "qwe":
            pass
        else:
            raise NameError

    def _compute_fdem_responce_dlf(self, direction, magnetic):
        # compute admittivity, impedivity, wavenumber in each layer
        admittivity, impedivity, k = self._compute_wavenumber(self.qss)
        self.admittivity = admittivity
        self.impedivity = impedivity
        self.k = k

        # フィルターの読み出し
        if self.ht_config["_user_def"]:
            self.phase = np.array(self.ht_config["phase"])
            self.bessel_j0 = np.array(self.ht_config["j0"])
            self.bessel_j1 = np.array(self.ht_config["j0"])
        else:
            filter_name = self.ht_config["filter"]
            self.phase, self.bessel_j0, self.bessel_j1 = \
                dlf.load_hankel_filter(filter_name)
        
        self.nphase = len(self.phase)
        
        # compute fdem responce
        fd_resp = self.source._compute_hankel_transform_dlf(
                                                    self, direction, magnetic)
        return fd_resp

    def _compute_wavenumber(self, qss):
        # 2D array (nfreq, nlayer)
        res = self.resistivity
        rep = self.rel_e_permittivity
        rmp = self.rel_m_peameability
        omega = self.omega

        cond = 1 / res
        eperm = rep * const.epsilon_0
        mperm = rmp * const.mu_0

        impedivity = 1j * omega.reshape(-1,1) @ mperm.reshape(1,-1)

        if qss:
            admittivity = np.ones((self.nfreq, 1)) @ cond.reshape(1,-1)
            admittivity = admittivity.astype(complex)
        else:
            admittivity = cond.reshape(1,-1) \
                            + 1j * omega.reshape(-1,1) @ eperm.reshape(1,-1)

        k = np.sqrt(-admittivity * impedivity)
        return admittivity, impedivity, k

    def _compute_kernel_components(self):
        self.ndims = (self.nfreq, self.nlayer, self.nphase)
        self.lambda_ = self.phase / self.divisor
        kk, ll, mm = self.ndims
        lambda_tensor = np.zeros((kk, ll, mm), dtype=complex)
        lambda_tensor[:,:] = self.lambda_
        k_tensor = np.zeros((mm, kk, ll), dtype=complex)
        k_tensor[:,:] = self.k
        k_tensor = k_tensor.transpose((1, 2, 0))
        self.u = np.sqrt(lambda_tensor ** 2 - k_tensor ** 2)

        te_dsign = self.source.kernel_te_down_sign
        tm_dsign = self.source.kernel_tm_down_sign
        te_usign = self.source.kernel_te_up_sign
        tm_usign = self.source.kernel_tm_up_sign

        inp = (
            self.thick_all, self.depth,
            self.zs, self.z,
            self.si, self.ri,
            self.ndims, self.u,
            self.admittivity, self.impedivity,
            te_dsign, tm_dsign, te_usign, tm_usign
        )

        u_te, d_te, u_tm, d_tm, e_up, e_down = \
                                damping.compute_coefficients(*inp)

        self.u_te, self.d_te = u_te, d_te
        self.u_tm, self.d_tm = u_tm, d_tm
        self.e_up, self.e_down = e_up, e_down

    def _compute_kernel_components_multi(self):
        self.ndist = len(self.divisor)
        self.ndims = (self.ndist, self.nfreq, self.nlayer, self.nphase)
        self.lambda_ = self.phase.reshape(1,-1) / self.divisor.reshape(-1,1)
        jj, kk, ll, mm = self.ndims
        lambda_tensor = np.zeros((kk, ll, jj, mm), dtype=complex)
        lambda_tensor[:,:] = self.lambda_
        lambda_tensor = lambda_tensor.transpose((2,0,1,3))
        k_tensor = np.zeros((jj, mm, kk, ll), dtype=complex)
        k_tensor[:,:] = self.k
        k_tensor = k_tensor.transpose((0,2,3,1))
        self.u = np.sqrt(lambda_tensor ** 2 - k_tensor ** 2)

        te_dsign = self.source.kernel_te_down_sign
        tm_dsign = self.source.kernel_tm_down_sign
        te_usign = self.source.kernel_te_up_sign
        tm_usign = self.source.kernel_tm_up_sign

        inp = (
            self.thick_all, self.depth,
            self.zs, self.z,
            self.si, self.ri,
            self.ndims, self.u,
            self.admittivity, self.impedivity,
            te_dsign, tm_dsign, te_usign, tm_usign
        )

        u_te, d_te, u_tm, d_tm, e_up, e_down = \
                            damping.compute_coefficients_multi(*inp)
        
        self.u_te, self.d_te = u_te, d_te
        self.u_tm, self.d_tm = u_tm, d_tm
        self.e_up, self.e_down = e_up, e_down