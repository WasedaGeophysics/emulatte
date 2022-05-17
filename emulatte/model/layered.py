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
from pandas import NA
import scipy.constants as const

# from Internal Packages
from ..utils.emu_object import Model, Source, DataArray
from ..dlf import loader as dlf

class Earth1DEM(Model):
    r"""Main UI class holding model parameter and results

    """
    def __init__(self, depth : ArrayLike, state : str = "qs") -> None:
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
        self.nug : int = len(self.depth)
        self.nlayer : int = self.nug + 1
        self.state = state

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

        rep = np.ones(self.N) if rep is None else \
              np.array(rep, dtype=float)
        rmp = np.ones(self.N) if rmp is None else \
              np.array(rmp, dtype=float)

        res = np.append(1e31, res)
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
    
    def set_source(self, source : Source, loc : ArrayLike) -> None:
        r"""
        Parameters
        ----------
        source : Source

        loc : array-like
            coordinate of specified source (x, y, z)
        """
        self.source = source
        self.source_loc = np.array(loc, dtype=float)
        self.source_type = source.__class__.__name__
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
            hankel_dlf = "key_201", 
            fourier_sin_cos_dlf : str | dict = "key_time_201"
            ) -> None:
            pass

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
        
        direction = [char for char in direction]
        self.rc = np.array(coordinate)

        # 計算実行可能性の判定
        self._certificate()
        # 空間環境の精査
        self._scan_placement()
        
        frequency = np.array(frequency)
        omega = 2 * np.pi * frequency

        

        # compute data in FD
        if fieldtype in {"E", "D", "J"}:
            data = self._compute_fdem_responce(direction, omega)
            if fieldtype == "D":
                data = self.eperm[self.m] * data
            elif fieldtype == "J":
                data = self.conductivity[self.m] * data
        elif fieldtype in {"H", "B"}:
            data = self._compute_fdem_responce(
                            direction, omega, magnetic = True)
            if fieldtype == "B":
                data = self.mperm[self.m] * data
        else:
            raise Exception

        if time_derivative:
            data = data * 1.j * omega

        if normalize:

        data = DataArray(data)
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



    def _certificate(self):
        # フィルター設定の確認
        pass
    
    def _scan_placement(self):
        # 送受信配置送受信配置
        point_source = [
            "VMD", "HMD", "AMD", "VED", "HED", "AED", "CircularLoop"
            ]
        pass

    def _conpute_fdem_responce(self, direction, omega, magnetic = False):
        if self.ht_config["method"] == "dlf":
            # フィルターの読み出し
            if self.ht_config["_user_def"]:
                lambda_phase = np.array(self.ht_config["phase"])
                bessel_j0 = np.array(self.ht_config["j0"])
                bessel_j1 = np.array(self.ht_config["j0"])
            else:
                filter_name = self.ht_config["filter"]
                lambda_phase, bessel_j0, bessel_j1 = \
                    dlf.load_hankel_filter(filter_name)
            
            # compute fdem responce
            if not magnetic:
                fd_resp = self.source._compute_hankel_input_e(
                                    self, direction, omega, 
                                    lambda_phase, bessel_j0, bessel_j1
                                    )
            else:
                fd_resp = self.source._compute_hankel_input_m(
                                    self, direction, omega, 
                                    lambda_phase, bessel_j0, bessel_j1
                                    )

        elif self.ht_config["method"] == "qwe":
            pass
        else:
            raise NameError

        return fd_resp
            

        



        


