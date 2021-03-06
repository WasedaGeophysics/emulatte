# Copyright 2022 Waseda Geophysics Laboratory
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

# -*- coding: utf-8 -*-
import numpy as np
from .model.layered import Earth1DEM

def create_model(thick_or_depth, state = None, stuck = True):
    r"""
    Parameters
    ----------
    thick : array_like
            layer thickness (m), except for the first (on the ground) & last layers
    state : str
            = 'qs' -> electroquasistatic mode, diplacement current doesn't affect magnetic field.
            = 'ip' -> Pelton()
    """
    if stuck:
        depth = [0, *np.cumsum(thick_or_depth)]
    else:
        depth = thick_or_depth

    if state is None:
        model = Earth1DEM(depth)
    elif state == 'unsteady':
        model = Earth1DEM(depth, qss=False)
    else:
        raise Exception
    return model