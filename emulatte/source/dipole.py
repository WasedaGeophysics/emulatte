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

from ..core import filter
from ..utils.converter import array, check_waveform


class VMD:
    def __init__(self, moment, ontime = None):
        self.moment = moment
        self.num_dipole = 1
        self.kernel_te_up_sign = 1
        self.kernel_te_down_sign = 1
        self.kernel_tm_up_sign = 0
        self.kernel_tm_down_sign = 0
        self.tx_type = check_waveform(ontime)

    def hankel_transform_e(self, model, direction, ):
        y_base, wt0, wt1 = filter.load_hankel_filter(model.hankel_filter)

    def hankel_transform_h(self, model, direction, ):
        y_base, wt0, wt1 = filter.load_hankel_filter(model.hankel_filter)

    def hankel_ex(self):


