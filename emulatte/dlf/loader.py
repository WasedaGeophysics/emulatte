# Copyright 2021 Waseda Geophysics Laboratory
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
"""
hankelフィルター係数のロード
"""
from .sin_cos_filter import (
    anderson_time_787, 
    key_time_201, 
    raito_time_250,
    werthmuller_time_201
)
from .hankel_filter import (
    kong_241,
    mizunaga_90, 
    anderson_801,
    werthmuller_201,
    key_201
)


# function for load hankel filter
def load_hankel_filter(hankel_filter_name):
    if hankel_filter_name == "anderson_801":
        base, j0, j1 = load_anderson_801()
    elif hankel_filter_name == "kong_241":
        base, j0, j1 = load_kong_241()
    elif hankel_filter_name == "mizunaga_90":
        base, j0, j1 = load_mizunaga_90()
    elif hankel_filter_name == "werthmuller_201":
        base, j0, j1 = load_werthmuller_201()
    elif hankel_filter_name == "key_201":
        base, j0, j1 = load_key_201()
    else:
        raise NameError('invalid hankel filter name')

    return base, j0, j1


def load_sin_cos_filter(sin_cos_filter_name):
    if sin_cos_filter_name == "anderson_time_787":
        base, cos, sin = load_anderson_time_787()
    elif sin_cos_filter_name == "key_time_201":
        base, cos, sin = load_key_time_201()
    elif sin_cos_filter_name == "werthmuller_time_201":
        base, cos, sin = load_werthmuller_time_201()
    elif sin_cos_filter_name == "raito_time_250":
        base, cos, sin = load_raito_time_250()
    else:
        raise NameError('invalid fft filter name')

    return base, cos, sin


# DLF
def load_anderson_801():
    base = anderson_801.base
    j0 = anderson_801.j0
    j1 = anderson_801.j1
    return base, j0, j1


def load_kong_241():
    base = kong_241.base
    j0 = kong_241.j0
    j1 = kong_241.j1
    return base, j0, j1


def load_mizunaga_90():
    base = mizunaga_90.base
    j0 = mizunaga_90.j0
    j1 = mizunaga_90.j1
    return base, j0, j1


def load_werthmuller_201():
    base = werthmuller_201.base
    j0 = werthmuller_201.j0
    j1 = werthmuller_201.j1
    return base, j0, j1


def load_key_201():
    base = key_201.base
    j0 = key_201.j0
    j1 = key_201.j1
    return base, j0, j1


# FDRIFT
def load_anderson_time_787():
    base = anderson_time_787.base
    cos = anderson_time_787.cos
    sin = anderson_time_787.sin
    return base, cos, sin


def load_key_time_201():
    base = key_time_201.base
    cos = key_time_201.cos
    sin = key_time_201.sin
    return base, cos, sin


def load_raito_time_250():
    base = raito_time_250.base
    j0 = raito_time_250.j0
    j1 = raito_time_250.j1
    return base, j0, j1


def load_werthmuller_time_201():
    base = werthmuller_time_201.base
    cos = werthmuller_time_201.cos
    sin = werthmuller_time_201.sin
    return base, cos, sin