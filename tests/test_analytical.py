import pytest
import sys
sys.path.append('')
from example.Ward_and_Hohmann_1988.vmd import calc_vmd_freq, calc_vmd_time
from example.Ward_and_Hohmann_1988.circularloop import calc_cl_freq, calc_cl_time
import numpy as np


def calc_rel_err(ans_num, ans_ana):
    rel_err = np.abs((ans_num - ans_ana) / ans_ana) * 100
    return rel_err


hankel_filter = ['anderson801', 'kong241', 'mizunaga90', 'werthmuller201', 'key201']


def test_vmd_freq(tol_err=1e1):
    freq = np.logspace(-1, 5, 301)
    filter_name = hankel_filter[3]
    [fhz_emu, fhz_emp, fhz_ana], freq = calc_vmd_freq(freq, filter_name=filter_name)

    # assert np.all(calc_rel_err(fhz_ana.real, fhz_emp.real) < tol_err)
    # assert np.all(calc_rel_err(fhz_ana.imag, fhz_emp.imag) < tol_err)
    assert np.all(calc_rel_err(fhz_ana.real, fhz_emu.real) < tol_err)
    assert np.all(calc_rel_err(fhz_ana.imag, fhz_emu.imag) < tol_err)


def test_vmd_time(tol_err=1e1):
    time = np.logspace(-8, 0, 301)
    filter_name = hankel_filter[3]
    [thz_emu, thz_emp, thz_ana], time_thz, [thzdt_emu, thzdt_emp, thzdt_ana], time_thzdt = calc_vmd_time(time, filter_name)

    # assert np.all(calc_rel_err(thz_ana, thz_emp) < tol_err)
    # assert np.all(calc_rel_err(thz_ana, thzdt_emp) < tol_err)
    assert np.all(calc_rel_err(thz_ana, thz_emu) < tol_err)
    assert np.all(calc_rel_err(thzdt_ana, thzdt_emu) < tol_err)

def test_cl_freq(tol_err=1e3):
    freq = np.logspace(-1, np.log10(250000), 301)
    filter_name = hankel_filter[3]
    [fhz_emu, fhz_emp, fhz_ana], freq = calc_cl_freq(freq, filter_name=filter_name)

    # assert np.all(calc_rel_err(fhz_ana.real, fhz_emp.real) < tol_err)
    # assert np.all(calc_rel_err(fhz_ana.imag, fhz_emp.imag) < tol_err)
    assert np.all(calc_rel_err(fhz_ana.real, fhz_emu.real) < tol_err)
    assert np.all(calc_rel_err(fhz_ana.imag, fhz_emu.imag) < tol_err)


def test_cl_time(tol_err=1e3):
    time = np.logspace(-8, 0, 301)
    filter_name = hankel_filter[3]
    [thz_emu, thz_emp, thz_ana], time_thz, [thzdt_emu, thzdt_emp, thzdt_ana], time_thzdt = calc_cl_time(time, filter_name)

    # assert np.all(calc_rel_err(thz_ana, thz_emp) < tol_err)
    # assert np.all(calc_rel_err(thz_ana, thzdt_emp) < tol_err)
    assert np.all(calc_rel_err(thz_ana, thz_emu) < tol_err)
    assert np.all(calc_rel_err(thzdt_ana, thzdt_emu) < tol_err)
