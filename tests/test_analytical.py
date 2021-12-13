import pytest
import sys
sys.path.append('')
from example.Ward_and_Hohmann_1988.vmd import calc_vmd_freq_err, calc_vmd_time_err
import numpy as np


def test_vmd_freq(tol_err=1e1):
    rel_err_emp_re, rel_err_emp_im, rel_err_emu_re, rel_err_emu_im = calc_vmd_freq_err()
    assert np.all(rel_err_emp_re < tol_err)
    assert np.all(rel_err_emp_im < tol_err)
    assert np.all(rel_err_emu_re < tol_err)
    assert np.all(rel_err_emu_im < tol_err)


def test_vmd_time(tol_err=1e1):
    rel_err_emp_hz, rel_err_emu_hz, rel_err_emp_hzdt, rel_err_emu_hzdt = calc_vmd_time_err()
    assert np.all(rel_err_emp_hz < tol_err)
    assert np.all(rel_err_emu_hz < tol_err)
    assert np.all(rel_err_emp_hzdt < tol_err)
    assert np.all(rel_err_emu_hzdt < tol_err)
