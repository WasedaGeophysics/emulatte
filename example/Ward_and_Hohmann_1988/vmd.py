import sys
sys.path.append('')
from analytical.analytical import VMD
import numpy as np
import empymod
import emulatte.forward as fwd


def calc_rel_err(ans_num, ans_ana):
    rel_err = np.abs((ans_num - ans_ana) / ans_ana) * 100
    return rel_err


# 送受信装置
src = [0, 0, 0, 0, 90]
rec = [100, 0, 0, 0, 90]

# 均質構造
depth = 0
thick = []
res = [2e14, 100]  # 空気層の比抵抗を無限大と近似

# for emulatte
emsrc_name = 'VMD'
props = {'res': res}
model = fwd.model(thick)
model.set_properties(**props)

# fro empymod
inp = {'src': src, 'rec': rec, 'depth': depth, 'res': res, 'verb': 1}

# 解析解
r = np.sqrt(rec[0]**2 + rec[1]**2)
vmd = VMD(res[1], r)


def calc_vmd_freq(freq, filter_name='key201'):
    # freq = np.logspace(-1, 5, 301)
    # emulatte
    emsrc = fwd.transmitter(emsrc_name, freq, moment=1)
    model.locate(emsrc, src[:3], rec[:3])
    fEMF = model.emulate(hankel_filter=filter_name)
    fhz_emu = fEMF['h_z']
    # empymod
    fhz_emp = empymod.loop(**inp, freqtime=freq)
    # 解析解
    fhz_ana = vmd.fd_hz(freq)

    return [fhz_emu, fhz_emp, fhz_ana], freq

    # # 相対誤差
    # rel_err_emp_re = calc_rel_err(fhz_ana.real, fhz_emp.real)
    # rel_err_emp_im = calc_rel_err(fhz_ana.imag, fhz_emp.imag)
    # rel_err_emu_re = calc_rel_err(fhz_ana.real, fhz_emu.real)
    # rel_err_emu_im = calc_rel_err(fhz_ana.imag, fhz_emu.imag)
    # return rel_err_emp_re, rel_err_emp_im, rel_err_emu_re, rel_err_emu_im


def calc_vmd_time(time, filter_name='key201'):
    # time = np.logspace(-8, 0, 301)
    # emulatte
    emsrc = fwd.transmitter(emsrc_name, time, moment=1)
    model.locate(emsrc, src[:3], rec[:3])
    tEMF, time_thz = model.emulate(hankel_filter=filter_name, td_transform='DLAG', time_diff=False, ignore_displacement_current=True)
    thz_emu = tEMF['h_z'].real
    tEMF_dt, time_thzdt = model.emulate(hankel_filter=filter_name, td_transform='DLAG', time_diff=True, ignore_displacement_current=True)
    thzdt_emu = tEMF_dt['h_z'].real
    # empymod
    epermH = [0, 1]
    thz_emp = empymod.loop(signal=-1, freqtime=time_thz, xdirect=True, epermH=epermH, **inp)  # スイッチオン応答(微分なし)
    thzdt_emp = empymod.loop(signal=0, freqtime=time_thzdt, xdirect=True, epermH=epermH, **inp)  # インパルス応答 (微分あり)
    # 解析解
    thz_ana = vmd.td_hz(time_thz)
    thzdt_ana = vmd.td_dhzdt(time_thzdt)

    return [thz_emu, thz_emp, thz_ana], time_thz, [thzdt_emu, thzdt_emp, thzdt_ana], time_thzdt

    # # 相対誤差
    # # 微分なし
    # rel_err_emp_thz = calc_rel_err(thz_ana, thz_emp)
    # rel_err_emu_thz = calc_rel_err(thz_ana, thz_emu)
    # # 微分あり
    # rel_err_emp_thzdt = calc_rel_err(thz_ana, thzdt_emp)
    # rel_err_emu_thzdt = calc_rel_err(thzdt_ana, thzdt_emu)

    # return rel_err_emp_thz, rel_err_emu_thz, rel_err_emp_thzdt, rel_err_emu_thzdt
