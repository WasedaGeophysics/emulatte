import sys
sys.path.append('')
from analytical.analytical import VMD
import numpy as np
import empymod
import emulatte.forward as fwd


def calc_rel_err(ans_num, ans_ana):
    rel_err = np.abs((ans_num - ans_ana) / ans_ana) * 100
    return rel_err


def calc_vmd_freq_err():
    # 探査条件
    freq = np.logspace(-1, 5, 301)
    # 送受信点位置
    src = [0, 0, 0, 0, 90]
    rec = [100, 0, 0, 0, 90]
    # 均質構造
    depth = 0
    thick = []
    res = [2e14, 100]  # 空気層の比抵抗を無限大と近似
    inp = {'src': src, 'rec': rec, 'depth': depth, 'res': res, 'freqtime': freq, 'verb': 1}

    # 解析解
    vmd = VMD()
    r = np.sqrt(rec[0]**2 + rec[1]**2)  # 送受信器間距離
    fhz_ana = vmd.fd_hz(res[1], r, freq)

    # empymod
    fhz_emp = empymod.loop(**inp)

    # emulatte
    # VMD
    emsrc_name = 'VMD'
    props = {'res': res}
    model = fwd.model(thick)
    model.set_properties(**props)
    emsrc = fwd.transmitter(emsrc_name, freq, moment=1)
    model.locate(emsrc, src[:3], rec[:3])
    EMF = model.emulate(hankel_filter='werthmuller201')
    fhz_emu = EMF['h_z']

    # 相対誤差
    rel_err_emp_re = calc_rel_err(fhz_ana.real, fhz_emp.real)
    rel_err_emp_im = calc_rel_err(fhz_ana.imag, fhz_emp.imag)
    rel_err_emu_re = calc_rel_err(fhz_ana.real, fhz_emu.real)
    rel_err_emu_im = calc_rel_err(fhz_ana.imag, fhz_emu.imag)
    return rel_err_emp_re, rel_err_emp_im, rel_err_emu_re, rel_err_emu_im


def calc_vmd_time_err():
    # 探査条件
    time = np.logspace(-8, 0, 301)
    # 送受信点位置
    src = [0, 0, 0, 0, 90]
    rec = [100, 0, 0, 0, 90]
    # 均質構造
    depth = 0
    thick = []
    res = [2e14, 100]  # 空気層の比抵抗を無限大と近似
    epermH = [0, 1]
    inp = {'src': src, 'rec': rec, 'depth': depth, 'res': res, 'verb': 1, 'xdirect': True, 'epermH': epermH}

    # emulatte
    emsrc_name = 'VMD'
    props = {'res': res}
    model = fwd.model(thick)
    model.set_properties(**props)
    emsrc = fwd.transmitter(emsrc_name, time, moment=1)
    model.locate(emsrc, src[:3], rec[:3])
    EMF, time_hz = model.emulate(hankel_filter='werthmuller201', td_transform='DLAG', time_diff=False, ignore_displacement_current=True)
    thz_emu = EMF['h_z'].real
    EMF_dt, time_dt = model.emulate(hankel_filter='werthmuller201', td_transform='DLAG', time_diff=True, ignore_displacement_current=True)
    thzdt_emu = EMF_dt['h_z'].real

    # 解析解
    vmd = VMD()
    r = np.sqrt(rec[0]**2 + rec[1]**2)  # 送受信器間距離
    thz_ana = vmd.td_hz(res[1], r, time_hz)
    thzdt_ana = vmd.td_hz(rec[1], r, time_dt)

    # empymod
    thz_emp = empymod.loop(signal=-1, freqtime=time_hz, **inp)  # スイッチオン応答(微分なし)
    thzdt_emp = empymod.loop(signal=0, freqtime=time_dt, **inp)  # インパルス応答 (微分あり)

    # 相対誤差
    # 微分なし
    rel_err_emp_hz = calc_rel_err(thz_ana, thz_emp)
    rel_err_emu_hz = calc_rel_err(thz_ana, thz_emu)
    # 微分あり
    rel_err_emp_hzdt = calc_rel_err(thz_ana, thzdt_emp)
    rel_err_emu_hzdt = calc_rel_err(thzdt_ana, thzdt_emu)

    return rel_err_emp_hz, rel_err_emu_hz, rel_err_emp_hzdt, rel_err_emu_hzdt
