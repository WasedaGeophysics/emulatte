import sys
sys.path.append('')
from analytical.analytical import CL
import numpy as np
import empymod
import emulatte.forward as fwd

# 均質構造
depth = 0
thick = []
res = [2e14, 100]  # 空気層の比抵抗を無限大と近似

# for emulatte
sc = [0, 0, 0]
rc = [0, 0, 0]
radius = 50
emsrc_name = 'CircularLoop'
props = {'res': res}
model = fwd.model(thick)
model.set_properties(**props)

# fro empymod
src = [radius, 0, 0, 90, 0]
rec = [0, 0, 0, 0, 90]
strength = 2 * np.pi * radius
mrec = True
inp = {'src': src, 'rec': rec, 'depth': depth, 'res': res, 'verb': 1, 'strength': strength, 'mrec': mrec}

# 解析解
cl = CL(res[1], radius)


def calc_cl_freq(freq, filter_name='key201'):
    # emulatte
    emsrc = fwd.transmitter(emsrc_name, freq, current=1, radius=radius, turns=1)
    model.locate(emsrc, sc, rc)
    fEMF = model.emulate(hankel_filter=filter_name)
    fhz_emu = fEMF['h_z']
    # empymod
    fhz_emp = empymod.bipole(freqtime=freq, **inp)
    # 解析解
    fhz_ana = cl.fd_hz(freq)

    return [fhz_emu, fhz_emp, fhz_ana], freq


def calc_cl_time(time, filter_name='key201'):
    # emulatte
    emsrc = fwd.transmitter(emsrc_name, time, current=1, radius=radius, turns=1)
    model.locate(emsrc, sc, rc)
    tEMF, time_thz = model.emulate(hankel_filter=filter_name, td_transform='DLAG', time_diff=False, ignore_displacement_current=True)
    thz_emu = tEMF['h_z'].real
    tEMF_dt, time_thzdt = model.emulate(hankel_filter=filter_name, td_transform='DLAG', time_diff=True, ignore_displacement_current=True)
    thzdt_emu = tEMF_dt['h_z'].real
    # empymod
    eperm = [0, 0]  # Reduce early time numerical noise (diffusive approx for air)
    thz_emp = empymod.bipole(signal=-1, freqtime=time_thz, epermH=eperm, **inp)  # スイッチオン応答(微分なし)
    thzdt_emp = empymod.bipole(signal=0, freqtime=time_thzdt, epermH=eperm, **inp)  # インパルス応答 (微分あり)
    # 解析解
    thz_ana = cl.td_hz(time_thz)
    thzdt_ana = cl.td_dhzdt(time_thzdt)

    return [thz_emu, thz_emp, thz_ana], time_thz, [thzdt_emu, thzdt_emp, thzdt_ana], time_thzdt
