import numpy as np
from ..emulatte.model import earth
from ..emulatte.source.dipole import VMD

src = [0,0,0]
thicks = []
res = [100]
freq = np.logspace(-1,7, 300)

model = earth.EM1D(thicks)
vmd = VMD(1)
# ここでFDTD, waveformを決める
# ontime > 0 : step-on, = 0 : impluse, < 0 : step-off, array-like :  arbitorary_wave, None : FD
# ontimeの設定はloopとlineのみ

model.set_params(res)
model.set_source(vmd, src)
# 今後のことを考えて、デジタルフィルタ（Hankel, Fourier）は外から適用できるように「も」する
model.set_filter(hankel_filter='key201')

Ex = model.field("E", "x", [-1,0,0], freq)