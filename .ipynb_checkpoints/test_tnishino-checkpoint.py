from w1dem.forward import model, transceiver, w1dem
import numpy as np
import matplotlib.pyplot as plt

#w1dem1.0
thicks = np.array([20, 5, 2, 5])
res = np.array([100, 80, 10, 80, 100])
sc = np.array([0, 0, 0])
rc = np.array([100, 0, 0])
freqtime = np.logspace(-2, 5, 301)

model = model.Subsurface1D(thicks)
model.add_resistivity(res)
vmd = transceiver.VMD()
model.locate(vmd, sc, rc)
ans = model.run(freqtime, hankel_filter='key201')

#w1dem0.0
fdtd = 1
transmitter = 1
tx = [0]
ty = [0]
tz = [0]
rx = [100]
ry = [0]
rz = [0]

dipole_mom = 1
thickness = [20, 5, 2, 5]
res = np.array([100, 80, 10, 80, 100])
freqtime = np.logspace(-2, 5, 301)
plot_number = len(freqtime)
hankel_filter = "key201"
dbdt = 1

w1dem_fdem = w1dem.Fdem(rx, ry, rz, tx, ty, tz, res, thickness, hankel_filter, fdtd, dbdt, plot_number, freqtime)

bns, _ = w1dem_fdem.vmd(dipole_mom)

ak = model.kernel.real
ai = model.kernel.imag
bk = w1dem_fdem.kernel.real
bi = w1dem_fdem.kernel.imag

i = 1
y1, y2 = model.u[i], w1dem_fdem.u[i]
print(y1==y2)
plt.plot(y1)
plt.plot(y2, linestyle='--')
#plt.plot(k[2])
#resp = abs(ans['h_z'].real)
#imsp = abs(ans['h_z'].imag)
#plt.plot(freqtime, imsp)
#plt.plot(freqtime, resp)
#plt.yscale('log')
#plt.xscale('log')
plt.show()