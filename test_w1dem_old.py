import numpy as np
from w1dem_old import w1dem
import matplotlib.pyplot as plt

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

ans, _ = w1dem_fdem.vmd(dipole_mom)

print(w1dem_fdem.kernel.shape)
kernel = w1dem_fdem.kernel

kernel1 = kernel[0]
kernel2 = kernel[1]
kernel3 = kernel[2]

plt.plot(kernel1)
plt.plot(kernel2)
plt.plot(kernel3)
plt.yscale('log')
plt.show()

#resp = abs(ans['h_z'].real)
#imsp = abs(ans['h_z'].imag)