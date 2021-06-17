from w1dem.forward import model, transceiver
import numpy as np
import matplotlib.pyplot as plt

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

resp = abs(ans['h_z'].real)
imsp = abs(ans['h_z'].imag)
plt.plot(freqtime, imsp)
plt.plot(freqtime, resp)
plt.yscale('log')
plt.xscale('log')
plt.show()