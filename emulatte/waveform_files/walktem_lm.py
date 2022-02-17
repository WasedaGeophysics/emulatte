import numpy as np

# 引用元
# https://empymod.emsig.xyz/en/stable/gallery/tdomain/tem_walktem.html#sphx-glr-gallery-tdomain-tem-walktem-py

# Off time
# Low moment
off_time = np.array([
    1.149E-05, 1.350E-05, 1.549E-05, 1.750E-05, 2.000E-05, 2.299E-05,
    2.649E-05, 3.099E-05, 3.700E-05, 4.450E-05, 5.350E-05, 6.499E-05,
    7.949E-05, 9.799E-05, 1.215E-04, 1.505E-04, 1.875E-04, 2.340E-04,
    2.920E-04, 3.655E-04, 4.580E-04, 5.745E-04, 7.210E-04
])

# Waveform
# Low moment
waveform_time = np.array([-1.041E-03, -9.850E-04, 0.000E+00, 4.000E-06])  # ランプタイムあり
waveform_current = np.array([0.0, 1.0, 1.0, 0.0])
