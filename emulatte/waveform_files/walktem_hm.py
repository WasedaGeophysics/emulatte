import numpy as np

# 引用元
# https://empymod.emsig.xyz/en/stable/gallery/tdomain/tem_walktem.html#sphx-glr-gallery-tdomain-tem-walktem-py

# Off time
# High moment
off_time = np.array([
    9.810e-05, 1.216e-04, 1.506e-04, 1.876e-04, 2.341e-04, 2.921e-04,
    3.656e-04, 4.581e-04, 5.746e-04, 7.211e-04, 9.056e-04, 1.138e-03,
    1.431e-03, 1.799e-03, 2.262e-03, 2.846e-03, 3.580e-03, 4.505e-03,
    5.670e-03, 7.135e-03
])

# Waveform
# High moment
waveform_time = np.array([-8.333E-03, -8.033E-03, 0.000E+00, 5.600E-06])  # ランプタイムあり
waveform_current = np.array([0.0, 1.0, 1.0, 0.0])
