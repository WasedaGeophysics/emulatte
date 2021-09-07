import numpy as np
import sys

def kroneckers_delta(ii, jj):
    if ii == jj:
        return 1
    else:
        return 0

def ndarray_filter(array_like, valiable_name):
    types = [list, tuple, np.ndarray]
    if type(array_like) in types:
        array = np.array(array_like, dtype=np.float64)
        return array
    else:
        print('TypeError : {} must be input as list, tuple or ndarray'.format(
            valiable_name
        ))
        sys.exit

