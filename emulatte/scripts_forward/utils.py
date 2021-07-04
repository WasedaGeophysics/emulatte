import numpy as np

def kroneckers_delta(ii, jj):
    if ii == jj:
        return 1
    else:
        return 0

def ndarray_filter(array_like, valiable_name):
    if type(array_like) == list:
        array = np.array(array_like)
    elif type(array_like) == tuple:
        array = np.array(array_like)
    elif type(array_like) == np.ndarray:
        array = array_like
    else:
        print('TypeError : {} must be input as list, tuple or ndarray'.format(
            valiable_name
        ))
        quit()
    return array

