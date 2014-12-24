from __future__ import print_function
import numpy as np
from numpy import zeros, asarray
from numpy cimport npy_intp, npy_uint8


def find_first(npy_uint8[:, :] X):
    """Find the index of the first element of X along axis 0 that
    evaluates to True.
    """
    cdef npy_intp i, j
    cdef npy_intp n = X.shape[0]
    cdef npy_intp m = X.shape[1]
    cdef npy_intp[::1] out = zeros(m, dtype=np.intp)

    for j in range(m):
        out[j] = -1
        for i in range(n):
            if X[i, j]:
                out[j] = i
                break
    return asarray(out)
