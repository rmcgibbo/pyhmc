from __future__ import print_function
import numpy as np
from numpy import zeros, asarray
from numpy cimport npy_intp, npy_uint8


def find_first(npy_uint8[:] X):
    """Find the index of the first element of X that evaluates to True.
    Returns -1 if not found.
    """
    cdef npy_intp i
    cdef npy_intp n = X.shape[0]

    for i in range(n):
        if X[i]:
            return i
    return -1
