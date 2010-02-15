"""Energy functions for phase model
"""

import numpy as np
cimport numpy as np
cimport cython

cdef extern from "math.h":
    double cos(double x)
    double sin(double x)
    double exp(double x)
    double tanh(double x)
    double atan2(double im, double re)
    double sqrt(double x)
    double abs(double x)

cdef extern from "stdlib.h":
    ctypedef unsigned long size_t
    void free(void *ptr)
    void *malloc(size_t size)
    void *realloc(void *ptr, size_t size)
    size_t strlen(char *s)
    char *strcpy(char *dest, char *src)


# @cython.boundscheck(False)
def f_multivariate_normal(np.ndarray[np.double_t, ndim=1] x,
                          np.ndarray[np.double_t, ndim=2] M):
    """Energy function for multivariate normal distribution
    """
    cdef unsigned int i,j
    cdef unsigned int n = len(x)
    cdef double out = 0
    for i in range(n):
        for j in range(n):
            out += .5*x[i]*M[i,j]*x[j]
    return out

# @cython.boundscheck(False)
def f_phasedist(np.ndarray[np.double_t, ndim=1] theta,
                np.ndarray[np.double_t, ndim=2] M):
    """Energy function for phase model
    """
    cdef unsigned int i,j,k
    cdef unsigned int n = len(theta)
    cdef double out = 0
    cdef double *x = <double *>malloc(2*n*sizeof(double))
    for i in range(n):
        j = 2*i
        x[j] = cos(theta[i])
        x[j+1] = sin(theta[i])
    for i in range(2*n):
        for j in range(2*n):
            out -= .5*x[i]*M[i,j]*x[j]
    free(x)
    return out

# @cython.boundscheck(False)
def f_phasedist_biased(np.ndarray[np.double_t, ndim=1] theta,
                       np.ndarray[np.double_t, ndim=2] M):
    """Energy function for phase model with bias term
    """
    cdef unsigned int i,j,k
    cdef unsigned int n = len(theta)+1
    cdef double out = 0
    cdef double *x = <double *>malloc(2*n*sizeof(double))
    x[0] = 1.0
    for i in range(1,n):
        j = 2*i
        x[j] = cos(theta[i-1])
        x[j+1] = sin(theta[i-1])
    for i in range(2*n):
        for j in range(2*n):
            out -= .5*x[i]*M[i,j]*x[j]
    free(x)
    return out

# @cython.boundscheck(False)
def g_multivariate_normal(np.ndarray[np.double_t, ndim=1] x,
                np.ndarray[np.double_t, ndim=2] M):
    """Energy gradient for multivariate normal distribution
    """
    cdef unsigned int i,j
    cdef unsigned int n = len(x)
    cdef np.ndarray[np.double_t, ndim=1] out = np.zeros(n)
    for i in range(n):
        for j in range(n): out[i] += .5*x[j]*(M[j,i]+M[i,j])
    return out


# @cython.boundscheck(False)
def g_phasedist(np.ndarray[np.double_t, ndim=1] theta,
                np.ndarray[np.double_t, ndim=2] M):
    """Energy gradient for phase model
    """
    cdef unsigned int i,j,k
    cdef unsigned int n = len(theta)
    cdef double xdot1, xdot2
    cdef np.ndarray[np.double_t, ndim=1] out = np.zeros(n)
    # cdef np.ndarray[np.double_t, ndim=1] x = np.empty(2*n)
    cdef double *x = <double *>malloc(2*n*sizeof(double))
    for i in range(n):
        j = 2*i
        x[j] = cos(theta[i])
        x[j+1] = sin(theta[i])
    for i in range(n):
        j = 2*i
        for k in range(2*n): out[i] += x[k]*(M[k,j]*x[j+1]-M[k,j+1]*x[j])
    free(x)
    return out


# @cython.boundscheck(False)
def g_phasedist_biased(np.ndarray[np.double_t, ndim=1] theta,
                       np.ndarray[np.double_t, ndim=2] M):
    """Energy gradient for phase model with bias term
    """
    cdef unsigned int i,j,k
    cdef unsigned int n = len(theta)+1
    cdef double xdot1, xdot2
    cdef np.ndarray[np.double_t, ndim=1] out = np.zeros(n-1)
    # cdef np.ndarray[np.double_t, ndim=1] x = np.empty(2*n)
    cdef double *x = <double *>malloc(2*n*sizeof(double))
    x[0] = 1.0
    for i in range(1,n):
        j = 2*i
        x[j] = cos(theta[i-1])
        x[j+1] = sin(theta[i-1])
    for i in range(1,n):
        j = 2*i
        for k in range(2*n): out[i-1] += x[k]*(M[k,j]*x[j+1]-M[k,j+1]*x[j])
    free(x)
    return out
