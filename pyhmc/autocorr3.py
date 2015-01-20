from __future__ import division, absolute_import, print_function
import numpy as np
from statsmodels.tsa.stattools import acf
from pyhmc._utils import find_first


def integrated_autocorr3(x):
    r"""Estimate the integrated autocorrelation time, :math:`\tau_{int}` of a
    time series.

    This method performancs a summation of empirical autocorrelation function,
    setting the window based on the initial sequence estimator (Geyer 1992),
    which stops when the sum of two consecutive elements in the empirica
    autocorrelation function become negative.

    Parameters
    ----------
    x : ndarray, shape=(n_samples, n_dims)
        The time series, with time along axis 0.

    References
    ----------
    .. [1] Geyer, Charles J. "Practical markov chain monte carlo." Statistical
       Science (1992): 473-483.

    Returns
    -------
    tau_int : ndarray, shape=(n_dims,)
        The estimated integrated autocorrelation time of each dimension in
        ``x``, considered independently.
    """
    if x.ndim == 1:
        x = x.reshape(-1, 1)

    tau = np.zeros(x.shape[1])
    for j in range(x.shape[1]):
        f = acf(x[:,j], nlags=2*(len(x)//2)-1, unbiased=False, fft=True)
        # reshape and thens sum over the second axis to get the sum of the pairs
        # [1,2,3,4,5,6,7,8] -> [[1,2], [3,4], [5,6], [7,8]] -> [3, 7, 11, 15]
        gamma = f.reshape(-1, 2).sum(axis=1)
        ind = find_first((gamma<0).astype(np.uint8))
        tau[j] = -1 + 2*np.sum(gamma[:ind])
    return tau
