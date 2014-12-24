from __future__ import division, absolute_import, print_function
import numpy as np
from statsmodels.tsa.stattools import acf
from pyhmc._utils import find_first


def integrated_autocorr6(x, c=6):
    r"""Estimate the integrated autocorrelation time, :math:`\tau_{int}` of a
    time series.

    This method performancs a summation of empirical autocorrelation function,
    using Sokal's "automatic windowing" procedure. The window length, ``M`` is
    chosen self-consistently to be the smallest value such that ``M`` is at
    least ``c`` times the estimated autocorrelation time, where ``c`` should
    be a constant in the range of 4, 6, or 10. See Appendix C of Sokal 1988.

    Parameters
    ----------
    x : ndarray, shape=(n_samples, n_dims)
        The time series, with time along axis 0.
    max_length : int
        The data ``x`` is aggregated if necessary by taking batch means so that
        the length of the series is less than ``max.length``.

    References
    ----------
    .. [1] Madras, Neal, and Alan D. Sokal. "The pivot algorithm: a highly
    efficient Monte Carlo method for the self-avoiding walk." J.
    Stat. Phys. 50.1-2 (1988): 109-186.

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
        f = acf(x[:,j], nlags=len(x), unbiased=False, fft=True)
        # vector of the taus, all with different choices of the window
        # length
        taus = 1 + 2*np.cumsum(f)[1:]
        ms = np.arange(len(f)-1)
        ind = find_first((ms > c*taus).astype(np.uint8))
        tau[j] = taus[ind]
    return tau




#     # reshape and thens sum over the second axis to get the sum of the pairs
    #     # [1,2,3,4,5,6,7,8] -> [[1,2], [3,4], [5,6], [7,8]] -> [3, 7, 11, 15]
    #     gamma = f.reshape(-1, 2).sum(axis=1)
    #     ind = find_first((gamma<0).astype(np.uint8))
    #     tau[j] = -1 + 2*np.sum(gamma[:ind])
    # return tau
