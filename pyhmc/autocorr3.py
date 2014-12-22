import numpy as np
import matplotlib.pyplot as pp
from pyhmc import autocorr
from pyhmc._hmc import find_first


def integrated_autocorr3(x):
    r"""Estimate the integrated autocorrelation time, :math:`\tau_{int}` of a
    time series.

    Parameters
    ----------
    x : ndarray, shape=(n_samples, n_dims)
        The time series, with time along axis 0.
    max_length : int
        The data ``x`` is aggregated if necessary by taking batch means so that
        the length of the series is less than ``max.length``.

    Notes
    -----
    This method uses the initial sequence estimator (Geyer 1992), which sums
    consecutive elements in the empirical autocorrelation function, truncated
    when adjacent sample ACF values become negative.

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

    acf = autocorr(x)
    # truncate acf to the nearest multiple of two
    acf = acf[:2*(len(acf)//2)]
    tau = np.zeros(x.shape[1])

    for j in range(x.shape[1]):
        # reshape and thens sum over the second axis to get the sum of the pairs
        # [1,2,3,4,5,6,7,8] -> [[1,2], [3,4], [5,6], [7,8]] -> [3, 7, 11, 15]
        gamma = acf[:,j].reshape(-1, 2).sum(axis=1)
        ind = find_first((gamma<0).reshape(-1,1).astype(np.uint8))[0]
        tau[j] = -1 + 2*np.sum(gamma[:ind])
    return tau
