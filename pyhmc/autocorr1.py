# This code is adapted from https://github.com/dfm/emcee (MIT license)
from __future__ import division, print_function, absolute_import
import sys
import numpy as np
from ._utils import find_first
from statsmodels.tsa.stattools import acf

__all__ = ["integrated_autocorr1"]


def integrated_autocorr1(x):
    r"""Estimate the integrated autocorrelation time, :math:`\tau_{int}` of a
    time series.

    Parameters
    ----------
    x : ndarray, shape=(n_samples, n_dims)
        The time series, with time along axis 0.

    Notes
    -----
    This method directly sums the first `k` entries of the ACF, where `k` is
    chosen to be the index of the first instance where the ACF crosses zero
    (Chodera 2007)

    References
    ----------
    .. [1] J. D. Chodera, W. C. Swope, J. W. Pitera, C. Seok, and K. A. Dill.
       JCTC 3(1):26-41, 2007.

    Returns
    -------
    tau_int : ndarray, shape=(n_dims,)
        The estimated integrated autocorrelation time of each dimension in
        ``x``, considered independently.
    """
    # Compute the autocorrelation function.
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    n = len(x)

    tau = np.zeros(x.shape[1])
    for j in range(x.shape[1]):
        f = acf(x[:,j], nlags=n, unbiased=False, fft=True)
        window = find_first((f < 0).astype(np.uint8))
        tau[j] = 1 + 2*f[:window].sum()

    return tau
