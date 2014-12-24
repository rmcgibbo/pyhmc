# This code is adapted from https://github.com/dfm/emcee (MIT license)
from __future__ import division, print_function, absolute_import
import sys
import numpy as np
from ._utils import find_first
from statsmodels.tsa.stattools import acf

__all__ = ["integrated_autocorr1"]


def integrated_autocorr1(x, acf_cutoff=0.0):
    r"""Estimate the integrated autocorrelation time, :math:`\tau_{int}` of a
    time series.

    This method performancs a summation of empirical autocorrelation function,
    using a window length, ``M``, to the smallest value such that
    ``ACF(m) <= acf_cutoff``. This procedure is used in (Chodera 2007) with
    ``acf_cutoff = 0``. In (Hoffman 2011, Hub 2010), this estimator is used
    with ``acf_cutoff = 0.05``.

    Parameters
    ----------
    x : ndarray, shape=(n_samples, n_dims)
        The time series, with time along axis 0.

    References
    ----------
    .. [1] J. D. Chodera, W. C. Swope, J. W. Pitera, C. Seok, and K. A. Dill.
       JCTC 3(1):26-41, 2007.
    .. [2] Hoffman, M. D., and A. Gelman. "The No-U-Turn sampler: Adaptively
       setting path lengths in Hamiltonian Monte Carlo." arXiv preprint
       arXiv:1111.4246 (2011).
    .. [3] Hub, J. S., B. L. De Groot, and D. V. Der Spoel. "g_wham: A Fre
       Weighted Histogram Analysis Implementation Including Robust Error and
       Autocorrelation Estimates." J. Chem. Theory Comput. 6.12 (2010):
       3713-3720.

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
        window = find_first((f <= acf_cutoff).astype(np.uint8))
        tau[j] = 1 + 2*f[1:window].sum()

    return tau
