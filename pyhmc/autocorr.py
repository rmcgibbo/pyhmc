# This code is adapted from https://github.com/dfm/emcee (MIT license)
from __future__ import division, print_function, absolute_import
import numpy as np
from ._hmc import find_first

__all__ = ["autocorr", "integrated_autocorr"]


def autocorr(x, axis=0, fast=False):
    """Estimate the autocorrelation function of a time series using the FFT.

    Parameters
    ----------
    x : ndarray, shape=(n_samples,) or shape=(n_samples, n_dims)
        The time series. If multidimensional, set the time axis using the
        ``axis`` keyword argument and the function will be computed for every
        other axis.
    axis :  int, optional
        The time axis of ``x``. Assumed to be the first axis if not specified.
    fast : bool, optional
        If ``True``, only use the largest ``2^n`` entries for efficiency.
        (default: False)

    Examples
    --------
    >>> # [ generate samples from ``hmc`` ]
    >>> acf = autocorr(samples)
    >>> import matplotlib.pyplot as plt
    >>> plt.semilogx(acf)

    .. image:: ../_static/autocorr.png

    Returns
    -------
    acf : ndarray, shape=(n_samples,) or shape=(n_samples, n_dims)
        The autocorrelation function of ``x``
    """
    x = np.atleast_1d(x)
    m = [slice(None), ] * len(x.shape)

    # For computational efficiency, crop the chain to the largest power of
    # two if requested.
    if fast:
        n = int(2**np.floor(np.log2(x.shape[axis])))
        m[axis] = slice(0, n)
        x = x
    else:
        n = x.shape[axis]

    # Compute the FFT and then (from that) the auto-correlation function.
    f = np.fft.fft(x-np.mean(x, axis=axis), n=2*n, axis=axis)
    m[axis] = slice(0, n)
    acf = np.fft.ifft(f * np.conjugate(f), axis=axis)[m].real
    m[axis] = 0
    return acf / acf[m]


def integrated_autocorr(x, window=None, fast=False):
    r"""Estimate the integrated autocorrelation time, :math:`\tau_{int}` of a
    time series.

    See `Sokal's notes <http://www.stat.unc.edu/faculty/cji/Sokal.pdf>`_ on
    MCMC and sample estimators for autocorrelation times. This qualtity is
    also called the statistical innefficiency [1], because the effective number
    of idependent samples in a positively-correlated MCMC timeseries is
    :math:`n_{samples}/\tau_{int}`.

    Parameters
    ----------
    x : ndarray, shape=(n_samples, n_dims)
        The time series, with time along axis 0.
    window : int, optional
        The size of the window to use. If not supplied, the window is chosen
        to be the index of the first time the autocorrelation function crosses
        zero.
    fast : bool, optional
        If ``True``, only use the largest ``2^n`` entries for efficiency.
        (default: False)

    References
    ----------
    .. [1] J. D. Chodera, W. C. Swope, J. W. Pitera, C. Seok, and K. A. Dill.
       JCTC 3(1):26-41, 2007.

    Returns
    -------
    t_int : shape=(n_dims,)
        The estimated integrated autocorrelation time of ``x``.
    """
    # Compute the autocorrelation function.
    if x.ndim == 1:
        x = x.reshape(-1, 1)

    f = autocorr(x, axis=0, fast=fast)
    if window is None:
        window = find_first((f < 0).astype(np.uint8))
    elif np.isscalar(window):
        window = window * np.ones(x.shape[1])
    else:
        raise NotImplementedError()

    tau = np.zeros(x.shape[1])
    for j in range(x.shape[1]):
        tau[j] = 1 + 2*np.sum(f[:window[j], j])
    return tau
