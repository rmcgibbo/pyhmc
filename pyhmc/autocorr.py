# The MIT License (MIT)
#
# Copyright (c) 2010-2013 Daniel Foreman-Mackey & contributors.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from __future__ import division, print_function, absolute_import
import numpy as np

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


def integrated_autocorr(x, axis=0, window=50, fast=False):
    """Estimate the integrated autocorrelation time of a time series.

    See `Sokal's notes <http://www.stat.unc.edu/faculty/cji/Sokal.pdf>`_ on
    MCMC and sample estimators for autocorrelation times.

    Parameters
    ----------
    x : ndarray, shape=(n_samples,) or shape=(n_samples, n_dims)
        The time series. If multidimensional, set the time axis using the
        ``axis`` keyword argument and the function will be computed for every
        other axis.
    axis :  int, optional
        The time axis of ``x``. Assumed to be the first axis if not specified.
    window : int, optional
        The size of the window to use. (default: 50)
    fast : bool, optional
        If ``True``, only use the largest ``2^n`` entries for efficiency.
        (default: False)

    Returns
    -------
    acftime : shape=(n_dims,)
        The estimated integrated autocorrelation time of ``x``.
    """
    # Compute the autocorrelation function.
    f = autocorr(x, axis=axis, fast=fast)

    # Special case 1D for simplicity.
    if len(f.shape) == 1:
        return 1 + 2*np.sum(f[1:window])

    # N-dimensional case.
    m = [slice(None), ] * len(f.shape)
    m[axis] = slice(1, window)
    tau = 1 + 2*np.sum(f[m], axis=axis)

    return tau
