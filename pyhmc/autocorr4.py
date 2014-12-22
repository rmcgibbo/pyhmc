from __future__ import division, absolute_import, print_function
import numpy as np
import statsmodels.api as sm


def integrated_autocorr4(x, max_length=200):
    """Estimate the integrated autocorrelation time, :math:`\tau_{int}` of a
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
    This method fits a linear regression to the lower frequency of the
    log spectrum, which is extrapolated to zero to estimate the zero
    frequency power spectrum.

    References
    ----------
    .. [1] Heidelberger, P., and P. D. Welch. "A spectral method for
       confidence interval generation and run length control in simulations."
       Communications of the ACM 24.4 (1981): 233-245.

    Returns
    -------
    tau_int : ndarray, shape=(n_dims,)
        The estimated integrated autocorrelation time of each dimension in
        ``x``, considered independently.
    """
    if x.ndim == 1:
        x = x.reshape(-1, 1)

    pvar = np.var(x, axis=0, ddof=1)

    v0 = np.zeros(x.shape[1])
    for j in range(x.shape[1]):
        xx = x[:, j]
        batch_size = 1
        if max_length is not None and len(x) > max_length:
            batch_size = int(len(x) / max_length)
            xx = xx[:max_length*batch_size].reshape(max_length, batch_size).mean(axis=1)

        v0[j] = _spectrum0(xx) * batch_size
    return v0 / pvar


def _spectrum0(x):
    """The spectral density at frequency zero is estimated by fitting a
    glm to the low-frequency end of the periodogram.

    The raw periodogram is calculated for the series 'x' and a
    generalized linear model with family 'Gamma' and log link is
    fitted to the periodogram. The predictor is proportional to frequency.

    Notes
    -----
    This code is translated from R's CODA package (do.spectrum0).
    """
    N = len(x)
    Nfreq = int(N/2)
    freq = 1 / N + np.arange(Nfreq) / N

    xfft = np.fft.fft(x)
    spec = ((xfft * np.conjugate(xfft)).real / N)[1:Nfreq+1]

    one = np.ones(Nfreq),
    f1 = np.sqrt(3) * (4 * freq - 1),
    df = np.vstack((one, f1)).T

    model = sm.GLM(spec, df, sm.families.Gamma(sm.families.links.log)).fit()
    v0 = model.predict([1, -np.sqrt(3)])[0]
    return v0



