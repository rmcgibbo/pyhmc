from __future__ import division, absolute_import, print_function
import numpy as np


def integrated_autocorr5(x, size='sqrt'):
    """Estimate the integrated autocorrelation time, :math:`\tau_{int}` of a
    time series.

    Parameters
    ----------
    x : ndarray, shape=(n_samples, n_dims)
        The time series, with time along axis 0.
    size : {'sqrt', 'cbrt'}
        The batch size. The default value is "sqroot", which uses the square
        root of the sample size. "cuberoot" will cause the function to use the
        cube root of the sample size. A numeric value may be provided if
        neither "sqrt" nor "cbrt" is satisfactory.

    Notes
    -----
    This method uses the constent batch means estimator [1] where the number
    of batches are chosen as functions of the overall run length.

    References
    ----------
    .. [1] Flegal, J. M., Haran, M. and Jones, G. L. (2008) Markov chain Monte
       Carlo: Can we trust the third significant figure? Statistical Science,
       23, 250-260.

    Returns
    -------
    tau_int : ndarray, shape=(n_dims,)
      The estimated integrated autocorrelation time of each dimension in
      ``x``, considered independently.
    """
    if x.ndim == 1:
        x = x.reshape(-1, 1)

    if size == 'sqrt':
        batch_size = int(np.sqrt(x.shape[0]))
    elif size == 'cbrt':
        batch_size = int(x.shape[0]**0.5)
    elif np.isscalar(size):
        batch_size = size
    else:
        raise NotImplementedError('unrecoginized argument, size=%s' % size)

    bigvar = np.var(x, axis=0, ddof=1)
    # leave off the extra bit at the end that's not a clean multiple
    x = x[:batch_size*(len(x)//batch_size)]
    bigmean = np.mean(x, axis=0)

    sigma2_bm = np.zeros(x.shape[1])
    for j in range(x.shape[1]):
        # reshape into the batches, and then compute the batch-means
        bm = x[:, j].reshape(-1, batch_size).mean(axis=1)
        sigma2_bm[j] = (batch_size / (len(bm)-1)) * np.sum((bm-bigmean[j])**2)

    return sigma2_bm / bigvar

