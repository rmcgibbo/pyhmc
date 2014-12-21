import numpy as np
from scipy.linalg import toeplitz

# for testing against R
from rpy2.robjects import r
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

from .hmc import _check_random_state


def generate_AR1(phi, sigma, n_steps, c=0, y0=0, random_state=None):
    y = np.zeros(n_steps)
    y[0] = y0
    random = _check_random_state(random_state)
    rand = random.normal(scale=sigma, size=(n_steps,))
    for i in range(1, n_steps):
        y[i] = c + phi*y[i-1] + rand[i]
    return y


def yule_walker(X, aic=True, order_max=None, demean=True):
    """Estimate AR(p) parameters from a sequence X using Yule-Walker equation.

    Parameters
    ----------
    X : array-like
        1d array
    aic: bool
        If ``True``, then the Akaike Information Criterion is used to choose
        the order of the autoregressive model. If ``False``, the model of order
        ``order.max`` is fitted.
    order_max : integer, optional
        Maximum order of model to fit. Defaults to the smaller of N-1 and
        10*log10(N) where N is the length of the sequence.
    demean : bool
        True, the mean is subtracted from `X` before estimation.

    Returns
    -------
    rho : array, shape=(order,)
        The autoregressive coefficients
    sigma2 : float
        Variance of the nosie term
    aic : float
        Akaike Information Criterion
    """
    # this code is adapted from https://github.com/statsmodels/statsmodels.
    # changes are made to increase compability with R's ``ar.yw``.
    X = np.array(X)
    if demean:
        X -= X.mean()
    n = X.shape[0]

    if X.ndim > 1 and X.shape[1] != 1:
        raise ValueError("expecting a vector to estimate AR parameters")

    if order_max is None:
        order_max = min(n - 1, int(10 * np.log10(n)))

    r = np.zeros(order_max+1, np.float64)
    r[0] = (X**2).sum() / n

    for k in range(1, order_max+1):
        r[k] = (X[0:-k]*X[k:]).sum() / n

    orders = np.arange(1, order_max+1) if aic else [order_max]
    aics = np.zeros(len(orders))
    sigmasqs = np.zeros(len(orders))
    rhos = [None for i in orders]

    for i, order in enumerate(orders):
        r_left = r[:order]
        r_right = r[1:order+1]

        # R = toeplitz(r[:-1])
        R = toeplitz(r_left)
        # rho = np.linalg.solve(R, r[1:])
        rho = np.linalg.solve(R, r_right)
        # sigmasq = r[0] - (r[1:]*rho).sum()
        sigmasq = r[0] - (r_right*rho).sum()
        aic = len(X) * (np.log(sigmasq) + 1) + 2*order + 2*demean
        # R compability
        sigmasq = sigmasq * len(X)/(len(X) - (order + 1))

        aics[i] = aic
        sigmasqs[i] = sigmasq
        rhos[i] = rho

    index = np.argmin(aics)
    return rhos[index], sigmasqs[index]


def integrated_autocorr(x):
    # fit an AR(p) model, with p selected by AIC
    rho, sigma2 = yule_walker(x, order_max=10)
    # power spectral density at zero frequency
    spec0 = sigma2 / (1 - np.sum(rho))**2
    # divide by the variance
    tau_int = spec0 / np.var(x, ddof=1)
    return tau_int


def test_1():
    r("require('coda')")

    random = np.random.RandomState(1)
    for i in range(10):
        x = generate_AR1(phi=0.95, sigma=1, n_steps=1000, c=0, y0=0, random_state=random)
        r.assign('x', x)
        tau = r('nrow(x)/effectiveSize(x)')[0]
        np.testing.assert_approx_equal(tau, integrated_autocorr(x))
