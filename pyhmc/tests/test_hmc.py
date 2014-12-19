import numpy as np
import scipy.stats
from scipy.optimize import approx_fprime
from pyhmc import hmc


def lnprob_gaussian(x, icov):
    logp = -np.dot(x, np.dot(icov, x)) / 2.0
    grad = -np.dot(x, icov)
    return logp, grad


def test_1():
    # test sampling from a highly-correlated gaussian
    dim = 2
    x0 = np.zeros(dim)
    cov = np.array([[1, 1.98], [1.98, 4]])
    icov = np.linalg.inv(cov)

    samples, logp, diag = hmc(lnprob_gaussian, x0, args=(icov,),
                  n_samples=10**4, n_burn=10**3,
                  n_steps=10, epsilon=0.20, return_diagnostics=True,
                  return_logp=True, random_state=2)

    C = np.cov(samples, rowvar=0, bias=1)
    np.testing.assert_array_almost_equal(cov, C, 1)
    for i in range(100):
        np.testing.assert_almost_equal(
            lnprob_gaussian(samples[i], icov)[0],
            logp[i])


def test_2():
    # test that random state is used correctly
    dim = 2
    x0 = np.zeros(dim)
    cov = np.array([[1, 1.98], [1.98, 4]])
    icov = np.linalg.inv(cov)

    samples1 = hmc(lnprob_gaussian, x0, args=(icov,),
                  n_samples=10, n_burn=0,
                  n_steps=10, epsilon=0.25, return_diagnostics=False,
                  random_state=0)

    samples2 = hmc(lnprob_gaussian, x0, args=(icov,),
                  n_samples=10, n_burn=0,
                  n_steps=10, epsilon=0.25, return_diagnostics=False,
                  random_state=0)
    np.testing.assert_array_almost_equal(samples1, samples2)


def test_3():
    rv = scipy.stats.loggamma(c=1)
    eps = np.sqrt(np.finfo(float).resolution)
    def logprob(x):
        return rv.logpdf(x), approx_fprime(x, rv.logpdf, eps)

    samples = hmc(logprob, [0], epsilon=1, n_steps=10, window=3, persistence=True)

    # import matplotlib.pyplot as pp
    (osm, osr), (slope, intercept, r) = scipy.stats.probplot(
        samples[:,0], dist=rv, fit=True)
    assert r > 0.99
    # pp.show()
