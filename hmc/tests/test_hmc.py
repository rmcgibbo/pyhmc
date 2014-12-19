import numpy as np
from ..hmc import hmc


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

    samples, diag = hmc(lnprob_gaussian, x0, args=(icov,),
                  n_samples=10**4, n_burn=10**3,
                  steps=10, epsilon=0.20, return_diagnostics=True,
                  random_state=2, window=1)

    C = np.cov(samples, rowvar=0, bias=1)
    np.testing.assert_array_almost_equal(cov, C, 1)
    # import matplotlib.pyplot as pp
    # pp.plot(samples[:, 0], samples[:, 1], 'x-')
    # pp.show()

def test_2():
    # test that random state is used correctly
    dim = 2
    x0 = np.zeros(dim)
    cov = np.array([[1, 1.98], [1.98, 4]])
    icov = np.linalg.inv(cov)

    samples1 = hmc(lnprob_gaussian, x0, args=(icov,),
                  n_samples=10, n_burn=0,
                  steps=10, epsilon=0.25, return_diagnostics=False,
                  random_state=0)

    samples2 = hmc(lnprob_gaussian, x0, args=(icov,),
                  n_samples=10, n_burn=0,
                  steps=10, epsilon=0.25, return_diagnostics=False,
                  random_state=0)
    np.testing.assert_array_almost_equal(samples1, samples2)
