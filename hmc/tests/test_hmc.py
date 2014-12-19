import numpy as np
from ..hmc import hmc
import matplotlib.pyplot as pp


def lnprob_gaussian(x, icov):
    logp = -np.dot(x, np.dot(icov, x)) / 2.0
    grad = -np.dot(x, icov)
    return logp, grad


def test_1():
    dim = 2
    x0 = np.zeros(dim)
    cov = np.array([[1, 1.98], [1.98, 4]])
    icov = np.linalg.inv(cov)

    samples = hmc(lnprob_gaussian, x0, args=(icov,),
                  n_samples=10**4, n_omit=10**3,
                  steps=15, epsilon=0.25, return_diagnostics=False,
                  display=False)

    C = np.cov(samples, rowvar=0, bias=1)
    np.testing.assert_array_almost_equal(cov, C, 1)
