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
    # cov = np.array([[1, 0], [0, 1]])
    icov = np.linalg.inv(cov)

    samples, diag = hmc(lnprob_gaussian, x0, args=(icov,),
                  n_samples=10**3, n_omit=0,
                  steps=10, epsilon=0.2, return_diagnostics=True,
                  display=True)

    pp.plot(samples[:, 0], samples[:, 1])
    pp.show()


# def test_hmc():
#     import f_energy as en
#     dim = 5
#     x0 = np.zeros(dim)
#     M = np.eye(dim)
#     samples = hmc(en.f_multivariate_normal, x0, en.g_multivariate_normal, args=(M,),
#                   nsamples=10**3,nomit=10**3,steps=100,stepadj=.05)
#     C = np.cov(samples,rowvar=0,bias=1)
#     np.testing.assert_array_almost_equal(M,np.linalg.inv(C),1)
