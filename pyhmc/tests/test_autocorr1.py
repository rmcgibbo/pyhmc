import numpy as np
from pyhmc._utils import find_first
from scipy.optimize import curve_fit
from pyhmc.tests.test_autocorr2 import generate_AR1
from pyhmc import hmc, integrated_autocorr1

SAMPLES = None
PHI = 0.95

def setup():
    global SAMPLES
    SAMPLES = generate_AR1(PHI, 1, 10000, random_state=0).reshape(-1,1)


def test_2():
    corr_time1 = integrated_autocorr1(SAMPLES)

    # http://www.hep.fsu.edu/~berg/teach/mcmc08/material/lecture07mcmc3.pdf
    # For a large exponential autocorrelation time t_exp, the approximation
    # is that the integrated autocorrelation time is equal to twice the
    # exponential autocorrelation time, which for a AR1 model is 1/log(\phi)

    expected = -2/np.log(PHI)
    assert corr_time1.shape == (1,)
    assert np.abs(corr_time1 - expected) < 0.3 * expected


def test_find_first():
    X = np.array([
         [0, 0, 1, 1, 1],
         [0, 0, 1, 0, 0],
         [0, 1, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [1, 1, 1, 1, 1]], dtype=np.uint8).T

    reference = [2, 2, 1, -1, 0]
    for j in range(X.shape[1]):
        np.testing.assert_array_equal(
            find_first(X[:,j]), reference[j])


def test_3():
    samples = np.hstack((SAMPLES, SAMPLES))
    corr_time = integrated_autocorr1(samples)
    assert corr_time.shape == (2, )
