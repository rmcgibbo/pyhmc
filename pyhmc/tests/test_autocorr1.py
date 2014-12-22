import numpy as np
from pyhmc._hmc import find_first
from scipy.optimize import curve_fit
from pyhmc import hmc, autocorr, integrated_autocorr1

SAMPLES = None
PHI = 0.95

def setup():
    global SAMPLES
    SAMPLES = generate_AR1(PHI, 1, 10000).reshape(-1,1)


def generate_AR1(phi, sigma, n_steps, c=0, y0=0):
    y = np.zeros(n_steps)
    y[0] = y0
    rand = np.random.normal(scale=sigma, size=(n_steps,))
    for i in range(1, n_steps):
        y[i] = c + phi*y[i-1] + rand[i]
    return y


def test_1():
    acf = autocorr(SAMPLES)
    assert acf.shape == (10000, 1)
    assert np.max(acf) == 1.0

    def exp(x, t):
        return np.exp(-x/t)

    ref = -1 / np.log(PHI)
    t_exp = curve_fit(xdata=np.arange(1000), ydata=acf[:1000,0], f=exp, p0=(1,))[0]
    assert np.abs(t_exp - ref) < 0.30 * ref


def test_2():
    corr_time1 = integrated_autocorr1(SAMPLES)
    corr_time2 = integrated_autocorr1(SAMPLES, window=50)

    # http://www.hep.fsu.edu/~berg/teach/mcmc08/material/lecture07mcmc3.pdf
    # For a large exponential autocorrelation time t_exp, the approximation
    # is that the integrated autocorrelation time is equal to twice the
    # exponential autocorrelation time, which for a AR1 model is 1/log(\phi)

    expected = -2/np.log(PHI)
    assert corr_time1.shape == (1,)
    assert corr_time1.shape == (1,)
    assert np.abs(corr_time1 - expected) < 0.3 * expected
    assert np.abs(corr_time2 - expected) < 0.3 * expected


def test_find_first():
    X = np.array([
         [0, 0, 1, 1, 1],
         [0, 0, 1, 0, 0],
         [0, 1, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [1, 1, 1, 1, 1]], dtype=np.uint8).T

    np.testing.assert_array_equal(
        find_first(X),
        [2, 2, 1, -1, 0])


def test_3():
    samples = np.hstack((SAMPLES, SAMPLES))
    corr_time = integrated_autocorr1(samples)
    assert corr_time.shape == (2, )
