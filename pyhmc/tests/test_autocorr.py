import numpy as np
#import matplotlib.pyplot as pp
from scipy.optimize import curve_fit
from pyhmc import hmc, autocorr, integrated_autocorr

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
    assert np.abs(t_exp - ref) < 0.25 * ref


def test_2():
    corr_time = integrated_autocorr(SAMPLES)

    # http://www.hep.fsu.edu/~berg/teach/mcmc08/material/lecture07mcmc3.pdf
    # For a large exponential autocorrelation time t_exp, the approximation
    # is that the integrated autocorrelation time is equal to twice the
    # exponential autocorrelation time, which for a AR1 model is 1/log(\phi)

    expected = -2/np.log(PHI)
    assert corr_time.shape == (1,)
    assert np.abs(corr_time - expected) < 0.25 * expected
