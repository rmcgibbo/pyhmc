from __future__ import division
import numpy as np
from pyhmc.autocorr4 import integrated_autocorr4
from pyhmc.tests.test_autocorr2 import generate_AR1

# for testing against R
from rpy2.robjects import r
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()


def integrated_autocorr4r(x):
    r.require('coda')
    # r.assign('x', x)
    # print(r('spectrum0(x, max.freq=0.1)'))
    # print(r('spectrum0(x, max.freq=0.5)'))
    # print(r('spectrum0(x, max.freq=0.5)'))
    value = r['spectrum0'](x)[0][0] / np.var(x, ddof=1)
    return value


def test_1():
    for i in range(10):
        y = generate_AR1(phi=0.98, sigma=1, n_steps=10000, c=0, y0=0)
        val = integrated_autocorr4(y)
        rval = integrated_autocorr4r(y)
        np.testing.assert_almost_equal(val, rval, decimal=2)


def test_2():
    y = generate_AR1(phi=0.98, sigma=1, n_steps=1000, c=0, y0=0)
    y2 = np.vstack((y, y)).T
    val = integrated_autocorr4(y2)
    print(val)
