import numpy as np
from pyhmc import integrated_autocorr3
from pyhmc.tests.test_autocorr2 import generate_AR1

# for testing against R
from rpy2.robjects import r
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()


def setup():
    with open(os.path.expanduser('~/.R/Makevars'), 'w') as f:
        f.write('''CC=gcc
CXX=g++''')
    r("install.packages('mcmc', repos='http://cran.us.r-project.org')")
    r.require('mcmc')


def test_1():
    y = generate_AR1(0.95, 1, 10000)
    tau = integrated_autocorr3(y)

    r.assign('x', y)
    r('popvar = (var(x)*(nrow(x)-1)/nrow(x))')
    r('init = initseq(x)')
    tau_ref = r('initseq(x)$var.pos / popvar')[0]
    print(tau, tau_ref)
    np.testing.assert_array_almost_equal(tau, tau_ref)


def test_2():
    y = generate_AR1(0.95, 1, 10000)
    y2 = np.vstack((y, y)).T
    tau = integrated_autocorr3(y2)
    assert tau.shape == (2,)
    assert tau[0] == tau[1]
