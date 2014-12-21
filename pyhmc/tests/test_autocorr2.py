import numpy as np
# for testing against R
from rpy2.robjects import r
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

from pyhmc.autocorr2 import integrated_autocorr
from pyhmc.hmc import _check_random_state

def generate_AR1(phi, sigma, n_steps, c=0, y0=0, random_state=None):
    y = np.zeros(n_steps)
    y[0] = y0
    random = _check_random_state(random_state)
    rand = random.normal(scale=sigma, size=(n_steps,))
    for i in range(1, n_steps):
        y[i] = c + phi*y[i-1] + rand[i]
    return y


def test_1():
    r("require('coda')")

    random = np.random.RandomState(1)
    for i in range(10):
        x = generate_AR1(phi=0.95, sigma=1, n_steps=1000, c=0, y0=0, random_state=random)
        r.assign('x', x)
        tau = r('nrow(x)/effectiveSize(x)')[0]
        np.testing.assert_approx_equal(tau, integrated_autocorr(x))
