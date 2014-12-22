import numpy as np
from pyhmc.tests.test_autocorr2 import generate_AR1
from pyhmc import integrated_autocorr5

# for testing against R
from rpy2.robjects import r
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()


def test_1():
    x = generate_AR1(0.95, 1, 50**2+25, random_state=0)
    val = integrated_autocorr5(x, size='sqrt')

    r.require('batchmeans')
    r.assign('x', x)
    ref = r('(bm(x)$se)^2 * length(x) / var(x)')[0]
    np.testing.assert_almost_equal(val, ref)


def test_2():
    x = generate_AR1(0.95, 1, 50**2+25, random_state=0)
    x2 = np.vstack((x, x)).T
    val = integrated_autocorr5(x2, size='sqrt')
    assert val.shape == (2,)
    assert val[0] == val[1]
