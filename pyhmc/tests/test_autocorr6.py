import numpy as np
from pyhmc.tests.test_autocorr2 import generate_AR1
from pyhmc import integrated_autocorr6, integrated_autocorr1


def test_1():
    x = generate_AR1(0.95, 1, 50**2+25, random_state=0)
    x2 = np.vstack((x, x)).T

    t1 = integrated_autocorr6(x)
    t2 = integrated_autocorr6(x2)
    assert t2.shape == (2,)
    assert t2[0] == t1 and t2[1] == t1[0]
