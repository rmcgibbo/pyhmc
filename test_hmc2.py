import numpy as np
from hmc2 import hmc

def test_checkgrad():
    import nose
    import f_energy as en
    dim = 10
    x0 = np.zeros(dim)
    M = np.random.rand(dim,dim)
    M += M.T.copy()
    error = hmc(en.f_multivariate_normal, x0, en.g_multivariate_normal, args=(M,), checkgrad=True)
    nose.tools.assert_almost_equal(error,0,5)


def test_hmc():
    import f_energy as en
    dim = 5
    x0 = np.zeros(dim)
    M = np.eye(dim)
    samples = hmc(en.f_multivariate_normal, x0, en.g_multivariate_normal, args=(M,),
                  nsamples=10**3,nomit=10**3,steps=100,stepadj=.05)
    C = np.cov(samples,rowvar=0,bias=1)
    np.testing.assert_array_almost_equal(M,np.linalg.inv(C),1)


def test_set_randstate():
    from hmc2 import get_state, set_state
    import nose
    state = get_state()
    rand = np.random.rand()
    set_state(state)
    nose.tools.assert_equal(np.random.rand(),rand)


def test_set_momentum():
    import hmc2
    from hmc2 import get_state, set_state
    hmc2.HMC_MOM = np.ones(3)
    state = get_state()
    hmc2.HMC_MOM = np.zeros(3)
    set_state(state)
    np.testing.assert_array_equal(np.ones(3),hmc2.HMC_MOM)


def test_addlogs():
    import nose
    from hmc2 import addlogs
    a,b = np.random.randn(2)
    nose.tools.assert_almost_equal(addlogs(a,b),np.log(np.exp(a)+np.exp(b)))

if __name__ == '__main__':
    import nose
    # nose.runmodule(exit=False,argv=['nose','-s','--pdb-failures'])
    nose.runmodule(exit=False,argv=['nose','-s'])
