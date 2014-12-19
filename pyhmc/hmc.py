from __future__ import print_function, division, absolute_import

import numbers
import numpy as np
from ._hmc import hmc_main_loop

__all__ = ['hmc', '__version__']


def hmc(fun, x0, n_samples=1000, args=(), display=False, n_steps=1, n_burn=0,
        persistence=False, decay=0.9, epsilon=0.2, window=1,
        return_logp=False, return_diagnostics=False, random_state=None):
    """Hamiltonian Monte Carlo sampler.

    Uses a Hamiltonian / Hybrid Monte Carlo algorithm to sample from the
    distribution P ~ exp(f). The Markov chain starts at the point x0. The
    callable ``fun`` should return the log probability and gradient of the
    log probability of the target density.

    Parameters
    ----------
    fun : callable
        A callable which takes a vector in the parameter spaces as input
        and returns the natural logarithm of the posterior probabily
        for that position, and gradient of the posterior probability with
        respect to the parameter vector, ``logp, grad = func(x, *args)``.
    x0 : 1-d array
      Starting point for the sampling Markov chain.

    Other Parameters
    ----------------
    n_samples : int
        The number of samples retained from the Markov chain.
    args : tuple
        additional arguments to be passed to fun().
    display : bool
        If True, enables verbose display output. Default: False
    n_steps : int
        Defines the trajectory length between Metropolized steps (i.e. the
        number of leapfrog steps at each iteration before accept/reject).
    n_burn : int
        The number of samples omitted from the start of the chain as 'burn in'.
    persistence : bool
        If True, momentum persistence is used (i.e. the momentum
        variables decay rather than being replaced). Default: False
    decay : float, default=0.9
        Defines the decay used when a persistent update of (leap-frog)
        momentum is used. Bounded to the interval [0, 1.).
    epsilon : float, default=0.2.
        The step adjustment used in the integrator. In physics-terms, this
        is the time step.
    window : int, default=1
        The size of the acceptance window (See [1], Section 5.4)
    return_logp : bool, default=False
        If True, the energy values for all samples are returned.
    return_diagnostics : bool, default=False
        If True, diagnostic information is returned (see below).

    Returns
    -------
    samples : array, shape=(n_samples, n_params)
        Array with data samples in rows.
    logp : array, shape=(n_samples)
        If ``return_logp`` is ``True``, also returns an array of the log
        probability for all samples.
    diagn : dict
        If ``return_diagnostics`` is ``True``, also returns a dictionary with
        diagnostic information (position, momentum and acceptance threshold)
        for each step of the chain in ``diagn['pos']``, ``diagn['mom']`` and
        ``diagn['acc']`` respectively. All candidate states (including
        rejected ones) are stored in ``diagn['pos']``. The diagn dictionary
        contains the following items:

          ``pos`` : array
             the position vectors of the dynamic process
          ``mom`` : array
             the momentum vectors of the dynamic process
          ``acc`` : array
             the acceptance thresholds
          ``rej`` : float
             the rejection rate
          ``stp`` : float
             the step size vectors

    References
    ----------
    .. [1] Neal, Radford. "MCMC using Hamiltonian dynamics." Handbook of
       Markov Chain Monte Carlo 2 (2011).
    """
    # check some options
    assert n_steps >= 1, 'step size has to be >= 1'
    assert n_samples >= 1, 'n_samples has to be >= 1'
    assert n_burn >= 0, 'n_burn has to be >= 0'
    assert decay >= 0, 'decay has to be >= 0'
    assert decay <= 1, 'decay has to be <= 1'
    assert window >= 0, 'window has to be >= 0'
    if window > n_steps:
        window = n_steps
        if display:
            print("setting window size to step size %d" % window)

    if persistence:
        alpha = decay
        salpha = np.sqrt(1-alpha**2)
    else:
        alpha = salpha = 0.

    x0 = np.asarray(x0, dtype=np.double)
    n_params = len(x0)

    # Initialize matrix of returned samples
    samples = np.zeros((n_samples, n_params))

    # Return energies?
    if return_logp:
        logp = np.zeros(n_samples)
    else:
        logp = np.zeros(0)

    # Return diagnostics?
    if return_diagnostics:
        diagn_pos = np.zeros((n_samples, n_params))
        diagn_mom = np.zeros((n_samples, n_params))
        diagn_acc = np.zeros(n_samples)
    else:
        diagn_pos = None
        diagn_mom = None
        diagn_acc = None

    random = _check_random_state(random_state)
    p = random.randn(n_params)

    # Main loop.
    all_args = [
        fun, x0, args, p, samples, logp,
        diagn_pos, diagn_mom, diagn_acc,
        n_samples, n_burn, window,
        n_steps, display, persistence,
        return_logp, return_diagnostics,
        alpha, salpha, epsilon, random,]

    n_reject = hmc_main_loop(*all_args)

    if display:
        print('\nFraction of samples rejected:  %g\n'%(n_reject / n_samples))

    # Store diagnostics
    if return_diagnostics:
        diagn = dict()
        diagn['pos'] = diagn_pos   # positions matrix
        diagn['mom'] = diagn_mom   # momentum matrix
        diagn['acc'] = diagn_acc   # acceptance treshold matrix
        diagn['rej'] = n_reject / n_samples   # rejection rate
        diagn['stps'] = epsilon    # stepsize vector

    if return_logp or return_diagnostics:
        out = (samples,)
    else:
        return samples

    if return_logp:
        out += (logp,)
    if return_diagnostics:
        out += (diagn,)
    return out



def _check_random_state(seed):
    """Turn seed into a np.random.RandomState instance

    If seed is None, return the RandomState singleton used by np.random.
    If seed is an int, return a new RandomState instance seeded with seed.
    If seed is already a RandomState instance, return it.
    Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)
