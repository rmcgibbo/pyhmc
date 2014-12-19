"""
pyhmc: Hamiltonain Monte Carlo Sampling in Python
=================================================

This package is a straight-forward port of the functions `hmc2.m` and
hmc2_opt.m from the [MCMCstuff](http://www.lce.hut.fi/research/mm/mcmcstuff/)
matlab toolbox written by Aki Vehtari. The code is originally based on the
functions `hmc.m` from the [netlab toolbox](http://www.ncrg.aston.ac.uk/netlab/index.php)
written by Ian T Nabney. The portion of algorithm involving "windows" is
derived from the C code for this function included in the
[Software for Flexible Bayesian Modeling](http://www.cs.toronto.edu/~radford/fbm.software.html)
written by Radford Neal.

The original Python [port](https://github.com/koepsell/pyhmc) was made
by Kilian Koepsell, and subsequently modernized by Robert T. McGibbon.

Authors
-------
- Kilian Koepsell <kilian@berkeley.edu>
- Robert T. McGibbon <rmcgibbo@gmail.com>
"""
from __future__ import print_function, division
import numbers
import numpy as np
from ._hmc import hmc_main_loop

__all__ = ['hmc']


def hmc(fun, x0, args=(), display=False, steps=1, n_samples=1, n_burn=0,
        persistence=False, decay=0.9, epsilon=0.2, window=1,
        return_energies=False, return_diagnostics=False, random_state=None):
    """Hybrid Monte Carlo sampling.

    Uses a hybrid Monte Carlo algorithm to sample from the distribution P ~
    exp(f), where fun is the first argument to hmc. The Markov chain starts
    at the point x, and the function gradf is the gradient of the `energy'
    function f.

    Parameters
    ----------
    fun : callable
        A callable which takes a vector in the parameter spaces as input
        and returns the natural logarithm of the posterior probabily
        for that position, and gradient of the posterior probability with
        respect to the parameter vector, ``logp, grad = func(x, *args)``.
    x0 : 1-d array
      Starting point for the sampling Markov chain.

    Optional Parameters
    -------------------
    args : tuple
        additional arguments to be passed to fun().
    display : bool
        If True, enables verbose display output. Default: False
    steps : int
        Defines the trajectory length (i.e. the number of leapfrog
        steps at each iteration).
    n_samples : int
        The number of samples retained from the Markov chain.
    n_burn : int
        The number of samples omitted from the start of the chain as 'burn in'.
    persistence : bool
        If True, momentum persistence is used (i.e. the momentum
        variables decay rather than being replaced). Default: False
    decay : float, default=0.9
        Defines the decay used when a persistent update of (leap-frog)
        momentum is used. Bounded to the interval [0, 1.).
    epsilon : float, default=0.2.
        The step adjustment used in leap-frogs
    window : int, default=1
        The size of the acceptance window.
    return_energies : bool, default=False
        If True, the energy values for all samples are returned.
    return_diagnostics : bool, default=False
        If True, diagnostic information is returned (see below).

    Returns
    -------
    samples : array
      Array with data samples in rows.
    energies : array
      If return_energies is True, also returns an array of the energy values
      (i.e. negative log probabilities) for all samples.
    diagn : dict
      If return_diagnostics is True, also returns a dictionary with diagnostic
      information (position, momentum and acceptance threshold) for each step
      of the chain in diagn.pos, diagn.mom and diagn.acc respectively.
      All candidate states (including rejected ones) are stored in
      diagn['pos']. The diagn dictionary contains the following items:

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
    """
    # check some options
    assert steps >= 1, 'step size has to be >= 1'
    assert n_samples >= 1, 'n_samples has to be >= 1'
    assert n_burn >= 0, 'n_burn has to be >= 0'
    assert decay >= 0, 'decay has to be >= 0'
    assert decay <= 1, 'decay has to be <= 1'
    assert window >= 0, 'window has to be >= 0'
    if window > steps:
        window = steps
        if display:
            print("setting window size to step size %d" % window)

    if persistence:
        alpha = decay
        salpha = np.sqrt(1-alpha**2)
    else:
        alpha = salpha = 0.

    n_params = len(x0)

    # Initialize matrix of returned samples
    samples = np.zeros((n_samples, n_params))

    # Return energies?
    if return_energies:
        energies = np.zeros(n_samples)
    else:
        energies = np.zeros(0)

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
        fun, x0, args, p, samples, energies,
        diagn_pos, diagn_mom, diagn_acc,
        n_samples, n_burn, window,
        steps, display, persistence,
        return_energies, return_diagnostics,
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

    if return_energies or return_diagnostics:
        out = (samples,)
    else:
        return samples

    if return_energies:
        out += (energies,)
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
