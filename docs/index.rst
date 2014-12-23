.. currentmodule:: pyhmc

pyhmc: Hamiltonain Monte Carlo in Python
========================================

Introduction
------------
Hamiltonian Monte Carlo or Hybrid Monte Carlo (HMC) is a Markov chain Monte Carlo (MCMC) algorithm. Hamiltonian dynamics can be used to produce distant proposals for the Metropolis algorithm, thereby avoiding the slow exploration of the state space that results from the diffusive behaviour of simple random-walk proposals. It does this by taking a series of steps informed by first-order gradient information.

This feature allows it to converge much more quickly to high-dimensional target distributions compared to simpler methods such as Metropolis, Gibbs sampling (and derivatives).

References
----------
`Neal, Radford. "MCMC using Hamiltonian dynamics." Handbook of Markov Chain Monte Carlo 2 (2011) <http://arxiv.org/pdf/1206.1901.pdf>`_.

Example
-------
If you wanted to draw samples from a 5 dimensional Gaussian, you would do something like:

.. code-block:: python

    # define your probability distribution
    import numpy as np
    def logprob(x, ivar):
        logp = -0.5 * np.sum(ivar * x**2)
        grad = -ivar * x
        return logp, grad


.. code-block:: python

    # run the sampler
    from pyhmc import hmc
    ivar = 1. / np.random.rand(5)
    samples = hmc(logprob, x0=np.random.randn(5), args=(ivar,), n_samples=1e4)

.. code-block:: python

    # Optionally, plot the results (requires an external package)
    import triangle  # pip install triangle_plot
    figure = triangle.corner(samples)
    figure.savefig('triangle.png')


.. image:: https://cloud.githubusercontent.com/assets/641278/5500865/09b6271c-8703-11e4-9c6b-78add8e96d87.png

.. raw:: html

   <div style="display:none">


.. toctree::
    :maxdepth: 1

    installation

.. autosummary::
    :template: function.rst
    :toctree: generated/

    ~hmc

.. toctree::
    :maxdepth: 1

    tuning
    autocorr

.. raw:: html

    </div>

