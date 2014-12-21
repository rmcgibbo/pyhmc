.. currentmodule:: pyhmc
Tuning HMC
==========

Goals
-----
If you care about the quality of the samples you obtain, you must tune the
sampler. It's simply unavoidable.

When the step size ``epsilon``, is too small, the system is too conservative,
and doesn't explore parameter space rapidly. On the other hand, when
``epsilon`` is too large, the trajectory is unstable and all of the steps
basically get rejected during the Metropolis step.

`Theoretical analysis <http://arxiv.org/abs/1001.4460>`_ indicates that the
optimal balance of these two factors comes when the acceptance probability is
:math:`\sim 0.651`. This isn't a strict number that you have to nail, but
getting the acceptance probability within the :math:`0.4 - 0.9` is a good
target.

Practical
---------
In practice, some simple advice is to set the number of steps per sample to
:math:`\sim 10`,  ``n_steps = 10``, and then adjust ``epsilon`` to tune the
acceptance rate to an acceptible value.

Then, use the :func:`autocorr` and :func:`integrated_autocorr` to estimate the
correlation time of the sampler. The integrated autocorrelation time determines
the statistical errors in Monte Carlo measurements of :math:`\langle f \rangle`,
which converge like :math:`\sim 1/\sqrt{\frac{n_{samples}}{\tau_{int}}}`. The
"effective" number of independent samples thus basically reduced by a factor
of :math:`\tau_{int}`.

To achieve a reasonably small statistical error it is necessary to make a run of length :math:`\approx 1000\tau_{int}`.

See `Sokal's notes <http://www.stat.unc.edu/faculty/cji/Sokal.pdf>`_ on MCMC and sample estimators for autocorrelation times for more details.


References
----------
`Beskos, Alexandros, et al. "Optimal tuning of the hybrid Monte Carlo algorithm." Bernoulli 19.5A (2013): 1501-1534. <http://arxiv.org/abs/1001.4460>`_
