.. currentmodule:: pyhmc
Autocorrelation Estimators
==========================

Functions
---------
Naive summation of the empirical autocorrelation function results in a catastrophic error from the tail of the integral. There are variety of different estimators in the literature for :math:`\tau_{inf}`. This module
implements a few of them.

.. autosummary::
    :template: function.rst
    :toctree: generated/

    ~autocorr
    ~integrated_autocorr1
    ~integrated_autocorr2
    ~integrated_autocorr3
    ~integrated_autocorr4
    ~integrated_autocorr5

Theory
------
The autocorrelation time quantifies the rate of convergence of the sample
mean of a function of an (aperiodic / stationary / ergodic, recurrent) Markov chain. Suppose that our Markov chain, :math:`X = {X_1, X_2, \ldots X_n}` in the state space :math:`\Omega` has invariant distribution :math:`\pi`. For some function, :math:`g`, the goal is to estimate :math:`\mu \equiv \int_\Omega g(x) \pi (dx)`. The ergodic theorem
guarentees (if :math:`E_\pi |g| < \infty`)

.. math::

    \newcommand{\cov}{\mathop{\rm Cov}\nolimits}
    \newcommand{\var}{\mathop{\rm Var}\nolimits}


    \bar{g}_n \equiv \frac{1}{n} \sum_{i=1}^n g(X_i) \rightarrow \pi.

Assume that we run our Markov chain for :math:`n` iterations. **How accurate is** :math:`\bar{g}_n` **?**

Define the autocovariance, :math:`C_g(t) = \cov(g(X_s), g(X_{s+t}))`, and the autocorrelation :math:`\rho_g(t) = C_g(t)/C_g(0)`. Then,

.. math::

   \var(\bar{g}_n) &= \frac{1}{n^2} \sum_{i,j=1}^n \cov(g(X_i), g(X_j)) \\
   &\approx \frac{1}{n} \tau_{int} C_g(0)  \hspace{1em}\text{for}\hspace{2pt} n \gg \tau

Where

.. math::

    \tau_{int} = 1 + 2\sum_{t=1}^\infty \rho_g(t)

If each iteration of the chain was i.i.d, the asymptotic variance would be :math:`C_g(0)/n`, so :math:`\tau_{int}` can be thought of as a reduction in the
effective number of independent samples due to autocorrelation,

.. math::

    n_{effective} = \frac{n}{\tau_{int}}.

This quantity is also referred to as the statistical inefficiency, IAT, or IACT.

References
----------
1. `Sokal, A. D. "Monte Carlo Methods in Statistical Mechanics Foundations and New Algorithms" <http://www.stat.unc.edu/faculty/cji/Sokal.pdf>`_
2. `Flegal, J. M., M. Haran, and G. L. Jones. "Markov chain Monte Carlo: Can we trust the third significant figure?." Statistical Science (2008): 250-260. <http://projecteuclid.org/euclid.ss/1219339116>`_
3. `J. D. Chodera, W. C. Swope, J. W. Pitera, C. Seok, and K. A. Dill. "Use of the weighted histogram analysis method for the analysis of simulated and parallel tempering simulations." JCTC 3(1):26-41, 2007. <http://pubs.acs.org/doi/abs/10.1021/ct0502864>`_
4. `Plummer, M, N. Best, K. Cowles, and K. Vines (2006). CODA: convergence diagnosis and output analysis for MCMC. R News, 6(1) pp. 7-11. <http://cran.r-project.org/doc/Rnews/Rnews_2006-1.pdf#page=7>`_
5. `Geyer, C. J. "Practical markov chain monte carlo." Statistical Science (1992): 473-483. <http://projecteuclid.org/euclid.ss/1177011137>`_
6. `Heidelberger, P, and P D. Welch. "A spectral method for confidence interval generation and run length control in simulations." Communications of the ACM 24.4 (1981): 233-245. <http://dl.acm.org/citation.cfm?id=358630>`_
7. `Thompson, M. B. "A Comparison of Methods for Computing Autocorrelation Time" arXiv preprint arXiv:1011.0175 (2010). <http://arxiv.org/pdf/1011.0175.pdf>`_
