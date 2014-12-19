pyhmc: Hamiltonain Monte Carlo in Python
=================================================
[![Build Status](https://travis-ci.org/rmcgibbo/pyhmc.svg)](https://travis-ci.org/rmcgibbo/pyhmc)
[![License](https://img.shields.io/badge/license-BSD-red.svg?style=flat)](https://pypi.python.org/pypi/pyhmc)
[![Docs](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](https://pythonhosted.org/pyhmc/)

This package is a straight-forward port of the functions `hmc2.m` and
`hmc2_opt.m` from the [MCMCstuff](http://www.lce.hut.fi/research/mm/mcmcstuff/) matlab toolbox written by Aki Vehtari. The code is originally based on the functions `hmc.m` from the [netlab toolbox](http://www.ncrg.aston.ac.uk/netlab/index.php)
written by Ian T Nabney. The portion of algorithm involving "windows" is derived from the C code for this function included in the [Software for Flexible Bayesian Modeling](http://www.cs.toronto.edu/~radford/fbm.software.html) written by Radford Neal.

The original Python [port](https://github.com/koepsell/pyhmc) was made by Kilian Koepsell, and subsequently modernized by Robert T. McGibbon.

Authors
-------
- Kilian Koepsell <kilian@berkeley.edu>
- Robert T. McGibbon <rmcgibbo@gmail.com>

This software is distributed under the BSD License (see LICENSE file).

Example
-------

If you wanted to draw samples from a 5 dimensional Gaussian, you would do
something like:

```
import numpy as np
def logprob(x, ivar):
    logp = -0.5 * np.sum(ivar * x**2)
    grad = -ivar * x
    return logp, grad
```

```
from pyhmc import hmc
ivar = 1. / np.random.rand(5)
samples = hmc(logprob, x0=np.random.randn(5), args=(ivar,), n_samples=1e4)
```

```
# Using the beautiful $ pip install triangle_plot
import triangle
figure = triangle.corner(samples)
figure.savefig('triangle.png')
```

![triangle](https://cloud.githubusercontent.com/assets/641278/5500865/09b6271c-8703-11e4-9c6b-78add8e96d87.png)

