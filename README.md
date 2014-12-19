pyhmc: Hamiltonain Monte Carlo Sampling in Python
=================================================

This package is a straight-forward port of the functions `hmc2.m` and
`hmc2_opt.m` from the [MCMCstuff](http://www.lce.hut.fi/research/mm/mcmcstuff/) matlab toolbox written by Aki Vehtari. The code is originally based on the functions `hmc.m` from the [netlab toolbox](http://www.ncrg.aston.ac.uk/netlab/index.php)
written by Ian T Nabney. The portion of algorithm involving "windows" is derived from the C code for this function included in the [Software for Flexible Bayesian Modeling](http://www.cs.toronto.edu/~radford/fbm.software.html) written by Radford Neal.

The original Python [port](https://github.com/koepsell/pyhmc) was made by Kilian Koepsell, and subsequently modernized by Robert T. McGibbon.

Authors
-------
- Kilian Koepsell <kilian@berkeley.edu>
- Robert T. McGibbon <rmcgibbo@gmail.com>

This software is distributed under the BSD License (see LICENSE file).
