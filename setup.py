"""Hybrid Monte Carlo Sampler for Python

This package is a straight-forward port of the functions hmc2.m and
hmc2_opt.m from the MCMCstuff matlab toolbox written by Aki Vehtari
<http://www.lce.hut.fi/research/mm/mcmcstuff/>.

The code is originally based on the functions hmc.m from the netlab toolbox
written by Ian T Nabney <http://www.ncrg.aston.ac.uk/netlab/index.php>.

The portion of algorithm involving "windows" is derived from the C code for
this function included in the Software for Flexible Bayesian Modeling
written by Radford Neal <http://www.cs.toronto.edu/~radford/fbm.software.html>.

This software is distributed under the BSD License (see LICENSE file).
"""
from setuptools import find_packages, setup, Extension
import numpy as np
from Cython.Build import cythonize

DOCLINES = __doc__.split("\n")
CLASSIFIERS = """\
Development Status :: 3 - Alpha
Intended Audience :: Science/Research
Intended Audience :: Developers
License :: OSI Approved :: BSD
Programming Language :: Python
Operating System :: OS Independent
"""

setup(
    name='hmc',
    author="Morgan Fouesneau",
    url="https://github.com/rmcgibbo/pyhmc",
    description=DOCLINES[0],
    long_description="\n".join(DOCLINES[2:]),
    version=0.1,
    license='MIT',
    install_requires=['numpy'],
    packages=find_packages(),
    ext_modules=cythonize(
        Extension('hmc/_hmc', ['hmc/_hmc.pyx'],
        include_dirs=[np.get_include()]))
)
