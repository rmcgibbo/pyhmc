"""Hybrid Monte Carlo for Python

This package is a straight-forward port of the functions ``hmc2.m`` and
``hmc2_opt.m`` from the
`MCMCstuff <http://www.lce.hut.fi/research/mm/mcmcstuff/>`__ matlab
toolbox written by Aki Vehtari. The code is originally based on the
functions ``hmc.m`` from the `netlab
toolbox <http://www.ncrg.aston.ac.uk/netlab/index.php>`__ written by Ian
T Nabney. The portion of algorithm involving "windows" is derived from
the C code for this function included in the `Software for Flexible
Bayesian
Modeling <http://www.cs.toronto.edu/~radford/fbm.software.html>`__
written by Radford Neal.

The original Python `port <https://github.com/koepsell/pyhmc>`__ was
made by Kilian Koepsell, and subsequently modernized by Robert T.
McGibbon.

Authors
-------

-  Kilian Koepsell kilian@berkeley.edu
-  Robert T. McGibbon rmcgibbo@gmail.com

This software is distributed under the BSD License (see LICENSE file).
"""
from setuptools import find_packages, setup, Extension
import numpy as np
from Cython.Distutils import build_ext
import versioneer
versioneer.VCS = 'git'
versioneer.versionfile_source = 'pyhmc/_version.py'
versioneer.versionfile_build = 'pyhmc/_version.py'
versioneer.tag_prefix = '' # tags are like 1.2.0
versioneer.parentdir_prefix = 'pyhmc-' # dirname like 'myproject-1.2.0'

DOCLINES = __doc__.split("\n")
CLASSIFIERS = """\
Development Status :: 3 - Alpha
Intended Audience :: Science/Research
Intended Audience :: Developers
License :: OSI Approved :: BSD
Programming Language :: Python
Operating System :: OS Independent
"""

cmdclass = versioneer.get_cmdclass()
cmdclass['build_ext'] = build_ext
extensions = [
    Extension('pyhmc._hmc', ['pyhmc/_hmc.pyx'],
              include_dirs=[np.get_include()]),
    Extension('pyhmc._utils', ['pyhmc/_utils.pyx'],
              include_dirs=[np.get_include()]),
]

setup(
    name='pyhmc',
    author="Robert T. McGibbon",
    author_email='rmcgibbo@gmail.com',
    version=versioneer.get_version(),
    cmdclass=cmdclass,
    url="https://github.com/rmcgibbo/pyhmc",
    description=DOCLINES[0],
    long_description="\n".join(DOCLINES[2:]),
    license='BSD',
    install_requires=['numpy'],
    packages=find_packages(),
    zip_safe=False,
    ext_modules=extensions,
)
