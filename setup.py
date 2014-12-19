"""Hybrid Monte Carlo for Python
"""
from setuptools import find_packages, setup, Extension
import numpy as np
from Cython.Distutils import build_ext
import versioneer
versioneer.VCS = 'git'
versioneer.versionfile_source = 'hmc/_version.py'
versioneer.versionfile_build = 'hmc/_version.py'
versioneer.tag_prefix = '' # tags are like 1.2.0
versioneer.parentdir_prefix = 'hmc-' # dirname like 'myproject-1.2.0'

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

setup(
    name='pyhmc',
    author="Robert T. McGibbon",
    author_email='rmcgibbo@gmail.com',
    version=versioneer.get_version(),
    cmdclass=cmdclass,
    url="https://github.com/rmcgibbo/pyhmc",
    description=DOCLINES[0],
    long_description="\n".join(DOCLINES[2:]),
    license='MIT',
    install_requires=['numpy'],
    packages=find_packages(),
    zip_safe=False,
    ext_modules=[Extension('pyhmc._hmc', ['pyhmc/_hmc.pyx'],
                           include_dirs=[np.get_include()])],
)
