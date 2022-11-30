# -*- coding: utf-8 -*-

import io
import os
import re

from setuptools import find_packages
from setuptools import setup

# version
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'bpl', '__init__.py'), 'r') as f:
  init_py = f.read()
version = re.search('__version__ = "(.*)"', init_py).groups()[0]

# obtain long description from README
with io.open(os.path.join(here, 'README.md'), 'r', encoding='utf-8') as f:
  README = f.read()

# installation packages
packages = find_packages()
if 'docs' in packages:
  packages.remove('docs')
if 'tests' in packages:
  packages.remove('tests')

# setup
setup(
  name='brainpy-largescale',
  version=version,
  description='brainpy-largescale depends on brainpy',
  long_description=README,
  long_description_content_type="text/markdown",
  author='NanHu Neuromorphic Computing Laboratory Team',
  author_email='nhnao@cnaeit.com',
  packages=packages,
  python_requires='>=3.7',
  install_requires=['numpy>=1.15', 'tqdm', 'brainpy',
                    'brainpylib', 'numba', 'mpi4py', 'mpi4jax', 'jax[cpu]==0.3.24'],
  url='https://github.com/NH-NCL/brainpy-largescale',
  project_urls={
    "Bug Tracker": "https://github.com/NH-NCL/brainpy-largescale/issues",
    "Documentation": "https://brainpy.readthedocs.io/",
    "Source Code": "https://github.com/NH-NCL/brainpy-largescale",
  },
  keywords=('brainpy largescale, '
            'computational neuroscience, '
            'brain-inspired computation, '
            'dynamical systems, '
            'differential equations, '
            'brain modeling, '
            'brain dynamics modeling, '
            'brain dynamics programming'),
  classifiers=[
    'Natural Language :: English',
    'Operating System :: OS Independent',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: Apache Software License',
    'Topic :: Scientific/Engineering :: Bio-Informatics',
    'Topic :: Scientific/Engineering :: Mathematics',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'Topic :: Software Development :: Libraries',
  ],
  license='Apache-2.0 license',
)
