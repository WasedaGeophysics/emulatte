# Copyright 2021 Waseda Geophysics Laboratory
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from setuptools import setup, find_packages

setup(
    name = 'emulatte',
    version = '0.1.0',
    description = 'A Primitive Tools for Electromagnetic Explorations',
    long_description = 'README',
    author = 'Takumi Ueda',
    author_email = 'takumi.ueda@waseda.jp',
    url = 'https://github.com/WasedaGeophysics/emulatte.git',
    license = license("Apache2.0"),
    packages = find_packages(exclude=('docs', 'tests', 'tutorials')),
    install_requires = open('requirements.txt').read().splitlines(),
)

#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
import os

from setuptools import setup, find_packages

try:
    with open('README.rst') as f:
        readme = f.read()
except IOError:
    readme = ''

def _requires_from_file(filename):
    return open(filename).read().splitlines()

# version
here = os.path.dirname(os.path.abspath(__file__))
version = next((line.split('=')[1].strip().replace("'", '')
                for line in open(os.path.join(here,
                                              'pypipkg',
                                              '__init__.py'))
                if line.startswith('__version__ = ')),
               '0.0.dev0')

setup(
    name="pypipkg",
    version=version,
    url='https://github.com/user/pypipkg',
    author='kinpira',
    author_email='kinpira@example.jp',
    maintainer='kinpira',
    maintainer_email='kinpira@example.jp',
    description='Package Dependency: Validates package requirements',
    long_description=readme,
    packages=find_packages(),
    install_requires=_requires_from_file('requirements.txt'),
    license="MIT",
    classifiers=[
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'License :: OSI Approved :: MIT License',
    ],
    entry_points="""
      # -*- Entry points: -*-
      [console_scripts]
      pkgdep = pypipkg.scripts.command:main
    """,
)