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