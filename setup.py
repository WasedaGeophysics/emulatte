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

PYTHON_REQUIRES = '>=3.6'

with open('README.md', 'r', encoding='utf-8') as fp:
    readme = fp.read()
LONG_DESCRIPTION = readme
LONG_DESCRIPTION_CONTENT_TYPE = 'text/markdown'

setup(
    name = 'emulatte',
    version = '0.1.0',
    description = 'A Primitive Tools for Electromagnetic Explorations',
    long_description=LONG_DESCRIPTION,
    long_description_content_type=LONG_DESCRIPTION_CONTENT_TYPE,\
    author = 'Takumi Ueda',
    author_email = 'takumi.ueda@waseda.jp',
    url = 'https://github.com/WasedaGeophysics/emulatte.git',
    download_url = 'https://github.com/WasedaGeophysics/emulatte.git',
    license = license("Apache2.0"),
    keywords = 'electromagnetics geophysics EM 1DEM'
    packages = find_packages(exclude=('docs', 'lab', 'example', 'analytical')),
    install_requires = open('requirements.txt').read().splitlines(),
)
