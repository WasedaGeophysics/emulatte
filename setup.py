from setuptools import setup, find_packages

setup(
    name = 'w1dem',
    version = '0.0.1',
    description = 'A Primitive Tools for Electromagnetic Explorations',
    long_description = 'README',
    author = 'Takumi Ueda',
    author_email = 'takumi.ueda@waseda.jp',
    url = 'https://github.com/WasedaGeophysics/w1dem',
    license = license("MIT"),
    packages = find_packages(exclude=('docs', 'tests', 'tutorials')),
    install_requires = open('requirements.txt').read().splitlines(),
)