#!/usr/bin/env python

from os.path import exists
from setuptools import setup

packages = ['symbolic_array']

tests = [p + '.tests' for p in packages]


setup(name='symbolic_array',
      version='0.1.0',
      description='Symbolic Numpy Arrays',
      maintainer='Matthew Rocklin',
      maintainer_email='mrocklin@gmail.com',
      license='BSD',
      keywords='symblic,numpy',
      packages=packages + tests,
      long_description=(open('README.rst').read() if exists('README.rst')
                        else ''),
      install_requires=list(open('requirements.txt').read().strip().split('\n')),
      zip_safe=False)
