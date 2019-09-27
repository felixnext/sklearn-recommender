'''
Setup Script for the Recommender Module

This will install the recommender module to the local python distribution.
'''


import os
from setuptools import setup
from setuptools import find_packages


__status__      = "Package"
__copyright__   = "Copyright 2019"
__license__     = "MIT License"
__version__     = "0.1.5"

# 01101100 00110000 00110000 01110000
__author__      = "Felix Geilert"


this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'readme.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='sklearn-recommender',
      version=__version__,
      description='Sklearn Extension to integration recommender functions',
      long_description=long_description,
      long_description_content_type="text/markdown",
      keywords='recommender systems',
      url='https://github.com/felixnext/sklearn-recommender',
      author='Felix Geilert',
      license='MIT License',
      packages=find_packages(),
      install_requires=[ 'numpy', 'scikit-learn', 'scipy', 'pandas', 'nltk' ],
      include_package_data=True,
      zip_safe=False)
