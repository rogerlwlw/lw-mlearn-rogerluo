# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 19:01:45 2019

@author: rogerluo
"""
# build dist
# setup.py bdist_wheel bdist_egg  sdist

import re
from setuptools import setup, find_packages

for line in open('lw_mlearn/__init__.py'):
    match = re.match("__version__ *= *'(.*)'", line)
    if match:
        __version__, = match.groups()

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="lw-mlearn-rogerluo",
    version=__version__,
    packages=find_packages(),
    inlcude_package_data=True,
    install_requires=[
        'sklearn-pandas>=1.8.0', 'numpy>=1.15.0', 'pandas>=0.24.0',
        'scikit-learn>=0.21.2', 'scipy>=1.1.0', 'matplotlib>=3.0.0',
        'yapf>=0.25.0', 'xgboost>=0.81', 'imbalanced-learn>=0.4.0'
    ],
    author="rogerluo",
    author_email="coolww@outlook.com",
    description=
    "Machine learning automatic workflow based on sklearn estimator",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='GNU/LGPLv3',
    url='https://github.com/rogerlwlw/lw-mlearn-rogerluo',
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
    keywords=[
        'pipeline optimization', 'hyperparameter optimization', 'data science',
        'machine learning', 'genetic programming', 'evolutionary computation'
    ],
)
