# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 19:01:45 2019

@author: rogerluo
"""
# build egg for lw_mlearn

import re
from setuptools import setup, find_packages


for line in open('lw_mlearn/__init__.py'):
    match = re.match("__version__ *= *'(.*)'", line)
    if match:
        __version__, = match.groups()
               
setup(
    name="lw-mlearn",
    version=__version__,
    packages=find_packages(include=['lw_mlearn']),
    install_requires=['sklearn-pandas', 
                      'scikit-learn', 
                      'pandas', 
                      'yapf'],
    description="This is a machine learning Package based on sklearn estimator",   
    # metadata to display on PyPI
    author="rogerluo",
    author_email="coolww@outlook.com", 
    
    license="PSF",
    keywords=['Machine learning', 'sklearn', 'pandas'],
    url= 'https://github.com/rogerlwlw/lw-mlearn'
    # could also include long_description, download_url, classifiers, etc.
)

print('setup run successfully')




