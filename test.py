# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 14:10:59 2019

@author: rogerluo
"""
import os
from shutil import rmtree
from  lw_mlearn.lw_model import get_default_estimators, train_models
from sklearn.datasets import make_classification

def _test_lw_model(X=None, y=None, delete=False):
    '''run test of ML_model class methods through train_models function
    '''
    # --
    if not os.path.exists('tests'):
        os.makedirs('tests', exist_ok=True)
    os.chdir('tests')
    if  X is None or y is None: 
        X, y = make_classification(300, n_redundant=5, n_features=50)
    l = get_default_estimators()   
    for i in l:
        train_models(i, (X, y), (X, y),
                     max_leaf_nodes=10, scoring=['KS', 'roc_auc'])
    if delete: 
        rmtree('tests')
 
    return


if __name__ == '__main__':
    _test_lw_model()
    

