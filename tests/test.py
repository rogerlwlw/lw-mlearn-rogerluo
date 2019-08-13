# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 14:10:59 2019

@author: rogerluo
"""
import os
from sklearn.datasets import make_classification

from lw_mlearn.lw_model import get_default_estimators, train_models

def _test_lw_model(X=None, y=None, delete=False):
    '''run test of ML_model class methods through train_models function
    '''
    # --
    if not os.path.exists('tests_result'):
        os.makedirs('tests_result', exist_ok=True)
    os.chdir('tests_result')
    if  X is None or y is None: 
        X, y = make_classification(300, n_redundant=5, n_features=50)
    l = get_default_estimators()[:2]   
    for i in l:
        m = train_models(i, (X, y), (X, y),
                     max_leaf_nodes=10, scoring=['KS', 'roc_auc'])
        if delete:
            m.delete_model()
 
    return


if __name__ == '__main__':
    _test_lw_model(delete=True)
    

