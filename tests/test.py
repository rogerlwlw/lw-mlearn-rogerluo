# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 14:10:59 2019

@author: rogerluo
"""
from sklearn.datasets import make_classification
from shutil import rmtree

from lw_mlearn.lw_model import  run_analy

def _test_lw_model(X=None, y=None, delete=False):
    '''run test of ML_model class methods through train_models function
    '''
    if  X is None or y is None: 
        X, y = make_classification(300, n_redundant=5, n_features=50)
        
        run_analy(X, y, (X, y), verbose=1, q=10, dirs='test_result')
        if delete:
            rmtree('test_result', ignore_errors=True)
    return


if __name__ == '__main__':
    _test_lw_model(delete=False)
#    X, y = make_classification(300, n_redundant=5, n_features=50)    
#    run_analy(X,y, (X, y), model_list=['cleanNA_woe5_LogisticRegression'])


