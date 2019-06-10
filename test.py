# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 14:10:59 2019

@author: rogerluo
"""
from lw_mlearn.lw_model import train_models
from lw_mlearn import pipe_main
from sklearn.datasets import make_classification


def test_model():
    '''with small fake data to walk through workflow
    '''
    X, y = make_classification(100, n_redundant=5, n_features=20)
    l = [
         'clean_oht_fxgb_cleanNN_XGBClassifier',
         'clean_oht_fxgb_cleanNN_RandomForestClassifier',
         'clean_oht_fxgb_cleanNN_GradientBoostingClassifier',
         'clean_oht_fxgb_cleanNN_DecisionTreeClassifier',
         'clean_oht_fxgb_cleanNN_BalancedRandomForestClassifier',
         'clean_oht_fxgb_cleanNN_RUSBoostClassifier',        
         'clean_oht_fRFElog_cleanNN_SVC',
         'clean_oht_fRFElog_cleanNN_GradientBoostingClassifier',

         ]
    # 
    for i in l:        
        model = train_models(i, (X, y), (X, y), max_leaf_nodes=10)
        model.delete_model()
        
if __name__ == '__main__':
    test_model()  