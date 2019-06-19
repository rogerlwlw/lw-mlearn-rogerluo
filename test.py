# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 14:10:59 2019

@author: rogerluo
"""
from lw_mlearn.lw_model import train_models
from sklearn.datasets import make_classification
import os


def test_model():
    '''with small fake data to walk through workflow
    '''
    X, y = make_classification(300, n_redundant=5, n_features=30)
    l = [
        'clean_oht_LDA_fxgb_cleanNN_AdaBoostClassifier',
        'clean_oht_fxgb_RUSBoostClassifier',
        'clean_oht_LDA_cleanNN_stdscale_GaussianProcessClassifier',
        'clean_oht_fRFErf_cleanNN_stdscale_GaussianProcessClassifier',
        'clean_oht_fRFErf_cleanNN_stdscale_SVC',
        'clean_oht_LDA_cleanNN_stdscale_SVC',
        'clean_oht_fxgb_cleanNN_XGBClassifier',
        'clean_oht_fxgb_oside_XGBClassifier',
        'clean_oht_frf_oside_XGBClassifier',
        'clean_oht_fxgb_cleanNN_RandomForestClassifier',
        'clean_oht_frf_RandomForestClassifier',
        'clean_oht_fxgb_BalancedRandomForestClassifier',
        'clean_oht_fxgb_cleanNN_GradientBoostingClassifier',
        'clean_oht_fRFElog_cleanNN_GradientBoostingClassifier',
        'clean_oht_fxgb_cleanNN_DecisionTreeClassifier',
    ]
    # --
    for i in l:
        model = train_models(i, (X, y), (X, y), max_leaf_nodes=10)
        model.delete_model()


if __name__ == '__main__':
    os.chdir('tests')
    test_model()
