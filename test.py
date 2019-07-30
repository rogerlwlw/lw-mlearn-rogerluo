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
    X, y = make_classification(300, n_redundant=5, n_features=50)
    l = [
            'cleanNA_woe_frf_LogisticRegression',
            'cleanNA_woe_frf20_LogisticRegression',
            'cleanNA_woe_cleanNN_fRFErf_LogisticRegression',
            'cleanNA_woe_frf10_LogisticRegression',
            'cleanMean_woe_frf20_Nys_SGDClassifier',
            'cleanMean_oht_stdscale_frf20_Nys_SGDClassifier',
            'clean_oht_XGBClassifier',
            'clean_oht_oside_frf_XGBClassifier',
            'clean_oht_cleanNN_fxgb_XGBClassifier',
            'clean_oht_cleanNN_inlierForest_fxgb_XGBClassifier',
            'clean_oht_frf_RandomForestClassifier',
            'clean_oht_fxgb_BalancedRandomForestClassifier',
            'clean_oht_cleanNN_frf_AdaBoostClassifier',
            'clean_oht_fxgb_RUSBoostClassifier',
            'clean_oht_fxgb_cleanNN_GradientBoostingClassifier',
            'clean_oht_fsvm_cleanNN_GradientBoostingClassifier',
            'clean_oht_fxgb_cleanNN_DecisionTreeClassifier',
    ]
    # --
    for i in l:
        model = train_models(i, (X, y), (X, y),
                             max_leaf_nodes=10,
                             scoring=['KS', 'roc_auc'])
        model.delete_model()


if __name__ == '__main__':
    os.chdir('tests')
    test_model()
