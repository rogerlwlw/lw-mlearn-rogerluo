# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 11:20:44 2019

@author: BBD
"""

import pandas as pd
import numpy as np

from mlens.ensemble import SuperLearner, Subsemble, BlendEnsemble
from mlens.preprocessing import Subset

from functools import wraps

from matplotlib import docstring

def build_stack(estimators, meta_estimator, propagate_features=None, 
                proba=True):
    '''
    
    '''
    
    
    if meta_estimator is not None:
        pass
    
    return

def get_score_fn(scoring):
    '''
    scoring: str or callable
    
    return
    ----
    scoring function that accepts an array of true values and an array 
    of predictions: score = f(y_true, y_pred).      
    '''
    if callable(scoring):
        return scoring
    else:
        return 
    
@docstring.Appender(SuperLearner.add.__doc__)
@wraps(SuperLearner)
def build_super(
             estimators,
             preprocessing,
             proba,
             meta, 
             propagate_features,                       
             folds=3,
             shuffle=True,
             random_state=0,
             
             scorer='roc_auc',
             raise_on_exception=True,
             n_jobs=2,
             model_selection=False,
             sample_size=20):
    '''return stack ensemble model instance
    
    params 
    -------
    see __doc__ of SuperLearner & SuperLearner.add, already wrapped in
    '''
    scorer = get_score_fn(scorer) # return scoring functions
    L = locals().copy()
    # --
    ens = SuperLearner().add()
    return ens