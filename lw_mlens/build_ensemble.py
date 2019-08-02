# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 11:20:44 2019

@author: BBD
"""

import pandas as pd
import numpy as np

from mlens.ensemble import SuperLearner, Subsemble, BlendEnsemble
from mlens.preprocessing import Subset

def build_stack(estimators, meta_estimator, propagate_features=None, 
                proba=True):
    '''
    
    '''
    
    
    if meta_estimator is not None:
        pass
    
    return