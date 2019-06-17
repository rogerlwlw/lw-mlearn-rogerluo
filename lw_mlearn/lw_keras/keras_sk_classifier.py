# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 12:54:01 2019

@author: rogerluo
"""
from functools import wraps
from keras.wrappers.scikit_learn import KerasClassifier

from . mlp import  bfn_dense

class KerasClassifier_(KerasClassifier):
    '''wrap keras model as scikit estimators
    '''

    @wraps(KerasClassifier.fit)
    def fit(self, X, y=None, sample_weight=None, **kwargs):
        '''
        rewrite fit method to check input shape of X
        '''        
        self.set_params(input_shape = X.shape[1:])
        return super().fit(X, y, sample_weight,
                     **kwargs)
