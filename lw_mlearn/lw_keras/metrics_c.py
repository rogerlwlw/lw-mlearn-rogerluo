# -*- coding: utf-8 -*-
"""
Created on Mon May 27 15:17:17 2019

@author: rogerluo
"""
import tensorflow as tf
from keras.metrics import get
from sklearn.metrics import (roc_auc_score, 
                             average_precision_score, f1_score, hinge_loss,
                             recall_score)


def get_metrics(identifier):
    '''return metrics
    '''    
    if isinstance(identifier, str):
        if metrics_collection.get(identifier) is not None:           
            return metrics_collection.get(identifier)
        else:
            return get(identifier)
    elif callable(identifier):
        return identifier


#%%
def sk_auc(y_true, y_pred):
    return tf.py_function(roc_auc_score, (y_true, y_pred), tf.double)

def sk_average_precision(y_true, y_pred):
    return tf.py_function(average_precision_score, (y_true, y_pred), tf.double)

def sk_hinge_loss(y_true, y_pred):
    return tf.py_function(hinge_loss, (y_true, y_pred), tf.double)  
      
metrics_collection = dict(
        sk_auc = sk_auc,
        sk_average_precision = sk_average_precision,
        sk_hinge_loss = sk_hinge_loss,
        )

