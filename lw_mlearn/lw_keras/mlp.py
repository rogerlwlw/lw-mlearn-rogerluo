# -*- coding: utf-8 -*-
"""
Keras Multi-layer perceptron model

Created on Fri May 17 15:34:51 2019

@author: rogerluo
"""
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import regularizers as reg

from .metrics_c import get_metrics
from .docstring import Substitution
from .keras_doc_string import Keras_doc_


@Substitution(optimizer=Keras_doc_.optimizer.strip(),
              loss=Keras_doc_.loss.strip(),
              metrics=Keras_doc_.metrics.strip(),
              input_shape=Keras_doc_.input_shape.strip())
def bfn_dense(dense_layer=(1, ),
              dropout=(0, ),
              input_shape=(32, ),
              out_activation='sigmoid',
              hidden_activation='relu',
              optimizer='adam',
              loss='binary_crossentropy',
              metrics=['binary_accuracy'],
              penalty=None,
              dropout_ratio=0.2,
              **kwargs):
    '''build function to return densely connected network
    
    parameters
    ----
    dense_layer:tuple
        The ith element represents the number of neurons in the ith hidden layer
    dropout: boolean tuple
        true of the ith element represents ith layer is connected with dropout layer
    {input_shape}
    
    out_activation:
        Activation function for the ouput layer
        
    hidden_activation:
        Activation function for the hidden layer  
        
    {optimizer}
    {loss}
    {metrics}   
    
    return
    ---- 
    keras muli-layer perceptron model compiled  
    '''
    model = Sequential()

    regularizer = {'l1': reg.l1(), 'l2': reg.l2(), 'l1_l2': reg.l1_l2()}

    layer_structure = []
    for i, item in enumerate(dense_layer):
        if i < len(dropout):
            layer_structure.append((item, dropout[i]))
        else:
            layer_structure.append(item, 0)

    n = 0
    for i, j in layer_structure:
        n += 1
        if n == 1:
            model.add(
                Dense(i,
                      input_shape=input_shape,
                      activation=hidden_activation,
                      kernel_regularizer=regularizer.get(penalty)))
        elif n == len(dense_layer):
            model.add(
                Dense(i,
                      activation=out_activation,
                      kernel_regularizer=regularizer.get(penalty)))
        else:
            model.add(
                Dense(i,
                      activation=hidden_activation,
                      kernel_regularizer=regularizer.get(penalty)))
        if bool(j):
            model.add(Dropout(dropout_ratio))

    metrics = [get_metrics(i) for i in metrics]
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model
