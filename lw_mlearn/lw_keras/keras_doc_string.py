# -*- coding: utf-8 -*-
"""
__doc__ for keras parameters commonly used
Created on Mon Jun  3 17:35:12 2019

@author: rogerluo
"""


class Keras_doc_():

    activation = '''
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        '''
    input_shape = '''
        # Input shape
            nD tensor with shape: `(batch_size, ..., input_dim)`.
            The most common situation would be
            a 2D input with shape `(batch_size, input_dim)`.
        '''

    output_shape = '''
        # Output shape
            nD tensor with shape: `(batch_size, ..., units)`.
            For instance, for a 2D input with shape `(batch_size, input_dim)`,
            the output would have shape `(batch_size, units)`..
        '''
    optimizer = '''
        optimizer: String (name of optimizer) or optimizer instance.
        See [optimizers](/optimizers).
        '''
    loss = '''
        loss: String (name of objective function) or objective function.
        See [losses](/losses).
        If the model has multiple outputs, you can use a different loss
        on each output by passing a dictionary or a list of losses.
        The loss value that will be minimized by the model
        will then be the sum of all individual losses.
        '''
    metrics = ''' 
        metrics: List of metrics to be evaluated by the model
        during training and testing.
        Typically you will use `metrics=['accuracy']`.
        To specify different metrics for different outputs of a
        multi-output model, you could also pass a dictionary,
        such as `metrics={'output_a': 'accuracy'}`.
        '''
    target_tensors = '''    
        target_tensors: By default, Keras will create placeholders for the model's
        target, which will be fed with the target data during training. 
        If instead you would like to use your own target tensors (in turn, Keras 
        will not expect external Numpy data for these targets at training time),
        you can specify them via the target_tensors argument. It can be a single 
        tensor (for a single-output model), a list of tensors, or a dict mapping 
        output names to target tensors.
        '''
    epochs = '''
        epochs: Integer. Number of epochs to train the model. An epoch is an
        iteration over the entire x and y data provided. Note that in conjunction
        with initial_epoch,  epochs is to be understood as "final epoch". The model
        is not trained for a number of iterations given by epochs, but merely until
        the epoch of index epochs is reached.
        '''
    batch_size = '''
        batch_size: Integer or None. Number of samples per gradient update. If
        unspecified, batch_size will default to 32.
        '''
    callbacks = '''
        callbacks: List of keras.callbacks.Callback instances. List of callbacks to
        apply during training and validation (if ). See callbacks.
        '''
    validation_split = '''
        validation_split: Float between 0 and 1. Fraction of the training data to be
        used as validation data. The model will set apart this fraction of the
        training data, will not train on it, and will evaluate the loss and any model
        metrics on this data at the end of each epoch. The validation data is
        selected from the last samples in the x and y data provided, before
        shuffling.
        '''
    validation_data = '''
        validation_data: tuple (x_val, y_val) or tuple  (x_val, y_val,
        val_sample_weights) on which to evaluate the loss and any model metrics at
        the end of each epoch. The model will not be trained on this data. 
        validation_data will override validation_split.
        '''
    shuffle = '''
        shuffle: Boolean (whether to shuffle the training data before each epoch) 
        or str (for 'batch'). 'batch' is a special option for dealing with the
        limitations of HDF5 data; it shuffles in batch-sized chunks. Has no effect
        when steps_per_epoch is not None.
        '''
