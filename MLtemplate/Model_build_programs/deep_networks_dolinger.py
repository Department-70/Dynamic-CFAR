# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 10:51:10 2022

@author: Geoffrey Dolinger
"""

import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import InputLayer, Dense, Dropout
from tensorflow.keras.models import Sequential

def deep_network_basic(n_inputs, n_hidden, n_output, activation='elu', lrate=0.001, 
                       opt=None, loss='mse', dropout=None, dropout_input =None,
                       kernel_regularizer=None, metrics =None):
    
    if dropout is not None:
        print("Dropout")
    if kernel_regularizer is not None:
        print("L2 Regularization")
        kernel_regularizer = tf.keras.regularizers.l2(kernel_regularizer)
    
    
    model = Sequential();
    
    model.add(InputLayer(input_shape=(n_inputs,)))
    
    if dropout_input is not None:
       model.add(Dropout(rate=dropout_input, name='dropout_input')) 
    
    
    for N in range(np.size(n_hidden)):
        model.add(Dense(n_hidden[N], use_bias=True, name="hidden%s"%(N), 
                  activation=activation,kernel_regularizer=kernel_regularizer))
        
        if dropout is not None:
            model.add(Dropout(rate=dropout, name="dropout_%02d"%(N)))
        
    model.add(Dense(n_output, use_bias=True, name="output", activation=activation))
    
    # Optimizer
    if opt is None:
        opt = tf.keras.optimizers.Adam(learning_rate=lrate, beta_1=0.9, beta_2=0.999,
                                epsilon=None, decay=0.0, amsgrad=False)
    
    # Bind the optimizer and the loss function to the model
    model.compile(loss=loss, optimizer=opt, metrics = metrics)
    

    return model