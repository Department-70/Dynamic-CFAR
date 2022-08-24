# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 10:11:33 2022

@author: Geoffrey Dolinger
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from keras.layers import SimpleRNN, Dense, LeakyReLU, InputLayer, GRU, LSTM, Embedding, Conv1D, Dropout,Input  
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import backend


def create_rnn_network(input_dim,
                         n_classes,
                         conv_layers=None,
                         rnn_layers=None,
                         gru_layers=None,
                         lstm_layers=None,
                         dense_layers=None,
                         activation='elu',
                         activation_dense='elu',
                         recurrent_dropout=0.0,
                         dropout_dense=None,
                         lambda_l2_dense=None,
                         lambda_l2_rnn=None,
                         lrate=None):
    
    '''
    This function is meant to work with the hw4_dolinger.py file and will build 
    a convolutional and/or inception network based on passed parameters.  
        inputDim = encoded size of input data.
        emb_out = output dimension for embedding layer
        n_classes = number of classes used by the final output classification
        conv_layers = dictionary of size N(number of layers) with information for:
            filters = individual layer's # of filters
            kernel_size = kernel size
            stride = stride or None
        rnn_layers = dictionary of size N(number of rnn layers) with information for:
            filters = internal layer's # of filters
        gru_layers = dictionary of size N(number of gru layers) with information for:
            filters = internal layer's # of filters
        lstm_layers = dictionary of size N(number of lstm layers) with information for:
            filters = internal layer's # of filters
        dense_layers = dictionary of size M(number of dense layers) with information for:
            units = fully connected(FC) layer nuerons. 
        activation = activation function for the recurrent layers
        activation_dense = activation function for the dense layers
        recurrent_dropout = dropout rate for recurrent layers or None
        dropout_dense = dropout rate for FC layers or None
        lambda_L2_dense = L2 regularization for FC layers or None
        lambda_L2_rnn= L2 regularization for recurrent layers or None
        lrate = learning rate used by optimizer
    ''' 

    kernel_regularizer_dense=None
    kernel_regularizer_rnn=None
    # provide a print function and set the regularizer if either dropout or L2_regularization is included
    if lambda_l2_dense is not None:
        # set up regularization given L2 is not None
        kernel_regularizer_dense = tf.keras.regularizers.l2(lambda_l2_dense)
    
    if lambda_l2_rnn is not None:
       # set up regularization given L2 is not None
       kernel_regularizer_rnn = tf.keras.regularizers.l2(lambda_l2_rnn)
    
    input_tensor = Input(shape =input_dim,name = "input")   
    #Convolutional layers (optional)
    tensor=input_tensor
    if conv_layers is not None:
        for i, conv in enumerate(conv_layers): #enumerate for different conditions for input layer   
            tensor=Conv1D(filters=conv['filters'],kernel_size=conv['kernel_size'],strides=conv['stride'], padding="valid", name="Con_%02d"%(i))(tensor)
    # for typical operation on one of RNN, GRU, or LSTM will be selected
    if rnn_layers is not None:
        for i, rnn in enumerate(rnn_layers): #enumerate for different conditions for input layer   
            #The final layer needs to have return sequences=False, but the intermediate layers should be True.
            if i == (len(rnn_layers)-1):
                ret_seq=False
            else:
                ret_seq=True
            tensor = SimpleRNN(units=rnn['units'],
                      activation=activation,
                      use_bias=True,
                      return_sequences=ret_seq,
                      kernel_initializer='random_uniform',
                      bias_initializer='random_uniform',
                      kernel_regularizer=kernel_regularizer_rnn,
                      unroll=False,name="RNN%02d"%(i))(tensor)
    if gru_layers is not None:
        for i, gru in enumerate(gru_layers): #enumerate for different conditions for input layer   
            if i == (len(gru_layers)-1):
                ret_seq=False
            else:
                ret_seq=True
            tensor=GRU(units=gru['units'],
                       activation=activation,
                       use_bias=True,
                       return_sequences=ret_seq,
                       kernel_initializer='random_uniform',
                       bias_initializer='random_uniform',
                       recurrent_dropout=recurrent_dropout,
                       kernel_regularizer=kernel_regularizer_rnn,
                       unroll=False,name="GRU%02d"%(i))(tensor)
    if lstm_layers is not None:
        for i, lstm in enumerate(lstm_layers): #enumerate for different conditions for input layer   
            if i == (len(lstm_layers)-1):
                ret_seq=False
            else:
                ret_seq=True
            tensor=LSTM(units=lstm['units'],
                       activation=activation,
                       use_bias=True,
                       return_sequences=ret_seq,
                       kernel_initializer='random_uniform',
                       bias_initializer='random_uniform',
                       recurrent_dropout=recurrent_dropout,
                       kernel_regularizer=kernel_regularizer_rnn,
                       unroll=False, name="LSTM%02d"%(i))(tensor)
    for i, dense in enumerate(dense_layers):
         #Add Dense layers based on input parameters for each layer
        tensor=Dense(dense['units'], use_bias=True, name="FC%02d"%(i), 
                   activation=activation_dense,kernel_regularizer=kernel_regularizer_dense)(tensor)
    
        if dropout_dense is not None:
           tensor=Dropout(rate=dropout_dense, name="dropout_%02d"%(i))(tensor)
    #Final Output layer       
    tensor = Dense(n_classes, use_bias=True, name="Output", activation="softmax")(tensor)
    
    opt = tf.keras.optimizers.Adam(learning_rate=lrate, beta_1=0.9, beta_2=0.999,
                                epsilon=None, decay=0.0, amsgrad=False)

    model=Model(inputs=input_tensor, outputs=tensor)
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), 
                  optimizer=opt, metrics =tf.keras.metrics.SparseCategoricalAccuracy()) 
    
    return model

    
    