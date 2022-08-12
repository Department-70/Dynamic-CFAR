# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 10:11:33 2022

@author: Geoffrey Dolinger
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from keras.layers import SimpleRNN, Dense, LeakyReLU, InputLayer, Flatten 
from keras.layers import GRU, LSTM, Embedding, Conv1D, Dropout,Input, MultiHeadAttention 
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import backend

def create_atten_network(input_dim,
                         emb_out,
                         n_classes,
                         conv_layers=None,
                         atten_layers=None,
                         dense_layers=None,
                         activation='elu',
                         activation_dense='sigmoid',
                         dropout_dense=None,
                         dropout_atten=None,
                         lambda_l2_dense=None,
                         L2_atten = None,
                         lrate=None):
    
    '''
    This function is meant to work with the hw7_dolinger.py file and will build 
    a multiheaded attention network based on passed parameters.  
        inputDim = encoded size of input data.
        emb_out = output dimension for embedding layer
        n_classes = number fo classes used by the final output classification
        conv_layers = dictionary of size N(number of layers) with information for:
            filters = individual layer's # of filters
            kernel_size = kernel size
            stride = stride or None
        atten_layers = dictionary of size N(number of MHA layers) with information for:
            units = internal layer's # of filtersheads
        dense_layers = dictionary of size M(number of dense layers) with information for:
            units = fully connected(FC) layer nuerons. 
        activation = activation function for the recurrent layers
        activation_dense = activation function for the dense layers
        attent_dropout = dropout rate for attention layers or None
        dropout_dense = dropout rate for FC layers or None
        lambda_L2_dense = L2 regularization for FC layers or None
        L2_atten= L2 regularization for attention layers or None
        lrate = learning rate used by optimizer
    ''' 

    kernel_regularizer_dense=None
    kernel_regularizer_atten=None
    # provide a print function and set the regularizer if either dropout or L2_regularization is included
    if lambda_l2_dense is not None:
        # set up regularization given L2 is not None
        kernel_regularizer_dense = tf.keras.regularizers.l2(lambda_l2_dense)
    
    if L2_atten is not None:
        # set up regularization given L2 is not None
        kernel_regularizer_atten = tf.keras.regularizers.l2(L2_atten)
    if dropout_atten is None:
        dropout_atten = 0.0
    
    input_tensor = Input(shape =(input_dim,),name = "input")
    #embedding layer set-up
    tensor = Embedding(
         input_dim=input_dim,
         output_dim=emb_out)(input_tensor)
    #Convolutional layers (optional)
    if conv_layers is not None:
        for i, conv in enumerate(conv_layers): #enumerate for different conditions for input layer   
            tensor=Conv1D(filters=conv['filters'],kernel_size=conv['kernel_size'],strides=conv['stride'], padding="valid", name="Con_%02d"%(i))(tensor)
    #Multi-headed Attention layers
    for i, atten in enumerate(atten_layers):
 
        tensor = MultiHeadAttention(num_heads=atten['units'],key_dim=tensor.shape[1], 
                                value_dim=tensor.shape[1],
                                kernel_regularizer=kernel_regularizer_atten, 
                                dropout=dropout_atten)(tensor,tensor)
    
    tensor= Flatten()(tensor) 
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

    
    