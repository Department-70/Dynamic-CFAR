# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 09:58:27 2022

@author: Geoffrey Dolinger
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Input
from tensorflow.keras.models import Sequential, Model
from inception_model import *

def create_inception_network(image_size,
                            n_channels,
                            inception_paths,
                            conv_layers=None,
                            inception_layers=None,
                            dense_layers=None,
                            activation='elu',
                            p_dropout=None,
                            lambda_l2=None,
                            lambda_l2_incep=None,
                            lrate=None, 
                            n_classes=None):
    
    '''
    This function is meant to work with the hw4_dolinger.py file and will build 
    a convolutional and/or inception network based on passed parameters.  
        image_size = X by Y image size.
        nchannels = number of channels
        conv_layers = dictionary of size N(number of layers) with information for:
            filters = individual layer's # of filters
            kernel_size = (k,k) kernel size
            pool_size = (p,p) pooling size or None
            stride = (s,s) stride or None
        inception_layers = dictionary of size N(number of layers) with information for:
            filters = internal layer's # of filters
            compress = boolean list in to compress inception layer
            inc_dropout = list of spacial dropout probabilities. 
        dense_layers = dictionary of size M(number of dense layers) with information for:
            units = fully connected(FC) layer nuerons. 
        p_dropout = dropout rate for FC layers or None
        lambda_L2 = L2 regularization for FC layers or None
        lambda_L2_incep= L2 regularization for inception layers or None
        lrate = learning rate used by optimizer
        n_classes = classes used for output tensor
    ''' 
    input_tensor = Input(shape =(image_size[0], image_size[1], n_channels),name = "input")
    kernel_regularizer=None
    # provide a print function and set the regularizer if either dropout or L2_regularization is included
    if lambda_l2 is not None:
        print("L2 Regularization")
        # set up regularization given L2 is not None
        kernel_regularizer = tf.keras.regularizers.l2(lambda_l2)
    if p_dropout is not None:
        print("Dropout")
    conv_tensors={}
    #loop all conv layers
    if conv_layers is not None:
        conv_tensors[0]=input_tensor
        for i, conv in enumerate(conv_layers):
            conv_tensors[i+1]= Conv2D(filters=conv['filters'],
                                        kernel_size=conv['kernel_size'],
                                        activation = "elu", 
                                        padding="same",
                                        name="Conv%s_%02d"%(conv['kernel_size'][0],i))(conv_tensors[i])
            if conv['pool_size'] is not None:
                pool_tensor=conv_tensors[i+1]
                conv_tensors[i+1]=MaxPooling2D(pool_size =conv['pool_size'],
                                               strides =conv['strides'], 
                                               name="MPool%s_%02d"%(conv['pool_size'][0],i))(pool_tensor)
    incep_tensors={}
    #loop all inception layers
    if inception_layers is not None:
        #if no conv layers then input tensor goes to inception layer
        if conv_layers is not None:
            incep_tensors[0] = conv_tensors[i+1]
        else:
            incep_tensors[0] = input_tensor
            
        for j, incep in enumerate(inception_layers):
            incep_tensors[j+1] = inception_model(incep_tensors[j],
                                                 nfilters=incep['filters'],
                                                 paths=inception_paths,
                                                 lambda_l2=lambda_l2_incep, 
                                                 name="inc%s"%(j),
                                                 compress=incep['compress'],
                                                 activation='elu',
                                                 elem_dropout=incep['inc_dropout'])
    dense_tensors = {}
    # make sure the last tensor is applied properly
    if inception_layers is not None:
        dense_tensors[0]=Flatten()(incep_tensors[j+1])
    else:
        dense_tensors[0]=Flatten()(conv_tensors[i+1])
    for k, dense in enumerate(dense_layers):
        #Add Dense layers based on input parameters for each layer
        dense_tensors[k+1] = Dense(dense['units'], use_bias=True, name="FC%02d"%(k), 
                  activation="elu",kernel_regularizer=kernel_regularizer)(dense_tensors[k])
        #If dropout rate exists then add dropout layer
        if p_dropout is not None:
            drop_tensor=dense_tensors[k+1]
            dense_tensors[k+1]=Dropout(rate=p_dropout, name="dropout_%02d"%(k))(drop_tensor) 
        
    output_tensor=Dense(n_classes, use_bias=True, name="Output", activation="softmax")(dense_tensors[k+1])
    
    opt = tf.keras.optimizers.Adam(learning_rate=lrate, beta_1=0.9, beta_2=0.999,
                                epsilon=None, decay=0.0, amsgrad=False)
    model=Model(inputs=input_tensor, outputs=output_tensor)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics =['categorical_accuracy'])    
    return model