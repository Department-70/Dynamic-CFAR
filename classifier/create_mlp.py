"""

@author: Geoffrey Dolinger
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from keras.layers import Conv2D, Dense, LeakyReLU, InputLayer, Flatten , SpatialDropout2D
from keras.layers import MaxPooling2D,Input, Dropout
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import backend

def create_mlp(input_shape, n_classes,                
                         dense_layers=None,
                         activation='elu',
                         dropout=None,
                         lambda_l2=None,
                         lrate=None):
    '''
    This function is meant to work with the hw8_dolinger.py file and will build 
    a U-net (or optional sequencial) network based on passed parameters.  
        input_shape = size of input image.
        n_classes = number of classes used by the final output classification
        conv_layers = dictionary of size N(number of layers) with information for:
            filters = individual layer's # of filters
            kernel_size = kernel size
            stride = stride or None
        unet_depth = the depth of the network. Same length as conv filters and 
             represents both up and down Unet architecture. Note set to all 1's 
             to make sequencial network. 
        activation = activation function for the Convolutional layers       
        dropout_spatial = spatial dropout rate for conv layers or None
        lambda_l2 = L2 regularization for conv layers or None
        lrate = learning rate used by optimizer
    '''
    
    kernel_regularizer=None
    if lambda_l2 is not None:
        # set up regularization given L2 is not None
        kernel_regularizer = tf.keras.regularizers.l2(lambda_l2)
    
    input_tensor = Input(shape=input_shape, name='input')
    
    tensor = input_tensor   
    #Dense Layers
    for i, dense in enumerate(dense_layers):
        tensor = Dense(dense['units'], use_bias=True, name="FullyConnected%s"%(i), 
                  activation="elu",kernel_regularizer=kernel_regularizer)(tensor)   
        if dropout is not None:
           tensor=Dropout(rate=dropout, name="dropout_%02d"%(i))(tensor)        
    #Final Output Layer    
    output_tensor=Dense(n_classes, use_bias=True, name="Output", activation="softmax")(tensor)
    
    opt = tf.keras.optimizers.Adam(learning_rate=lrate, beta_1=0.9, beta_2=0.999,
                                epsilon=None, decay=0.0, amsgrad=False)
    model=Model(inputs=input_tensor, outputs=output_tensor)
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), 
                  optimizer=opt, metrics =tf.keras.metrics.SparseCategoricalAccuracy()) 
    
    return model