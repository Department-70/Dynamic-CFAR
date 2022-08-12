"""
Advanced Machine Learning, 2022
HW 3 Original CNN model building code

@author: Geoffrey Dolinger
"""

import tensorflow as tf
import numpy as np

from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import InputLayer, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential

def create_cnn_classifier_network(image_size, 
                                  nchannels,
                                  conv_layers,
                                  dense_layers,
                                  p_dropout=None,
                                  lambda_l2=None,
                                  lrate=None, 
                                  n_classes=None):
    '''
    This function is meant to work with the hw3_dolinger.py file and will build 
    a convolutional nueral network based on passed parameters.  
        image_size = X by Y image size.
        nchannels = number of channels
        conv_layers = dictionary of size N(number of layers) with information for:
            filters = individual layer's # of filters
            kernel_size = (k,k) kernel size
            pool_size = (p,p) pooling size or None
            stride = (s,s) stride or None
        dense_layers = dictionary of size M(number of dense layers) with information for:
            units = fully connected(FC) layer nuerons. 
        p_dropout = dropout rate for FC layers or None
        lambda_L2 = L2 regularization for FC layers or None
        

    '''    
    kernel_regularizer=None
    # provide a print function and set the regularizer if either dropout or L2_regularization is included
    if p_dropout is not None:
        print("Dropout")
    if lambda_l2 is not None:
        print("L2 Regularization")
        # set up regularization given L2 is not None
        kernel_regularizer = tf.keras.regularizers.l2(lambda_l2)
    
    #Establish a sequencial model
    model = Sequential();
    #Input layer set with the input image and channel sizes
    model.add(InputLayer(input_shape=(image_size[0], image_size[1], nchannels)))
    #Loop through the convolutional and pooling layer. 
    for i, conv in enumerate(conv_layers): #enumerate for different conditions for input layer        
        #Add Convulutional layers based on input parameters for each layer
        model.add(Conv2D(filters=conv['filters'],kernel_size=conv['kernel_size'],activation = "elu", padding="same", name="Con_%02d"%(i)))
        #If pooling option exists then include a Maxpooling layer
        if conv['pool_size'] is not None:
            model.add(MaxPooling2D(pool_size =conv['pool_size'],strides =conv['strides'], name="Pool_%02d"%(i)))
    
    #Flatten the data to enter the Dense Layer
    model.add(Flatten())
    #Loop through the dense layers with droupout/L2_regularization.
    for i, dense in enumerate(dense_layers):
        #Add Dense layers based on input parameters for each layer
        model.add(Dense(dense['units'], use_bias=True, name="FC%02d"%(i), 
                  activation="elu",kernel_regularizer=kernel_regularizer))
        #If dropout rate exists then add dropout layer
        if p_dropout is not None:
           model.add(Dropout(rate=p_dropout, name="dropout_%02d"%(i))) 
        
    model.add(Dense(n_classes, use_bias=True, name="Output", activation="softmax"))
    
    # Optimizer
    opt = tf.keras.optimizers.Adam(learning_rate=lrate, beta_1=0.9, beta_2=0.999,
                                epsilon=None, decay=0.0, amsgrad=False)
    
    # Bind the optimizer and the loss function to the model
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics =['categorical_accuracy'])
    

    return model