"""

@author: Geoffrey Dolinger
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from keras.layers import Conv2D, Dense, LeakyReLU, InputLayer, Flatten , SpatialDropout2D
from keras.layers import MaxPooling2D, UpSampling2D, Embedding,Input,Concatenate, Add
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import backend

def create_unet(input_shape, n_classes, conv_layers=None,
                         unet_depth=None,
                         maxpool=None,
                         activation='elu',
                         dropout_spatial=None,
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
    #tesor stack for efficient Unet code. 
    tensor_stack =[]
    
    input_tensor = Input(shape=input_shape, name='input')
    
    tensor = input_tensor
    #Down the Unet
    d=1 #depth counter
    for i, conv in enumerate(conv_layers): #loop through conv layers
        #when depth increases then store tensor for skip connection and maxpool
        if unet_depth[i]>d:
            tensor_stack.append(tensor)
            d+=1
            tensor = MaxPooling2D(pool_size =maxpool,strides =maxpool, name="Pooling%s"%(i))(tensor)
        #add the Conv layer for each entry in conv_layer ist
        tensor = Conv2D(filters=conv['filters'],
                               kernel_size=conv['kernel_size'],strides=conv['stride'], 
                               padding="same", activation=activation,
                               kernel_regularizer=kernel_regularizer,
                               name="Con_down_%02d"%(i))(tensor)
        #Add spatial dropout is provided
        if dropout_spatial is not None:
            tensor = SpatialDropout2D(dropout_spatial)(tensor)
    
    # Up the Unet loop in reverse  
    for i,conv in reversed(list(enumerate(conv_layers))):
        #when depth decreases then upsample, add convolution and add skip connection
        if unet_depth[i]<d:
            tensor = UpSampling2D(size=maxpool,name="Up_sample%s"%(i))(tensor)
            tensor = Conv2D(filters=conv['filters'],
                                   kernel_size=conv['kernel_size'],strides=conv['stride'], 
                                   padding="same",  activation=activation,
                                   kernel_regularizer=kernel_regularizer,
                                   name="Con_up_%02d"%(i))(tensor)
            if dropout_spatial is not None:
                tensor = SpatialDropout2D(dropout_spatial)(tensor)
            # tensor = Concatenate()([tensor,tensor_stack.pop()])
            tensor = Add()([tensor,tensor_stack.pop()])
            d-=1 
        #If same depth then only add conv/dropout
        else:           
            tensor = Conv2D(filters=conv['filters'],
                                   kernel_size=conv['kernel_size'],strides=conv['stride'], 
                                   padding="same", activation=activation,
                                   kernel_regularizer=kernel_regularizer, 
                                   name="Con_up_%02d"%(i))(tensor)
            if dropout_spatial is not None:
                tensor = SpatialDropout2D(dropout_spatial)(tensor)
    
    #Final Output Layer
    output_tensor=Conv2D(filters=n_classes, kernel_size=(1,1), strides=(1,1), padding="same",name="Output", activation="softmax")(tensor)
    
    opt = tf.keras.optimizers.Adam(learning_rate=lrate, beta_1=0.9, beta_2=0.999,
                                epsilon=None, decay=0.0, amsgrad=False)
    model=Model(inputs=input_tensor, outputs=output_tensor)
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), 
                  optimizer=opt, metrics =tf.keras.metrics.SparseCategoricalAccuracy()) 
    
    return model