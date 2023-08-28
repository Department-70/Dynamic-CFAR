### Calculates: a probability of each distribution (Gaussian, K-Low, K-Medium, K-High, P-Low, P-Medium, P-High) to fit current data point (current data return)  
#
## Input: the Data file specified by the Args Class 
# data_ss is a data array with a reduse by one dim
# model_disc is a path to the model's discriminators
# test_num is number of test that will run
# disc_vector is a vector with the numbers each of which corresponds to certain distribution
# args is a parameter from Arg Class Module arguments
# args.max_test is the number of tests that run. If args.max_test is None that the whole data set will be trained.
# 
## Additional variables
# numDist is the number of distributions: Gaussian, K-low, K-medium, K-high, P-low, P-medium, P-high 
#
## Output: distribution_tensors.csv which is the probability list for each data point corresponding to the likelihood of each radar distribution.

### Load necessary packages.
# complex math functions
import numpy as np
# for ML building AI models
import tensorflow as tf
# for building layers in the tensorflow models
from tensorflow.keras import layers
# components for the models
from keras.layers import Conv2D, Dense, LeakyReLU, InputLayer, Flatten, SpatialDropout2D, MaxPooling2D, Input, Dropout
# iports models
from tensorflow.keras.models import Sequential, Model
# for the computational engine that performs the actual computations
from tensorflow.keras import backend
# loading matLab files
import scipy.io
# loading matLab files when a file is too big
import mat73
# for converting tensor to numpy array and saving to csv
from numpy import savetxt
# loading local file for configuration such as data location
import Args_Class_Module
###

######*****************************************************########
# returns the softmax probabilities from the discriminator 
#####*****************************************************########
def get_disc_vector(data_ss, model_disc, test_num):
    
    # add one dim for rows
    temp = np.expand_dims(data_ss[test_num,:],0)

    # the data has passed through the model to predict the distribution 
    #   that fit the data in temp
    disc_vector = model_disc.predict(temp,verbose = 0)
    return disc_vector
    
######*****************************************************########
# executes tests
#####*****************************************************########    
def execute_exp(args=None):
    '''
    :param args: Arg Class Module arguments
    ''' 

    # distribution number
    numDist = 7
        
    # Loads in the data and formats it as necessary.
    # Note depending on the file you are loading defines the dimensions of the data
    # The SIR_Sweep will have 3D and the normal data will have 2D
    try: 
        # Use this for small mat data files.
        data = scipy.io.loadmat(args.data)               
    except NotImplementedError:
        # loading big matLab files
        data = mat73.loadmat(args.data)  

     # reduce dim by one size   
    data_ss = np.squeeze(data.get("data"))     
    
    #---------------------------------------------------
    # This is distinguishing between  
    # cut = only clutter, no target (args.target = FAllS)
    # and
    # cut_target = has target in the data (args.target = TRUE)
    #---------------------------------------------------
    if (args.target): # args.target == TRUE
        z_full = np.squeeze(data.get("cut_target"))
    else: # args.target == FAllS
        z_full = np.squeeze(data.get("cut"))
    
    # Load the models in model_disc.  args.discriminator is the path where models are 
    model_disc = tf.keras.models.load_model(args.discriminator)

    #*****************************************************
    # Stores all the probabilities of the discriminator distributions
    if args.max_test is not None:
        # creates zeros array with args.max_test column and numDist rows
        distribution_tensors = np.zeros(shape=(args.max_test, numDist))
    else: # creates zeros array with the whole data set column and numDist rows
        distribution_tensors = np.zeros(shape=(len(data_ss), numDist))
    
    ## Converts resulting softmax tensor into Numpy Array for saving to file
    for test_num in range(len(data_ss) if args.max_test is None else args.max_test): # Run the data through the combined system.
        
        temp_distribution = get_disc_vector(data_ss, model_disc, test_num) # the softmax probabilities from the discriminator 
        proto_temp_distribution = tf.make_tensor_proto(temp_distribution)  # convert to numpy takes a prototensor as parameter
        distribution_tensors[test_num] = tf.make_ndarray(proto_temp_distribution) # distribution's array that came from tensor
        
    ##
    ##*****************************************************

    #*****************************************************
    # Saves all the discriminator distributions to the given file 


    if args.show_output is True:
        print(distribution_tensors)

    savetxt('/app/docker_bind/distribution_tensors.csv', distribution_tensors, delimiter=',')
    ##*****************************************************

### MAIN program
args = Args_Class_Module.Args_Class()
execute_exp(args)
###