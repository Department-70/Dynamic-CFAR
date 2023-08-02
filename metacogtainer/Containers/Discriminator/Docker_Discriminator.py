### Calculates: a probability of each distribution (Gaussian, K-Low, K-Medium, K-High, P-Low, P-Medium, P-High) to fit current data point (current data return)  
#
## Input: the Data file specified by the Args Class 
# data_ss is a data array
# model_disc is a path to the model's discriminators
# test_num is number of test that will run
# disc_vector is a vector with the numbers each of which corresponds to certain distribution
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
    
    
def execute_exp(args=None):
    '''
    Execute Tests
    
    :param args: Arg Class Module arguments
    ''' 
        
    # Load in our data and format it as necessary.
    # Note depending on the file you are loading defines the dimensions of the data
    # The SIR_Sweep will have 3D and the normal data will have 
    try: 
        data = scipy.io.loadmat(args.data)               #Use this for small mat data files.
    except NotImplementedError:
        data = mat73.loadmat(args.data)  
        
    data_ss = np.squeeze(data.get("data"))     
    
    #---------------------------------------------------
    # NOTE TO SELF: JOE
    # This is not where we are distinguishing between 3 vs 2 dimensions 
    # 
    # cut = only clutter, no target
    # cut_target = has target in the data
    # HERP DERP
    #---------------------------------------------------
    if (args.target):
        z_full = np.squeeze(data.get("cut_target"))
    else:
        z_full = np.squeeze(data.get("cut"))
    
    
    #Load in the discriminator agent.
    model_disc = tf.keras.models.load_model(args.discriminator)

    ######***************************************########
    # NEW : Stores all the discriminator distributions #
    #####***************************************########
    if args.max_test is not None:
        distribution_tensors = np.zeros(shape=(args.max_test, 7))
    else:
        distribution_tensors = np.zeros(shape=(len(data_ss), 7))
    
    #Run the data through the combined system.
    for test_num in range(len(data_ss) if args.max_test is None else args.max_test):
        
        #####*****##### New : Converts resulting softmax tensor into Numpy Array for saving to file
        temp_distribution = get_disc_vector(data_ss, model_disc, test_num)
        proto_temp_distribution = tf.make_tensor_proto(temp_distribution)  # convert to numpy takes a prototensor as parameter
        distribution_tensors[test_num] = tf.make_ndarray(proto_temp_distribution)

    ######********************************************************########
    # NEW : Saves all the discriminator distributions to the given file #
    #####********************************************************########

    if args.show_output is True:
        print(distribution_tensors)
    savetxt('/app/docker_bind/distribution_tensors.csv', distribution_tensors, delimiter=',')


### MAIN program
args = Args_Class_Module.Args_Class()
execute_exp(args)
###