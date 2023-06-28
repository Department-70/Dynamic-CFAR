#Load necessary packages.
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from keras.layers import Conv2D, Dense, LeakyReLU, InputLayer, Flatten , SpatialDropout2D
from keras.layers import MaxPooling2D,Input, Dropout
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import backend
import scipy.io
import mat73

# NEW : for converting tensor to numpy array and saving to csv
from numpy import savetxt
import Args_Class_Module

######*****************************************************########
# NEW : returns the softmax probabilities from the discriminator #
#####*****************************************************########
def get_disc_vector(data_ss, model_disc, test_num):
    
    temp = np.expand_dims(data_ss[test_num,:],0)
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
    for i in range(len(data_ss) if args.max_test is None else args.max_test):
        
        #####*****##### New : Converts resulting softmax tensor into Numpy Array for saving to file
        temp_distribution = get_disc_vector(data_ss, model_disc, i)
        proto_temp_distribution = tf.make_tensor_proto(temp_distribution)  # convert to numpy takes a prototensor as parameter
        distribution_tensors[i] = tf.make_ndarray(proto_temp_distribution)

    ######********************************************************########
    # NEW : Saves all the discriminator distributions to the given file #
    #####********************************************************########

    if args.show_output is True:
        print(distribution_tensors)
    savetxt('/app/docker_bind/distribution_tensors.csv', distribution_tensors, delimiter=',')
    
# MAIN - Not a part of a function below
args = Args_Class_Module.Args_Class()
execute_exp(args)