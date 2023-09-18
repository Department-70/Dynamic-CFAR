# For each probability distribution finds the max (most likely distribution) and saves to new file

# args.max_test is the number of tests that run. If args.max_test is None that the whole data set will be trained.
# args.show_output is as show_output in Args_Class_Module.Args_Class()
# data_ss is a data array with a reduse by one dim
# disc_vector is a vector with the numbers each of which corresponds to certain distribution. In our case, each disc_vector
#   has 7 elements and the total number of disc_vector is the length of  data_ss or args.max_test
# max_disc_list is the list with the the indices of the maximum values in each disc_vector

## output
# max_disc_list.csv in /app/docker_bind/max_disc_list.csv
###

# for ML building AI models
import tensorflow as tf

# complex math functions
import numpy as np

# for load txt file
from numpy import loadtxt

# for save txt file
from numpy import savetxt

import Args_Class_Module

# for load matLab data
import scipy.io

# loading big matLab files
import mat73


args = Args_Class_Module.Args_Class()

if args.max_test is not None:
    # create column-array with max_test zeros elements
    max_disc_list = np.zeros(shape=(args.max_test, 1))
else:
    try: 
        # load all args.data set
        data = scipy.io.loadmat(args.data) # Use this for small mat data files.
    except NotImplementedError: # if you have error NotImplementedError then use next line
        data = mat73.loadmat(args.data) # Use this for big mat data files. 

    # reduce dim by one size      
    data_ss = np.squeeze(data.get("data"))

    # create column-array with len(data_ss) zeros elements
    max_disc_list = np.zeros(shape=(len(data_ss), 1))

# load distribution_tensors.csv that was created in Dockerfile_disc wher Docker_Discriminator.py was running
distribution_tensors = loadtxt('/app/docker_bind/distribution_tensors.csv', delimiter=',')
#print(distribution_tensors)

#*****************************************************
# Creates a list, max_disc_list, with the indices
#   of the maximum values in each disc_vector
i = 0
for disc_vector in distribution_tensors:
     # Returns the indices of the maximum values
     max_disc = np.argmax(disc_vector)

     max_disc_list[i] = max_disc
     i = i + 1
##*****************************************************

if args.show_output is True:
    print(max_disc_list)
savetxt('/app/docker_bind/max_disc_list.csv', max_disc_list, delimiter=',')
