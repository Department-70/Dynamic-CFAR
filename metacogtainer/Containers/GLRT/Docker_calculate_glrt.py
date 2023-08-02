import numpy as np
import tensorflow as tf
import scipy.io
import mat73
from numpy import loadtxt
from numpy import savetxt
import Args_Class_Module
import Cognitive_Detector as cog


args=Args_Class_Module.Args_Class() # Args Configuration File

model_0_thresholds = loadtxt('/app/docker_bind/model_0_thresholds.csv', delimiter=',')
model_1_thresholds = loadtxt('/app/docker_bind/model_1_thresholds.csv', delimiter=',')
model_2_thresholds = loadtxt('/app/docker_bind/model_2_thresholds.csv', delimiter=',')
model_3_thresholds = loadtxt('/app/docker_bind/model_3_thresholds.csv', delimiter=',')
model_4_thresholds = loadtxt('/app/docker_bind/model_4_thresholds.csv', delimiter=',')
model_5_thresholds = loadtxt('/app/docker_bind/model_5_thresholds.csv', delimiter=',')
model_6_thresholds = loadtxt('/app/docker_bind/model_6_thresholds.csv', delimiter=',')

thresholds = np.array([[0,0]]) # Add temporary data to ensure data format

# Add to same array so they can all be iterated over in the same loop
big_model_array = [ model_0_thresholds, model_1_thresholds, model_2_thresholds, model_3_thresholds, model_4_thresholds, model_5_thresholds, model_6_thresholds ]

# Only concatenate data points if the array is not empty, i.e. if the size is not 0 then it has data points to concatenate
for small_array in big_model_array:
     if small_array.size != 0:
          thresholds = np.concatenate( (thresholds, small_array), axis=0 )
thresholds = thresholds[1:] # Delete temporary data



pulse_num = np.linspace(1,args.sample_len,args.sample_len)
p = np.exp(-1j*2*np.pi*pulse_num*args.f_d*args.PRI)


try: 
    data = scipy.io.loadmat(args.data)               #Use this for small mat data files.
except NotImplementedError:
    data = mat73.loadmat(args.data)  
    

data_ss = np.squeeze(data.get("data"))
S = np.squeeze(data.get("covar"))


if (args.target):
        z_full = np.squeeze(data.get("cut_target"))
else:
    z_full = np.squeeze(data.get("cut"))


# NOTE : Original formatting
# Goes through the necessary iteration and runs the runDet function code
# for i in range(len(data_ss) if args.max_test is None else args.max_test):
#     detection = np.asarray(cog.cogDetector(args.algorithm, z_full[i], p, S[i,:,:], theshold))
#     false_alarms = false_alarms + detection

false_alarms = 0

# index_and_threshold : [index, threshold] - that is, index_and_threshold[0] = index / index_and_threshold[1] = threshold
# the numpy array is of type float, however, an index has to be an integer so we use int(index_and_threshold[0]) to get an integer representation of the index
for index_and_threshold in thresholds:
    detection = np.asarray(cog.cogDetector(args.algorithm, z_full[ int(index_and_threshold[0]) ], p, S[ int(index_and_threshold[0]) ,:,:], index_and_threshold[1]))
    false_alarms = false_alarms + detection

print('False Alarms:', false_alarms)

false_alarms = np.array([false_alarms])
savetxt('/app/docker_bind/false_alarms.csv', false_alarms, delimiter=',')