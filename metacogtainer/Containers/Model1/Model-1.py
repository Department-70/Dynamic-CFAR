import numpy as np
import tensorflow as tf
import scipy.io
import mat73
from numpy import loadtxt
from numpy import savetxt
import Args_Class_Module
import Cognitive_Detector as cog
import os.path


MODEL_NUMBER = 1


args=Args_Class_Module.Args_Class() # Args Configuration File

model_thresh1 = tf.keras.models.load_model(args.model_thresh1, compile=False) # Model 1 
model_thresh1.compile()


# Used to compare with MODEL_NUMBER, if max_dist_list[i] == MODEL_NUMBER then generate threshold
max_disc_list = loadtxt('/app/docker_bind/max_disc_list.csv', delimiter=',')
     


# These five variables are used in the function call
#data_ss = loadtxt('/app/docker_bind/data_ss.csv', delimiter=',')
#z_full = loadtxt('/app/docker_bind/z_full.csv', delimiter=',')

# Because S was saved as 2D it needs to be expanded to its original 3D shape
#S = loadtxt('/app/docker_bind/S.csv', delimiter=',')
#S = S.reshape( S.shape[0], S.shape[1] // S.shape[2], S.shape[2] )

#p = loadtxt('/app/docker_bind/p.csv', delimiter=',')
#test_num = loadtxt('/app/docker_bind/test_num.csv', delimiter=',')


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


# NOTE: This block of comments is the old system where GLRT and Threshold were calculated in the same container

# Checks if values already exist for FA classes, GLRT is not used here because it only uses Model 0
#if os.path.isfile('/app/docker_bind/FA_ideal.csv'):
#   FA_ideal = loadtxt('/app/docker_bind/FA_ideal.csv', delimiter=',')
#else:
#   FA_ideal = 0

#FA_ideal = 0

# Loading is commented out here, because it added to previous run totals, FA_cd is initialized in Model 0

#if os.path.isfile('/app/docker_bind/FA_cd.csv'):
#   FA_cd = loadtxt('/app/docker_bind/FA_cd.csv', delimiter=',')
#else:
#   FA_cd = 0

#FA_cd=0





# # Goes through the necessary iteration and runs the runDet function code
# for i in range(len(data_ss) if args.max_test is None else args.max_test):
#     temp = np.expand_dims(data_ss[i,:],0) # Called in run det
#     model1_thresh = model_thresh1.predict(temp,verbose = 0) # Called in function 'one'
#     det = np.asarray(cog.cogDetector(args.algorithm, z_full[i], p, S[i,:,:], model1_thresh))
#     FA_ideal = FA_ideal + det

#     max_disc = max_disc_list[i]
#     if max_disc == MODEL_NUMBER:
#         FA_cd = FA_cd + det

# # Data needs to be in 1D or 2D format to use savetxt, so these lines format the numbers
# FA_cd = np.array([FA_cd])
# FA_ideal = np.array([FA_ideal])

# savetxt('/app/docker_bind/FA_cd_model_1.csv', FA_cd, delimiter=',')
# savetxt('/app/docker_bind/FA_ideal.csv', FA_ideal, delimiter=',')

#print("Model 1 Ideal:",FA_ideal[0])
#print("Model 1 CD:",FA_cd[0])


model_1_thresholds = []

for i in range(len(data_ss) if args.max_test is None else args.max_test):

    max_disc = max_disc_list[i]
    if max_disc == MODEL_NUMBER:

        temp = np.expand_dims(data_ss[i,:],0) # Called in run det
        model1_thresh = model_thresh1.predict(temp,verbose = 0) # Called in function 'one'

        model_1_thresholds.append( [i, model1_thresh] )

savetxt('/app/docker_bind/model_1_thresholds.csv', np.array(model_1_thresholds), delimiter=',')