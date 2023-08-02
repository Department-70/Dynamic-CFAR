import numpy as np
import tensorflow as tf
import scipy.io
import mat73
from numpy import loadtxt
from numpy import savetxt
import Args_Class_Module
import Cognitive_Detector as cog
import os.path

MODEL_NUMBER = 0


args=Args_Class_Module.Args_Class() # Args Configuration File

# Used to compare with MODEL_NUMBER, if max_dist_list[i] == MODEL_NUMBER then generate threshold
max_disc_list = loadtxt('/app/docker_bind/max_disc_list.csv', delimiter=',')

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

# Checks if values already exist for FA classes, GLRT is not used here because it only uses Model 0 / IDEAL is Model 1
#if os.path.isfile('/app/docker_bind/FA_cd.csv'):
#    FA_cd = loadtxt('/app/docker_bind/FA_cd.csv', delimiter=',')
#else:
#    FA_cd = 0

# FA_cd = 0
# FA_glrt = 0




# # Goes through the necessary iteration and runs the runDet function code
# for i in range(len(data_ss) if args.max_test is None else args.max_test):
#     temp = np.expand_dims(data_ss[i,:],0) # Called in run det
#     thresh = np.asarray(cog.cogThreshold(args.algorithm,args.P_fa,args.K,args.N)) # MODEL 0 FUNCTION
#     det = np.asarray(cog.cogDetector(args.algorithm, z_full[i], p, S[i,:,:], thresh)) # MODEL 0 FUNCTION
#     FA_glrt = FA_glrt + det

#     max_disc = max_disc_list[i]
#     if max_disc == MODEL_NUMBER:
#         FA_cd = FA_cd + det

# if args.show_output is True:
#         print(FA_cd)
#         print(FA_glrt)

# # Data needs to be in 1D or 2D format to use savetxt, so these lines format the numbers
# FA_cd = np.array([FA_cd])
# FA_glrt = np.array([FA_glrt])

# savetxt('/app/docker_bind/FA_cd_model_0.csv', FA_cd, delimiter=',')
# savetxt('/app/docker_bind/FA_glrt.csv', FA_glrt, delimiter=',')

#print("Model 6 CD:",FA_cd[0])


model_0_thresholds = []

for i in range(len(data_ss) if args.max_test is None else args.max_test):

    max_disc = int(max_disc_list[i])
    if max_disc == MODEL_NUMBER:

        # temp = np.expand_dims(data_ss[i,:],0) # Called in run det
        thresh = np.asarray(cog.cogThreshold(args.algorithm,args.P_fa,args.K,args.N)) # MODEL 0 FUNCTION

        model_0_thresholds.append( [i, thresh] )

savetxt('/app/docker_bind/model_0_thresholds.csv', np.array(model_0_thresholds), delimiter=',')