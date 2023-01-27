# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 08:15:14 2022

@author: Alex
"""

#Load necessary packages.
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from keras.layers import Conv2D, Dense, LeakyReLU, InputLayer, Flatten , SpatialDropout2D
from keras.layers import MaxPooling2D,Input, Dropout
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import backend
import scipy.io
import Cognitive_Detector as cog
# import mat73
from sklearn.utils import shuffle
import pickle


#Define our constants and parameters.
alg = 'amf'
N = 16
K = 32
P_fa = 1/np.power(10,4)
sample_len = 16
PRI = 1e-8     #Pulse repetition interval.
f_d = 2e7      #Doppler frequency.
pulse_num = np.linspace(1,sample_len,sample_len)

p = np.exp(-1j*2*np.pi*pulse_num*f_d*PRI)

#Load in our data and format it as necessary.
# data = scipy.io.loadmat('clutter_final_K_SIR_Sweep')
# data = scipy.io.loadmat('clutter_final_P_SIR_Sweep')
data = scipy.io.loadmat('clutter_final_G_SIR_Sweep')               #Use this for small mat data files.
# data = scipy.io.loadmat('clutter_final_K_test')               #Use this for small mat data files.
# data = mat73.loadmat('')                  #Use this for the larger mat73 files.
data_ss = np.squeeze(data.get("data"))
S = np.squeeze(data.get("covar"))
# z_nt = data.get("data_cut")
# z = np.squeeze(data.get("cut"))
z_full = np.squeeze(data.get("cut_target"))
sir = np.squeeze(data.get("cut_target_SIR"))
# label = np.squeeze(data.get("label"))
# shape = data.get('shape')

# train_data_len = int(np.round(0.7*len(label)))
# val_data_len = int(np.round(0.2*len(label)))
# test_data_len = int(np.round(0.1*len(label)))

#Load in the discriminator agent.
model_disc = tf.keras.models.load_model('./classifier/results/ordered_a_Dense200_50_drop_0_100_LR_0_000100_model')


#Load in the threshold setting models.
##These are the models that I trained on the order statistics for PFA=10^-4.
model_thresh4 = tf.keras.models.load_model('./classifier/results/P_14_Dense1000_200_50_drop_0_100_LR_0_000100_model')
model_thresh5 = tf.keras.models.load_model('./classifier/results/P_44_Dense1000_200_50_drop_0_100_LR_0_000100_model')
model_thresh6 = tf.keras.models.load_model('./classifier/results/P_84_Dense1000_200_50_drop_0_100_LR_0_000100_model')

##These are the models that I trained on the order statistics for PFA=10^-4.
model_thresh1 = tf.keras.models.load_model('./classifier/results/K_54_Dense1000_200_50_drop_0_100_LR_0_000100_model')
model_thresh2 = tf.keras.models.load_model('./classifier/results/K_14_Dense1000_200_50_drop_0_100_LR_0_000100_model')
model_thresh3 = tf.keras.models.load_model('./classifier/results/K_44_Dense1000_200_50_drop_0_100_LR_0_000100_model')

model_final = tf.keras.models.load_model('./classifier/results/G_GF4_Dense500_50_drop_0_100_LR_0_000100_model')

def zero(os,z,S):
    thresh = np.asarray(cog.cogThreshold(alg,P_fa,K,N))
    det = np.asarray(cog.cogDetector(alg, z, p, S, thresh))
    # print("Detector 0")
    return det

def one(os,z,S):
    thresh = model_thresh1.predict(os)
    det = np.asarray(cog.cogDetector(alg, z, p, S, thresh))
    # print("Detector 1")
    return det

def two(os,z,S):
    thresh = model_thresh2.predict(os)
    det = np.asarray(cog.cogDetector(alg, z, p, S, thresh))
    # print("Detector 2")
    return det

def three(os,z,S):
    thresh = model_thresh3.predict(os)
    det = np.asarray(cog.cogDetector(alg, z, p, S, thresh))
    # print("Detector 3")
    return det

def four(os,z,S):
    thresh = model_thresh4.predict(os)
    det = np.asarray(cog.cogDetector(alg, z, p, S, thresh))
    # print("Detector 4")
    return det

def five(os,z,S):
    thresh = model_thresh5.predict(os)
    det = np.asarray(cog.cogDetector(alg, z, p, S, thresh))
    # print("Detector 5")
    return det

def six(os,z,S):
    thresh = model_thresh6.predict(os)
    det = np.asarray(cog.cogDetector(alg, z, p, S, thresh))
    # print("Detector 6")
    return det

options = {0 : zero,
           1 : one,
           2 : two,
           3 : three,
           4 : four,
           5 : five,
           6 : six}

output = np.zeros([len(data_ss),14])
output_full = np.zeros([len(data_ss),21])
det_final = np.zeros([len(data_ss),1])
results = np.zeros([len(sir),4])
FA_CD = 0
FA_glrt = 0
FA_ideal = 0
#Run the data through the combined system.
for j in range(len(sir)):
    # z = z_full[j,:,:]
    z = z_full[j+10,:,:]
    FA_CD = 0
    FA_glrt = 0
    FA_ideal = 0
    for i in range(len(data_ss)):
        temp = np.expand_dims(data_ss[i,:],0)
        disc_vector = model_disc.predict(temp)
        det = options[np.argmax(disc_vector)](temp, z[i,:], S[i,:,:])
        det_glrt = options[0](temp, z[i,:], S[i,:,:])
        det_ideal = options[1](temp, z[i,:], S[i,:,:])
        
        FA_CD = FA_CD+det 
        FA_ideal = FA_ideal + det_ideal
        FA_glrt = FA_glrt+det_glrt
        print('------')
        print(i)
        print(np.argmax(disc_vector))
        print("SIR")
        # print(sir[j,i])
        print(sir[j+10,i])
        # print(det_final[i])
        print("Cognitive Detector")
        print(FA_CD)
        print("GLRT")
        print(FA_glrt)
        print("Ideal")
        print(FA_ideal)
        print('------')
    # results[j,:] = [sir[j,i],FA_CD,FA_glrt,FA_ideal]
    results[j+10,:] = [sir[j+10,i],FA_CD,FA_glrt,FA_ideal]

# train_data = output[0:train_data_len,:]
# val_data = output[train_data_len:train_data_len+val_data_len,:]
# test_data = output[train_data_len+val_data_len:,:]

# train_data_full = output_full[0:train_data_len,:]
# val_data_full = output_full[train_data_len:train_data_len+val_data_len,:]
# test_data_full = output_full[train_data_len+val_data_len:,:]

# train_label = label[0:train_data_len]
# val_label = label[train_data_len:train_data_len+val_data_len]
# test_label = label[train_data_len+val_data_len:]

# train_data_ss = data_ss[0:train_data_len,:]
# val_data_ss = data_ss[train_data_len:train_data_len+val_data_len,:]
# test_data_ss = data_ss[train_data_len+val_data_len:,:]

# train_S = S[0:train_data_len,:,:]
# val_S = S[train_data_len:train_data_len+val_data_len,:,:]
# test_S = S[train_data_len+val_data_len:,:,:]

# train_z = z[0:train_data_len,:]
# val_z = z[train_data_len:train_data_len+val_data_len,:]
# test_z = z[train_data_len+val_data_len:,:]

# train_shape = shape[0:train_data_len,:]
# val_shape = shape[train_data_len:train_data_len+val_data_len,:]
# test_shape = shape[train_data_len+val_data_len:,:]


# train_data,train_data_full,train_label,train_data_ss,train_S,train_z,train_shape = shuffle(train_data,train_data_full,train_label,train_data_ss,train_S,train_z,train_shape)
# val_data,val_data_full,val_label,val_data_ss,val_S,val_z,val_shape = shuffle(val_data,val_data_full,val_label,val_data_ss,val_S,val_z,val_shape)
# test_data,test_data_full,test_label,test_data_ss,test_S,test_z,test_shape = shuffle(test_data,test_data_full,test_label,test_data_ss,test_S,test_z,test_shape)

# training_data = [train_data,train_data_full,train_label,train_data_ss,train_S,train_z,train_shape]
# validation_data = [val_data,val_data_full,val_label,val_data_ss,val_S,val_z,val_shape]
# testing_data = [test_data,test_data_full,test_label,test_data_ss,test_S,test_z,test_shape]
# full_data = [output,output_full,label,data_ss,S,z,shape]

# with open('Final_Agent_Data.pkl','wb') as file:
#     pickle.dump(full_data,file)

# with open('Final_Agent_Training_Data.pkl','wb') as file:
#     pickle.dump(training_data,file)
    
# with open('Final_Agent_Validation_Data.pkl','wb') as file:
#     pickle.dump(validation_data,file)
    
# with open('Final_Agent_Testing_Data.pkl','wb') as file:
#     pickle.dump(testing_data,file)
    
# with open('Final_Agent_Training_Data.pkl','rb') as file:
#     training_stuff = pickle.load(file)

# with open('Final_Agent_Validation_Data.pkl','rb') as file:
#     validation_stuff = pickle.load(file)
    
# with open('Final_Agent_Testing_Data.pkl','rb') as file:
#     testing_stuff = pickle.load(file)