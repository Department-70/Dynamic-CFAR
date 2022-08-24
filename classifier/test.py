# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 13:47:20 2022

@author: Geoffrey Dolinger
"""
import pickle
import scipy.io
import numpy as np
from sklearn.utils import shuffle

import pandas as pd
import sklearn
from sklearn import metrics
import seaborn

import os
import fnmatch
import matplotlib.pyplot as plt


from CFAR_classifier import *

mat = scipy.io.loadmat('clutter_order.mat')
data =mat['data']
label = mat['label']
# data = np.reshape(data,(8000,1024))
data2 = np.reshape(data,(8000,8,8))
# dataset = np.concatenate((data,label), axis=1)
# # dataset_shuffle = np.random.shuffle(dataset)
# data_shuffle,label_shuffle = shuffle(data,label)

# b = np.reshape(np.array([0,1,2,3,4,5,6]),(7,1))
# a= np.array([[0.1,0.2,0.3,0.4],[1.1,1.2,1.3,1.4],[2.1,2.2,2.3,2.4],[3.1,3.2,3.3,3.4],[4.1,4.2,4.3,4.4],[5.1,5.2,5.3,5.4],[6.1,6.2,6.3,6.4]])

# a_shuffle,b_shuffle = shuffle(a,b)
# train_a = a_shuffle[0:4,:]
# train_b = b_shuffle[0:4,:]
# val_a = a_shuffle[5,:]
# val_b = b_shuffle[5,:]
# test_a = a_shuffle[6,:]
# test_b = b_shuffle[6,:]
# class_size = int(np.round(data.shape[0]/4))
# train_portion = int(np.round(.75*data.shape[0]/4))
# val_portion = int(np.round(.125*data.shape[0]/4))
# test_portion =int(np.round(.125*data.shape[0]/4)) 
# data = np.reshape(data,(8000,32,32))
# class_size = int(np.round(data.shape[0]/4))
# train_portion = int(np.round(.75*data.shape[0]/4))
# val_portion = int(np.round(.125*data.shape[0]/4))
# test_portion =int(np.round(.125*data.shape[0]/4)) 
# train_x0=[]
# train_y0=[]
# val_x0=[]
# val_y0=[]
# test_x0=[]
# test_y0=[]
# for i in range(4):
#     train_x0.append(data[i*class_size:i*class_size+train_portion])
#     train_y0.append(label[i*class_size:i*class_size+train_portion])
#     val_x0.append(data[i*class_size+train_portion:i*class_size+train_portion+val_portion])
#     val_y0.append(label[i*class_size+train_portion:i*class_size+train_portion+val_portion])
#     test_x0.append(data[i*class_size+train_portion+val_portion:(i+1)*class_size])
#     test_y0.append(label[i*class_size+train_portion+val_portion:(i+1)*class_size])
            
# train_x0=np.reshape(train_x0,(4*train_portion,data.shape[1],data.shape[2]))
# train_y0=np.reshape(train_y0,(4*train_portion,label.shape[1]))
# val_x0=np.reshape(val_x0,(4*val_portion,data.shape[1],data.shape[2]))
# val_y0=np.reshape(val_y0,(4*val_portion,label.shape[1]))
# test_x0=np.reshape(test_x0,(4*test_portion,data.shape[1],data.shape[2]))
# test_y0=np.reshape(test_y0,(4*test_portion,label.shape[1]))

# train_x,train_y = shuffle(train_x0,train_y0)
# val_x,val_y = shuffle(val_x0,val_y0)
# test_x,test_y = shuffle(test_x0,test_y0)

parser = create_parser()
args = parser.parse_args(['@exp.txt','@order.txt'])
print(args)
# execute_exp(args)
display_Metrics(args,test=True,save=True)

args_str = augment_args(args)
fbase = generate_fname(args, args_str)
# fbase_1,drop = fbase.rsplit('rot',1)
fbase_1 = '%s_results.pkl'%(fbase)
dir, fbase_0 = fbase_1.rsplit('/',1)
# dir, fbase_save = fbase_1.rsplit('/',1)

#check results folder for all rotations with shallow network args
results = read_all_rotations(dir, fbase_0)

# dirname = './results'
# filebase = 'Raw2_Dense500_250_50_drop_0.200_LR_0_001000_results.pkl'

# results=read_all_rotations(dirname, filebase)

# labels = [0,1,2,3]
# test_label = results[0]['test_label']
# test_pred = results[0]['predict_testing']
# test_preds = np.argmax(test_pred, axis=1)

# train_label = results[0]['train_label']
# train_pred = results[0]['predict_training']
# train_preds = np.argmax(train_pred, axis=1)

# val_label = results[0]['val_label']
# val_pred = results[0]['predict_validation']
# val_preds = np.argmax(val_pred, axis=1)
# #Generate the confusion matrix 
# conf_matrix1=sklearn.metrics.confusion_matrix(test_label, test_preds ,labels=labels, sample_weight=None, normalize='true')
# conf_matrix2=sklearn.metrics.confusion_matrix(train_label, train_preds ,labels=labels, sample_weight=None, normalize='true')
# conf_matrix3=sklearn.metrics.confusion_matrix(val_label, val_preds ,labels=labels, sample_weight=None, normalize='true')
# #Generate and format plots for confusion matricies
# # subplot_args = { 'nrows': 1, 'ncols': 1, 'figsize': (20, 8),
# #                               'subplot_kw': {'xticks': [], 'yticks': []} }
# # f, (ax1,ax2) = plt.subplots(**subplot_args)                                   
# plt.figure()
# g1=seaborn.heatmap(conf_matrix1,annot=True, linewidths=4, cmap='magma_r')
# g1.set_title('Testing Confusion Matrix (Normalized true classes)', fontsize=20)
# g1.set_xlabel("Predicted Class", fontsize=16)
# g1.set_ylabel("True Class", fontsize=16)
# plt.savefig('Test_Confusion_Matrix',facecolor='white',bbox_inches='tight')
# plt.figure()
# g2=seaborn.heatmap(conf_matrix2,annot=True, linewidths=4, cmap='magma_r')
# g2.set_title('Training Confusion Matrix (Normalized true classes)', fontsize=20)
# g2.set_xlabel("Predicted Class", fontsize=16)
# g2.set_ylabel("True Class", fontsize=16)
# plt.savefig('Training_Confusion_Matrix',facecolor='white',bbox_inches='tight')
# plt.figure()
# g3=seaborn.heatmap(conf_matrix3,annot=True, linewidths=4, cmap='magma_r')
# g3.set_title('Validation Confusion Matrix (Normalized true classes)', fontsize=20)
# g3.set_xlabel("Predicted Class", fontsize=16)
# g3.set_ylabel("True Class", fontsize=16)
# plt.savefig('Val_Confusion_Matrix',facecolor='white',bbox_inches='tight')
