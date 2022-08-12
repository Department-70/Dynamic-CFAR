'''
Data loader for the Chesapeake Bay Watershed "patches" data set

Author: Andrew H. Fagg
2022-04-26

'''
import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import pickle

import tensorflow as tf
from tensorflow import keras

def load_single_file(fname):
    '''
    Load one patches file of the Chesapeake Watershed data set and return the imagery and pixel labels
    
    For the returned imagery, the features are scaled to the 0...1 range
    
    Pixel labels are 8-bit integers representing the following classes:
    0: No class (should generally not be seen in this data set, if at all)
    1: water
    2: tree canopy / forest
    3: low vegetation / field
    4: barren land
    5: impervious (other)
    6: impervious (road) 
    
    :param fname: Absolute file name
    
    :return: Tuple of image (256x256x24) and labels (256x256).  The outputs are proper TF Tensors
    '''
    # Load data
    fname = fname.numpy().decode('utf-8')
    
    dat = np.load(fname)
    
    # Extract the example and place it into standard TF format
    dat = dat['arr_0']
    dat = np.transpose(dat, [0, 2, 3, 1])
    
    # Labels: 8th element
    outs = dat[0, :, :, 8]
    outs = np.int_(outs)
    
    # 15 = no data case; set these to weight 0; all others are weight 1.0
    weights = np.logical_not(np.equal(outs, 15)) * 1.0
    
    # Set all class 15 to class 0
    np.equal(outs, 15, where=0)
    
    # Image data
    images = dat[0, :, :, 0:8]/255.0
    
    # Landsat data
    # Unclear what the max is over the data set, but at least this gets us into the ballpark
    landsat = dat[0, :, :, 10:28]/4000.0
    
    # Building mask
    mask = dat[0, :, :, 28]
    
    ins = np.concatenate([images, landsat], axis=2)
    
    # Some basic checks
    assert not np.isnan(ins).any(), "File contains NaNs (%s)"%(fname)
    assert np.min(outs) >= 0, "Labels out of bounds (%s, %d)"%(fname, np.min(outs))
    assert np.max(outs) < 7, "Labels out of bounds (%s, %d)"%(fname, np.max(outs))
    
    # Translate from numpy to TF Tensors
    return tf.cast(ins, tf.float32), tf.cast(outs, tf.int8)#, tf.cast(mask, tf.float32), tf.cast(weights, tf.float32)

def create_dataset(base_dir='/home/fagg/datasets/radiant_earth/pa', partition='train', fold=0, filt='*', 
                   batch_size=8, prefetch=2, num_parallel_calls=4):
    '''
    Files are located in <base_dir>/<partition>/F<fold>/
    
    Create a TF Dataset that can be used for training and evaluating a model
    :param base_dir: Location of the dataset partitions 
    :param partition: Partition subdirectory of base_dir that contains the folds to be loaded (possibilities are "train" and "valid")
    :param fold: Fold to load (0 ... 9)
    :param filt: Regular expression filter for the files within the fold directory. 
                    '*' means use all files
                    '*0' means use all files ending in zero
                    '*[012]' means use all files ending zero, one or two
    :param batch_size: Size of the batches to be produced by this data set
    :param prefetch: Number of batches to prefetch in parallel with training
    :param num_parallel_calls: Number of threads to use for the data loading process
    
    :return: TF Dataset that emits tuples (ins, outs)
                ins is a TF Tensor of shape batch_size x 256 x 256 x 24
                outs is a TF Tensor of shape batch_size x 256 x 256
    '''
    
    # Full list of files in the dataset
    data = tf.data.Dataset.list_files(['%s/%s/F%d/%s.npz'%(base_dir, partition, fold, f) for f in filt], shuffle=False)
    
    # Load each file
    #  - py_function allows eager execution
    #  - we must declare here the return types of the Dataset
    data = data.map(lambda x: tf.py_function(func=load_single_file, inp=[x], Tout=(tf.float32, tf.int8)), #, tf.float32, tf.float32)), 
                num_parallel_calls=num_parallel_calls)
    
    # Batch the individual elements
    data = data.batch(batch_size)
    
    # Buffer multiple batches
    data = data.prefetch(prefetch)
    
    return data

