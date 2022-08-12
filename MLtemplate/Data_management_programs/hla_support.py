'''
Support for reading in and preparing the HLA data set
'''

import tensorflow as tf
import pandas as pd
import numpy as np
import os
import fnmatch
import matplotlib.pyplot as plt
import random
from tensorflow import keras
from sklearn.model_selection import train_test_split

def load_data(fold, dir_base='HLAs', allele='1501'):
    '''
    Load the specified HLA data set
    - There are both training and testing data
    - Inputs are strings (what is returned are strings that are pre-padded 
       with zeros so they are all the same length)
    - Outputs are log affinities
    
    :param fold: Fold number (1 ... 5)
    :param dir_base: Directory that contains the Folds
    :param allele: '1501' or '1301'
    :return: len_max = Length of the longest input string
             ins_train = List of strings for training
             outs_train = Log affinity
             ins_test = List of strings for testing
             outs_test = Log affinity
             
    '''
    
    # Locations of the files
    fname_base = '%s/Fold_%d/'%(dir_base,fold)
    fname_train = fname_base + 'train/train_%s_fold%d.csv'%(allele,fold)
    fname_test = fname_base + 'test/test_%s_fold%d.csv'%(allele,fold)
    
    # Load training data
    data_train = pd.read_csv(fname_train, header=None)
    ins_train = data_train[0].tolist()
    outs_train = np.array(data_train[1].tolist())
    
    # Load test data
    data_test = pd.read_csv(fname_test, header=None)
    ins_test = data_test[0].tolist()
    outs_test = np.array(data_test[1].tolist())
    
    # Normalize string lengths
    len_max = np.max(np.array([len(s) for s in ins_train + ins_test]))
    ins_train = [s.rjust(len_max, '0') for s in ins_train]
    ins_test = [s.rjust(len_max, '0') for s in ins_test]
    
    return len_max, ins_train, outs_train, ins_test, outs_test

def prepare_data_set(fold, dir_base='HLAs', allele='1501', seed=100, valid_size=0.25):
    '''
    Load and prepare specified HLA data fold
    - Load training/testing data
    - Tokenize all of the strings (these tokens become the inputs to the network)
    - Split the training set into a proper training/validation set
    
    :param fold: Fold number (1 ... 5)
    :param dir_base: Directory that contains the Folds
    :param allele: '1501' or '1301'
    :param seed: Random seed for train/validation set splitting
    :param valid_size: Fraction of original training set that is to become the validation set
    
    :return: tokenizer = Tokenizer object that has been trained
             len_max = Length of the longest input string
             n_tokens = Number of tokens across the entire data set
             ins_train = List of strings for training
             outs_train = Log affinity
             ins_valid = List of strings for validation
             outs_valid = Log affinity
             ins_test = List of strings for testing
             outs_test = Log affinity
    '''
    len_max, ins_train, outs_train, ins_test, outs_test = load_data(fold, 
                                                                    dir_base=dir_base, 
                                                                    allele=allele)
    
    # Convert strings to lists of indices
    tokenizer = keras.preprocessing.text.Tokenizer(char_level=True)
    # Note: a bit of dis
    tokenizer.fit_on_texts(ins_train)
    
    # Do the conversion: we will create token lists that are big enough for both train / test sets
    #  (using test set is a bit abusive, but not a lot)
    ins_train = np.array(tokenizer.texts_to_sequences(ins_train))-1
    ins_test = np.array(tokenizer.texts_to_sequences(ins_test))-1
    
    # Number of tokens includes one value for 'unknown token'
    n_tokens = np.max(ins_train) + 2

    # Reshape the outputs into 2D tensors
    outs_train = np.reshape(outs_train, newshape=(outs_train.shape[0],1))
    outs_test = np.reshape(outs_test, newshape=(outs_test.shape[0],1))
    
     # Split training set into train and validation
    ins_train, ins_valid, outs_train, outs_valid = train_test_split(ins_train, outs_train, test_size=valid_size, random_state=seed)
    
    return tokenizer, len_max, n_tokens, ins_train, outs_train, ins_valid, outs_valid, ins_test, outs_test

    