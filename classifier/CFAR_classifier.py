'''
Machine Learning Template

Author: Geoffrey Dolinger, with supplied code by Andrew H. Fagg (andrewhfagg@gmail.com)
'''
'''
library inclusion based off of application
'''
import tensorflow as tf
import random
import sys
import argparse
import pickle
from tensorflow import keras
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model
from sklearn.utils import shuffle

import scipy.io
import os
import fnmatch
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
import sklearn
from sklearn import metrics
import seaborn

from keras.datasets import mnist
# # Provided
# from symbiotic_metrics import *
from job_control import *

# Import application specific functions for data system/ML design
# from chesapeake_loader import *
# from create_unet import *
from create_mlp import *
from create_cnn import *
from create_RNN_network import *
#################################################################


def create_parser():
    '''
    Create argument parser
    -This function will create the parser structure and include parameter 
    setting given the specific application
    '''
    # Parse the command-line arguments
    parser = argparse.ArgumentParser(description='MLP', fromfile_prefix_chars='@')

    # High-level commands
    parser.add_argument('--check', action='store_true', help='Check results for completeness')
    parser.add_argument('--nogo', action='store_true', help='Do not perform the experiment')
    parser.add_argument('--verbose', '-v', action='count', default=2, help="Verbosity level")
    parser.add_argument('--testing', action='store_true',help='Do to perform testing evaluation')
    # CPU/GPU
    parser.add_argument('--cpus_per_task', type=int, default=None, help="Number of threads to consume")
    parser.add_argument('--gpu', action='store_true', help='Use a GPU')

    # High-level experiment configuration
    parser.add_argument('--exp_type', type=str, default=None, help="Experiment type")    
    parser.add_argument('--label', type=str, default='Raw2', help="Extra label to add to output files")
    parser.add_argument('--dataset', type=str, default='clutter.mat', help='Data set directory')    
    parser.add_argument('--results_path', type=str, default='./results', help='Results directory')

    # Specific experiment configuration
    parser.add_argument('--exp_index', type=int, default=None, help='Experiment index')
    parser.add_argument('--epochs', type=int, default=1000, help='Training epochs')
    parser.add_argument('--lrate', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--image_size', nargs=3, type=int, default=[32,32], help="Size of input images (rows, cols, channels)")
    parser.add_argument('--nclasses', type=int, default=4, help='Number of classes in maxpool output')    
   
    # Individual layer parameters (convolution in this example)
    parser.add_argument('--conv_size', nargs='+', type=int, default=None, help='kernel size of leading conv layers') 
    parser.add_argument('--conv_nfilters', nargs='+', type=int, default=[20,20], help='Convolution filters per layer (sequence of ints)')
    parser.add_argument('--conv_stride', nargs='+', type=int, default=[1,1], help='stride amount in conv layers') 
    parser.add_argument('--maxpool',type=int, default=None, help='size of pooling')
    parser.add_argument('--dropout_spatial', nargs='+', type=float, default=None, help='Inception Layer compression flag')
    
    #RNN layer parameters
    parser.add_argument('--rnn_size', nargs='+', type=int, default=None, help='kernel size of leading Simple RNN layers')
    parser.add_argument('--gru_size', nargs='+', type=int, default=None, help='kernel size of leading GRU layers')
    parser.add_argument('--lstm_size', nargs='+', type=int, default=None, help='kernel size of leading LSTM layers')
    parser.add_argument('--dropout_rnn', type=float, default=0.0, help='Dropout rate')
    parser.add_argument('--L2_rnn', type=float, default=None, help="L2 regularization in inception parameter")
    
    parser.add_argument('--dense', nargs='+', type=int, default=[500, 250,50], help='Number of hidden units per layer (sequence of ints)')
    parser.add_argument('--dropout_dense', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--L2_dense', type=float, default=None, help="L2 regularization parameter")
    # Early stopping
    parser.add_argument('--min_delta', type=float, default=0.0, help="Minimum delta for early termination")
    parser.add_argument('--patience', type=int, default=100, help="Patience for early termination")

    # Training parameters
    parser.add_argument('--batch', type=int, default=1, help="Training set batch size")
    parser.add_argument('--steps_per_epoch', type=int, default=None, help="Number of gradient descent steps per epoch")    
    
    '''
    Examples Parser inputs:
    # High-level experiment configuration

    parser.add_argument('--Nfolds', type=int, default=5, help='Maximum number of folds')    
   
    # Specific experiment configuration 
    parser.add_argument('--rotation', type=int, default=1, help='Cross-validation rotation')
    parser.add_argument('--Ntraining', type=int, default=3, help='Number of training folds')    
    parser.add_argument('--filts_train', nargs='+', type=str, default=['*0','*1','*2','*3','*4','*5','*6','*7','*8'], help='filter list for training')
    parser.add_argument('--filts_valid', nargs='+', type=str, default=['*9'], help='filter list for validation')
    parser.add_argument('--filts_test', nargs='+', type=str, default=['*'], help='filter list for testing') 
    
    # Training parameters
    parser.add_argument('--validation_fraction', type=float, default=0.1, help="Fraction of available validation set to actually use for validation")
    parser.add_argument('--testing_fraction', type=float, default=0.5, help="Fraction of available testing set to actually use for testing")
    parser.add_argument('--generator_seed', type=int, default=42, help="Seed used for generator configuration")
    
    Examples of ML parameters:
    
    #Embedding parameters
    parser.add_argument('--emb_size',type=int, default=5, help='diminsion to reduce in embedding')
    
    
    #RNN layer parameters
    parser.add_argument('--rnn_size', nargs='+', type=int, default=None, help='kernel size of leading Simple RNN layers')
    parser.add_argument('--gru_size', nargs='+', type=int, default=None, help='kernel size of leading GRU layers')
    parser.add_argument('--lstm_size', nargs='+', type=int, default=None, help='kernel size of leading LSTM layers')
    parser.add_argument('--dropout_rnn', type=float, default=0.0, help='Dropout rate')
    parser.add_argument('--L2_rnn', type=float, default=None, help="L2 regularization in inception parameter")
    
    # Inception Model parameters
    parser.add_argument('--path1', nargs='+', type=str, default=['c1'], 
                        help='First path in inception layer (c=conv, #=kernel size, mp=max pooling, #pooling size)')
    parser.add_argument('--path2', nargs='+', type=str, default=['c3'], 
                        help='Forth path in inception layer (c=conv, #=kernel size, mp=max pooling, #pooling size)')
    parser.add_argument('--path3', nargs='+', type=str, default=['c1','c3'], 
                        help='Second path in inception layer (c=conv, #=kernel size, mp=max pooling, #pooling size)')
    parser.add_argument('--path4', nargs='+', type=str, default=['c1','c5'], 
                        help='Third path in inception layer (c=conv, #=kernel size, mp=max pooling, #pooling size)')
    parser.add_argument('--path5', nargs='+', type=str, default=['mp3','c1'], 
                        help='Forth path in inception layer (c=conv, #=kernel size, mp=max pooling, #pooling size)')    
    
    #Individual convolution layer parameters
    parser.add_argument('--conv_size', nargs='+', type=int, default=None, help='kernel size of leading conv layers') 
    parser.add_argument('--conv_nfilters', nargs='+', type=int, default=None, help='Convolution filters per layer (sequence of ints)')
    parser.add_argument('--pool', nargs='+', type=int, default=None, help='Max pooling size (1=None)')   
    
    #Inception layer parameters
    parser.add_argument('--inception', nargs='+', type=int, default=None, help='Inception Layer compression flag')    
    parser.add_argument('--compress', nargs='+', type=bool, default=None, help='Inception Layer compression flag')
    parser.add_argument('--dropout_spacial', nargs='+', type=float, default=None, help='Inception Layer compression flag') 
    parser.add_argument('--L2_incep', type=float, default=None, help="L2 regularization in inception parameter")
    
    #Dense layer parameters 
    parser.add_argument('--dense', nargs='+', type=int, default=[100, 20], help='Number of hidden units per layer (sequence of ints)')    
    parser.add_argument('--dropout_dense', type=float, default=None, help='Dropout rate')
    parser.add_argument('--L2_dense', type=float, default=None, help="L2 regularization parameter")
    
    #Augmentation parameters
    parser.add_argument('--augment', action='store_true', help='determine to use augmentation or not')
    parser.add_argument('--aug_rotation',type=float, default=None, help="Rotation range for augmentation")
    parser.add_argument('--aug_hflip',action='store_true', help="Rotation range for augmentation")
    parser.add_argument('--aug_vflip',action='store_true', help="Rotation range for augmentation")
    parser.add_argument('--aug_shear', type=float, default=0.2, help="Shear range for augmentation")
    parser.add_argument('--aug_zoom', type=float, default=[.5,1], help="Zoom range for augmentation")
    
    #Unet info
    parser.add_argument('--unet_depth', nargs='+', type=int, default=None, help='depth level for Unet design')
    parser.add_argument('--dropout_spatial', type=float, default=None, help='Dropout rate for Conv layers')
    parser.add_argument('--lambda_l2', type=float, default=None, help="L2 regularization parameter")
    
    '''
    
    return parser


def exp_type_to_hyperparameters(args):
    '''
    Translate the exp_type into a hyperparameter set

    :param args: ArgumentParser
    :return: Hyperparameter set (in dictionary form)
    -This function can allow you to set ranges for parser variables given a set 
    exp_type label.  p is used to implement multiple experiments with a set or 
    sweep of parameter values
    '''
    if args.exp_type is None:
        p=None
    elif args.exp_type =='CNN_sweep':
        p = {'conv_nfilters':[5,10,20],
             'lrate':[0.1, 0.25, 0.3, 0.4]}
    elif args.exp_type =='CNN':
        p = {'rotation': range(5)}
    else:
        assert False, "Unrecognized exp_type"

    return p


#################################################################
def check_args(args):
    '''
    This function will assert limits on arguments from the parser in case a 
    value is input that is out of acceptable ranges

    '''
    #assert (args.rotation >= 0 and args.rotation < args.Nfolds), "Rotation must be between 0 and Nfolds"
    #assert (args.Ntraining >= 1 and args.Ntraining <= (args.Nfolds-1)), "Ntraining must be between 1 and Nfolds-2"
    #assert (args.dropout is None or (args.dropout > 0.0 and args.dropout < 1)), "Dropout must be between 0 and 1"
    assert (args.lrate > 0.0 and args.lrate < 1), "Lrate must be between 0 and 1"
    #assert (args.L1_regularizer is None or (args.L1_regularizer > 0.0 and args.L1_regularizer < 1)), "L2_regularizer must be between 0 and 1"
    #assert (args.L2_regularizer is None or (args.L2_regularizer > 0.0 and args.L2_regularizer < 1)), "L2_regularizer must be between 0 and 1"
    assert (args.cpus_per_task is None or args.cpus_per_task > 1), "cpus_per_task must be positive or None"
    
def augment_args(args):
    '''
    Use the jobiterator to override the specified arguments based on the experiment index.

    Modifies the args

    :param args: arguments from ArgumentParser
    :return: A string representing the selection of parameters to be used in the file name
    '''
    
    # Create parameter sets to execute the experiment on.  This defines the Cartesian product
    #  of experiments that we will be executing
    p = exp_type_to_hyperparameters(args)

    # Check index number
    index = args.exp_index
    if(index is None):
        return ""
    
    # Create the iterator, Function within job_control.py, comment away if 
    # not using job_control for multiple jobs.
    ji = JobIterator(p)
    print("Total jobs:", ji.get_njobs())
    
    # Check bounds
    assert (args.exp_index >= 0 and args.exp_index < ji.get_njobs()), "exp_index out of range"

    # Print the parameters specific to this exp_index
    print(ji.get_index(args.exp_index))
    
    # Push the attributes to the args object and return a string that describes these structures
    return ji.set_attributes_by_index(args.exp_index, args)
 
    
#################################################################

def generate_fname(args, params_str):
    '''
    Generate the base file name for output files/directories.
    
    The approach is to encode the key experimental parameters in the file name.  This
    way, they are unique and easy to identify after the fact.

    :param args: from argParse
    :params_str: String generated by the JobIterator
    
    This function will generate file naming related to the experiment.  Any 
    parameter in the args parser can be updated to be part of the naming 
    convention.
    '''
    
    # RNN information
    if args.rnn_size is None:
        rnn_str = ''
        rnn_size_string =''
    else:
        rnn_str = 'RNN'
        rnn_size_string = '_'.join(str(x) for x in args.rnn_size)
    
    if args.gru_size is None:
        gru_str = ''
        gru_size_string =''
    else:
        gru_str = 'GRU'
        gru_size_string = '_'.join(str(x) for x in args.gru_size)
        
    if args.lstm_size is None:
        lstm_str = ''
        lstm_size_string =''
    else:
        lstm_str = 'LSTM'
        lstm_size_string = '_'.join(str(x) for x in args.lstm_size)
    # Conv configuration
    if args.conv_size is None:
        conv_size_str = ''
        conv_filter_str = ''
    else:
        conv_size_str = '_'.join(str(x) for x in args.conv_size)
        conv_filter_str = '_'.join(str(x) for x in args.conv_nfilters)   
    # Dense
    if args.dense is None:
        dense_str = ''
        dense_size_str = ''
    else:
        dense_str = 'Dense'
        dense_size_str = '_'.join(str(x) for x in args.dense)
    # Dropout
    if args.dropout_dense is None:
        dropout_str = ''
    else:
        dropout_str = '_drop_%0.3f_'%(args.dropout_dense)
        dropout_str = dropout_str .replace('.','_') 
    # Label
    if args.label is None:
        label_str = ""
    else:
        label_str = "%s_"%args.label
    # Conv dropout
    # if args.dropout_spatial is None:
    #     dropoutS_str = ''
    # else:
    #     dropoutS_str = 'dropS_%0.3f_'%(args.dropout_spatial)
    #     dropoutS_str= dropout_str.replace('.','_')
     
     # L2 regularization
    if args.L2_dense is None:
        regularizer_l2_str = ''
    else:
        regularizer_l2_str = 'L2_%0.6f_'%(args.L2_dense)
        regularizer_l2_str= regularizer_l2_str.replace('.','_')    
    # Experiment type
    if args.exp_type is None:
        experiment_type_str = ""
    else:
        experiment_type_str = "%s_"%args.exp_type

    # learning rate
    lrate_str = "LR_%0.6f"%args.lrate
    lrate_str= lrate_str.replace('.','_')
    # Put it all together, include a %s for each included string or argument
    return "%s/%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s"%(args.results_path,
                                        experiment_type_str, label_str,
                                        conv_size_str,conv_filter_str,
                                        rnn_str,rnn_size_string,
                                        gru_str,gru_size_string,
                                        lstm_str, lstm_size_string,
                                        dense_str, dense_size_str,
                                        dropout_str, 
                                        regularizer_l2_str,                                                                           
                                        lrate_str)
    '''
    Other naming examples:
    # Inception
    if args.inception is None:
        incep_str =''
    else:
        incep_str = '_'.join(str(x) for x in args.inception)
    
    
        
    # Conv dropout
    if args.dropout_spatial is None:
        dropout_str = ''
    else:
        dropout_str = 'drop_%0.3f_'%(args.dropout_spatial)
        dropout_str= dropout_str.replace('.','_')
    
    # L2 regularization
    if args.lambda_l2 is None:
        regularizer_l2_str = ''
    else:
        regularizer_l2_str = 'L2_%0.6f_'%(args.lambda_l2)
        regularizer_l2_str= regularizer_l2_str.replace('.','_')
    '''

#################################################################
def execute_exp(args=None):
    '''
    Perform the training and evaluation for a single model
    
    :param args: Argparse arguments
    '''
    

    # Check the arguments
    if args is None:
        # Case where no args are given (usually, because we are calling from within Jupyter)
        #  In this situation, we just use the default arguments
        parser = create_parser()
        args = parser.parse_args([])        
    print(args.exp_index)
    
    # Override arguments if we are using exp_index

    args_str = augment_args(args)
    
    # Set number of threads, if it is specified
    if args.cpus_per_task is not None:
        tf.config.threading.set_intra_op_parallelism_threads(args.cpus_per_task)
        tf.config.threading.set_inter_op_parallelism_threads(args.cpus_per_task)   
        
    #Importing Data, Many ways to do this.  Included is a data generator method 
    #using the function from suplement file (multiple methods available including
    # pickle files, folder structures, .mat files)     
    
    mat = scipy.io.loadmat(args.dataset)
    data =mat['data']
    label = mat['label']
    # data = np.reshape(dat,(8000,1024))
    if args.conv_size is None:
        class_size = int(np.round(data.shape[0]/args.nclasses))
        train_portion = int(np.round(.75*data.shape[0]/args.nclasses))
        val_portion = int(np.round(.125*data.shape[0]/args.nclasses))
        test_portion =int(np.round(.125*data.shape[0]/args.nclasses)) 
        train_x0=[]
        train_y0=[]
        val_x0=[]
        val_y0=[]
        test_x0=[]
        test_y0=[]
        for i in range(args.nclasses):
            train_x0.append(data[i*class_size:i*class_size+train_portion])
            train_y0.append(label[i*class_size:i*class_size+train_portion])
            val_x0.append(data[i*class_size+train_portion:i*class_size+train_portion+val_portion])
            val_y0.append(label[i*class_size+train_portion:i*class_size+train_portion+val_portion])
            test_x0.append(data[i*class_size+train_portion+val_portion:(i+1)*class_size])
            test_y0.append(label[i*class_size+train_portion+val_portion:(i+1)*class_size])
                    
        train_x0=np.reshape(train_x0,(args.nclasses*train_portion,data.shape[1]))
        train_y0=np.reshape(train_y0,(args.nclasses*train_portion,label.shape[1]))
        val_x0=np.reshape(val_x0,(args.nclasses*val_portion,data.shape[1]))
        val_y0=np.reshape(val_y0,(args.nclasses*val_portion,label.shape[1]))
        test_x0=np.reshape(test_x0,(args.nclasses*test_portion,data.shape[1]))
        test_y0=np.reshape(test_y0,(args.nclasses*test_portion,label.shape[1]))

        train_x,train_y = shuffle(train_x0,train_y0)
        val_x,val_y = shuffle(val_x0,val_y0)
        test_x,test_y = shuffle(test_x0,test_y0)               
    else:
        data = np.reshape(data,(data.shape[0],32,32))
        class_size = int(np.round(data.shape[0]/args.nclasses))
        train_portion = int(np.round(.75*data.shape[0]/args.nclasses))
        val_portion = int(np.round(.125*data.shape[0]/args.nclasses))
        test_portion =int(np.round(.125*data.shape[0]/args.nclasses)) 
        train_x0=[]
        train_y0=[]
        val_x0=[]
        val_y0=[]
        test_x0=[]
        test_y0=[]
        for i in range(args.nclasses):
            train_x0.append(data[i*class_size:i*class_size+train_portion])
            train_y0.append(label[i*class_size:i*class_size+train_portion])
            val_x0.append(data[i*class_size+train_portion:i*class_size+train_portion+val_portion])
            val_y0.append(label[i*class_size+train_portion:i*class_size+train_portion+val_portion])
            test_x0.append(data[i*class_size+train_portion+val_portion:(i+1)*class_size])
            test_y0.append(label[i*class_size+train_portion+val_portion:(i+1)*class_size])
                    
        train_x0=np.reshape(train_x0,(args.nclasses*train_portion,data.shape[1],data.shape[2]))
        train_y0=np.reshape(train_y0,(args.nclasses*train_portion,label.shape[1]))
        val_x0=np.reshape(val_x0,(args.nclasses*val_portion,data.shape[1],data.shape[2]))
        val_y0=np.reshape(val_y0,(args.nclasses*val_portion,label.shape[1]))
        test_x0=np.reshape(test_x0,(args.nclasses*test_portion,data.shape[1],data.shape[2]))
        test_y0=np.reshape(test_y0,(args.nclasses*test_portion,label.shape[1]))
    
        train_x,train_y = shuffle(train_x0,train_y0)
        val_x,val_y = shuffle(val_x0,val_y0)
        test_x,test_y = shuffle(test_x0,test_y0)
    
    
    
    #Pull data given fold
    # dataset_train = create_dataset(base_dir=args.dataset,
    #                          partition='train', fold=args.rotation, filt=args.filts_train, 
    #                          batch_size=8, prefetch=2, num_parallel_calls=4)
    # dataset_valid = create_dataset(base_dir=args.dataset,
    #                          partition='train', fold=args.rotation, filt=args.filts_valid, 
    #                          batch_size=8, prefetch=2, num_parallel_calls=4)
    # dataset_test= create_dataset(base_dir=args.dataset,
    #                          partition='valid', fold=args.rotation, filt=args.filts_test, 
    #                          batch_size=8, prefetch=2, num_parallel_calls=4)

    # Parameter configuratuion
    if args.conv_size is not None:
        #dictionary of conv parameters
        conv_layers = [{'filters': f, 'kernel_size': s, 'stride': st}
                   for s, f, st, in zip(args.conv_size, args.conv_nfilters, args.conv_stride)]
    else:
        conv_layers = None  
    if args.rnn_size is not None:
        #dictionary of rnn parameters
        rnn_layers = [{'units': i} for i in args.rnn_size]
    else:
        rnn_layers = None
    
    if args.gru_size is not None:
        #dictionary of gru parameters
        gru_layers = [{'units': i} for i in args.gru_size]
    else:
        gru_layers = None
    
    if args.lstm_size is not None:
        #dictionary of lstm parameters
        lstm_layers = [{'units': i} for i in args.lstm_size]
    else:
        lstm_layers = None
    dense_layers = [{'units': i} for i in args.dense]
    
    #function to build model given layer parameters provided above
    if args.conv_size is None:
        if (args.lstm_size or args.gru_size) is None:
            model=create_mlp(data.shape[1], args.nclasses,
                                  dense_layers=dense_layers,
                                  activation='elu',
                                  dropout=args.dropout_dense,                             
                                  lambda_l2=args.L2_dense,
                                  lrate=args.lrate)
        else:
             model=create_rnn_network(input_dim=(data.shape[1],data.shape[2]),
                                     n_classes=args.nclasses,
                                     conv_layers=conv_layers,
                                     rnn_layers=rnn_layers,
                                     gru_layers=gru_layers,
                                     lstm_layers=lstm_layers,
                                     dense_layers=dense_layers,
                                     activation='elu',
                                     activation_dense='elu',
                                     recurrent_dropout=args.dropout_rnn,
                                     dropout_dense=args.dropout_dense,
                                     lambda_l2_dense=args.L2_dense,
                                     lambda_l2_rnn=args.L2_rnn,
                                     lrate=args.lrate)
    else:
        model=create_cnn(args.image_size, args.nclasses,
                                conv_layers=conv_layers,
                                dense_layers=dense_layers,
                                activation='elu',
                                dropout=args.dropout_dense,                             
                                lambda_l2=args.L2_dense,
                                lrate=args.lrate)
    
    # Report model structure if verbosity is turned on
    fbase = generate_fname(args, args_str)
    if args.verbose >= 1:
        print(model.summary())
        plot_model(model, to_file='%s_model_plot.png'%fbase, show_shapes=True, show_layer_names=True)
    print(args)

    # Output file base and pkl file

    fname_out = "%s_results.pkl"%fbase
    
    # Perform the experiment?
    if(args.nogo):
        # No!
        print("NO GO")
        print(fbase)
        return

    # Check if output file already exists
    if os.path.exists(fname_out):
            # Results file does exist: exit
            print("File %s already exists"%fname_out)
            return
            
    # Callbacks
    early_stopping_cb = keras.callbacks.EarlyStopping(patience=args.patience,
                                                      restore_best_weights=True,
                                                      min_delta=args.min_delta)

    # Learn
    #  steps_per_epoch: how many batches from the training set do we use for training in one epoch?
    history = model.fit(train_x, train_y,
                        epochs=args.epochs,
                        steps_per_epoch=args.steps_per_epoch,
                        use_multiprocessing=False, 
                        verbose=args.verbose>=2,
                        validation_data=(val_x,val_y),
                        validation_steps=args.steps_per_epoch, 
                        callbacks=[early_stopping_cb])

    # Calculate and store the important input/output and prediction information from the testing set
    results = {}
    results['args'] = args
    results['train_in'] = train_x
    results['train_label'] = train_y
    results['val_in'] = val_x
    results['val_label'] = val_y
    results['test_in'] = test_x
    results['test_label'] = test_y
    results['predict_validation'] = model.predict(val_x)
    results['predict_validation_eval'] = model.evaluate(val_x,val_y)
    
    if args.testing is not None:
        results['predict_testing'] = model.predict(test_x)
        results['predict_testing_eval'] = model.evaluate(test_x,test_y)
        
    results['predict_training'] = model.predict(train_x)
    results['predict_training_eval'] = model.evaluate(train_x,train_y)
    results['history'] = history.history

    
    # Save results
    fbase = generate_fname(args, args_str)
    results['fname_base'] = fbase
    with open("%s_results.pkl"%(fbase), "wb") as fp:
        pickle.dump(results, fp)
    
    # Save model
    print(fbase)
    model.save("%s_model"%(fbase))
   
    return model

#################################################################
def read_all_rotations(dirname, filebase):
    '''Read results from dirname from files matching filebase
    This function can open multiple files with the same basic naming
    convention.  Useful to pull information for metrics and plotting given
    parameter sweep or multiple experiments. 
    '''

    # The set of files in the directory
    files = fnmatch.filter(os.listdir(dirname), filebase)
    files.sort()
    results = []

    # Loop over matching files
    for f in files:
        fp = open("%s/%s"%(dirname,f), "rb")
        r = pickle.load(fp)
        fp.close()
        results.append(r)
    return results

def display_Metrics(args1, test=False,save=False):    
    '''
    This function will take in the args for the shallow and deep network of 
    choice and then generate the figures (This is application specific)
    '''
    #pull the directory and base file name for shallow args 
    args_str = augment_args(args1)
    fbase = generate_fname(args1, args_str)
    # fbase_1,drop = fbase.rsplit('rot',1)
    fbase_1 = '%s_results.pkl'%(fbase)
    dir, fbase_0 = fbase_1.rsplit('/',1)
    dir, fbase_save = fbase.rsplit('/',1)
    
    #check results folder for all rotations with shallow network args
    results = read_all_rotations(dir, fbase_0)
    
    #empty lists for appending
    val_acc1=[]
    train_acc1=[]
    
    #extract validation accuracy, and testing accuracy
    for res_temp in results:
        train_acc1.append(res_temp['history']['sparse_categorical_accuracy'])
        val_acc1.append(res_temp['history']['val_sparse_categorical_accuracy'])      
        
    # colors = ['r','b','g','y','m' ]
    
    #Plot figure 2 Validation sparce accuracy vs. Epoch
    plt.figure(1)
    for i in range(np.size(results)):
        plt.plot(train_acc1[i],linestyle='-', label = 'Training')
        plt.plot(val_acc1[i],linestyle='-', label = 'Validation')

        # plt.plot(val_acc2[i],linestyle='--',color = colors[i], label ='Complex_Rotation_%s'%(i+1))
    plt.title("Validation/Training ACC vs. Epoch")
    plt.ylabel('Sparse Categorical Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.3), ncol= 5)
    plt.grid(True)
    plt.gcf().set_size_inches(10, 5)
    # plt.yticks(np.arange(0.9,1,.05))
    if save:
        plt.savefig('Val_ACC_%s'%(fbase_save),facecolor='white',bbox_inches='tight')        
    # Load all 5 rotation models and iterate across the data to generate the confusion matricies
    if args1.nclasses ==7:
        labels = [0,1,2,3,4,5,6]
        if args1.dataset == 'clutter_order_7.mat':
            xylabels = ['Gaussian', 'K-dist(a=0.5)','K-dist(a=1.5)','K-dist(a=4.5)', 'Pareto(a=1)','Pareto(a=3)','Pareto(a=10)']
        else:
            xylabels = ['Gaussian', 'K-dist(a=0.5)','K-dist(a=1.5)','K-dist(a=4.5)', 'Pareto(a=2)','Pareto(a=3)','Pareto(a=10)']
    elif args1.nclasses == 3:
        labels = [0,1,2]
        xylabels = ['Gaussian', 'K-dist(a=0.5)','K-dist(a=4.5)']
    else:
        labels = [0,1,2,3]
        xylabels = ['Gaussian', 'K-dist(a=0.5)','K-dist(a=1.5)','K-dist(a=4.5)']
    # test_label = results['test_label'] 
    test_label = results[0]['test_label']     
    test_pred = results[0]['predict_testing']
    test_preds = np.argmax(test_pred, axis=1)

    train_label = results[0]['train_label']
    train_pred = results[0]['predict_training']
    train_preds = np.argmax(train_pred, axis=1)

    val_label = results[0]['val_label']
    val_pred = results[0]['predict_validation']
    val_preds = np.argmax(val_pred, axis=1)
    #Generate the confusion matrix 
    conf_matrix3=sklearn.metrics.confusion_matrix(test_label, test_preds ,labels=labels, sample_weight=None, normalize='true')
    conf_matrix1=sklearn.metrics.confusion_matrix(train_label, train_preds ,labels=labels, sample_weight=None, normalize='true')
    conf_matrix2=sklearn.metrics.confusion_matrix(val_label, val_preds ,labels=labels, sample_weight=None, normalize='true')
    plt.figure(2)
    g1=seaborn.heatmap(conf_matrix1,annot=True, linewidths=4, cmap='magma_r', xticklabels=xylabels,yticklabels=xylabels)
    g1.set_title('Training Confusion Matrix (Normalized true classes)', fontsize=20)
    g1.set_xlabel("Predicted Class", fontsize=16)
    g1.set_ylabel("True Class", fontsize=16)
    plt.gcf().set_size_inches(10, 5)
    if save:
        plt.savefig('Train_Confusion_Matrix_%s'%(fbase_save),facecolor='white',bbox_inches='tight')
    plt.figure(3)
    g2=seaborn.heatmap(conf_matrix2,annot=True, linewidths=4, cmap='magma_r', xticklabels=xylabels,yticklabels=xylabels)
    g2.set_title('Validation Confusion Matrix (Normalized true classes)', fontsize=20)
    g2.set_xlabel("Predicted Class", fontsize=16)
    g2.set_ylabel("True Class", fontsize=16)
    plt.gcf().set_size_inches(10, 5)
    if save:
        plt.savefig('Val_Confusion_Matrix_%s'%(fbase_save),facecolor='white',bbox_inches='tight')
    plt.figure(4)
    g3=seaborn.heatmap(conf_matrix3,annot=True, linewidths=4, cmap='magma_r', xticklabels=xylabels,yticklabels=xylabels)
    g3.set_title('Testing Confusion Matrix (Normalized true classes)', fontsize=20)
    g3.set_xlabel("Predicted Class", fontsize=16)
    g3.set_ylabel("True Class", fontsize=16)
    plt.gcf().set_size_inches(10, 5)
    if save:
        plt.savefig('Test_Confusion_Matrix_%s'%(fbase_save),facecolor='white',bbox_inches='tight')
#     if test:                            
#         base_dir = 'radiant_earth/pa'
#         dataset_test = create_dataset(base_dir=base_dir,
#                            partition='valid', fold=0, filt=['*'], 
#                            batch_size=8, prefetch=2, num_parallel_calls=4)
#         #excrate the comman model naming
#         mbase,drop = model_name.rsplit('rot',1)
#         #Loop across all folds
#         for folds in range(5):
#             #Load the trained model
#             model = tf.keras.models.load_model('%srot_0%s_model'%(mbase,folds))
#             #lists for appending
#             ins_test = []
#             outs_test = []
#             preds_test = []
#             labels_test = []
#             #Loop the dataset and store testing inputs/outputs and predictions
#             for i in dataset_test:
#                 ins = i[0].numpy()
#                 outs = i[1].numpy()
#                 ins_test.append(ins)
#                 outs_test.append(outs)
#                 preds = model.predict(ins)
#                 preds_test.append(preds)
#                 labels = np.argmax(preds, axis=3)
#                 labels_test.append(labels)    
#             # Build final lists after looping testing generator
#             ins_test = np.concatenate(ins_test, axis=0)
#             outs_test =  np.concatenate(outs_test, axis=0) 
#             preds_test = np.concatenate(preds_test, axis=0)
#             labels_test = np.concatenate(labels_test, axis=0)
#             ins_test = ins_test[:,:,:,0:3]                    
            
#             # convert to vector to generate the confusion matrix
#             conf_true = np.reshape(outs_test, -1)
#             conf_pred = np.reshape(labels_test, -1)
#             #label class to ensure 0 case.  
#             labels = [0,1,2,3,4,5,6]
            
#             #Generate the confusion matrix 
#             conf_matrix1=sklearn.metrics.confusion_matrix(conf_true, conf_pred ,labels=labels, sample_weight=None, normalize='true')
#             conf_matrix2=sklearn.metrics.confusion_matrix(conf_true, conf_pred ,labels=labels, sample_weight=None, normalize=None)
            
#             #Generate and format plots for confusion matricies
#             subplot_args = { 'nrows': 1, 'ncols': 2, 'figsize': (20, 8),
#                                          'subplot_kw': {'xticks': [], 'yticks': []} }
#             f, (ax1,ax2) = plt.subplots(**subplot_args)                                   
#             g1=seaborn.heatmap(conf_matrix1,annot=True, linewidths=4, cmap='magma_r',ax=ax1)
#             g1.set_title('Fold %s Testing Confusion Matrix (Normalized true classes)'%(folds), fontsize=20)
#             g1.set_xlabel("Predicted Class", fontsize=16)
#             g1.set_ylabel("True Class", fontsize=16)
#             g2=seaborn.heatmap(conf_matrix2,annot=True, linewidths=4, cmap='magma_r',ax=ax2)
#             g2.set_title('Fold %s Testing Confusion Matrix(Raw data)'%(folds), fontsize=20)
#             g2.set_xlabel("Predicted Class", fontsize=16)
#             g2.set_ylabel("True Class", fontsize=16)
#             if save:
#                 plt.savefig('Fig%s_Confusion_Matrix_Fold%s'%(folds+3,folds),facecolor='white',bbox_inches='tight')
        
#         #Plot figure 8 Testing Histogram for sparce accuracy
#         plt.figure(8)
#         plt.rcParams["figure.figsize"] = (12,8)
#         plt.hist(test_acc1, 20, color='r',alpha=.5)
#         plt.axvline(x=np.mean(test_acc1),color='r', label ='Mean at %0.5f'%(np.mean(test_acc1)) )
#         plt.title("Testing Sparse Categorical Accuracy")
#         plt.xlabel('Sparse Categorical Accuracy')
#         plt.ylabel('Number of Rotations')
#         plt.yticks(np.arange(0,2,1))
#         plt.legend()
#         if save:
#             plt.savefig('Fig8_Testing_Histogram_ACC_%s'%(fbase_save),facecolor='white',bbox_inches='tight')
        
#         #Plot figure 9 Examples of classifier
#         model = tf.keras.models.load_model(model_name)
#         ins_test = []
#         outs_test = []
#         preds_test = []
#         labels_test = []
#         for i in dataset_test:
#             ins = i[0].numpy()
#             outs = i[1].numpy()
#             ins_test.append(ins)
#             outs_test.append(outs)
#             preds = model.predict(ins)
#             preds_test.append(preds)
#             labels = np.argmax(preds, axis=3)
#             labels_test.append(labels)    

#         ins_test = np.concatenate(ins_test, axis=0)
#         outs_test =  np.concatenate(outs_test, axis=0) 
#         preds_test = np.concatenate(preds_test, axis=0)
#         labels_test = np.concatenate(labels_test, axis=0)
#         ins_test = ins_test[:,:,:,0:3]
        
#         #Random Selection of examples (I tried to find examples with multiple classes)
#         batch = [18,151,201]    
#         subplot_args = { 'nrows': 3, 'ncols': 3, 'figsize': (20, 20),
#                  'subplot_kw': {'xticks': [], 'yticks': []} }
#         f, ax = plt.subplots(**subplot_args)

#         for j,num in enumerate(batch):
#             ax[j,0].imshow(ins_test[num,:,:,0:3])
#             ax[j,0].set_title('RGB Example %s'%(num), fontsize=14)
#             ax[j,1].imshow(outs_test[num,:,:],vmin=0, vmax=6)
#             ax[j,1].set_title('Truth Example %s'%(num), fontsize=14)
#             ax[j,2].imshow(labels_test[num,:,:],vmin=0, vmax=6)
#             ax[j,2].set_title('Prediction Example %s'%(num), fontsize=14)
#             if save:
#                 plt.savefig('Fig9_Examples',facecolor='white',bbox_inches='tight')
            
            
def model_summary(args):
    '''
    This function follows the same process used in experiment to build the 
    model given the parameters.  
    The main purpose of thie function is to easily display the model summary 
    and any model diagrams.
    '''
    # Parameter configuratuion
    if args.conv_size is not None:
        #dictionary of conv parameters
        conv_layers = [{'filters': f, 'kernel_size': s, 'stride': st}
                   for s, f, st, in zip(args.conv_size, args.conv_nfilters, args.conv_stride)]
    else:
        conv_layers = None  
    dense_layers = [{'units': i} for i in args.dense]
    
    #function to build model given layer parameters provided above
    model=create_cnn(args.image_size, args.nclasses, conv_layers=conv_layers,                             
                              maxpool=args.maxpool,
                              dense_layers=dense_layers,
                              activation='elu',
                              dropout=args.dropout_dense,
                              dropout_spatial=args.dropout_spatial,
                              lambda_l2=args.L2_dense,
                              lrate=args.lrate)
    
    # Report model structure if verbosity is turned on
    print(model.summary())
    args_str = augment_args(args)
    fbase = generate_fname(args, args_str)
    plot_model(model, to_file='%s_model_plot.png'%fbase, show_shapes=True, show_layer_names=True)

    
    
if __name__ == "__main__":
    # Parse and check incoming arguments
    parser = create_parser()
    args = parser.parse_args()
    check_args(args)
    
    # Turn off GPU?
    if not args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        
    # GPU check
    physical_devices = tf.config.list_physical_devices('GPU') 
    n_physical_devices = len(physical_devices)
    if(n_physical_devices > 0):
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print('We have %d GPUs\n'%n_physical_devices)
    else:
        print('NO GPU')

    if(args.check):
        # Just check to see if all experiments have been executed
        check_completeness(args)
    else:
        # Execute the experiment
        execute_exp(args)
        
