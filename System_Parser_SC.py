# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 10:39:12 2023

@author: Timmy
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
import os
import Cognitive_Detector as cog
#import mat73
from sklearn.utils import shuffle
import pickle
import argparse
from job_control import *



def create_parser():
    '''
    Create argument parser
    -This function will create the parser structure and include parameter 
    setting given the specific application
    '''
    # Parse the command-line arguments
    parser = argparse.ArgumentParser(description='Threshold', fromfile_prefix_chars='@')
    
    # High-level experiment configuration
    parser.add_argument('--exp_index', type=int, default=None, help='Experiment index')
    parser.add_argument('--exp_type', type=str, default=None, help="Experiment type")    
    parser.add_argument('--label', type=str, default='', help="Extra label to add to output files")
    parser.add_argument('--partition', type=int, default=None, help="Number of partitions of the dataset to iterate")    
    parser.add_argument('--rotation', type=int, default=1, help="rotation to run from 1 to partition")
    
    parser.add_argument('--results_path', type=str, default='./results', help='Results directory')
    parser.add_argument('--verbose', '-v', action='count', default=0, help="Verbosity level")
    # CPU/GPU
    parser.add_argument('--cpus_per_task', type=int, default=None, help="Number of threads to consume")
    parser.add_argument('--gpu', action='store_true', help='Use a GPU')
    #Expirement parameters
    parser.add_argument('--algorithm', type=str, default='glrt', help='Algorithm type you want to use')
    parser.add_argument('--N', type=int, default=16, help="N parameter")    
    parser.add_argument('--K', type=int, default=32, help="K parameter")
    parser.add_argument('--P_fa', type=float, default=1/np.power(10,4), help="Probability of false alarm ")
    parser.add_argument('--sample_len', type=int, default=16, help="Length of the sample")
    parser.add_argument('--PRI', type=float, default=1e-8, help="Pulse repetition interval")
    parser.add_argument('--f_d', type=float, default=2e7, help="Doppler frequency")
    parser.add_argument('--target', action='store_true', help="If you want the cut_target (true) or just cut (false)")
    parser.add_argument('--max_test', type=int, default=None, help="Limit on the maximum number of runs")
    
    parser.add_argument('--discriminator', type=str, default='./models/ordered_a_Dense200_50_drop_0_100_LR_0_000100_model', help='This is the file path to the Discriminator Model')
    parser.add_argument('--data', type=str, default='./datasets/clutter_final_G.mat', help='File path to the data mat file')
    parser.add_argument('--model_name', type=str, default='./models/*_Dense200_50_drop_0_100_LR_0_000100_model', help='File naming convention for threshold model')
    # parser.add_argument('--model_thresh4', type=str, default='./classifier/results/P_14_Dense1000_200_50_drop_0_100_LR_0_000100_model', help='File path to a threshold model')
    # parser.add_argument('--model_thresh5', type=str, default='./classifier/results/P_44_Dense1000_200_50_drop_0_100_LR_0_000100_model', help='File path to a threshold model')
    # parser.add_argument('--model_thresh6', type=str, default='./classifier/results/P_84_Dense1000_200_50_drop_0_100_LR_0_000100_model', help='File path to a threshold model')

    # parser.add_argument('--model_thresh1', type=str, default='./classifier/results/K_54_Dense1000_200_50_drop_0_100_LR_0_000100_model', help='File path to a threshold model')
    # parser.add_argument('--model_thresh2', type=str, default='./classifier/results/K_14_Dense1000_200_50_drop_0_100_LR_0_000100_model', help='File path to a threshold model')
    # parser.add_argument('--model_thresh3', type=str, default='./classifier/results/K_44_Dense1000_200_50_drop_0_100_LR_0_000100_model', help='File path to a threshold model')
    
    parser.add_argument('--model_final', type=str, default='./models/G_GF4_Dense500_50_drop_0_100_LR_0_000100_model', help='File path to a threshold model')

    parser.add_argument('--alg', type=str, default='glrt', help="Adaptive detection algorithm to use.")
    parser.add_argument('--dist', type=str, default='gaussian', help="The clutter distribution to use.")
    
    
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
    elif args.exp_type =='sys_sweep':        
        p = {'P_fa':[0.0001,0.0002,0.0003,0.0004,0.0005,0.0006,0.0007,0.0008,0.0009,0.001],
             'target':[False,True]}
    elif args.exp_type =='CNN':
        p = {'rotation': range(5)}
    else:
        assert False, "Unrecognized exp_type"

    return p

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
    
    # Experiment type
    if args.exp_type is None:
        experiment_type_str = ""
    else:
        experiment_type_str = "%s_"%args.exp_type
        # Experiment type
    if args.label is None:
        label_str = ""
    else:
        label_str = "%s_"%args.label
    if args.target:
        target_str='tgt_'
    else:
        target_str='no_tgt_'
    if args.partition is None:
        rot_str=""
    else:
        rot_str="rot_%sof%s"%(args.rotation,args.partition)

    # Put it all together, include a %s for each included string or argument
    return "%s/%s%s%s_%s%s"%(args.results_path,
                                        experiment_type_str, label_str, args.P_fa, target_str, rot_str)





def zero(args,p,os,z,S,model_thresh):
    thresh = np.asarray(cog.cogThreshold(args.algorithm,args.P_fa,args.K,args.N))
    det = np.asarray(cog.cogDetector(args.algorithm, z, p, S, thresh))
    # print("Detector 0")
    return det

def one(args,p,os,z,S,model_thresh1):
    thresh = model_thresh1.predict(os,verbose=0)
    det = np.asarray(cog.cogDetector(args.algorithm, z, p, S, thresh))
    # print("Detector 1")
    return det

def two(args,p,os,z,S,model_thresh2):
    thresh = model_thresh2.predict(os,verbose=0)
    det = np.asarray(cog.cogDetector(args.algorithm, z, p, S, thresh))
    # print("Detector 2")
    return det

def three(args,p,os,z,S,model_thresh3):
    thresh = model_thresh3.predict(os,verbose=0)
    det = np.asarray(cog.cogDetector(args.algorithm, z, p, S, thresh))
    # print("Detector 3")
    return det

def four(args,p,os,z,S,model_thresh4):
    thresh = model_thresh4.predict(os,verbose=0)
    det = np.asarray(cog.cogDetector(args.algorithm, z, p, S, thresh))
    # print("Detector 4")
    return det

def five(args,p,os,z,S,model_thresh5):
    thresh = model_thresh5.predict(os,verbose=0)
    det = np.asarray(cog.cogDetector(args.algorithm, z, p, S, thresh))
    # print("Detector 5")
    return det

def six(args,p,os,z,S,model_thresh6):
    thresh = model_thresh6.predict(os,verbose=0)
    det = np.asarray(cog.cogDetector(args.algorithm, z, p, S, thresh))
    # print("Detector 6")
    return det


def cogThreshold(alg,P_fa,K,N):
    # Calculates the threshold for the GLRT detector. This calculation is exact, but assumes a Gaussian distribution for the interference.
    if alg == 'glrt':
        l_0 = 1/(np.power(P_fa,(1/(K+1-N))));         #Set threshold l_0 from desired PFA, sample support K, and CPI pulse number N
        eta_0 = (l_0-1)/l_0;  
        return eta_0
    # Calculates the threshold for the AMF detector. This calculation is an approximation that loses fidelity the lower the 
    # values of N and K and assumes a Gaussian distribution for the interference.
    elif alg == 'amf':
        eta_0 = ((K+1)/(K-N+1))*(np.power(P_fa,(-1/(K+2-N)))-1)
        return eta_0
    # Calculates the threshold for the ACE detector. This calculation is an approximation that loses fidelity the lower the 
    # values of N and K and assumes a Gaussian distribution for the interference.
    elif alg == 'ace':
        num = 1-(np.power(P_fa,(1/(K+1-N))));
        den = 1-(((K-N+1)/(K+1))*(np.power(P_fa,(1/(K+1-N)))))
        eta_0 = num/den
        return eta_0
    else:
        print('Unrecognized detector type.')
        



def execute_exp(args=None):
    '''
    Execute Tests
    
    :param args: Argparse arguments
    '''
    # Check the arguments
    if args is None:
        # Case where no args are given (usually, because we are calling from within Jupyter)
        #  In this situation, we just use the default arguments
        parser = create_parser()
        args = parser.parse_args([])        
    #  print(args.exp_index)
    
    args_str = augment_args(args)
    
    if args.cpus_per_task is not None:
        tf.config.threading.set_intra_op_parallelism_threads(args.cpus_per_task)
        tf.config.threading.set_inter_op_parallelism_threads(args.cpus_per_task)
    
    pulse_num = np.linspace(1,args.sample_len,args.sample_len)
    p = np.exp(-1j*2*np.pi*pulse_num*args.f_d*args.PRI)
    
    
    
        
    # Load in our data and format it as necessary.
    # Note depending on the file you are loading defines the dimensions of the data
    # The SIR_Sweep will have 3D and the normal data will have 
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
        
        
    if args.partition is not None:
        data_partition = int(len(data_ss)/args.partition)
        if args.rotation > args.partition:  
            print("rotation (%s) is out of range for partition size (%s)"%(args.rotation, args.partition))
            return
        elif args.rotation < args.partition:
            data_ss = data_ss[(args.rotation-1)*data_partition:(args.rotation)*data_partition,:]
            S = S[(args.rotation-1)*data_partition:(args.rotation)*data_partition,:,:]
            z_full =  z_full[(args.rotation-1)*data_partition:(args.rotation)*data_partition,:]        
        else:
            data_ss = data_ss[(args.rotation-1)*data_partition:,:]
            S = S[(args.rotation-1)*data_partition:,:,:]
            z_full =  z_full[(args.rotation-1)*data_partition:,:]  
        
    
    
    #Load in the discriminator agent.
    model_disc = tf.keras.models.load_model(args.discriminator, compile=False)
    model_disc.compile(optimizer='adam')
    thmodel = args.model_name
    thmodel_0, thmodel_1 = thmodel.rsplit('*')
    model_th1 = '%s%s_%s_%s%s'%(thmodel_0,'K','L',args.P_fa,thmodel_1)
    model_th2 = '%s%s_%s_%s%s'%(thmodel_0,'K','M',args.P_fa,thmodel_1)
    model_th3 = '%s%s_%s_%s%s'%(thmodel_0,'K','H',args.P_fa,thmodel_1)
    model_th4 = '%s%s_%s_%s%s'%(thmodel_0,'P','L',args.P_fa,thmodel_1)
    model_th5 = '%s%s_%s_%s%s'%(thmodel_0,'P','M',args.P_fa,thmodel_1)
    model_th6 = '%s%s_%s_%s%s'%(thmodel_0,'P','H',args.P_fa,thmodel_1)
    
    
    #Load in the threshold setting models.
    ##These are the models that I trained on the order statistics for PFA=10^-4.
    model_thresh4 = tf.keras.models.load_model(model_th4, compile=False)
    model_thresh4.compile(optimizer='adam')
    model_thresh5 = tf.keras.models.load_model(model_th5, compile=False)
    model_thresh5.compile(optimizer='adam')
    model_thresh6 = tf.keras.models.load_model(model_th6, compile=False)
    model_thresh6.compile(optimizer='adam')
    
    ##These are the models that I trained on the order statistics for PFA=10^-4.
    model_thresh1 = tf.keras.models.load_model(model_th1, compile=False)
    model_thresh1.compile(optimizer='adam')
    model_thresh2 = tf.keras.models.load_model(model_th2, compile=False)
    model_thresh2.compile(optimizer='adam')
    model_thresh3 = tf.keras.models.load_model(model_th3, compile=False)
    model_thresh3.compile(optimizer='adam')
    
    model_final = tf.keras.models.load_model(args.model_final, compile=False)
    model_final.compile(optimizer='adam')
        
    
    models = {  0: model_thresh1,
                1: model_thresh1,
                2: model_thresh2,
                3: model_thresh3,
                4: model_thresh4,
                5: model_thresh5,
                6: model_thresh6}
    

    options = {0 : zero,
               1 : one,
               2 : two,
               3 : three,
               4 : four,
               5 : five,
               6 : six}
    
    # Sets up data collection and output
    results = np.zeros([len(z_full),4])
    FA_CD = 0
    FA_glrt = 0
    FA_ideal = 0
    #Run the data through the combined system.
    
    
    # If statment distingishes between the SIR_Sweep (3 dim) and the "normal" data (2 dim)
    if( z_full.ndim==3):
        for j in range(len(z_full)):
            # z = z_full[j,:,:]
            print(z_full.shape)
            z = z_full[j+10,:]
            print(z.shape)
            FA_CD = 0
            FA_glrt = 0
            FA_ideal = 0
            for i in range( len(data_ss) if args.max_test is None else args.max_test):
                det, det_glrt, det_ideal = runDet(data_ss, z, S,p, models,model_disc ,options,i)
                
                FA_CD = FA_CD+det 
                FA_ideal = FA_ideal + det_ideal
                FA_glrt = FA_glrt+det_glrt
                
                if(args.v>=1):
                    print('------')
                    print(i)
                if (args.v>=2):
                    print(np.argmax(disc_vector))
                    print("SIR")
                    # print(sir[j,i])
                    print(z_full[j+10,i])
                    # print(det_final[i])
                    print("Cognitive Detector")
                    print(FA_CD)
                    print("GLRT")
                    print(FA_glrt)
                    print("Ideal")
                    print(FA_ideal)
                    print('------')
            results[j+10,:] = [z_full[j+10,i],FA_CD,FA_glrt,FA_ideal]
    else:
        FA_CD = 0
        FA_glrt = 0
        FA_ideal = 0
        for i in range(len(data_ss) if args.max_test is None else args.max_test):
            det, det_glrt, det_ideal, shape_disc = runDet(args,data_ss, z_full, S,p, models,model_disc,options, i)
            
            FA_CD = FA_CD+det 
            FA_ideal = FA_ideal + det_ideal
            FA_glrt = FA_glrt+det_glrt
            
            if (args.verbose>=1):
                print('------')
                print(i)
            if (args.verbose >=2):
                print(shape_disc)
                print("cut")
                # print(sir[j,i])
                print(z_full[i])
                # print(det_final[i])
                print("Cognitive Detector")
                print(FA_CD)
                print("GLRT")
                print(FA_glrt)
                print("Ideal")
                print(FA_ideal)
                print('------')
        results = [FA_CD,FA_ideal,FA_glrt]
    directory= os.getcwd()
    if not os.path.isdir('%s%s'%(directory,'/results/')):
        os.makedirs('%s%s'%(directory,'/results/'))    
    from scipy.io import savemat
    fbase = generate_fname(args, args_str)
    fname_out = "%s_results.mat"%fbase
    total_run=len(data_ss) if args.max_test is None else args.max_test
    outputData = {'FA_CD':results[0],'FA_KL':results[1],'FA_gauss':results[2],'data_run':total_run}
    savemat(fname_out,outputData)
    
    
    
    
def runDet(args,data_ss, z, S,p, models,model_disc,options, test_num):
    temp = np.expand_dims(data_ss[test_num,:],0)
    disc_vector = model_disc.predict(temp,verbose=0)
    max_disc = np.argmax(disc_vector)
    det = options[max_disc](args,p,temp, z[test_num], S[test_num,:,:],models[max_disc])
    det_glrt = options[0](args,p,temp, z[test_num], S[test_num,:,:],models[0])
    det_ideal = options[1](args,p,temp, z[test_num], S[test_num,:,:],models[1])
    return det, det_glrt, det_ideal, np.argmax(disc_vector)
    
    
    
    

if __name__ == "__main__":
    # Parse and check incoming arguments
    parser = create_parser()
    args = parser.parse_args()
    # check_args(args)
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
    
    
    execute_exp(args)
