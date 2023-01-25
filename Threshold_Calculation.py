# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 14:37:51 2023

@author: Alex
"""
import numpy as np
import argparse

import Clutter_Sim as cs
import Cognitive_Detector as cd

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
    parser.add_argument('--label', type=str, default='Test', help="Extra label to add to output files")
    
    parser.add_argument('--results_path', type=str, default='./results', help='Results directory')
    
    parser.add_argument('--alg', type=str, default='glrt', help="Adaptive detection algorithm to use.")
    parser.add_argument('--dist', type=str, default='gaussian', help="The clutter distribution to use.")
    parser.add_argument('--target_PFA', type=float, default=1e-2, help="Target PFA for threshold selection.") 
    parser.add_argument('--PFA_margin', type=float, default=0.1, help="PFA margin used to select threshold.")     
    parser.add_argument('--sim_ord', type=int, default=10, help='Used with target_PFA to determine the number of simulated datasets to generate.')    
    parser.add_argument('--N', type=int, default=16, help='Number of pulses per sample.')
    parser.add_argument('--K', type=int, default=32, help='Number of training samples simulation.')       
    parser.add_argument('--a', type=int, default=1, help='Clutter distribution shape parameter.') 
    parser.add_argument('--b', type=int, default=1, help='Clutter distribution scale parameter.') 
    
    # parser.add_argument('--TH_init', nargs='+', type=float, default=[0.5,0.5,0.5,0.5,0.5], help="List of initial threshold values.") 
    # parser.add_argument('--delta', nargs='+', type=float, default=[0.05,0.05,0.05,0.05,0.05], help="List of delta values to use for threshold searches.") 
    # parser.add_argument('--a_list', nargs='+', type=float, default=[0.5,0.75,1,1.25,1.5], help="List of clutter distribution shape parameter values to use.") 
    parser.add_argument('--TH_init', nargs='+', type=float, default=[0.2,0.2], help="List of initial threshold values.") 
    parser.add_argument('--delta', nargs='+', type=float, default=[0.05,0.05], help="List of delta values to use for threshold searches.") 
    parser.add_argument('--a_list', nargs='+', type=float, default=[0.5,0.75], help="List of clutter distribution shape parameter values to use.") 
    
    
    parser.add_argument('--rho', type=float, default=0.9, help='Clutter correlation coefficient.') 
    parser.add_argument('--max_count', type=int, default=40, help='Maximum number of iterations to run during threshold search.')
    parser.add_argument('--PRI', type=float, default=1e-8, help='Pulse repetition interval.') 
    parser.add_argument('--f_d', type=float, default=2e7, help='Center frequency.')
    
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

    # Put it all together, include a %s for each included string or argument
    return "%s/%s%s"%(args.results_path,
                                        experiment_type_str, label_str)


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
    #  print(args.exp_index)
    
    args_str = augment_args(args)
    
    PFA_max = args.target_PFA + args.PFA_margin*args.target_PFA
    PFA_min = args.target_PFA - args.PFA_margin*args.target_PFA
    sim_num = int(np.round(args.sim_ord/args.target_PFA))
    
    TH_init = np.asarray(args.TH_init)
    delta = np.asarray(args.delta)
    a_list = np.asarray(args.a_list)
    
    threshold_list = np.zeros(np.size(TH_init))
    count_list = np.zeros(np.size(TH_init))
    PFA_list = np.zeros(np.size(TH_init))
    
    for i in range(np.size(TH_init)):
        threshold_list[i],count_list[i],PFA_list[i] = glrt_TH_optimizer(args, TH_init[i], args.alg, delta[i], args.dist, a_list[i], sim_num, args.N, args.K, args.target_PFA, PFA_max, PFA_min)
    
    data = {'thresholds':threshold_list,'counts':count_list,'PFAs':PFA_list}
    from scipy.io import savemat
    fbase = generate_fname(args, args_str)
    fname_out = "%s_results.mat"%fbase
    savemat(fname_out,data)
    

# alg = 'glrt'
# dist = 'gaussian'
# # target_PFA = 1e-4
# target_PFA = 1e-2
# sim_num = int(np.round(100/target_PFA))
# N = 16
# K = 2*N
# PFA_margin = 0.1
# PFA_max = target_PFA + PFA_margin*target_PFA
# PFA_min = target_PFA - PFA_margin*target_PFA
# a = 1.5
# b = 1

# TH_init = np.asarray([.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5,.5])
# delta = np.asarray([.05,.05,.05,.05,.05,.05,.05,.05,.05,.05,.05,.05,.05,.05,.05,.05])
# a_list = np.asarray([3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10,10.5])
# TH_init = np.asarray(0.2)
# delta = np.asarray(0.05)
# a_list = np.asarray(3)


# threshold_list = np.zeros(np.size(TH_init))
# count_list = np.zeros(np.size(TH_init))
# PFA_list = np.zeros(np.size(TH_init))

def glrt_PFA(args, alg, dist, a, sim_num, N, K, P_fa, threshhold):
    i = np.linspace(1,N,N)
    # PRI = args.PRI
    # f_d = 2e7
    # rho = 0.9
    # b = 1
    p = np.exp(-1j*2*np.pi*i*args.f_d*args.PRI)
    FA = 0
    
    d = cs.Clutter_Sim(dist,sim_num,K,N,args.rho,a,args.b)
    
    for ii in range(sim_num):
        cov_est = np.zeros((K,N,N))
        cov_est = cov_est.astype('complex128')
        
        for jj in range(K):
            cov_est[jj,:,:] = (np.expand_dims(d[ii+jj+1,:],0))*(np.expand_dims(d[ii+jj+1,:],0).conj().T)
    
        S = np.sum(cov_est,axis=0)
        M_est = S/K
        z = d[ii,:]
        
        det = cd.cogDetector(alg,z,p,S,threshhold)
        FA = FA + det
    
    return FA/sim_num
        

def glrt_TH_optimizer(args, TH_init, alg, delta, dist, a, sim_num, N, K, target_PFA, PFA_max, PFA_min):
    threshold = TH_init
    count=1
    flip=0
    # max_count=40
    PFA_history = np.zeros((args.max_count+1,1))
    PFA_history[0,0] = 1
    PFA = glrt_PFA(args, alg, dist, a, sim_num, N, K, PFA_max, threshold)
    
    while(PFA>PFA_max or PFA<PFA_min):
        if PFA>PFA_max:
            threshold = threshold+delta
            PFA = glrt_PFA(args, alg, dist, a, sim_num, N, K, PFA_max, threshold)
            PFA_history[count+1,0] = abs(PFA-target_PFA)
            if (PFA<=PFA_min):
                flip = 1
        else:
            threshold = threshold-delta
            PFA = glrt_PFA(args, alg, dist, a, sim_num, N, K, PFA_max, threshold)
            PFA_history[count+1,0] = abs(PFA-target_PFA)
            if (PFA<=PFA_min):
                flip = 1
        if flip==1:
            delta=delta/2
        count=count+1
        if count > args.max_count-1:
            break
    return threshold,count,PFA


# threshold,count,PFA = glrt_TH_optimizer(TH_init, alg, delta, dist, a, sim_num, N, K, target_PFA, PFA_max, PFA_min)    
# PFA = glrt_PFA(alg, dist, a, sim_num, N, K, target_PFA, threshhold=0.65)
if __name__ == "__main__":
    # Parse and check incoming arguments
    parser = create_parser()
    args = parser.parse_args()
    # check_args(args)
    
    execute_exp(args)