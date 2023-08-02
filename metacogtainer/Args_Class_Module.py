# Used for calculating P_fa with power function
import numpy as np

# Implementation of the original argument parser as a Class File
class Args_Class:
    def __init__(self, 
    exp_index=None, 
    exp_type=None, 
    label='Test', 
    results_path='./results', 
    v=0, 
    algorithm='amf',
    N=16, 
    K=32, 
    P_fa=1/np.power(10,4), 
    sample_len=16, 
    PRI=1e-8, 
    f_d=2e7, 
    target=False, 
    max_test=100, 
    discriminator='./classifier/ordered_a_Dense200_50_drop_0_100_LR_0_000100_model', 
    data='./Datasets/clutter_final_G.mat' , 
    model_thresh4='./classifier/P_14_Dense1000_200_50_drop_0_100_LR_0_000100_model', 
    model_thresh5='./classifier/P_44_Dense1000_200_50_drop_0_100_LR_0_000100_model',
    model_thresh6='./classifier/P_84_Dense1000_200_50_drop_0_100_LR_0_000100_model',
    model_thresh1='./classifier/K_54_Dense1000_200_50_drop_0_100_LR_0_000100_model', 
    model_thresh2='./classifier/K_14_Dense1000_200_50_drop_0_100_LR_0_000100_model', 
    model_thresh3='./classifier/K_44_Dense1000_200_50_drop_0_100_LR_0_000100_model', 
    model_final='./classifier/G_GF4_Dense500_50_drop_0_100_LR_0_000100_model', 
    alg='glrt', 
    dist='gaussian',
    run_ideal=True,
    show_output=False):
        self.exp_index=exp_index 
        self.exp_type=exp_type 
        self.label=label 
        
        self.results_path=results_path 
        self.v=v
        
        self.algorithm=algorithm
        self.N=N 
        self.K=K 
        self.P_fa=P_fa 
        self.sample_len=sample_len        
        self.PRI=PRI
        self.f_d=f_d 
        self.target=target 
        self.max_test=max_test 
        
        self.discriminator=discriminator 
        self.data=data 
        
        self.model_thresh4=model_thresh4
        self.model_thresh5=model_thresh5
        self.model_thresh6=model_thresh6
        
        self.model_thresh1=model_thresh1
        self.model_thresh2=model_thresh2 
        self.model_thresh3=model_thresh3 
        
        self.model_final=model_final 
        
        self.alg=alg 
        self.dist=dist 
        
        self.run_ideal=run_ideal
        self.show_output=show_output