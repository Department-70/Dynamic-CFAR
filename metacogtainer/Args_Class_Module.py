# Parmeters and variables for all programs that could be changed here
# (Implementation of the original argument parser as a Class File)

# complex math functions
import numpy as np

# all the parmeters and the variables below belong to Args_Class
class Args_Class:

    ######*****************************************************########
    # set up the parameters and initializing the variables
    #####*****************************************************######## 
    def __init__(self, 
                 # experiment index: tells about combinations of parameters that will be overwritten
                 exp_index = None, 
                 #  experiment type: tells what experiment to run
                 exp_type = None, 
                 # name of the result file
                 label = 'Test', 
                 # path where to save the result file
                 results_path = './results', 
                 # how much output will be printed 
                 v = 0, 
                 # the radar algorithm
                 algorithm = 'amf',
                 # sample size
                 N = 16, 
                 # how many samples
                 K = 32, 
                 # probability of false alarm that we are trying to reach in our test 1/10^4
                 P_fa = 1/np.power(10,4), 
                 # sample size
                 sample_len = 16, 
                 # pulse repetition interval in sec
                 PRI = 1e-8, 
                 # frequency of the radar in Hz
                 f_d = 2e7, 
                 # if there is a target in the data or not
                 target = False, 
                 # the max amount of tests that we want to run
                 max_test = 50, 
                 # the path to the discriminator
                 discriminator='./classifier/ordered_a_Dense200_50_drop_0_100_LR_0_000100_model', 
                 # location of the data
                 data = './Datasets/clutter_final_G.mat' , 
                 #*****************************************************
                 # threshold selector file path
                 model_thresh4 = './classifier/P_14_Dense1000_200_50_drop_0_100_LR_0_000100_model', 
                 model_thresh5 = './classifier/P_44_Dense1000_200_50_drop_0_100_LR_0_000100_model',
                 model_thresh6 = './classifier/P_84_Dense1000_200_50_drop_0_100_LR_0_000100_model',
                 model_thresh1 = './classifier/K_54_Dense1000_200_50_drop_0_100_LR_0_000100_model', 
                 model_thresh2 = './classifier/K_14_Dense1000_200_50_drop_0_100_LR_0_000100_model', 
                 model_thresh3 = './classifier/K_44_Dense1000_200_50_drop_0_100_LR_0_000100_model', 
                 model_final = './classifier/G_GF4_Dense500_50_drop_0_100_LR_0_000100_model', 
                 ##*****************************************************
                 # the radar algorithm
                 alg ='glrt', 
                 # the testing distribution
                 dist = 'gaussian',
                 # if we want to run the ideal detector with gaussian distribution
                 run_ideal = True,
                 # if we want to show the  final results
                 show_output = False):
        
        #*****************************************************
        # saving all inputs in the Args_Class
        self.exp_index = exp_index 
        self.exp_type = exp_type 
        self.label = label 
        
        self.results_path = results_path 
        self.v = v
        
        self.algorithm = algorithm
        self.N = N 
        self.K = K 
        self.P_fa = P_fa 
        self.sample_len = sample_len        
        self.PRI = PRI
        self.f_d  =f_d 
        self.target = target 
        self.max_test = max_test 
        
        self.discriminator = discriminator 
        self.data = data 
        
        self.model_thresh4 = model_thresh4
        self.model_thresh5 = model_thresh5
        self.model_thresh6 = model_thresh6
        self.model_thresh1 = model_thresh1
        self.model_thresh2 = model_thresh2 
        self.model_thresh3 = model_thresh3 
        self.model_final = model_final 
        
        self.alg = alg 
        self.dist = dist 
        
        self.run_ideal = run_ideal
        self.show_output = show_output
        ##*****************************************************