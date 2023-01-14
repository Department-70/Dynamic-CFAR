# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 14:43:39 2022

@author: Alex
"""

import numpy as np
import scipy.io


def cogDetector(alg,z,p,S,T):
    
    # test_p = scipy.io.loadmat("test_p.mat")
    # test_p = test_p.get('p')
    
    
    if alg == 'glrt':
        num = np.power(np.linalg.norm(np.dot(np.dot(np.conjugate(z),np.linalg.inv(S)),np.transpose(p))),2)
        den = (1+np.dot(np.dot(np.conjugate(z),np.linalg.inv(S)),np.transpose(z)))*(np.dot(np.dot(np.conjugate(p),np.linalg.inv(S)),np.transpose(p)))
        test_stat = np.linalg.norm(num/den)
        if test_stat > T:
            det = 1
        else:
            det = 0
        return det
        # return test_stat
    elif alg == 'amf':
        num = np.power(np.linalg.norm(np.dot(np.dot(np.conjugate(z),np.linalg.inv(S)),np.transpose(p))),2)
        den = np.linalg.norm(np.dot(np.dot(np.conjugate(p),np.linalg.inv(S)),np.transpose(p)))
        test_stat = np.linalg.norm(num/den)
        if test_stat > T:
            det = 1
        else:
            det = 0
        return det
        # return test_stat
    elif alg == 'ace':
        num = np.power(np.linalg.norm(np.dot(np.dot(np.conjugate(z),np.linalg.inv(S)),np.transpose(p))),2)
        den = np.linalg.norm(np.dot(np.dot(np.conjugate(p),np.linalg.inv(S)),np.transpose(p)))*np.linalg.norm(np.dot(np.dot(np.conjugate(z),np.linalg.inv(S)),np.transpose(z)))
        test_stat = np.linalg.norm(num/den)
        if test_stat > T:
            det = 1
        else:
            det = 0
        return det
        # return test_stat
    elif alg == 'abort':
        num_amf = np.power(np.linalg.norm(np.dot(np.dot(np.conjugate(z),np.linalg.inv(S)),np.transpose(p))),2)
        den_amf = np.linalg.norm(np.dot(np.dot(np.conjugate(p),np.linalg.inv(S)),np.transpose(p)))
        test_stat_amf = np.linalg.norm(num_amf/den_amf)
        num = 1+test_stat_amf
        den = 1+np.linalg.norm(np.dot(np.dot(np.conjugate(z),np.linalg.inv(S)),np.transpose(z)))-test_stat_amf
        test_stat = np.linalg.norm(num/den)
        if test_stat > T:
            det = 1
        else:
            det = 0
        return det
        # return test_stat
    elif alg == 'rao':
        S_rao = np.dot(np.conjugate(z),np.transpose(z))+S
        num = np.power(np.linalg.norm(np.dot(np.dot(np.conjugate(z),np.linalg.inv(S_rao)),np.transpose(p))),2)
        den = np.linalg.norm(np.dot(np.dot(np.conjugate(p),np.linalg.inv(S_rao)),np.transpose(p)))
        test_stat = np.linalg.norm(num/den)
        if test_stat > T:
            det = 1
        else:
            det = 0
        return det
        # return test_stat
    else:
        print('Unrecognized detector type.')

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
        

# test_p = scipy.io.loadmat("test_p.mat")
# test_p = test_p.get('p')
# test_s = scipy.io.loadmat("test_s.mat")
# test_s = test_s.get('S')
# test_z = scipy.io.loadmat("test_z.mat")
# test_z = test_z.get('z')
# test = cogDetector('rao',test_z,test_p,test_s,1.32)
# print(test)
# test = cogThreshold('ace',10**-4,32,16)
# print(test)