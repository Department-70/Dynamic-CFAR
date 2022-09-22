# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 14:43:39 2022

@author: Alex
"""

import numpy as np
# import scipy.io


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
    else:
        print('Unrecognized detector type.')
    
# test_p = scipy.io.loadmat("test_p.mat")
# test_p = test_p.get('p')
# test_s = scipy.io.loadmat("test_s.mat")
# test_s = test_s.get('S')
# test_z = scipy.io.loadmat("test_z.mat")
# test_z = test_z.get('z')
# test = cogDetector('glrt',test_z,test_p,test_s,0.41)