# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 10:22:32 2023

@author: Alex
"""

import numpy as np
import scipy.io
import scipy.stats as ss
import cmath



def Clutter_Sim(type='gaussian',J=1000,K=32,N=16,rho=0.9,a=1,b=1):
    mu = np.zeros((N))
    M = np.zeros((N,N))
    x_mv = np.zeros((J+K,N))
    x_mv = x_mv.astype('complex128')
    
    
    for i in range(N):
        for j in range(N):
            M[i,j] = rho**(np.absolute(i-j))
    
    for k in range(J+K):
        x_mv_re = ss.multivariate_normal.rvs(mu,M)
        x_mv_co = ss.multivariate_normal.rvs(mu,M)
        x_mv[k] = np.asarray([complex(x_mv_re[i],x_mv_co[i]) for i in range(N)])
    
    if type == 'gaussian':
        tau = 1;
    elif type == 'K':
        tau = np.tile(ss.gamma.rvs(a,0,b,size=J+K),(N,1)).T
    elif type == 'pareto1':
        tau = np.tile(1/ss.gamma.rvs(a,0,b,size=J+K),(N,1)).T
    x = np.multiply(np.sqrt(tau),x_mv)
    return x

# temp = Clutter_Sim('K')