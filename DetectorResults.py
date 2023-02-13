# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 08:39:24 2023

@author: Timmy
"""

import scipy.io 
import fnmatch
import os
from natsort import os_sorted
import re
import numpy as np
from scipy.io import savemat

directory = "./results/"
data_name="*_sweep_*_results.mat"

files = fnmatch.filter(os.listdir(directory), data_name)
flist = data_name.rsplit('*',2)
#Structure of data:
# [data_runs, FA_CD, FA_gauss]
count = {}
for fi in files:          
        full_f = '%s%s'%(directory, fi)
        f_rang0 = fi.rsplit(flist[1],1)
        f_rang = f_rang0[1].rsplit(flist[2])
        f = scipy.io.loadmat(full_f)
        num_range =re.findall(r'\d+', f_rang[0])
        dataset = re.findall(r'^\w', f_rang[0])
        dictValue=f_rang0[0] + "_"+dataset[0]+"_0." +num_range[1]   
        dict_short = 'all' + "_"+dataset[0]+"_0." +num_range[1]
        if (f_rang0 [0] == 'amf'):
            if ("no_tgt" in fi):
                count[dict_short] =count.get(dict_short,np.zeros(12))+[np.squeeze(f['data_run']),np.squeeze(f['FA_amf']),0,0,0,0,0,0,0,0,0,0]
            else:
                count[dict_short] =count.get(dict_short,np.zeros(12))+[0,0,0,0,0,0,np.squeeze(f['data_run']),np.squeeze(f['FA_amf']),0,0,0,0]
        else:
            if ("no_tgt" in fi):
                count[dict_short] =count.get(dict_short,np.zeros(12))+[0,0,np.squeeze(f['data_run']),np.squeeze(f['FA_CD']),np.squeeze(f['FA_gauss']),np.squeeze(f['FA_KL']),0,0,0,0,0,0]
            else:
                count[dict_short] =count.get(dict_short,np.zeros(12))+[0,0,0,0,0,0,0,0,np.squeeze(f['data_run']),np.squeeze(f['FA_CD']),np.squeeze(f['FA_gauss']),np.squeeze(f['FA_KL'])]

keys= list(count.keys())
keys.sort()
pfa_lst = [0.0001,0.0002,0.0003,0.0004,0.0005,0.0006,0.0007,0.0008,0.0009,
       0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,
       0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1]
dist=['G','K','P']
count_sub={}
for p in pfa_lst:
    key_full = '%s_%s_%0.4f'%('all','full',p)
    key_G = '%s_%s_%0.4f'%('all',dist[0],p)
    key_K = '%s_%s_%0.4f'%('all',dist[1],p)
    key_P = '%s_%s_%0.4f'%('all',dist[2],p)
    count_sub[key_full] = count.get(key_G)+count.get(key_K)+count.get(key_P)
    count_sub[key_G] = count.get(key_G)
Algorithm=['CD']*28 + ['glrt']*28+['amf']*28 + ["K_L"]*28+['CD']*28 + ['glrt']*28+['amf']*28 + ["K_L"]*28
Sweep =['Full']*112 + ['G']*112
PFA_target = np.tile(pfa_lst,(1,8))
PFA_target = PFA_target.T
Exp_PFA = np.zeros((224,1))
Exp_Det = np.zeros((224,1))    
for i, p in enumerate(pfa_lst):
    Full_name = 'all_full_%0.4f'%(p)
    G_name = 'all_G_%0.4f'%(p) 
    full_data = count_sub.get(Full_name)
    G_data = count_sub.get(G_name) 
    Exp_PFA[i] = full_data[3]/full_data[2]
    Exp_PFA[i+28] = full_data[4]/full_data[2] 
    Exp_PFA[i+56] = full_data[1]/full_data[0]
    Exp_PFA[i+84] = full_data[5]/full_data[2]
    Exp_PFA[i+112] = G_data[3]/G_data[2]
    Exp_PFA[i+140] = G_data[4]/G_data[2] 
    Exp_PFA[i+168] = G_data[1]/G_data[0]
    Exp_PFA[i+196] = G_data[5]/G_data[2]
    
    Exp_Det[i] = full_data[9]/full_data[8]
    Exp_Det[i+28] = full_data[10]/full_data[8] 
    Exp_Det[i+56] = full_data[7]/full_data[6]
    Exp_Det[i+84] = full_data[11]/full_data[8]
    Exp_Det[i+112] = G_data[9]/G_data[8]
    Exp_Det[i+140] = G_data[10]/G_data[8] 
    Exp_Det[i+168] = G_data[7]/G_data[6]
    Exp_Det[i+196] = G_data[11]/G_data[8]  
fname_out = 'PFA_sweep_results.mat'
outputData = {'alg':Algorithm,'data':Sweep,'PFA_target':PFA_target,'exp_PFA':Exp_PFA,'exp_Det':Exp_Det}
savemat(fname_out,outputData)    
sub_files = fnmatch.filter(os.listdir(directory), data_name)
for i in keys:
    pfa= re.findall(r'\d\.\d+',i)[0]
    data=(np.rint(count[i])).astype(int)
    alg = i[0:3]
    dist = i[4]
    # for
    if alg == 'amf':
        exp_pfa_amf=data[1]/data[0]
        exp_det_amf =data[5]/data[4]
    else:
        exp_pfa_cd=data[1]/data[0]
        exp_pfa_glrt=data[2]/data[0]
        exp_pfa_kl=data[3]/data[0]
        exp_pfa_cd=data[5]/data[4]
        exp_pfa_glrt=data[6]/data[4]
        exp_pfa_kl=data[7]/data[4]
    
    print("Data:")
    print("Alg: %s, Dist: "%(i[0:3]) + i[4] + ", PFA: " + pfa)
    print("Number of test: %s, Number of system detections: %s"%(np.format_float_scientific(data[0]) ,data[1]))
    print("Target pfa was: %s, and calculated pfa is: %s"%(pfa,data[1]/data[0]))
    print("Performed pfa was %0.3f times the expected pfa"%(float(pfa)*(data[0]/data[1])))
    #print("Variation from expected detections: %s"%(int(np.abs(data[0]*float(pfa)-data[1] ))))
    print("------------------------------")
    # outputData = {'FA_%s'%(args.algorithm):results,'data_run':total_run}
    # savemat(fname_out,outputData)
    
    