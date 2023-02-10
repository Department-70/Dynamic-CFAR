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

directory = "./results/"
data_name="sys_sweep_*of10_results.mat"

files = fnmatch.filter(os.listdir(directory), data_name)
flist = data_name.rsplit('*',2)
#Structure of data:
# [data_runs, FA_CD, FA_gauss]
count = {"G_0.001":np.zeros(3)}
for fi in files:
    if ("no_tgt" in fi):
        full_f = '%s%s'%(directory, fi)
        f_rang0 = fi.rsplit(flist[0],1)
        f_rang = f_rang0[1].rsplit(flist[1])
        f = scipy.io.loadmat(full_f)
        num_range =re.findall(r'\d+', f_rang[0])
        dataset = re.findall(r'^\w', f_rang[0])
        dictValue=dataset[0]+"_0." +num_range[1]
        count[dictValue] =count.get(dictValue,np.zeros(3))+[np.squeeze(f['data_run']),np.squeeze(f['FA_CD']),np.squeeze(f['FA_gauss'])]
        

for i in count.keys():
    pfa= re.findall(r'\d\.\d+',i)[0]
    data=(np.rint(count[i])).astype(int)
    print("Data:")
    print("Distribution: " +i[0] + ", PFA: " + pfa)
    print("Number of test: %s, Number of system detections: %s"%(np.format_float_scientific(data[0]) ,data[1]))
    print("Target pfa was: %s, and calculated pfa is: %s"%(pfa,data[1]/data[0]))
    print("Variation from expected detections: %s"%(np.abs(data[0]*float(pfa)-data[1] )))
    print("------------------------------")
    
    