# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 09:15:43 2023

@author: Alex
"""


from Threshold_Calculation import *


# Run a halving algorithm Parser call
parser = create_parser()
args = parser.parse_args(['@Test.txt'])
print(args.TH_init)
execute_exp(args)

# weights= [[1,1,1,1,1,1]]
# label = ['TR1_repeat_decay_rate_0_8']
# # label = ['TR1_series_decay_rate_0_8']
# # label = ['TR1_repeat']
# for i,w in enumerate(weights):
#     # args = parser.parse_args(['@correction_TR1_series.txt', '--label', label[i], '--obj_weight', '%s'%(w[0]), '%s'%(w[1]), '%s'%(w[2]),'%s'%(w[3]) ,'%s'%(w[4]), '%s'%(w[5])])
#     args = parser.parse_args(['@correction_TR1_repeat.txt', '--label', label[i], '--obj_weight', '%s'%(w[0]), '%s'%(w[1]), '%s'%(w[2]),'%s'%(w[3]) ,'%s'%(w[4]), '%s'%(w[5])])
#     #execute_exp(args)
#     display_Metrics(args, save=True)