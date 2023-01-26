# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 14:40:45 2023

@author: Timmy
"""

from Threshold_Calculation import *
import sys
if __name__ == "__main__":
    # Run a halving algorithm Parser call
    parser = create_parser()
    args = parser.parse_args(['@%s'%(sys.argv[1])])
    print(args)
    execute_exp(args)