# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 14:40:45 2023

@author: Timmy
"""

from System_Parser_SC import *
import sys
import tensorflow as tf 
if __name__ == "__main__":
    # Run a halving algorithm Parser call
    with tf.device('/CPU:0'):
        parser = create_parser()
        args = parser.parse_args(['@%s'%(sys.argv[1])])
        print(args)
        execute_exp(args)