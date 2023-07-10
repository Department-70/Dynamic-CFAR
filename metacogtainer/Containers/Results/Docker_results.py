import numpy as np
from numpy import loadtxt
from numpy import savetxt
import Args_Class_Module

args = Args_Class_Module.Args_Class()

fa_cd_model_0 = loadtxt('/app/docker_bind/FA_cd_model_0.csv', delimiter=',')
fa_cd_model_1 = loadtxt('/app/docker_bind/FA_cd_model_0.csv', delimiter=',')
fa_cd_model_2 = loadtxt('/app/docker_bind/FA_cd_model_0.csv', delimiter=',')
fa_cd_model_3 = loadtxt('/app/docker_bind/FA_cd_model_0.csv', delimiter=',')
fa_cd_model_4 = loadtxt('/app/docker_bind/FA_cd_model_0.csv', delimiter=',')
fa_cd_model_5 = loadtxt('/app/docker_bind/FA_cd_model_0.csv', delimiter=',')
fa_cd_model_6 = loadtxt('/app/docker_bind/FA_cd_model_0.csv', delimiter=',')

fa_cd = fa_cd_model_0 + fa_cd_model_1 + fa_cd_model_2 + fa_cd_model_3 + fa_cd_model_4 + fa_cd_model_5 + fa_cd_model_6

fa_glrt = loadtxt('/app/docker_bind/FA_glrt.csv', delimiter=',')

if args.run_ideal is True:
    fa_ideal = loadtxt('/app/docker_bind/FA_ideal.csv', delimiter=',')
    print("CD:",fa_cd,"/ GLRT:",fa_glrt,"/ Ideal:",fa_ideal)
    
    results = np.array([fa_cd, fa_glrt, fa_ideal])
    savetxt('/app/docker_bind/metacog_results.csv', results, delimiter=',')
else:
    print("CD:",fa_cd,"/ GLRT:",fa_glrt)
    
    results = np.array([fa_cd, fa_glrt, 0])
    savetxt('/app/docker_bind/metacog_results.csv', results, delimiter=',')

