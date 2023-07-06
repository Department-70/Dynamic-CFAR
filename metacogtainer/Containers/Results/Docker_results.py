from numpy import loadtxt
import Args_Class_Module

args = Args_Class_Module.Args_Class()

fa_cd = loadtxt('/app/docker_bind/FA_cd.csv', delimiter=',')
fa_glrt = loadtxt('/app/docker_bind/FA_glrt.csv', delimiter=',')

if args.run_ideal is True:
    fa_ideal = loadtxt('/app/docker_bind/FA_ideal.csv', delimiter=',')
    print("CD:",fa_cd,"/ GLRT:",fa_glrt,"/ Ideal:",fa_ideal)
else:
    print("CD:",fa_cd,"/ GLRT:",fa_glrt)

