import numpy as np
from numpy import loadtxt
from numpy import savetxt
import Args_Class_Module
import scipy.io
import mat73

# For each probability distribution finds the max (most likely distribution) and saves to new file
args = Args_Class_Module.Args_Class()

if args.max_test is not None:
    max_disc_list = np.zeros(shape=(args.max_test, 1))
else:
    try: 
        data = scipy.io.loadmat(args.data)               #Use this for small mat data files.
    except NotImplementedError:
        data = mat73.loadmat(args.data)  
        
    data_ss = np.squeeze(data.get("data"))
    max_disc_list = np.zeros(shape=(len(data_ss), 1))

distribution_tensors = loadtxt('/app/docker_bind/distribution_tensors.csv', delimiter=',')
#print(distribution_tensors)

i = 0
for disc_vector in distribution_tensors:
     max_disc = np.argmax(disc_vector)
     max_disc_list[i] = max_disc
     i = i + 1


if args.show_output is True:
    print(max_disc_list)

    
savetxt('/app/docker_bind/max_disc_list.csv', max_disc_list, delimiter=',')
