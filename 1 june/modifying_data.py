import numpy as np
import pickle
import data_loader
import get_parameters
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from get_parameters import *

############################################
####Initialization of Parameters############
############################################
def modify(lattice):
    u = np.random.uniform(size = 1)
    A = lattice + u
    b= A>1
    A[b] = A[b]-1
    lattice = A
    return lattice

l=8
lattice_shape=(l,l)
nsamples=1000
index_set=range(0,32,1)
T_vals=np.linspace(0.01,2,32)
pkl_file=open('./DATA/8by8lattices.pkl','rb')
allTlattices = pickle.load(pkl_file)
pkl_file.close()
new_lattices =[]
for index in index_set:
    print(index)
    temp=T_vals[index]
    lat = []
    for lattice in allTlattices[index]:
        lat.append(modify(lattice))
    new_lattices.append(lat)
    print('Done :)')
output = open('8by8modifiedlattices.pkl', 'wb')
pickle.dump(new_lattices, output)
output.close()
