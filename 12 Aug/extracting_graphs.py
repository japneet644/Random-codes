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
l=16
lattice_shape=(l,l)
nsamples = 500
index_set=range(0,19,1)
B_vals=np.linspace(0.1,1.9,19)
energy = []
S=[]
sp_heat=[]
mag=[]
mag_err=[]
thetas=[]
energy_mean=[]
energy_err = []
######################################
#########Opening saved data###########
######################################
pkl_file=open('./DATA/16or16lattices.pkl','rb')
allTlattices= pickle.load(pkl_file)
pkl_file.close()
#allTlattices contains 32 lists for each temperature
#Each list contains 5000 spin configurations
for index in index_set:
    temp=B_vals[index]
    lattices=allTlattices[index][-nsamples:]
    f = np.array(get_energy(lattices))/256.0
    print(f.shape)
    energy.append(f)
    thetas.append(get_parameters.get_magnetization_direction(lattices))
    sp_heat.append(get_specific_heat(lattices,temp))
    [[ax],mag_mean,mag_std]=get_mean_magnetization(lattices)
    mag.append(mag_mean)
    mag_err.append(mag_std)
    energy_mean.append(np.mean(np.array(f)))
    energy_err.append(np.std(np.array(f)))
    # print(lattices[0])
    # print(np.mean(lattices[100]), np.std(lattices[100]))

#################################
######Observing vortices#########
#################################
# data=(get_vorticity_configuration(allTlattices[20][9999])) #first index indicates the temperature index, second index is a no between 1-10000
# im = plt.imshow(data, interpolation='none')
# plt.figure(figsize=(8,4))
# values=range(-7,8)
# colors = [ im.cmap(im.norm(value)) for value in values]
# patches = [ mpatches.Patch(color=colors[i], label="Level {l}".format(l=values[i]) ) for i in range(len(values)) ]
# plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
# plt.grid(True)
# plt.show()
#
# ###########################
# ######Specific Heat########
# ###########################
print(sp_heat)
# plt.plot(B_vals,sp_heat)
# plt.xlabel('Temperature')
# plt.ylabel('Specific Heat')
# plt.show()

plt.plot(B_vals,(np.array(mag_err)**2)*B_vals,color='b',label='Samples')
plt.show()
plt.errorbar(B_vals,energy_mean,energy_err)
plt.xlabel('Temperature')
plt.ylabel('Energy')
plt.show()

#################################
##########Magnetization##########
#################################
print(mag)
plt.errorbar(B_vals,mag,mag_err)
plt.xlabel('Temperature')
plt.ylabel('Magnetization')
plt.show()
bins=np.histogram(np.concatenate(energy), bins=100)[1]
for i in range(0,19):
    plt.hist(thetas[i],bins =100,label = '%f'%B_vals[i] )
    print(np.mean(thetas[i]),np.std(thetas[i]))
plt.ylabel('No of times')
plt.legend(loc = 'best')
plt.show()
for i in range(19):
    plt.hist(energy[i],bins,label = '%f'%B_vals[i])
plt.ylabel('No of times')
plt.legend(loc = 'best')
plt.show()
