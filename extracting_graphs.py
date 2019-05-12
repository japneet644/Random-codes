import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from get_parameters import *

############################################
####Initialization of Parameters############
############################################
l=32
lattice_shape=(l,l)
nsamples=1000
index_set=range(0,32,1)
T_vals=np.linspace(0.01,2,32)
S=[]
sp_heat=[]
mag=[]
mag_err=[]

######################################
#########Opening saved data###########
######################################
pkl_file=open(str(lattice_shape)+'lattices.pkl','rb')
allTlattices=pickle.load(pkl_file)
pkl_file.close()
#allTlattices contains 32 lists for each temperature
#Each list contains 10000 spin configurations
for index in index_set:
        temp=T_vals[index]
        lattices=allTlattices[index][-nsamples:]
        sp_heat.append(get_specific_heat(lattices,temp))
        [mag_mean,mag_std]=get_mean_magnetization(lattices)
        mag.append(mag_mean)
        mag_err.append(mag_std)

#################################
######Observing vortices#########
#################################
data=(get_vorticity_configuration(allTlattices[20][9999])) #first index indicates the temperature index, second index is a no between 1-10000
im = plt.imshow(data, interpolation='none')
plt.figure(figsize=(8,4))
values=range(-7,8)
colors = [ im.cmap(im.norm(value)) for value in values]
patches = [ mpatches.Patch(color=colors[i], label="Level {l}".format(l=values[i]) ) for i in range(len(values)) ]
plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
plt.grid(True)
plt.show()

###########################
######Specific Heat########
###########################
plt.plot(T_vals,sp_heat)
plt.xlabel('Temperature')
plt.ylabel('Specific Heat')
plt.show()

#################################
##########Magnetization##########
#################################
plt.errorbar(T_vals,mag,mag_err)
plt.xlabel('Temperature')
plt.ylabel('Magnetization')
plt.show()


        
