import numpy as np
import pickle
import matplotlib.pyplot as plt
from get_parameters import *

#########################################
#########Setting parameters##############
#########################################
l=32
lattice_shape=(l,l)
nsamples=1000 #maximum 10000
T_vals=np.linspace(0.01,2,32)
index_set=range(0,8,1)
index_set=range(0,32,1)

#################################
##########Getting Data###########
#################################
pkl_file=open(str(lattice_shape)+'lattices.pkl','rb')
allTlattices=pickle.load(pkl_file)
pkl_file.close()

#####################################
#########Making data matrix##########
#####################################
S=[]
S_vor=[]
for index in index_set:
        temp=T_vals[index]
        lattices=allTlattices[index][-nsamples:]
        for lattice in lattices:
                S.append(lattice.reshape(l*l,))
                S_vor.append((get_vorticity_configuration(lattice)).reshape(l*l,)) 
S=np.array(S)
print(S.shape)
S_vor=np.array(S_vor)
print(S_vor.shape)

###########################################
#######PCA on raw spin configurations######
###########################################
S1=np.cos(2*np.pi*S)
S2=np.sin(2*np.pi*S)
final_S=np.concatenate((S1,S2),axis=1)
mean_along_column=(np.mean(final_S,axis=0)).reshape(1,2*l*l)
X=final_S-mean_along_column
XTX=np.matmul(np.transpose(X),X)
[A,B]=np.linalg.eig(XTX)
#print(A[0],A[1])
w1=B[:,0].reshape(2*l*l,1)
w2=B[:,1].reshape(2*l*l,1)
pc1=np.matmul(final_S,w1)
pc2=np.matmul(final_S,w2)
pc=(pc1*pc1+pc2*pc2)**0.5

##############################################
#######Correlating magnetization with pc######
##############################################
magn=[]
for index in index_set:
        lattices=allTlattices[index][-nsamples:]
        for lattice in lattices:
                magn.append(get_magnetization(lattice))
plt.scatter(pc,magn)
plt.show()

##############################################################
########Plotting Principal Component with temperature#########
##############################################################
T=[]
for index in index_set:
        for i in range(1000):
                T.append(T_vals[index])
T=np.array(T)
plt.plot(T,pc)
plt.show()


############################################
#######PCA on vorticity configurations######
############################################
final_S=S_vor+0
#You can choose absolute vorticity or squared vorticity here.
#In this case, final_S and S_vor are same
mean_along_column=(np.mean(final_S,axis=0)).reshape(1,l*l)
X=final_S-mean_along_column
XTX=np.matmul(np.transpose(X),X)
[A,B]=np.linalg.eig(XTX)
w=B[:,0].reshape(l*l,1)
pc=np.matmul(final_S,w)
plt.plot(T,pc)
plt.show()




        
