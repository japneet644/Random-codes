import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from get_parameters import *
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import to_categorical

###################################
#######Lattice Parameters##########
###################################
l=32
nsamples=1000
lattice_shape=(l,l) #This is small l, not 1
pkl_file=open(str(lattice_shape)+'lattices.pkl','rb') 
allTlattices=pickle.load(pkl_file)
pkl_file.close()
index_set=range(0,32,1)
T_vals=np.linspace(0.01,2,32)
assumed_crit_temp=1 #to get a graph, different assumed_crit_temp should be used
S=[]
T=[]
for index in index_set:
        temp=T_vals[index]
        lattices=allTlattices[index][-nsamples:]
        for lattice in lattices:
                S.append([np.array(lattice)])
                T.append(temp)
S=np.array(S)
S=S.swapaxes(1,3)
print(S.shape)

###############################################
######Dividing data into train and test########
###############################################
train_image=S+0
test_image=train_image+0 #test image needs to be different from test image
y_train=to_categorical(np.greater(assumed_crit_temp,T)) 
y_test=y_train+0 #y_test needs to be different from y_train

########################
######Keras model######
########################
model = Sequential()
model.add(Conv2D(8, (3, 3), activation='relu', input_shape=(32, 32, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(2, activation='sigmoid'))
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)
model.fit(train_image, y_train, batch_size=32, epochs=10)
score = model.evaluate(test_image, y_test, batch_size=32)




