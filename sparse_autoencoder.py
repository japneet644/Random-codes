import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import to_categorical
import pprint, pickle
import matplotlib.pyplot as plt
from get_parameters import *
from keras import regularizers

#################
##Training Data##
#################
all_matrices=[]
cos_matrix=[]
sin_matrix=[]
vortices=[] # list of winding numbers of central element
for r in range(10000):
        all_matrices.append(np.random.rand(3,3))
        vortices.append((get_vorticity_configuration(all_matrices[-1]))[1,1])
        cos_matrix.append(np.cos(2*np.pi*all_matrices[-1].reshape(9,)))
        sin_matrix.append(np.sin(2*np.pi*all_matrices[-1].reshape(9,)))
all_matrices=np.array(all_matrices)
vortices=np.array(vortices)
cos_matrix=np.array(cos_matrix)
sin_matrix=np.array(sin_matrix)

train_data=np.concatenate((cos_matrix,sin_matrix),axis=1)
train_ground_truth=train_data+0

#############
##Test Data##
#############
all_test_matrices=[]
cos_test_matrix=[]
sin_test_matrix=[]
vortices_test=[] # list of winding numbers of central element
for r in range(1000):
        all_test_matrices.append(np.random.rand(3,3))
        vortices_test.append((get_vorticity_configuration(all_test_matrices[-1]))[1,1])
        cos_test_matrix.append(np.cos(2*np.pi*all_test_matrices[-1].reshape(9,)))
        sin_test_matrix.append(np.sin(2*np.pi*all_test_matrices[-1].reshape(9,)))
all_test_matrices=np.array(all_test_matrices)
vortices_test=np.array(vortices_test)
cos_test_matrix=np.array(cos_test_matrix)
sin_test_matrix=np.array(sin_test_matrix)

test_data=np.concatenate((cos_test_matrix,sin_test_matrix),axis=1)
test_ground_truth=test_data+0

###############
##Keras model##
###############
epochs=50
model = Sequential()
model.add(Dense(30,activation='sigmoid',input_shape=(18,)))
model.add(Dense(20,activation='sigmoid'))
model.add(Dense(9,activation='sigmoid',activity_regularizer=regularizers.l1(1))) # bottleneck layer
model.add(Dense(20,activation='sigmoid'))
model.add(Dense(30,activation='sigmoid'))
model.add(Dense(18,activation='linear'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd,metrics=['mean_squared_error'])
model_train=model.fit(train_data, train_ground_truth, batch_size=32, epochs=epochs,validation_data=(test_data, test_ground_truth))

##############################################
##Evaluating the model (based on loss trend)##
##############################################
loss = model_train.history['loss']
val_loss=model_train.history['val_loss']
epochs = range(epochs)
plt.figure()
plt.plot(epochs[1:], loss[1:], 'bo', label='Training loss')
plt.plot(epochs[1:], val_loss[1:], 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

##############################################################################
##Evaluating the model (across different values of regularization parameter)##
##############################################################################
score = model.evaluate(test_data, test_ground_truth, batch_size=32)
print(score)

	

