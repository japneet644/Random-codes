import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.layers import Input,Conv2D,MaxPooling2D,UpSampling2D,Dense
from keras.models import Model
from keras.optimizers import RMSprop
##################################
#######Lattice parameters#########
##################################
l=32 # l x l is the dimension of the lattice (Note, it is L, not 1)
nsamples=1000 #no of samples per temperature
lattice_shape=(l,l)
pkl_file=open(str(lattice_shape)+'lattices.pkl','rb')
allTlattices=pickle.load(pkl_file)
pkl_file.close()
index_set=range(3,32,4)
index_set=range(0,32,1)
T_vals=np.linspace(0.01,2,32)

###############################################################
###########Division into training data and test data###########
###############################################################
S_train=[]
S_test=[]
for index in index_set:
        temp=T_vals[index]
        lattices=allTlattices[index][-nsamples:]
        for i in range(len(lattices)):
                if i<0.8*len(lattices):
                    S_train.append(lattices[i])
                else:
                    S_test.append(lattices[i])
S_train=np.array(S_train)
S_test=np.array(S_test)
S_train=S_train.reshape(-1, 32,32, 1)
S_test=S_test.reshape(-1, 32,32, 1)
train_X,valid_X,train_ground,valid_ground = train_test_split(S_train,
                                                             S_train, 
                                                             test_size=0.2, 
                                                             random_state=13)
###########################################
##########Keras Autoencoder Model##########
###########################################
batch_size = 128
epochs = 100
inChannel = 1
x, y = 32, 32
input_img = Input(shape = (x, y, inChannel))
def autoencoder(input_img):
    #encoder
    #input = 28 x 28 x 1 (wide and thin)
    print (1)    
    conv1 = Conv2D(16, (3, 3), activation='relu', padding='same',strides=1)(input_img) 
    print (2)
    pool1 = MaxPooling2D(pool_size=(2, 2),strides=2)(conv1)
    
    conv2 = Conv2D(8, (3, 3), activation='relu', padding='same',strides=1)(pool1) 
    pool2 = MaxPooling2D(pool_size=(2, 2),strides=2)(conv2) 
    conv3 = Conv2D(8, (3, 3), activation='relu', padding='same')(pool2)
    print(3)
    lin=Dense(2, input_shape=(4,4,8))(conv3)
    print (1) 
    #decoder
    conv4 = Conv2D(8, (3, 3), activation='relu', padding='same',strides=1)(lin) #7 x 7 x 128
    up1 = UpSampling2D((2,2))(conv4) # 14 x 14 x 128
    conv5 = Conv2D(8, (3, 3), activation='relu', padding='same',strides=1)(up1) # 14 x 14 x 64
    up2 = UpSampling2D((2,2))(conv5) # 28 x 28 x 64
    conv6 = Conv2D(16, (3, 3), activation='relu', padding='same',strides=1)(up2) # 14 x 14 x 64
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(conv6) # 28 x 28 x 1
    return decoded
autoencoder = Model(input_img, autoencoder(input_img))
autoencoder.compile(loss='binary_crossentropy', optimizer = RMSprop())
autoencoder.summary()
autoencoder_train = autoencoder.fit(train_X, train_ground, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_X, valid_ground))

loss = autoencoder_train.history['loss']
val_loss = autoencoder_train.history['val_loss']
epochs = range(epochs)
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()



        
