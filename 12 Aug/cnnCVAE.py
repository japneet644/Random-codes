import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import data_loader_special
import random
import math
import get_parameters
import pickle
import gzip
# nvidia-smi | grep 'python' | awk '{ print $3 }' | xargs -n1 kill -9

lattice_size  = 16
learning_rate = 1e-4
batch_size    = 100
n_z           = 2 # no of temp variables(generally 1)
n_zrand       = 1 # no of noise variables

# 4D tensors with dimension coding [batch, height, width, in_channels]
x = tf.placeholder(name='y', dtype=tf.float32, shape=[None, lattice_size+2,lattice_size+2,2])
y = tf.placeholder(name='x', dtype=tf.float32, shape=[None, lattice_size,lattice_size,1])
temp = tf.placeholder(name='temp', dtype=tf.float32, shape=[None, n_z])
loss_type  = 'log_gaussian' #'Binary_crossentropy'
datapoints = 190000
n_temps    = 19
B_vals = np.linspace(0.1,1.9,n_temps)
Is_train = True
k = False
l = True

def periodic_padding(x, padding=1):
    d1 = x.shape[1] # dimension 1: height
    d2 = x.shape[2] # dimension 2: width
    p = padding
    top_left = x[:, -p:, -p:] # top left
    top_center = x[:, -p:, :] # top center
    top_right = x[:, -p:, :p] # top right
    middle_left = x[:, :, -p:] # middle left
    middle_center = x # middle center
    middle_right = x[:, :, :p] # middle right
    bottom_left = x[:, :p, -p:] # bottom left
    bottom_center = x[:, :p, :] # bottom center
    bottom_right = x[:, :p, :p] # bottom right
    top = np.concatenate([top_left, top_center, top_right], axis=2)
    middle = np.concatenate([middle_left, middle_center, middle_right], axis=2)
    bottom = np.concatenate([bottom_left, bottom_center, bottom_right], axis=2)
    padded_x = np.concatenate([top, middle, bottom], axis=1)
    return padded_x

with tf.variable_scope('Encoder'):
    Conv1 = tf.contrib.layers.conv2d(inputs=x, num_outputs=16, kernel_size=[3, 3], stride=1, padding='VALID', activation_fn=tf.tanh)#(?,16,16,2) -> (?,16,16,16)
    pool1 = tf.layers.max_pooling2d(inputs=Conv1, pool_size=[2, 2], strides=2)# (?,16,16,10)->(?,8,8,10)
    Conv2 = tf.contrib.layers.conv2d(inputs=pool1, num_outputs=16, kernel_size=[3, 3], stride=1, padding='VALID', activation_fn=tf.tanh)# (?,4,4,16) -> (?,2,2,16)
    Conv3 = tf.contrib.layers.conv2d(inputs=Conv2, num_outputs=32, kernel_size=[3, 3], stride=1, padding='VALID', activation_fn=tf.tanh)# (?,4,4,16) -> (?,2,2,32)
    Conv4 = tf.contrib.layers.conv2d(inputs=Conv3, num_outputs=32, kernel_size=[3, 3], stride=1, padding='VALID', activation_fn=tf.tanh)# (?,4,4,32) -> (?,2,2,32)

    flat  = tf.reshape(Conv4, [-1, 2*2*32])
    net   = tf.contrib.layers.fully_connected(inputs=flat, num_outputs=40, activation_fn=tf.tanh,scope='FC')
    net1  = tf.contrib.layers.fully_connected(inputs=net , num_outputs=10, activation_fn=tf.tanh,scope='hid')
    Z_mu  = tf.contrib.layers.fully_connected(inputs=net1, num_outputs=n_zrand,activation_fn=None,scope="mu")
    Z_sg  = tf.contrib.layers.fully_connected(inputs=net1, num_outputs=n_zrand,activation_fn=None,scope="sg")
    eps = tf.random_normal(shape=tf.shape(Z_mu),mean=0, stddev=1.0, dtype=tf.float32)
    z   = Z_mu + tf.sqrt(tf.exp(Z_sg)) * eps

with tf.variable_scope('Decoder'):
    net   = tf.contrib.layers.fully_connected(inputs= tf.concat([temp,z],axis=1),num_outputs=40, activation_fn=tf.tanh,scope="inp")
    net1  = tf.contrib.layers.fully_connected(inputs = net                      ,num_outputs=128,activation_fn=tf.tanh,scope="hid1")
    net2  = tf.contrib.layers.fully_connected(inputs = net1                     ,num_outputs=256,activation_fn=tf.tanh,scope="hid2")

    Conv  = tf.reshape(net1, [-1,8,8,4])
    Conv1 = tf.contrib.layers.conv2d_transpose(inputs=Conv, num_outputs=10, kernel_size=[3,3], stride=1, padding='VALID', activation_fn=tf.tanh, scope='Conv1') #(?,1,2,2) ->(?,6,6,1)
    Conv2 = tf.contrib.layers.conv2d_transpose(inputs=Conv1, num_outputs=10, kernel_size=[3,3], stride=1, padding='VALID', activation_fn=tf.tanh, scope='Conv2') #(?,1,2,2) ->(?,6,6,1)
    Conv3 = tf.contrib.layers.conv2d_transpose(inputs=Conv2, num_outputs=10, kernel_size=[3,3], stride=1, padding='VALID', activation_fn=tf.tanh, scope='Conv3') #(?,1,2,2) ->(?,6,6,1)
    Convmu= tf.contrib.layers.conv2d_transpose(inputs=Conv3,num_outputs=1 , kernel_size=[3,3], stride=1, padding='VALID', activation_fn= tf.sigmoid, scope='Conv2mu') #(?,1,2,2) ->(?,6,6,1)
    Convsg= tf.contrib.layers.conv2d_transpose(inputs=Conv3,num_outputs=1 , kernel_size=[3,3], stride=1, padding='VALID', activation_fn=tf.nn.relu, scope='Conv2sg') #(?,1,2,2) ->(?,6,6,1)
    epsilon= tf.random_normal(shape = tf.shape(Convmu),mean =0.0 ,stddev = 1.0, dtype = tf.float32 )
    sample = Convmu + epsilon*tf.sqrt(tf.exp(-5*Convsg))#tf.random_normal(shape = tf.shape(net_mu),mean =net_mu ,stddev = tf.sqrt(tf.exp(-net_sg)), dtype = tf.float32 )#

with tf.variable_scope('loss'):
    # Reconstruction Loss
    recon_loss = 0.5*tf.reduce_sum( (1-tf.math.cos(y-Convmu))/tf.exp(-5*Convsg) + 1.837 -5*Convsg ,axis=1) #1.837= ln(2*pi)
    recon_loss = tf.reduce_mean(recon_loss)
    # Latent loss
    KL_loss = -0.5*tf.reduce_sum( 1 + Z_sg - tf.square(Z_mu) - tf.exp(Z_sg), axis=1)
    KL_loss = tf.reduce_mean(KL_loss)

total_loss = recon_loss + KL_loss
train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(total_loss)

losses = {
    'recon_loss': recon_loss,
    'total_loss': total_loss,
    'KL_loss':  KL_loss,
}

def return_intersection(hist_1, hist_2):# For evaluation : Calculates the % Overlap between two Histograms
    minima = np.minimum(hist_1, hist_2)
    intersection = np.true_divide(np.sum(minima), np.sum(hist_2))
    return intersection

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

saver = tf.train.Saver()
with tf.Session() as sess:
    if Is_train == False:
        saver.restore(sess,'./CVAE_xy2.ckpt')
        # writer = tf.summary.FileWriter('./graphs', sess.graph)
    if Is_train == True:
        training_data = data_loader_special.load_data_wrapper() #uploading the data
        bvals = np.repeat(np.linspace(0.1,1.9,19),10000)
        c = list(zip(training_data,bvals))
        random.shuffle(c) # pairing and shuffling the data and temeperature
        training_data, bvals = zip(*c)
        print(len(training_data),len(bvals))

        m = tf.placeholder(tf.float32,[datapoints, lattice_size+2, lattice_size+2,2])
        n = tf.placeholder(tf.float32,[datapoints, lattice_size,   lattice_size,1])
        b = tf.placeholder(tf.float32,[datapoints, n_z])
        # Uploading the data to prevent memeory issues prefetch and batching
        dataset = tf.data.Dataset.from_tensor_slices((m,n,b))
        dataset = dataset.prefetch(buffer_size=100)
        dataset = dataset.batch(batch_size)
        iterator = dataset.make_initializable_iterator()
        next = iterator.get_next()

        print("============< WARNING >===============")
        sess.run(tf.global_variables_initializer())
        print("==========< Model DELETED >===========")
        padded_training_data = periodic_padding(np.array(training_data)) #.todense()
        print(padded_training_data.shape)
        # pprint( np.repeat(np.array(bvals),(lattice_size+2)**2).reshape(datapoints,lattice_size+2,lattice_size+2,1)), axis = -1))
        sess.run(iterator.initializer,feed_dict = { m:np.concatenate((padded_training_data, np.repeat(np.array(bvals),(lattice_size+2)**2).reshape(datapoints,lattice_size+2,lattice_size+2,1)), axis = -1),n:training_data , b:np.repeat(np.array(bvals),n_z).reshape(datapoints,n_z) })
        print("Session initialized :)")
        print("Iterator initialized :)")

        for i in range(40000):
            if i>0 and i % (datapoints // batch_size) == 0:
                sess.run(iterator.initializer,feed_dict={ m:np.concatenate( [padded_training_data, np.repeat(np.array(bvals),(lattice_size+2)**2).reshape(datapoints,lattice_size+2,lattice_size+2,1)], axis = -1),n:training_data, b:np.repeat(np.array(bvals),n_z).reshape(datapoints,n_z) })
            g,h,j = sess.run(next)
            _, Losses = sess.run([train_op, losses],feed_dict={x: g,y:h ,temp:j })
            if i%1000==0:
                print(Losses)
        save_path = saver.save(sess, "./CVAE_xy2.ckpt")
        print("Model saved in path: %s" % save_path)
    #Evaluation (most commands execute if Is_train is FALSE)

    n = 30
    zsample = np.linspace(-1.5,1.5,n).reshape(n,1)#np.random.normal(size=[n,n_zrand])#
    Mdist = []
    Edist = []
    mean_magnetization = []
    var_magnetization = []
    mean_magnetization_data = []
    var_magnetization_data = []
    mean_energy = []
    var_energy = []
    mean_energy_data = []
    var_energy_data = []
    tvd_mag = []
    tvd_energy = []

    if Is_train == False:
        f = open('./DATA/16and16lattices.pkl', 'rb')
        if (f.read(2) == '\x1f\x8b'):
            f.seek(0)
            gzip.GzipFile(fileobj=f)
        else:
            f.seek(0)
        training_inputs = pickle.load(f, encoding="latin1")
        training_inputs = np.reshape(training_inputs,(n_temps*10000, lattice_size**2))
    for i in range(0,n_temps,3):
        t = np.repeat(B_vals[i],n_z*n).reshape(n,n_z)
        gsample = sess.run(sample, feed_dict={z:zsample,temp:t})
        gsample   = gsample.reshape(n,lattice_size,lattice_size)
        print(B_vals[i],  360*np.mean(gsample[0]),360*np.std(gsample[0]))
        Magnetization           = get_parameters.get_mean_magnetization(gsample)
        Magnetization_direction = get_parameters.get_magnetization_direction(gsample)
        energy                  = get_parameters.get_energy(gsample)
        if Is_train == k:
            fig1 = plt.figure(1)
            plt.plot(Magnetization[0][0],label = (B_vals[i]))
            fig2 = plt.figure(2)
            plt.plot(Magnetization_direction,label = (B_vals[i]))
        if Is_train == l:
            lattices = np.array(training_inputs[i*10000+8000:i*10000+n+8000]).reshape(n,lattice_size,lattice_size)
            energy_data = get_parameters.get_energy(lattices)
            thetas_data = get_parameters.get_magnetization_direction(lattices)
            [mag_data,mag_mean,mag_std]=get_parameters.get_mean_magnetization(lattices)

            mean_magnetization.append(Magnetization[1])
            var_magnetization.append(Magnetization[2])
            mean_magnetization_data.append(mag_mean)
            var_magnetization_data.append(mag_std)
            mean_energy.append(np.mean(energy))
            var_energy.append(np.std(energy))
            mean_energy_data.append(np.mean(energy_data))
            var_energy_data.append(np.std(energy_data))

            plt.subplot(3,1,1)
            tvd_mag.append( np.mean(np.abs(plt.hist(Magnetization[0][0],bins =20,range=[0, 1])[0]-plt.hist(mag_data,bins =20,range=[0, 1])[0] )) )
            plt.hist(Magnetization[0][0],bins =20,color='b',range=[0, 1],alpha=0.5)
            plt.hist(mag_data           ,bins =20,color='g',range=[0, 1],alpha=0.5)
            plt.ylabel('Magnetization')

            plt.subplot(3,1,2)
            plt.plot(Magnetization_direction,linestyle='dotted',color='b')
            plt.plot(thetas_data,            linestyle='dotted',color='g')
            plt.ylabel('Magnetization direction')
            plt.ylim((-360,0))

            plt.subplot(3,1,3)
            tvd_energy.append(np.mean(np.abs( plt.hist(energy,bins =300,range=[-130, 20])[0] - plt.hist(energy_data,bins =300,range=[-130, 20])[0]  )))
            plt.hist(energy     ,bins =300,color='b',range=[-130, 20],alpha =0.5)
            plt.hist(energy_data,bins =300,color='g',range=[-130, 20],alpha=0.5)
            plt.ylabel('Energy')

            # plt.show()
            plt.savefig('./out/combined@ %f.png'%((B_vals[i])), bbox_inches='tight')
            plt.close()
            Mhist_1,_ = np.histogram(Magnetization[0][0],bins =20,range=[0, 1])
            Mhist_2,_ = np.histogram(mag_data           ,bins =20,range=[0, 1])
            Mdist.append(return_intersection(Mhist_1,Mhist_2))

            Ehist_1,_ = np.histogram(energy     ,bins =300,range=[-130, 20])
            Ehist_2,_ = np.histogram(energy_data,bins =300,range=[-130, 20])
            Edist.append(return_intersection(Ehist_1,Ehist_2))
    if Is_train == k:
        plt.plot(zsample,Magnetization_direction,label = (B_vals[i]))
        plt.xlabel('Latent Variable value', fontsize=12)
        plt.ylabel('Magnetization of a single sample generated by the network', fontsize=10)
        plt.legend()
        plt.show()
    if Is_train == l:
        print("Magnetization Accuracy")
        print(Mdist)
        print(mean_magnetization)
        print(mean_magnetization_data)
        print(var_magnetization)
        print(var_magnetization_data)
        print(tvd_mag)
        plt.errorbar(B_vals,mean_magnetization,var_magnetization,color='b',label='VAE Samples')
        plt.errorbar(B_vals,mean_magnetization_data,var_magnetization_data,color = 'g',label='MC samples')
        plt.xlabel("Temperature")
        plt.ylabel('Magnetization')
        plt.legend()
        plt.savefig('../../Desktop/HGVAE-Magnetization.png', bbox_inches='tight')
        plt.title('HGVAE')
        plt.show()

        print("Energy Accuracy")
        print(Edist)
        print(mean_energy)
        print(mean_energy_data)
        print(var_energy)
        print(var_energy_data)
        print(tvd_energy)
        plt.errorbar(B_vals,mean_energy,var_energy,color='b',label='VAE Samples')
        plt.errorbar(B_vals,mean_energy_data,var_energy_data,color = 'g',label='MC samples')
        plt.xlabel("Temperature")
        plt.ylabel('Energy')
        plt.title('HGVAE')
        plt.legend()
        plt.savefig('../../Desktop/HGVAE-Energy.png', bbox_inches='tight')
        plt.show()

        print("Specfic Heat")
        plt.plot(B_vals,(np.array(var_energy)**2)/((B_vals)**2),color='b',label='Samples')
        plt.plot(B_vals,(np.array(var_energy_data)**2)/((B_vals)**2),color='g',label='Data')
        plt.ylabel("Specific Heat")
        plt.xlabel('Temperature')
        plt.title('HGVAE')
        plt.legend()
        plt.savefig('../../Desktop/HGVAE-Specific Heat.png', bbox_inches='tight')
        plt.show()

        print("Magnetic Susceptibility")
        plt.plot(B_vals,(np.array(var_magnetization)**2)/(B_vals),color='b',label='Samples')
        plt.plot(B_vals,(np.array(var_magnetization_data)**2)/(B_vals),color='g',label='Data')
        plt.ylabel("Magnetic Susceptibility")
        plt.xlabel('Temperature')
        plt.title('HGVAE')
        plt.legend()
        plt.savefig('../../Desktop/HGVAE-Magnetic_Susceptibility.png', bbox_inches='tight')
        plt.show()
