import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import data_loader
import random
import math
import get_parameters
import pickle
import gzip

lattice_size  = 8
learning_rate = 1e-4
batch_size    = 20
n_z           = 1
x = tf.placeholder(name='x', dtype=tf.float32, shape=[None, 64])
y = tf.placeholder(name='y', dtype=tf.float32, shape=[None,1])
loss_type  = 'log_gaussian' #'Binary_crossentropy'
datapoints = 320000
n_temps    = 32
T_vals = np.linspace(-1.0,1.0,n_temps)
global_step = tf.train.get_or_create_global_step()
Is_train = False

def generator(z,reuse= None):
    with tf.variable_scope('Generator', reuse=reuse):
        net   = tf.contrib.layers.fully_connected(inputs=z   , num_outputs=20, activation_fn=tf.tanh,scope="inp")
        net1  = tf.contrib.layers.fully_connected(inputs=net , num_outputs=40, activation_fn=tf.tanh,scope="hid1")
        net2  = tf.contrib.layers.fully_connected(inputs=net1, num_outputs=80, activation_fn=tf.tanh,scope="hid2")
        net_mu= tf.contrib.layers.fully_connected(inputs=net2, num_outputs=64, activation_fn=tf.tanh,scope="mu_sin")
        net_sg= tf.contrib.layers.fully_connected(inputs=net2, num_outputs=1 , activation_fn=tf.nn.relu,scope="sgg") #-ve log(sigma_sq)
        epsilon= tf.random_normal(shape = tf.shape(net_mu),mean =0.0 ,stddev = 1.0, dtype = tf.float32 )
        sample = net_mu + epsilon*tf.sqrt(tf.exp(-10*net_sg))#tf.random_normal(shape = tf.shape(net_mu),mean =net_mu ,stddev = tf.sqrt(tf.exp(-net_sg)), dtype = tf.float32 )#
    return sample

def discriminator(g,reuse = None):
    with tf.variable_scope('Discriminator', reuse=reuse):
        net   = tf.contrib.layers.fully_connected(inputs=g   , num_outputs=80,  activation_fn=tf.tanh   ,scope="inp")
        net1  = tf.contrib.layers.fully_connected(inputs=net , num_outputs=32,  activation_fn=tf.tanh   ,scope="hid1")
        net2  = tf.contrib.layers.fully_connected(inputs=net1, num_outputs=20,  activation_fn=tf.tanh   ,scope="hid2")
        d_out = tf.contrib.layers.fully_connected(inputs=net2, num_outputs=1 ,  activation_fn=None      ,scope="prob")
    return d_out

with tf.name_scope('noise_sample'):
    z_rand = tf.placeholder(name='z_r', dtype=tf.float32, shape = [None, n_z])
    z = tf.concat([z_rand,y], axis=1)

G = generator(z)
d_real = discriminator(g=tf.concat([x,y],axis=1))
d_fake = discriminator(g=tf.concat([G,y],axis=1),reuse = True)

def loss_func(logits_in,labels_in):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_in,labels=labels_in))

def return_intersection(hist_1, hist_2):
    minima = np.minimum(hist_1, hist_2)
    intersection = np.true_divide(np.sum(minima), np.sum(hist_2))
    return intersection

with tf.variable_scope('lossG'):
    train_g_loss = loss_func(d_fake,tf.ones_like(d_fake))

with tf.variable_scope('lossD'):
    D_real_loss=loss_func(d_real,tf.ones_like(d_real)*0.9) #Smoothing for generalization
    D_fake_loss=loss_func(d_fake,tf.zeros_like(d_real))
    train_d_loss=D_real_loss+D_fake_loss

# 4. Update weights
g_param = tf.trainable_variables(scope='Generator')
d_param = tf.trainable_variables(scope='Discriminator')
print(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))

with tf.variable_scope('optimizer'):
    d_optim = tf.train.AdamOptimizer(learning_rate= 1e-4 ).minimize(train_d_loss, var_list=d_param)
    g_optim = tf.train.AdamOptimizer(learning_rate= 1e-4 ).minimize(train_g_loss, var_list=g_param)
    saver   = tf.train.Saver()

with tf.Session() as sess:
    if Is_train == False:
        saver.restore(sess,'./GANmodel.ckpt')
        # writer = tf.summary.FileWriter('./graphs', sess.graph)
    if Is_train == True:
        training_data = data_loader.load_data_wrapper()
        tvals = np.repeat(np.linspace(-1.0,1.0,32),10000)
        c = list(zip(training_data,tvals))
        random.shuffle(c)
        training_data, tvals = zip(*c)
        print(len(training_data),len(tvals))
        m = tf.placeholder(tf.float32,[datapoints, 64])
        n = tf.placeholder(tf.float32,[datapoints, 1])
        dataset = tf.data.Dataset.from_tensor_slices((m,n))
        dataset = dataset.prefetch(buffer_size=1000)
        dataset = dataset.batch(batch_size)
        iterator = dataset.make_initializable_iterator()
        next = iterator.get_next()
        print("============< WARNING >===============")
        sess.run(tf.global_variables_initializer())
        print("==========< Model DELETED >===========")
        sess.run(iterator.initializer,feed_dict = {m:training_data, n:np.array(tvals).reshape(datapoints,1) + 0.01*np.random.randn(datapoints,1)})
        print("Session initialized :)")
        print("Iterator initialized :)")

        for i in range(32000):
            if i>0 and i % (datapoints // batch_size) == 0:
                sess.run(iterator.initializer, feed_dict = {m:training_data, n:np.array(tvals).reshape(datapoints,1) + 0.01*np.random.randn(datapoints,1) })
            g,h = sess.run(next)

            for _ in range(3):
                z_c = np.random.normal(size=[batch_size, n_z])
                D_loss_curr,_ = sess.run([train_d_loss,d_optim], feed_dict={ x:g, z_rand:z_c,y:h })
            z_c = np.random.normal(size=[batch_size, n_z])
            G_loss_curr,_ = sess.run([train_g_loss,g_optim], feed_dict={ z_rand:z_c,y:h })

            if i % 1000 == 0:
                print('Iter: {}'.format(i),'  D loss: {:.4}'. format(D_loss_curr),'  G_loss: {:.4}'.format(G_loss_curr))
        save_path = saver.save(sess, "./GANmodel.ckpt")
        print("Model saved in path: %s" % save_path)

    n = 500
    zsample =np.linspace(0,1.5,n).reshape(n,1)# np.random.normal(size=[n,n_z])#
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

    if Is_train == False:
        f = open('./DATA/8by8lattices.pkl', 'rb')
        if (f.read(2) == '\x1f\x8b'):
            f.seek(0)
            gzip.GzipFile(fileobj=f)
        else:
            f.seek(0)
        training_inputs = pickle.load(f, encoding="latin1")
        training_inputs = np.reshape(training_inputs,(320000, 64))
    for i in range(0,32):
        t = np.repeat(T_vals[i],n).reshape(n,1)
        gsample = sess.run(generator(z,reuse=True), feed_dict={z_rand:zsample,y:t-0.5})
        gsample   = gsample.reshape(n,lattice_size,lattice_size)
        print(T_vals[i],  360*np.mean(gsample[10]),360*np.std(gsample[10]))
        Magnetization           = get_parameters.get_mean_magnetization(gsample)
        Magnetization_direction = get_parameters.get_magnetization_direction(gsample)
        energy                  = get_parameters.get_energy(gsample)
        if Is_train == True:
            fig1 = plt.figure(1)
            plt.plot(zsample,Magnetization[0][0],label = (T_vals[i]+1.0))
            fig2 = plt.figure(2)
            plt.plot(zsample,Magnetization_direction,label = (T_vals[i]+1.0))
        if Is_train == False:
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
            plt.hist(Magnetization[0][0],bins =20,color='b',range=[0, 1],alpha=0.5)
            plt.hist(mag_data           ,bins =20,color='g',range=[0, 1],alpha=0.5)
            plt.ylabel('Magnetization')

            plt.subplot(3,1,2)
            plt.plot(Magnetization_direction,linestyle='dotted',color='b')
            plt.plot(thetas_data,            linestyle='dotted',color='g')
            plt.ylabel('Magnetization direction')
            plt.ylim((-360,0))

            plt.subplot(3,1,3)
            plt.hist(energy     ,bins =300,color='b',range=[-130, 20],alpha =0.5)
            plt.hist(energy_data,bins =300,color='g',range=[-130, 20],alpha=0.5)
            plt.ylabel('Energy')

            # plt.show()
            plt.savefig('./out/combined@ %f.png'%((T_vals[i]+1.0)), bbox_inches='tight')
            plt.close()
            Mhist_1,_ = np.histogram(Magnetization[0][0],bins =20,range=[0, 1])
            Mhist_2,_ = np.histogram(mag_data           ,bins =20,range=[0, 1])
            Mdist.append(return_intersection(Mhist_1,Mhist_2))

            Ehist_1,_ = np.histogram(energy     ,bins =300,range=[-130, 20])
            Ehist_2,_ = np.histogram(energy_data,bins =300,range=[-130, 20])
            Edist.append(return_intersection(Ehist_1,Ehist_2))
    if Is_train == True:
        plt.plot(zsample,Magnetization_direction,label = (T_vals[i]+1.0))
        plt.xlabel('Latent Variable value', fontsize=12)
        plt.ylabel('Magnetization of a single sample generated by the network', fontsize=10)
        plt.legend()
        plt.show()
    if Is_train == False:
        print("Magnetization Accuracy")
        print(Mdist)
        print(mean_magnetization)
        print(mean_magnetization_data)
        print(var_magnetization)
        print(var_magnetization_data)
        plt.errorbar(T_vals+1.1,mean_magnetization,var_magnetization,color='b',label='Samples')
        plt.errorbar(T_vals+1.1,mean_magnetization_data,var_magnetization_data,color = 'g',label='Data')
        plt.xlabel("Temperature")
        plt.ylabel('Magnetization')
        plt.legend()
        plt.savefig('../../Desktop/CGAN-Magnetization.png', bbox_inches='tight')
        plt.title('C-GAN')
        plt.show()

        print("Energy Accuracy")
        print(Edist)
        print(mean_energy)
        print(mean_energy_data)
        print(var_energy)
        print(var_energy_data)
        plt.errorbar(T_vals+1.1,mean_energy,var_energy,color='b',label='Samples')
        plt.errorbar(T_vals+1.1,mean_energy_data,var_energy_data,color = 'g',label='Data')
        plt.xlabel("Temperature")
        plt.ylabel('Energy')
        plt.title('C-GAN')
        plt.legend()
        plt.savefig('../../Desktop/CGAN-Energy.png', bbox_inches='tight')
        plt.show()

        print("Specfic Heat")
        plt.plot(T_vals+1.1,(np.array(var_energy)**2)/((1.1+T_vals)**2),color='b',label='Samples')
        plt.plot(T_vals+1.1,(np.array(var_energy_data)**2)/((1.1+T_vals)**2),color='g',label='Data')
        plt.ylabel("Specific Heat")
        plt.xlabel('Temperature')
        plt.title('C-GAN')
        plt.legend()
        plt.savefig('../../Desktop/CGAN-Specific Heat.png', bbox_inches='tight')
        plt.show()

        print("Magnetic Susceptibility")
        plt.plot(T_vals+1.1,(np.array(var_magnetization)**2)/(T_vals+1.1),color='b',label='Samples')
        plt.plot(T_vals+1.1,(np.array(var_magnetization_data)**2)/(T_vals+1.1),color='g',label='Data')
        plt.ylabel("Magnetic Susceptibility")
        plt.xlabel('Temperature')
        plt.title('C-GAN')
        plt.legend()
        plt.savefig('../../Desktop/CGAN-Magnetic_Susceptibility.png', bbox_inches='tight')
        plt.show()
