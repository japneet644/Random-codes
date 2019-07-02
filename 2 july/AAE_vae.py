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
batch_size    = 100
n_z           = 1
x = tf.placeholder(name='x', dtype=tf.float32, shape=[None, 64])
y = tf.placeholder(name='y', dtype=tf.float32, shape=[None,1])
z_prior = tf.placeholder(name='y', dtype=tf.float32, shape=[1,batch_size])
loss_type  = 'log_gaussian' #'Binary_crossentropy'
datapoints = 320000
n_temps    = 32
T_vals = np.linspace(-1.0,0.9,n_temps)
Is_train   = True
with tf.variable_scope('Encoder'):
    fully_connected1 = tf.contrib.layers.fully_connected(inputs=x, num_outputs=128, activation_fn=tf.tanh,scope="Fully_Conn1")
    fully_connected2 = tf.contrib.layers.fully_connected(inputs=fully_connected1, num_outputs=64, activation_fn=tf.tanh,scope="Fully_Conn2")
    fully_connected3 = tf.contrib.layers.fully_connected(inputs=fully_connected2, num_outputs=20, activation_fn=tf.tanh,scope="Fully_Conn3")
    z_mu             = tf.contrib.layers.fully_connected(inputs=fully_connected3, num_outputs=n_z, activation_fn=None,scope="Fully_Conn2_mu")
    z_log_sigma_sq   = tf.contrib.layers.fully_connected(inputs=fully_connected3, num_outputs=n_z, activation_fn=None,scope="Fully_Conn2_sig")

    eps = tf.random_normal(shape=tf.shape(z_log_sigma_sq),mean=0, stddev=1.0, dtype=tf.float32)
    z = z_mu + tf.sqrt(tf.exp(z_log_sigma_sq)) * eps

epsilon = 1e-10
u=tf.concat([y,z],axis=1)
with tf.variable_scope('Decoder'):
    fully_connected_decoder1 = tf.contrib.layers.fully_connected(inputs=u, num_outputs=20, activation_fn=tf.tanh,scope="Fully_Conn1_decoder")
    fully_connected_decoder2 = tf.contrib.layers.fully_connected(inputs=fully_connected_decoder1,num_outputs=64, activation_fn=tf.tanh,scope="Fully_Conn2_decoder")
    fully_connected_decoder3 = tf.contrib.layers.fully_connected(inputs=fully_connected_decoder2,num_outputs=128,activation_fn=tf.tanh,scope="Fully_Conn3_decoder")

    if loss_type == 'Binary_crossentropy':
        x_hat                = tf.contrib.layers.fully_connected(inputs=fully_connected_decoder3, num_outputs=64, activation_fn=tf.sigmoid,scope="Fully_Conn2_decoder_out")
        recon_loss           =-1*tf.reduce_sum(    x*tf.log(epsilon+x_hat) +(1-x)*tf.log(epsilon+1-x_hat),axis=1)#
    elif loss_type == 'log_gaussian':
        x_mu                 = tf.contrib.layers.fully_connected(inputs=fully_connected_decoder3, num_outputs=64, activation_fn=tf.sigmoid,scope="Fully_Conn2_decoder_mu")
        x_log_sigma_sq       = tf.contrib.layers.fully_connected(inputs=fully_connected_decoder3, num_outputs=64, activation_fn=tf.tanh,scope="Fully_Conn2_decoder_std")
        x_hat =tf.random_normal(shape = tf.shape(x_mu) ,mean = x_mu, stddev =tf.sqrt(tf.exp(-8*x_log_sigma_sq)), dtype = tf.float32 )

with tf.variable_scope('vae_loss'):
    recon_loss = 0.5*tf.reduce_mean(tf.reduce_sum( ((x-x_mu)**2)/tf.exp(-8*x_log_sigma_sq)+1.837- 8*x_log_sigma_sq ,axis=1)) #1.837= ln(2*pi)

def discriminator(g,reuse = None):
    with tf.variable_scope('Discriminator', reuse=reuse):
        net   = tf.contrib.layers.fully_connected(inputs=g   , num_outputs=40,  activation_fn=tf.nn.relu   ,scope="inp")
        net1  = tf.contrib.layers.fully_connected(inputs=net , num_outputs=32,  activation_fn=tf.nn.relu   ,scope="hid1")
        net2  = tf.contrib.layers.fully_connected(inputs=net1, num_outputs=10,  activation_fn=tf.nn.relu   ,scope="hid2")
        d_out = tf.contrib.layers.fully_connected(inputs=net2, num_outputs=1 ,  activation_fn=None         ,scope="prob")
    return d_out

def loss_func(logits_in,labels_in):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_in,labels=labels_in))

def return_intersection(hist_1, hist_2):
    minima = np.minimum(hist_1, hist_2)
    intersection = np.true_divide(np.sum(minima), np.sum(hist_2))
    return intersection

# z_prior = tf.random_normal(shape = tf.shape(z) ,mean = 0, stddev =1, dtype = tf.float32 )
d_real = discriminator(g=z_prior)
z_cap = tf.reshape(tf.squeeze(z),[1,batch_size])
d_fake = discriminator(g=z_cap,reuse = True)

with tf.variable_scope('lossD'):
    D_real_loss=10*loss_func(d_real,tf.ones_like(d_real)*0.9) #Smoothing for generalization
    D_fake_loss=10*loss_func(d_fake,tf.zeros_like(d_real))
    train_d_loss=D_real_loss+D_fake_loss
with tf.variable_scope('JSloss'):
    JS_loss = 10*loss_func(d_fake,tf.ones_like(d_fake))

encoder_param = tf.trainable_variables(scope='Encoder')
decoder_param = tf.trainable_variables(scope='Decoder')
d_param = tf.trainable_variables(scope='Discriminator')
print(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))

with tf.variable_scope('optimizer'):
    vae_optim = tf.train.AdamOptimizer(learning_rate= 1e-4 ).minimize(recon_loss,var_list=[encoder_param,decoder_param])
    d_optim   = tf.train.AdamOptimizer(learning_rate= 1e-4 ).minimize(train_d_loss, var_list=d_param)
    enc_optim = tf.train.AdamOptimizer(learning_rate= 1e-4 ).minimize(JS_loss, var_list=encoder_param)
    saver     = tf.train.Saver()

with tf.Session() as sess:
    if Is_train == False:
        saver.restore(sess,'./GANmodel.ckpt')
    if Is_train == True:
        writer = tf.summary.FileWriter('./graphs', sess.graph)
        training_data = data_loader.load_data_wrapper()
        tvals = np.repeat(np.linspace(-1.0,0.9,32),10000)
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

        for i in range(20000):
            if i>0 and i % (datapoints // batch_size) == 0:
                sess.run(iterator.initializer, feed_dict = {m:training_data, n:np.array(tvals).reshape(datapoints,1) + 0.01*np.random.randn(datapoints,1) })
            g,h = sess.run(next)

            VAE_loss_curr,_ = sess.run([recon_loss,vae_optim], feed_dict={x:g,y:h})
            # for _ in range(3):
            z_p = np.random.uniform(-1,1,size=[1,batch_size ])
            D_loss_curr,_   = sess.run([train_d_loss,d_optim], feed_dict={x:g,z_prior:z_p})
            # z_p = np.random.normal(size=[batch_size, n_z])
            JS_loss_curr,_  = sess.run([train_d_loss,d_optim], feed_dict={x:g,z_prior:z_p})
            if i % 1000 == 0:
                print('Iter: {}'.format(i),'  D loss: {:.4}'. format(D_loss_curr),'  Rec_loss: {:.4}'.format(VAE_loss_curr))

        save_path = saver.save(sess, "./GANmodel.ckpt")
        print("Model saved in path: %s" % save_path)

    n_samples = (32,20)
    if n_z == 1:
        T_vals = np.linspace(-1,0.9,32)
        zsample =  np.mgrid[-1.0:0.9:0.059375, -2.0:6.0:0.4].reshape(2,-1).T#np.random.normal(size = [n_samples[0]*n_samples[1],n_z])
        print(zsample)
        if loss_type == 'Binary_crossentropy':
            Gsample    = sess.run(fully_connected_decoder1, feed_dict={z: zsample[:,1].reshape(zsample.shape[0],1), y:zsample[:,0].reshape(zsample.shape[0],1) + 0.01*np.random.randn(zsample.shape[0],1)})
            Gsample2   = sess.run(fully_connected_decoder2, feed_dict={fully_connected_decoder1: Gsample  })
            gsample    = sess.run(x_hat,                    feed_dict={fully_connected_decoder2: Gsample2 })
        elif loss_type == 'log_gaussian':
            gsample    = sess.run(x_hat,feed_dict={z:zsample[:,1].reshape(zsample.shape[0],1),y:zsample[:,0].reshape(zsample.shape[0],1) } )
    else:
        T_vals = np.linspace(-1,0.9,32)
        n_samples = (32,20)
        zsample =  np.random.normal(size = [n_samples[0]*n_samples[1],n_z])#np.mgrid[-1.0:0.9:0.059375, -2.0:2.0:0.2].reshape(2,-1).T
        print(zsample)
        if loss_type == 'Binary_crossentropy':
            Gsample    = sess.run(fully_connected_decoder1, feed_dict={z: zsample[:,1].reshape(zsample.shape[0],1), y:zsample[:,0].reshape(zsample.shape[0],1) + 0.01*np.random.randn(zsample.shape[0],1)})
            Gsample2   = sess.run(fully_connected_decoder2, feed_dict={fully_connected_decoder1: Gsample  })
            gsample    = sess.run(x_hat,                    feed_dict={fully_connected_decoder2: Gsample2 })
        elif loss_type == 'log_gaussian':
            gsample    = sess.run(x_hat,feed_dict={z:zsample,y:np.repeat(T_vals,n_samples[1]).reshape(n_samples[0]*n_samples[1],1) } )

    gsample = gsample.reshape(zsample.shape[0],lattice_size,lattice_size)

    mean_magnetization = []
    Magnetization           = get_parameters.get_mean_magnetization(gsample)
    Magnetization_direction = get_parameters.get_magnetization_direction(gsample)
    energy                  = get_parameters.get_energy(gsample)
    if n_z == 1:
        for d in range(0,n_samples[0],5):
            plt.plot(zsample[:n_samples[1],1],Magnetization[0][0][d*n_samples[1]:(d+1)*n_samples[1]],label = d) #all values
        plt.xlabel('Latent Variable value', fontsize=12)
        plt.ylabel('Magnetization of a single sample generated by the network', fontsize=10)
        plt.legend()
        plt.show()
        for d in range(n_samples[0]):
            plt.plot(zsample[:n_samples[1],1],Magnetization_direction[d*n_samples[1]:(d+1)*n_samples[1]])
        plt.show()
        plt.hist(energy,bins =100)
        plt.legend()
        plt.show()
        n_maps = 2000 #no of mappings per temp
        for d in range(0,n_samples[1]):
            plt.plot(zsample[d:zsample.shape[0]:n_samples[1],0],Magnetization[0][0][d:zsample.shape[0]:n_samples[1]],label = d)
        plt.legend(loc='best')
        plt.show()
        if Is_train == False:
            f = open('./DATA/8by8lattices.pkl', 'rb')
            if (f.read(2) == '\x1f\x8b'):
                f.seek(0)
                gzip.GzipFile(fileobj=f)
            else:
                f.seek(0)
            training_data = pickle.load(f, encoding="latin1")
            training_data = np.reshape(training_data,(320000, 64))
            for d in range(0,32,4):
                sampledz = sess.run(u,feed_dict={x:training_data[10000*d+500:10000*d+500+n_maps],y:np.repeat(T_vals[d],n_maps).reshape(n_maps,1)})
                plt.scatter(sampledz[:,0],sampledz[:,1],label = d)
            plt.legend()
            plt.show()
    else:
        for d in range(0,n_samples[0],5):
            plt.plot(Magnetization[0][0][d*n_samples[1]:(d+1)*n_samples[1]],label = d) #all values
        plt.xlabel('Latent Variable value', fontsize=12)
        plt.ylabel('Magnetization of a single sample generated by the network', fontsize=10)
        plt.legend()
        plt.show()
        for d in range(n_samples[0]):
            plt.plot(Magnetization_direction[d*n_samples[1]:(d+1)*n_samples[1]])
        plt.show()
        plt.hist(energy,bins =100)
        plt.legend()
        plt.show()
        n_maps = 2000 #no of mappings per temp
        for d in range(0,n_samples[1]):
            plt.plot(Magnetization[0][0][d:zsample.shape[0]:n_samples[1]],label = d)
        plt.legend(loc='best')
        plt.show()
        if Is_train == False:
            f = open('./DATA/8by8lattices.pkl', 'rb')
            if (f.read(2) == '\x1f\x8b'):
                f.seek(0)
                gzip.GzipFile(fileobj=f)
            else:
                f.seek(0)
            training_data = pickle.load(f, encoding="latin1")
            training_data = np.reshape(training_data,(320000, 64))
            for d in range(0,32,4):
                sampledz = sess.run(u,feed_dict={x:training_data[10000*d+500:10000*d+500+n_maps],y:np.repeat(T_vals[d],n_maps).reshape(n_maps,1)})
                plt.scatter(sampledz[:,0],sampledz[:,1],label = d)
            plt.legend()
            plt.show()
    # Mdist = []
    # Edist = []
    # mean_magnetization = []
    # var_magnetization = []
    # mean_magnetization_data = []
    # var_magnetization_data = []
    # mean_energy = []
    # var_energy = []
    # mean_energy_data = []
    # var_energy_data = []
    #
    # n = n_samples[1]
    # if Is_train == False:
    #     f = open('./DATA/8by8lattices.pkl', 'rb')
    #     if (f.read(2) == '\x1f\x8b'):
    #         f.seek(0)
    #         gzip.GzipFile(fileobj=f)
    #     else:
    #         f.seek(0)
    #     training_data = pickle.load(f, encoding="latin1")
    #     training_data = np.reshape(training_data,(320000, 64))
    # for i in range(0,n_samples[0]):
    #     Magnetization           = get_parameters.get_mean_magnetization(gsample[i*n_samples[1]:(i+1)*n_samples[1]])
    #     Magnetization_direction = get_parameters.get_magnetization_direction(gsample[i*n_samples[1]:(i+1)*n_samples[1]])
    #     energy                  = get_parameters.get_energy(gsample[i*n_samples[1]:(i+1)*n_samples[1]])
    #     print(i)
    #     if Is_train == False:
    #         fig1 = plt.figure(1)
    #         plt.plot(zsample[:n_samples[1],1],Magnetization[0][0],label = (T_vals[i]+1.1))
    #         fig2 = plt.figure(2)
    #         plt.plot(zsample[:n_samples[1],1],Magnetization_direction,label = (T_vals[i]+1.1))
    #     if Is_train == True:
    #         lattices = np.array(training_data[i*10000:i*10000+n]).reshape(n,lattice_size,lattice_size)
    #         energy_data = get_parameters.get_energy(lattices)
    #         thetas_data = get_parameters.get_magnetization_direction(lattices)
    #         [mag_data,mag_mean,mag_std] = get_parameters.get_mean_magnetization(lattices)
    #         plt.subplot(3,1,1)
    #         plt.hist(Magnetization[0][0][i*n_samples[1]:(i+1)*n_samples[1]],bins =20,color='b',range=[0, 1],alpha=0.5)
    #         plt.hist(mag_data           ,bins =20,color='g',range=[0, 1],alpha=0.5)
    #         plt.ylabel('Magnetization ')
    #
    #         mean_magnetization.append(Magnetization[1])
    #         var_magnetization.append(Magnetization[2])
    #         mean_magnetization_data.append(mag_mean)
    #         var_magnetization_data.append(mag_std)
    #
    #         mean_energy.append(np.mean(energy))
    #         var_energy.append(np.std(energy))
    #         mean_energy_data.append(np.mean(energy_data))
    #         var_energy_data.append(np.std(energy_data))
    #
    #         plt.subplot(3,1,2)
    #         plt.plot(Magnetization_direction,linestyle='dotted',color='b')
    #         plt.plot(thetas_data,            linestyle='dotted',color='g')
    #         plt.ylabel('Magnetization direction')
    #         plt.ylim((-360,0))
    #         # plt.title('Magnetization direction')
    #
    #         plt.subplot(3,1,3)
    #         plt.hist(energy     ,bins =300,color='b',range=[-130, 20],alpha =0.5)
    #         plt.hist(energy_data,bins =300,color='g',range=[-130, 20],alpha=0.5)
    #         plt.ylabel('Energy')
    #
    #         # plt.show()
    #         plt.savefig('./out/combined@ %f.png'%((T_vals[i]+1.1)), bbox_inches='tight')
    #         plt.close()
    #         Mhist_1,_ = np.histogram(Magnetization[0][0],bins =20,range=[0, 1])
    #         Mhist_2,_ = np.histogram(mag_data           ,bins =20,range=[0, 1])
    #         Mdist.append(return_intersection(Mhist_1,Mhist_2))
    #
    #         Ehist_1,_ = np.histogram(energy     ,bins =300,range=[-130, 20])
    #         Ehist_2,_ = np.histogram(energy_data,bins =300,range=[-130, 20])
    #         Edist.append(return_intersection(Ehist_1,Ehist_2))
    # if Is_train ==False:
    #     plt.xlabel('Latent Variable value', fontsize=12)
    #     plt.ylabel('Magnetization of a single sample generated by the network', fontsize=10)
    #     plt.legend()
    #     plt.show()
    # if Is_train == True:
    #     print("Magnetization Accuracy")
    #     print(Mdist)
    #     print(mean_magnetization)
    #     print(mean_magnetization_data)
    #     print(var_magnetization)
    #     print(var_magnetization_data)
    #     plt.errorbar(T_vals+1.1,mean_magnetization,var_magnetization,color='b',label='Samples')
    #     plt.errorbar(T_vals+1.1,mean_magnetization_data,var_magnetization_data,color = 'g',label='Data')
    #     plt.xlabel("Temperature")
    #     plt.ylabel('Magnetization')
    #     plt.title('AAE VAE')
    #     plt.legend()
    #     plt.savefig('../../Desktop/AAE_VAE-Magnetization.png', bbox_inches='tight')
    #     plt.show()
    #
    #     print("Energy Accuracy")
    #     print(Edist)
    #     print(mean_energy)
    #     print(mean_energy_data)
    #     print(var_energy)
    #     print(var_energy_data)
    #     plt.errorbar(T_vals+1.1,mean_energy,var_energy,color='b',label='Samples')
    #     plt.errorbar(T_vals+1.1,mean_energy_data,var_energy_data,color = 'g',label='Data')
    #     plt.xlabel("Temperature")
    #     plt.ylabel('Energy')
    #     plt.title('AAE_VAE')
    #     plt.legend()
    #     plt.savefig('../../Desktop/AAE_VAE-Energy.png', bbox_inches='tight')
    #     plt.show()
    #
    #     print("Specfic Heat")
    #     plt.plot(T_vals+1.1,(np.array(var_energy)**2)/((T_vals+1.1)**2),color='b',label='Samples')
    #     plt.plot(T_vals+1.1,(np.array(var_energy_data)**2)/((T_vals+1.1)**2),color='g',label='Data')
    #     plt.ylabel("Specific Heat")
    #     plt.xlabel('Temperature')
    #     plt.title('AAE_VAE')
    #     plt.legend()
    #     plt.savefig('../../Desktop/AAE_VAE-Specific Heat.png', bbox_inches='tight')
    #     plt.show()
    #
    #     print("Magnetic Susceptibility")
    #     plt.plot(T_vals+1.1,(np.array(var_magnetization)**2)/(T_vals+1.1),color='b',label='Samples')
    #     plt.plot(T_vals+1.1,(np.array(var_magnetization_data)**2)/(T_vals+1.1),color='g',label='Data')
    #     plt.ylabel("Magnetic Susceptibility")
    #     plt.xlabel('Temperature')
    #     plt.title('AAE_VAE')
    #     plt.legend()
    #     plt.savefig('../../Desktop/AAE_VAE-Magnetic_Susceptibility.png', bbox_inches='tight')
    #     plt.show()
