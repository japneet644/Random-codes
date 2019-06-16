import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import data_loader
import random
import math
import get_parameters

lattice_size  = 8
learning_rate = 1e-4
batch_size    = 20
n_z           = 1
x = tf.placeholder(name='x', dtype=tf.float32, shape=[None, 64])
y = tf.placeholder(name='y', dtype=tf.float32, shape=[None,1])
loss_type  = 'log_gaussian' #'Binary_crossentropy'
datapoints = 320000
n_temps    = 32
T_vals = np.linspace(0,2,n_temps)
global_step = tf.train.get_or_create_global_step()

def generator(z,reuse= None):
    with tf.variable_scope('Generator', reuse=reuse):
        net   = tf.contrib.layers.fully_connected(inputs=z   , num_outputs=10, activation_fn=tf.nn.relu,scope="inp")
        net1  = tf.contrib.layers.fully_connected(inputs=net , num_outputs=20, activation_fn=tf.nn.relu,scope="hid1")
        net2  = tf.contrib.layers.fully_connected(inputs=net1, num_outputs=32, activation_fn=tf.nn.relu,scope="hid2")
        net_mu_sin = tf.contrib.layers.fully_connected(inputs=net2, num_outputs=64, activation_fn=tf.tanh,scope="mu_sin")
        net_mu_cos = tf.contrib.layers.fully_connected(inputs=net2, num_outputs=64, activation_fn=tf.tanh,scope="mu_cos")
        net_sg= tf.contrib.layers.fully_connected(inputs=net2, num_outputs=64, activation_fn=tf.sigmoid,scope="sg") #-ve log(sigma_sq)
        epsilon= tf.random_normal(shape = tf.shape(net_mu),mean =0.0 ,stddev = 1.0, dtype = tf.float32 )
        sample = net_mu + epsilon*tf.sqrt(tf.exp(-net_sg))
        sample_wrapped = tf.math.atan2(sample_sin,sample_cos)/(2*np.pi)
    return sample_wrapped

def discriminator(g,reuse = None):
    with tf.variable_scope('Discriminator', reuse=reuse):
        net   = tf.contrib.layers.fully_connected(inputs=g   , num_outputs=32,  activation_fn=tf.nn.relu   ,scope="inp")
        net1  = tf.contrib.layers.fully_connected(inputs=net , num_outputs=20,  activation_fn=tf.nn.relu   ,scope="hid1")
        net2  = tf.contrib.layers.fully_connected(inputs=net1, num_outputs=10,  activation_fn=tf.nn.relu   ,scope="hid2")
        d_out = tf.contrib.layers.fully_connected(inputs=net2, num_outputs=1 ,  activation_fn=None,scope="prob")
    return d_out

with tf.name_scope('noise_sample'):
    z_rand = tf.placeholder(name='z_r', dtype=tf.float32, shape = [None, n_z])
    z = tf.concat([z_rand,y], axis=1)

G = generator(z)
d_real = discriminator(g=tf.concat([x,y],axis=1))
d_fake = discriminator(g=tf.concat([G,y],axis=1),reuse = True)

def loss_func(logits_in,labels_in):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_in,labels=labels_in))

with tf.variable_scope('lossG'):
    train_g_loss = loss_func(d_fake,tf.ones_like(d_fake))

with tf.variable_scope('lossD'):
    D_real_loss=loss_func(d_real,tf.ones_like(d_real)*0.9) #Smoothing for generalization
    D_fake_loss=loss_func(d_fake,tf.zeros_like(d_real))
    train_d_loss=D_real_loss+D_fake_loss

# 4. Update weights
g_param = tf.trainable_variables(scope='Generator')
d_param = tf.trainable_variables(scope='Discriminator')

with tf.name_scope('optimizer'):
    d_optim = tf.train.AdamOptimizer(learning_rate= 1e-4 ).minimize(train_d_loss, var_list=d_param)
    g_optim = tf.train.AdamOptimizer(learning_rate= 1e-4 ).minimize(train_g_loss, var_list=g_param)
    saver   = tf.train.Saver()

with tf.Session() as sess:
    # saver.restore(sess,'./GANmodel.ckpt')
    writer = tf.summary.FileWriter('./graphs', sess.graph)
    training_data = data_loader.load_data_wrapper()
    tvals = np.repeat(np.linspace(0.1,2.0,32),10000)
    c = list(zip(training_data,tvals))
    random.shuffle(c)
    training_data, tvals = zip(*c)
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

    for i in range(30000):
        if i>0 and i % (datapoints // batch_size) == 0:
            sess.run(iterator.initializer, feed_dict = {m:training_data, n:np.array(tvals).reshape(datapoints,1) + 0.01*np.random.randn(datapoints,1) })
        g,h = sess.run(next)

        z_c = np.random.uniform(-1, 1, size=[batch_size, n_z])

        D_loss_curr,_ = sess.run([train_d_loss,d_optim], feed_dict={ x:g, z_rand:z_c,y:h })
        G_loss_curr,_ = sess.run([train_g_loss,g_optim], feed_dict={ z_rand:z_c,y:h })

        if i % 1000 == 0:
            print('Iter: {}'.format(i),'  D loss: {:.4}'. format(D_loss_curr),'  G_loss: {:.4}'.format(G_loss_curr))

    save_path = saver.save(sess, "./GANmodel.ckpt")
    print("Model saved in path: %s" % save_path)

    n = 100
    zsample = np.linspace(-1,1,n).reshape(n,1)

    for i in range(0,n_temps,2):
        t = np.repeat(T_vals[i],n).reshape(n,1)
        gsample = sess.run(G, feed_dict={z_rand:zsample,y:t})
        gsample = gsample.reshape(n,lattice_size,lattice_size)
        print(gsample[0],360*np.mean(gsample[0]),360*np.std(gsample[0]))
        Magnetization           = get_parameters.get_mean_magnetization(gsample)
        # Magnetization_direction = get_parameters.get_magnetization_direction(gsample)
        # energy                  = get_parameters.get_energy(gsample)
        plt.plot(Magnetization[0][0],label = T_vals[i]) #all values
    plt.xlabel('Latent Variable value', fontsize=12)
    plt.ylabel('Magnetization of a single sample generated by the network', fontsize=10)
    plt.legend()
    plt.show()

        # for d in range(n_samples[0]):
        #     plt.plot(Magnetization_direction)
        # plt.show()
        # plt.hist(energy,bins =100)
        # plt.legend()
        # plt.show()
        # n_maps = 2000 #no of mappings per temp

        # for u in range(0,20,2):
        #     sampledz = sess.run(z,feed_dict={x:training_data[10000*u+500:10000*u+500+n_maps],y:np.array(tvals[10000*u+500:10000*u+500+n_maps]).reshape(n_maps,1)-1})
        #     plt.hist(sampledz,bins=100,label = u)
        # plt.legend()
        # plt.show()

        # for d in range(0,n_samples[1]):
        #     plt.plot(zsample[d:zsample.shape[0]:n_samples[1],0],Magnetization[0][0][d:zsample.shape[0]:n_samples[1]],label = d)
        # plt.legend(loc='best')
        # plt.show()
    # checkpoint = tf.train.latest_checkpoint(self.model_dir)
    # if checkpoint:
    #     print('Load checkpoint {}...'.format(checkpoint))
    #     self.saver.restore(self.sess, checkpoint)
    #
    #         summary, global_step = self.sess.run([self.summary_op, self.global_step],
    #                                              feed_dict={self.x: x_batch, self.z_cat: z_cat,
    #                                                         self.z_cont: z_cont, self.z_rand: z_rand})
    #         if step % 100 == 0:
    #             print('Epoch[{}/{}] Step[{}/{}] g_loss:{:.4f}, d_loss:{:.4f}'.format(epoch, self.args.epoch, step,
    #                                                                                  steps_per_epoch, g_loss,
    #                                                                                  d_loss))

    # summary_writer.add_summary(summary, global_step)
    # self.save(global_step)
