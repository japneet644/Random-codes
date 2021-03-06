import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import data_loader
# import pickle, pprint
# import PIL.Image as pil
# from PIL import Image, ImageDraw, ImageFont
import random
import math
import get_parameters

lattice_size = 8
learning_rate=1e-4
batch_size=20
n_z=2
x = tf.placeholder(name='x', dtype=tf.float32, shape=[None, 64])
u_mu = tf.placeholder(name='u_mu', dtype=tf.float32, shape=[None, 64])
u_sg = tf.placeholder(name='u_sg', dtype=tf.float32, shape=[None, 64])
loss_type = 'log_gaussian' #'Binary_crossentropy'
datapoints = 320000
n_temps = 32
#
# fully_connected1 = tf.contrib.layers.fully_connected(inputs=x, num_outputs=32, activation_fn=tf.nn.relu,scope="Fully_Conn1")
# fully_connected2 = tf.contrib.layers.fully_connected(inputs=fully_connected1, num_outputs=10, activation_fn=tf.nn.relu,scope="Fully_Conn2")
# z_mu             = tf.contrib.layers.fully_connected(inputs=fully_connected2, num_outputs=n_z, activation_fn=None,scope="Fully_Conn2_mu")
# z_log_sigma_sq   = tf.contrib.layers.fully_connected(inputs=fully_connected2, num_outputs=n_z, activation_fn=None,scope="Fully_Conn2_sig")
#
# eps = tf.random_normal(shape=tf.shape(z_log_sigma_sq),mean=0, stddev=1, dtype=tf.float32)
# z = z_mu + tf.sqrt(tf.exp(z_log_sigma_sq)) * eps
# epsilon = 1e-10
# fully_connected_decoder1 = tf.contrib.layers.fully_connected(inputs=z, num_outputs=10, activation_fn=tf.nn.relu,scope="Fully_Conn1_decoder")
# fully_connected_decoder2 = tf.contrib.layers.fully_connected(inputs=fully_connected_decoder1, num_outputs=32, activation_fn=tf.nn.relu,scope="Fully_Conn2_decoder")
# if loss_type == 'Binary_crossentropy':
#     x_hat = tf.contrib.layers.fully_connected(inputs=fully_connected_decoder2, num_outputs=64, activation_fn=tf.sigmoid,scope="Fully_Conn2_decoder_out")
#     recon_loss = -1*tf.reduce_sum(    x * tf.log(epsilon+x_hat) +(1-x) * tf.log(epsilon+1-x_hat),axis=1)#
# elif loss_type == 'log_gaussian':
#     x_mu = tf.contrib.layers.fully_connected(inputs=fully_connected_decoder2, num_outputs=64, activation_fn=None,scope="Fully_Conn2_decoder_mu")
#     x_log_sigma_sq = tf.contrib.layers.fully_connected(inputs=fully_connected_decoder2, num_outputs=64, activation_fn=None,scope="Fully_Conn2_decoder_std")
#     recon_loss = 0.5*tf.reduce_sum(1.837+ ((x-x_mu)**2)/tf.exp(x_log_sigma_sq) + x_log_sigma_sq ,axis=1)
#     # x_hat =tf.random_normal(shape = tf.shape(x_mu) ,mean = x_mu, stddev =tf.sqrt(tf.exp(x_log_sigma_sq)), dtype = tf.float32 )
#
# # Reconstruction Loss
# recon_loss = tf.reduce_mean(recon_loss)
# # Latent loss
# KL_loss = -0.5 * tf.reduce_sum(    1 + z_log_sigma_sq - tf.square(z_mu) -tf.exp(z_log_sigma_sq), axis=1)
# KL_loss = tf.reduce_mean(KL_loss)
#
# total_loss = recon_loss + KL_loss
# train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(total_loss)
#
# losses = {
#     'recon_loss': recon_loss,
#     'total_loss': total_loss,
#     'KL_loss': KL_loss,
# }

tf.train.Saver.restore(sess,'./VAE_xy2.ckpt')
# saver =  tf.train.Saver()

temp  = tf.placeholder(name='temp', dtype = tf.float32)
fc1   = tf.contrib.layers.fully_connected(inputs=x, num_outputs=4, activation_fn=tf.nn.relu,scope="FC1")
fc2   = tf.contrib.layers.fully_connected(inputs=fc1, num_outputs=4, activation_fn=tf.nn.relu,scope="FC2")
z_hat_mu = tf.contrib.layers.fully_connected(inputs=fc2, num_outputs=n_z, activation_fn=tf.nn.relu,scope="output_mu")
z_hat_sg = tf.contrib.layers.fully_connected(inputs=fc2, num_outputs=n_z, activation_fn=tf.nn.relu,scope="output_mu")


loss_mu = -1*tf.reduce_mean(    u_mu*tf.log(epsilon+z_hat_mu) + (1-u_mu)*tf.log(epsilon+1-z_hat_mu))
loss_sg = -1*tf.reduce_mean(    u_sg*tf.log(epsilon+z_hat_sg) + (1-u_sg)*tf.log(epsilon+1-z_hat_sg))
total_loss = loss_mu + loss_sg

regression_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(total_loss)


with tf.Session() as sess:
    training_data = data_loader.load_data_wrapper()
    tvals = np.repeat(np.linspace(0.1,2.0,32),10000)
    c = list(zip(training_data,tvals))
    random.shuffle(c)
    training_data, tvals = zip(*c)
    a = tf.placeholder(tf.float32,[320000, 256])
    t = tf.placeholder(tf.float32,[320000,1])
    dataset = tf.data.Dataset.from_tensor_slices(a)
    dataset = dataset.prefetch(buffer_size=1000)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_initializable_iterator()
    next = iterator.get_next()

    sess.run(tf.global_variables_initializer())

    print("Session initialized :)")
    sess.run(iterator.initializer, feed_dict = {a:training_data, t:tvals})
    print("Iterator initialized :)")

    for i in range(10000):
            # print(i)
        if i>0 and i % (320000 // batch_size) == 0:
            sess.run(iterator.initializer, feed_dict = {a:training_data, t:tvals})
        m,n = sess.run(next)
        Z_mu, Z_sg = sess.run([z_mu,z_log_sigma_sq],feed_dict={x:m})
        _, Loss = sess.run([regression_op, total_loss],feed_dict={temp: n,u_mu : Z_mu, u_sg : Z_sg  })
        if i%1000==0:
            print(Loss)
            # Gsample  = sess.run(Ehidden1, feed_dict={x: training_data[0:100]})
            # Gsample2 = sess.run(Ehidden2, feed_dict={Ehidden1: Gsample})
            # gsample  = sess.run(z,    feed_dict={Ehidden2:Gsample2})
            # fig = plt.figure()
            # ax = fig.add_subplot(111)
            # for u,y in zip(gsample,tvals[:100]):
            #     ax.scatter(u[0],u[1],label='%f' %y)
            # plt.legend(loc='best')
            # plt.savefig("./out/ae_%d.png" % i)
            # plt.close()
            # print('fig svaed in ./out/ae_%d.png' % y)
