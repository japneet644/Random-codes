import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import data_loader
import pickle, pprint
import matplotlib.pyplot as plt
import PIL.Image as pil
import random
import mnist_loader

learning_rate=1e-4
batch_size=20
n_z=20
x = tf.placeholder(name='x', dtype=tf.float32, shape=[None, 784])

fully_connected1 = tf.contrib.layers.fully_connected(inputs=x, num_outputs=256, activation_fn=tf.nn.relu,scope="Fully_Conn1")
fully_connected2 = tf.contrib.layers.fully_connected(inputs=fully_connected1, num_outputs=128, activation_fn=tf.nn.relu,scope="Fully_Conn2")
z_mu             = tf.contrib.layers.fully_connected(inputs=fully_connected2, num_outputs=n_z, activation_fn=tf.nn.relu,scope="Fully_Conn2_mu")
z_log_sigma_sq   = tf.contrib.layers.fully_connected(inputs=fully_connected2, num_outputs=n_z, activation_fn=tf.nn.relu,scope="Fully_Conn2_sig")

eps = tf.random_normal(shape=tf.shape(z_log_sigma_sq),mean=0, stddev=1, dtype=tf.float32)
z = z_mu + tf.sqrt(tf.exp(z_log_sigma_sq)) * eps

#decoder
fully_connected_decoder1 = tf.contrib.layers.fully_connected(inputs=z, num_outputs=128, activation_fn=tf.nn.relu,scope="Fully_Conn1_decoder")
fully_connected_decoder2 = tf.contrib.layers.fully_connected(inputs=fully_connected_decoder1, num_outputs=256, activation_fn=tf.nn.relu,scope="Fully_Conn2_decoder")
# x_hat = tf.contrib.layers.fully_connected(inputs=fully_connected_decoder1, num_outputs=64, activation_fn=tf.sigmoid,scope="Fully_Conn2_decoder")
x_mu = tf.contrib.layers.fully_connected(inputs=fully_connected_decoder2, num_outputs=784, activation_fn=tf.nn.relu,scope="Fully_Conn2_decoder_mu")
x_log_sigma_sq = tf.contrib.layers.fully_connected(inputs=fully_connected_decoder2, num_outputs=784, activation_fn=None,scope="Fully_Conn2_decoder_std")
# x_hat = tf.contrib.layers.fully_connected(inputs=fully_connected_decoder1, num_outputs=784, activation_fn=tf.sigmoid,scope="Fully_Conn2_decoder")

# Reconstruction Loss

epsilon = 1e-10
recon_loss = 0.5*tf.reduce_sum(((x-x_mu)**2) ,axis=1)#-1*tf.reduce_sum(    x * tf.log(epsilon+x_hat) +(1-x) * tf.log(epsilon+1-x_hat),axis=1)
recon_loss = tf.reduce_mean(recon_loss)+0.5*1.837

# Latent loss

KL_loss = -0.5 * tf.reduce_sum(    1 + z_log_sigma_sq - tf.square(z_mu) -tf.exp(z_log_sigma_sq), axis=1)
KL_loss = tf.reduce_mean(KL_loss)

total_loss = recon_loss + KL_loss
train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(total_loss)

losses = {
    'recon_loss': recon_loss,
    'total_loss': total_loss,
    'KL_loss': KL_loss,
}

saver = tf.train.Saver()
with tf.Session() as sess:
    # saver.restore(sess,'./VAE_mnist.ckpt')
    training_data = mnist_loader.load_data_wrapper()
    random.shuffle(training_data)
    a = tf.placeholder(tf.float32,[50000, 784])
    dataset = tf.data.Dataset.from_tensor_slices(a)
    dataset = dataset.prefetch(buffer_size=1000)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_initializable_iterator()
    next = iterator.get_next()
    sess.run(tf.global_variables_initializer())
    print("session initialized 1")
    sess.run(iterator.initializer, feed_dict = {a:training_data})

    print("iterator initialized ")
    for i in range(000):
            # print(i)
        if i>0 and i % (50000 // batch_size) == 0:
            sess.run(iterator.initializer, feed_dict = {a:training_data})
        b = sess.run(next)
        _, Losses = sess.run([train_op, losses],feed_dict={x: b })
        if i%1000==0:
            print(Losses)
    #

    save_path = saver.save(sess, "./VAE_mnist.ckpt")
    print("Model saved in path: %s" % save_path)

    zsample =  np.random.random([10,n_z])
    Gsample  = sess.run(fully_connected_decoder1, feed_dict={z: zsample})
    Gsample2  = sess.run(fully_connected_decoder2, feed_dict={fully_connected_decoder1: Gsample})
    Gsamplemu = sess.run(x_mu, feed_dict={fully_connected_decoder2: Gsample2})
    Gsamplesig  = sess.run(x_log_sigma_sq,feed_dict={fully_connected_decoder2:Gsample2})
    gsample = sess.run(tf.random_normal(shape = tf.shape(Gsamplemu) ,mean = Gsamplemu,stddev = 0.01, dtype = tf.float32 ))#tf.sqrt(tf.exp(Gsamplesig))
    print(gsample)
    # print(training_data[0].reshape(28,28))
    img = pil.fromarray(np.uint8(training_data[0].reshape(28,28) * 255) , 'L')
    img.show()
    for i in range(10):
        img_gen = pil.fromarray(np.uint8(gsample[i].reshape(28,28) * 255) , 'L')
        img_gen.show()
