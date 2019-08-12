import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import data_loader
import pickle, pprint
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import math

class VariantionalAutoencoder(object):
    def __init__(self, learning_rate=1e-2, batch_size=10, n_z=1):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_z = n_z
        tf.reset_default_graph()
        self.build()

        self.sess = tf.InteractiveSession()
        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())

    def build(self):
        self.x = tf.placeholder(name='x', dtype=tf.float32, shape=[None, 64])

        fully_connected1 = tf.contrib.layers.fully_connected(inputs=self.x, num_outputs=32, activation_fn=tf.nn.relu,scope="Fully_Conn1")
        self.z_mu        = tf.contrib.layers.fully_connected(inputs=fully_connected1, num_outputs=self.n_z, activation_fn=None,scope="Fully_Conn2_mu")
        self.z_log_sigma_sq = tf.contrib.layers.fully_connected(inputs=fully_connected1, num_outputs=self.n_z, activation_fn=None,scope="Fully_Conn2_sig")

        eps = tf.random_normal(
            shape=tf.shape(self.z_log_sigma_sq),
            mean=0, stddev=1, dtype=tf.float32)
        self.z = self.z_mu + tf.sqrt(tf.exp(self.z_log_sigma_sq)) * eps

        #decoder
        self.fully_connected_decoder1 = tf.contrib.layers.fully_connected(inputs=self.z, num_outputs=32, activation_fn=tf.nn.relu,scope="Fully_Conn1_decoder")
        self.x_hat = tf.contrib.layers.fully_connected(inputs=self.fully_connected_decoder1, num_outputs=64, activation_fn=tf.sigmoid,scope="Fully_Conn2_decoder")

        # Reconstruction Loss

        epsilon = 1e-10
        recon_loss = -tf.reduce_sum(
            self.x * tf.log(epsilon+self.x_hat) +
            (1-self.x) * tf.log(epsilon+1-self.x_hat),
            axis=1
        )
        self.recon_loss = tf.reduce_mean(recon_loss)

        # Latent loss

        KL_loss = -0.5 * tf.reduce_sum(
            1 + self.z_log_sigma_sq - tf.square(self.z_mu) -
            tf.exp(self.z_log_sigma_sq), axis=1)
        self.KL_loss = tf.reduce_mean(KL_loss)

        self.total_loss = self.recon_loss + self.KL_loss
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.total_loss)

        self.losses = {
            'recon_loss': self.recon_loss,
            'total_loss': self.total_loss,
            'KL_loss': self.KL_loss,
        }
        return

    # Execute the forward and the backward pass
    def run_single_step(self):
        data = data_loader.load_data_wrapper() # IMPORT FROM  data_loader
        dataset = tf.data.Dataset.from_tensor_slices(data).batch(self.batch_size)
        iterator = dataset.make_initializable_iterator()
        next = iterator.get_next()

        with tf.Session() as sess:
            # all_vars= tf.global_variables()
            # def get_var(name):
            #     for i in range(len(all_vars)):
            #         if all_vars[i].name.startswith(name):
            #             return all_vars[i]
            #     return None
            # fc1_var_w = get_var('Fully_Conn1_decoder/weights')
            # fc1_var_b = get_var('Fully_Conn1_decoder/biases')
            # fc2_var_w = get_var('Fully_Conn2_decoder/weights')
            # fc2_var_b = get_var('Fully_Conn2_decoder/biases')
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()

            sess.run(iterator.initializer)
            for i in range(320000):
                # print(val)
                if i % (320000 // self.batch_size) == 0:
                    sess.run(iterator.initializer)
                val = sess.run(next)
                _, losses = self.sess.run([self.train_op, self.losses],feed_dict={self.x: val })
                if i % 4000 == 0:
                    print(losses)
            # save_path = saver.save(sess, "./VAEmodel.ckpt")
            # print("Model saved in path: %s" % save_path)
            zsample =  np.linspace(0,1, 10).reshape(10,1)
            G_sample = sess.run(self.fully_connected_decoder1, feed_dict={self.z: zsample})
            gsample = sess.run(self.x_hat, feed_dict={self.fully_connected_decoder1:G_sample})
            # print(gsample.shape)
            for i in range(10):
                self.draw_grid(8, gsample[i].reshape(8,8) , zsample[i])
        return

def draw_grid(lattice_size=8,angle=[],beta=0):
    height = 640
    width = 640
    image = Image.new(mode='L', size=(height, width), color=255)

    # Draw some lines
    draw = ImageDraw.Draw(image)
    y_start = 0
    y_end = image.height
    step_size = int(image.width / lattice_size)

    for x in range(0, image.width, step_size):
        line = ((x, y_start), (x, y_end))
        draw.line(line, fill=128)

    x_start = 0
    x_end = image.width

    for y in range(0, image.height, step_size):
        line = ((x_start, y), (x_end, y))
        draw.line(line, fill=128)
    # del draw
    draw.text((10,10), "LV %s" %(str(beta)), fill=(100))
    a = step_size//2
    pi = 3.141592654
    for i in range(0, image.width, step_size):
        for j in range(0, image.height, step_size):
            draw.line(((i+a, j+a) , ( i + a + a*math.cos(2*pi*angle[j//step_size,i//step_size]), j + a - a*math.sin(2*pi*angle[j//step_size,i//step_size]))))
            draw.line(((i+a, j+a) , ( i + a - a*math.cos(2*pi*angle[j//step_size,i//step_size]), j + a + a*math.sin(2*pi*angle[j//step_size,i//step_size]))))
    image.show()
