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
# from sklearn.model_selection import train_test_split
# from keras.layers import Input,Conv2D,MaxPooling2D,UpSampling2D,Dense
# from keras.models import Model
# from keras.optimizers import RMSprop

class VAEcnn(object):
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

        self.x = tf.placeholder(name='x', dtype=tf.float32, shape=[None, 32,32,1])

        conv1 = tf.layers.conv2d(inputs=self.x,filters=8,kernel_size=[5, 5],padding = 'SAME',activation=tf.nn.relu)#(?,32,32,1) -> (?,32,32,8)
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)# (?,32,32,8)->(16,16,8)
        conv2 = tf.layers.conv2d(inputs=pool1,filters=16,kernel_size=[5, 5],padding = 'VALID',activation=tf.nn.relu)# (16,16,8) -> (12,12,16)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)# (?,12,12,16) -> (?,6,6,16)
        mu    = tf.layers.conv2d(inputs = pool2,filters=1,kernel_size=[5, 5],padding = 'VALID',activation=None)#(?,6,6,16) ->mu(?,2,2,1)
        log_sigma_sq = tf.layers.conv2d(inputs=pool2, filters=1,kernel_size=[5,5 ],padding='VALID',activation = None)
        eps = tf.random_normal(shape=tf.shape(log_sigma_sq),mean=0, stddev=1, dtype=tf.float32)

        self.z = mu + tf.sqrt(tf.exp(log_sigma_sq)) * eps #(?,2,2,1)

        Dup_sampling1 = tf.contrib.layers.conv2d_transpose(inputs=self.z,num_outputs = 1, kernel_size=[4,4],stride = 2,padding = 'VALID',activation_fn  =tf.nn.relu) #(?,2,2,1) ->(?,6,6,1)
        Dconv1 = tf.contrib.layers.conv2d(inputs = Dup_sampling1,num_outputs = 16,kernel_size =[5,5],padding = 'SAME',activation_fn=tf.nn.relu) #(?,6,6,1)->(?,6,6,16)
        Dup_sampling2 = tf.contrib.layers.conv2d_transpose(inputs=Dconv1,num_outputs = 16, kernel_size=[6,6],stride = 2,padding = 'VALID',activation_fn  =tf.nn.relu) #(?,6,6,16) ->(?,16,16,16)
        Dconv2 = tf.contrib.layers.conv2d(inputs = Dup_sampling2,num_outputs = 8 , kernel_size =[5,5],padding = 'SAME',activation_fn=tf.nn.relu) # (?,16,16,16) -> (?,16,16,8)
        D_xhat = tf.contrib.layers.conv2d_transpose(inputs = Dconv2,num_outputs = 1,kernel_size =[2,2],stride=2,padding = 'SAME',activation_fn=tf.nn.sigmoid)

        # Reconstruction Loss

        epsilon = 1e-10
        recon_loss = -tf.reduce_sum(self.x * tf.log(epsilon+D_xhat) +(1-self.x) * tf.log(epsilon+1-D_xhat))
        self.recon_loss = tf.reduce_mean(recon_loss)
        # Latent loss
        KL_loss = -0.5 * tf.reduce_sum(1 + log_sigma_sq - tf.square(mu) - tf.exp(log_sigma_sq))
        self.KL_loss = tf.reduce_mean(KL_loss)

        self.total_loss = self.recon_loss + self.KL_loss
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.total_loss)
        self.losses = {'recon_loss': self.recon_loss,'total_loss': self.total_loss,'KL_loss': self.KL_loss,}
        return

    # Execute the forward and the backward pass
    def run_single_step(self):
        data = data_loader.load_data_wrapper() # IMPORT FROM  data_loader
        dataset = tf.data.Dataset.from_tensor_slices(data).batch(self.batch_size)
        iterator = dataset.make_initializable_iterator()
        next = iterator.get_next()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()

            sess.run(iterator.initializer)
            for i in range(320000):
                # print(val)
                if i>0 and i % (320000 // self.batch_size) == 0:
                    sess.run(iterator.initializer)
                val = sess.run(next)
                _, losses = self.sess.run([self.train_op, self.losses],feed_dict={self.x: val })
                if i % 4000 == 0:
                    print(losses)
            # save_path = saver.save(sess, "./VAEcnn_model.ckpt")
            # print("Model saved in path: %s" % save_path)
            # zsample =  np.linspace(0,1, 10).reshape(10,1)
            # G_sample = sess.run(self.fully_connected_decoder1, feed_dict={self.z: zsample})
            # gsample = sess.run(self.x_hat, feed_dict={self.fully_connected_decoder1:G_sample})
            # # print(gsample.shape)
            # for i in range(10):
            #     self.draw_grid(8, gsample[i].reshape(8,8) , zsample[i])
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
