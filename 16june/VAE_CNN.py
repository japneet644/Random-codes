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

learning_rate=1e-2
batch_size=10
n_z=1

x = tf.placeholder(name='x', dtype=tf.float32, shape=[None, 32,32,1])

conv1 = tf.layers.conv2d(inputs=x,filters=8,kernel_size=[5, 5],padding = 'SAME',activation=tf.nn.relu)#(?,32,32,1) -> (?,32,32,8)
pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)# (?,32,32,8)->(16,16,8)
conv2 = tf.layers.conv2d(inputs=pool1,filters=16,kernel_size=[5, 5],padding = 'VALID',activation=tf.nn.relu)# (16,16,8) -> (12,12,16)
pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)# (?,12,12,16) -> (?,6,6,16)
mu    = tf.layers.conv2d(inputs = pool2,filters=1,kernel_size=[5, 5],padding = 'VALID',activation=None)#(?,6,6,16) ->mu(?,2,2,1)
log_sigma_sq = tf.layers.conv2d(inputs=pool2, filters=1,kernel_size=[5,5 ],padding='VALID',activation = None)
eps   = tf.random_normal(shape=tf.shape(log_sigma_sq),mean=0, stddev=1, dtype=tf.float32)

z = mu + tf.sqrt(tf.exp(log_sigma_sq)) * eps #(?,2,2,1)

Dup_sampling1 = tf.contrib.layers.conv2d_transpose(inputs=z,num_outputs = 1, kernel_size=[4,4],stride = 2,padding = 'VALID',activation_fn  =tf.nn.relu) #(?,2,2,1) ->(?,6,6,1)
Dconv1 = tf.contrib.layers.conv2d(inputs = Dup_sampling1,num_outputs = 16,kernel_size =[5,5],padding = 'SAME',activation_fn=tf.nn.relu) #(?,6,6,1)->(?,6,6,16)
Dup_sampling2 = tf.contrib.layers.conv2d_transpose(inputs=Dconv1,num_outputs = 16, kernel_size=[6,6],stride = 2,padding = 'VALID',activation_fn  =tf.nn.relu) #(?,6,6,16) ->(?,16,16,16)
Dconv2 = tf.contrib.layers.conv2d(inputs = Dup_sampling2,num_outputs = 8 , kernel_size =[5,5],padding = 'SAME',activation_fn=tf.nn.relu) # (?,16,16,16) -> (?,16,16,8)
D_xhat = tf.contrib.layers.conv2d_transpose(inputs = Dconv2,num_outputs = 1,kernel_size =[2,2],stride=2,padding = 'SAME',activation_fn=tf.nn.sigmoid)

# Reconstruction Loss

epsilon = 1e-10
recon_loss = -tf.reduce_sum(x * tf.log(epsilon+D_xhat) +(1-x) * tf.log(epsilon+1-D_xhat))
recon_loss = tf.reduce_mean(recon_loss)
# Latent loss
KL_loss = -0.5 * tf.reduce_sum(1 + log_sigma_sq - tf.square(mu) - tf.exp(log_sigma_sq))
KL_loss = tf.reduce_mean(KL_loss)

total_loss = recon_loss + KL_loss
train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(total_loss)
losses = {'recon_loss': recon_loss,'total_loss': total_loss,'KL_loss': KL_loss,}

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess,'./VAE_CNN.ckpt')
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


    for i in range(320000):
        if i>0 and i % (320000 // batch_size) == 0:
            sess.run(iterator.initializer)
        val = sess.run(next)
        _, losses = sess.run([train_op, losses],feed_dict={x: val })
        if i % 4000 == 0:
            print(losses)
    # save_path = saver.save(sess, "./VAEcnn_model.ckpt")
    # print("Model saved in path: %s" % save_path)
    # zsample =  np.linspace(0,1, 10).reshape(10,1)
    # G_sample = sess.run(fully_connected_decoder1, feed_dict={z: zsample})
    # gsample = sess.run(x_hat, feed_dict={fully_connected_decoder1:G_sample})
    # # print(gsample.shape)
    # for i in range(10):
    #     draw_grid(8, gsample[i].reshape(8,8) , zsample[i])
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
