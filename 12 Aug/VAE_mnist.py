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

fully_connected1 = tf.contrib.layers.fully_connected(inputs=x, num_outputs=512, activation_fn=tf.nn.relu,scope="Fully_Conn1")
z_mu             = tf.contrib.layers.fully_connected(inputs=fully_connected1, num_outputs=n_z, activation_fn=None,scope="Fully_Conn2_mu")
z_log_sigma_sq   = tf.contrib.layers.fully_connected(inputs=fully_connected1, num_outputs=n_z, activation_fn=None,scope="Fully_Conn2_sig")

eps = tf.random_normal(shape=tf.shape(z_log_sigma_sq),mean=0, stddev=1, dtype=tf.float32)
z = z_mu + tf.sqrt(tf.exp(z_log_sigma_sq)) * eps

#decoder
fully_connected_decoder1 = tf.contrib.layers.fully_connected(inputs=z, num_outputs=512, activation_fn=tf.nn.relu,scope="Fully_Conn1_decoder")
x_hat = tf.contrib.layers.fully_connected(inputs=fully_connected_decoder1, num_outputs=784, activation_fn=tf.sigmoid,scope="Fully_Conn2_decoder")
# x_mu = tf.contrib.layers.fully_connected(inputs=Dhidden2, num_outputs=256, activation_fn=None,scope="Fully_Conn3_decodermu")
# x_log_sigma_sq = tf.contrib.layers.fully_connected(inputs=Dhidden2, num_outputs=256, activation_fn=None,scope="Fully_Conn3_decodersigma")
# x_hat = tf.random_normal(shape = tf.shape(x_log_sigma_sq), mean = x_mu, stddev=tf.sqrt(tf.exp(x_log_sigma_sq)), dtype = tf.float32)
# Reconstruction Loss

epsilon = 1e-10
recon_loss = -1*tf.reduce_sum(    x * tf.log(epsilon+x_hat) +(1-x) * tf.log(epsilon+1-x_hat),axis=1)
recon_loss = tf.reduce_mean(recon_loss)

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
    for i in range(5000):
            # print(i)
        if i>0 and i % (50000 // batch_size) == 0:
            sess.run(iterator.initializer, feed_dict = {a:training_data})
        b = sess.run(next)
        _, Losses = sess.run([train_op, losses],feed_dict={x: b })
        if i%1000==0:
            print(Losses)


    save_path = saver.save(sess, "./VAE_mnist.ckpt")
    print("Model saved in path: %s" % save_path)
    zsample =  np.random.random([10,n_z])
    G_sample = sess.run(fully_connected_decoder1, feed_dict={z: zsample})
    gsample = sess.run(x_hat, feed_dict={fully_connected_decoder1:G_sample})
    # print(gsample)
    # print(training_data[0].reshape(28,28))
    img = pil.fromarray(np.uint8(training_data[0].reshape(28,28) * 255) , 'L')
    img.show()
    for i in range(10):
        img_gen = pil.fromarray(np.uint8(gsample[i].reshape(28,28) * 255) , 'L')
        img_gen.show()

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
