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
import random
import get_parameters

lattice_size = 8
def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1.0/tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

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
    for i in range(0, image.width, step_size):
        for j in range(0, image.height, step_size):
            draw.line(((i+a, j+a) , ( i + a + a*math.cos(angle[i//step_size,j//step_size]), j + a + a*math.sin(angle[i//step_size,j//step_size]))))
            draw.line(((i+a, j+a) , ( i + a - a*math.cos(angle[i//step_size,j//step_size]), j + a - a*math.sin(angle[i//step_size,j//step_size]))))

    image.show()


X = tf.placeholder(tf.float32, shape=[None, 64])

D_W1 = tf.Variable(xavier_init([64, 32]))
D_b1 = tf.Variable(tf.zeros(shape=[32]))

D_W2 = tf.Variable(xavier_init([32, 1]))
D_b2 = tf.Variable(tf.zeros(shape=[1]))

theta_D = [D_W1, D_W2, D_b1, D_b2]

Z_dim = 1
Z = tf.placeholder(tf.float32, shape=[None, Z_dim])

G_W1 = tf.Variable(xavier_init([Z_dim, 32]))
G_b1 = tf.Variable(tf.zeros(shape=[32]))

G_W2 = tf.Variable(xavier_init([32, 64]))
G_b2 = tf.Variable(tf.zeros(shape=[64]))

theta_G = [G_W1, G_W2, G_b1, G_b2]


def sample_Z(m, n):
    return np.random.uniform(0., 1., size=[m, n])


def generator(z):
    G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
    G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
    G_prob = tf.nn.sigmoid(G_log_prob)

    return G_prob


def discriminator(x):
    D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
    D_logit = tf.matmul(D_h1, D_W2) + D_b2
    D_prob = tf.nn.sigmoid(D_logit)

    return D_prob, D_logit


def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')
    return fig


G_sample = generator(Z)
D_real, D_logit_real = discriminator(X)
D_fake, D_logit_fake = discriminator(G_sample)

D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))
G_loss = -tf.reduce_mean(tf.log(D_fake))

#
# -------------------
# D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
# D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
# D_loss = D_loss_real + D_loss_fake
# G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))

D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)

mini_batch_size = 10


saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess,'./GANmodel.ckpt')
    training_data = data_loader.load_data_wrapper()
    random.shuffle(training_data)
    a = tf.placeholder(tf.float32,[320000, 64])
    dataset = tf.data.Dataset.from_tensor_slices(a)
    dataset = dataset.prefetch(buffer_size=1000)
    dataset = dataset.batch(mini_batch_size)
    iterator = dataset.make_initializable_iterator()
    next = iterator.get_next()

    # sess.run(tf.global_variables_initializer())

    print("Session initialized :)")
    sess.run(iterator.initializer, feed_dict = {a:training_data})
    print("Iterator initialized :)")

    # it = 0

    for i in range(0000):      #total data 32*5000 = 160000
        # if i % 1000 == 0:
        #     samples = sess.run(G_sample, feed_dict={Z: sample_Z(16, Z_dim)})
        #     fig = plot(samples)
        #     plt.savefig('samples/{}.png'.format(str(it).zfill(3)), bbox_inches='tight')
        #     it += 1
        #     plt.close(fig)

        if i>0 and i % (160000 // mini_batch_size) == 0:
            sess.run(iterator.initializer, feed_dict = {a:training_data})
        val = sess.run(next)
        _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: val, Z: sample_Z(mini_batch_size, Z_dim)})
        _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: sample_Z(mini_batch_size, Z_dim)})

        if i % 1000 == 0:
            print('Iter: {}'.format(i),'  D loss: {:.4}'. format(D_loss_curr),'  G_loss: {:.4}'.format(G_loss_curr))
    save_path = saver.save(sess, "./GANmodel.ckpt")
    print("Model saved in path: %s" % save_path)
    n_samples = 32
    zsample =  np.linspace(0.01,2.5, n_samples).reshape(n_samples,1)
    gsample = sess.run([G_sample], feed_dict={Z:zsample})
    gsample = np.array(gsample).reshape(n_samples,lattice_size,lattice_size)
    print("Specific Heat")
    print(get_parameters.get_specific_heat(gsample,zsample))
    print()
    print(get_parameters.get_specific_heat(gsample,zsample))
    print("Mean magnetization and its Standard Deviation")
    Magnetization = get_parameters.get_mean_magnetization(gsample)
    print(Magnetization[0][0])
    fig = plt.figure()
    plt.plot(zsample,Magnetization[0][0])
    plt.xlabel('Latent Variable value', fontsize=12)
    plt.ylabel('Magnetization of a single sample generated by the network', fontsize=10)
    plt.show()
    # for i in range(10):
        # draw_grid(8,gsample[10*i].reshape(8,8),zsample[10*i])
        # print(get_vorticity_configuration(gsample[0][i]))
