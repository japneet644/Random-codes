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

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


X = tf.placeholder(tf.float32, shape=[None, 784])

D_W1 = tf.Variable(xavier_init([784, 200]))
D_b1 = tf.Variable(tf.zeros(shape=[200]))

D_W2 = tf.Variable(xavier_init([200, 20]))
D_b2 = tf.Variable(tf.zeros(shape=[20]))

theta_D = [D_W1, D_W2, D_b1, D_b2]

Z_dim = 20
Z = tf.placeholder(tf.float32, shape=[None, Z_dim])

G_W1 = tf.Variable(xavier_init([Z_dim, 200]))
G_b1 = tf.Variable(tf.zeros(shape=[200]))

G_W2 = tf.Variable(xavier_init([200, 784]))
G_b2 = tf.Variable(tf.zeros(shape=[784]))

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

epsilon = 1e-10
D_loss = -tf.reduce_mean(tf.log(epsilon+D_real) + tf.log(1.+epsilon - D_fake))
G_loss = -tf.reduce_mean(tf.log(epsilon+D_fake))

D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)

batch_size = 20
#==========================
saver = tf.train.Saver()

with tf.Session() as sess:
    # saver.restore(sess,'./GAN_mnist.ckpt')
    training_data = mnist_loader.load_data_wrapper() #======== IMPORT FROM  data_loader
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

    for i in range(10000):
        if i>0 and i % (50000 // batch_size) == 0:
            sess.run(iterator.initializer, feed_dict = {a:training_data})
        b = sess.run(next)
        _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: b, Z: sample_Z(batch_size, Z_dim)})
        _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: sample_Z(batch_size, Z_dim)})

        if i % 100 == 0:
            print('Iter: {}'.format(i),'  D loss: {:.4}'. format(D_loss_curr),'  G_loss: {:.4}'.format(G_loss_curr))


    save_path = saver.save(sess, "./GAN_mnist.ckpt")
    print("Model saved in path: %s" % save_path)
    zsample = np.random.random([10,Z_dim])
    gsample = sess.run([G_sample], feed_dict={Z:zsample})
    print(gsample[0].shape)
    for i in range(10):
        img_gen = pil.fromarray(np.uint8(gsample[0][i].reshape(28,28) * 255) , 'L')
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
    for i in range(0, image.width, step_size):
        for j in range(0, image.height, step_size):
            draw.line(((i+a, j+a) , ( i + a + a*math.cos(angle[i//step_size,j//step_size]), j + a + a*math.sin(angle[i//step_size,j//step_size]))))
            draw.line(((i+a, j+a) , ( i + a - a*math.cos(angle[i//step_size,j//step_size]), j + a - a*math.sin(angle[i//step_size,j//step_size]))))

    image.show()
