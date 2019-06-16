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
n_zcat        = 1
n_zc          = 1
n_zr          = 1
x = tf.placeholder(name='x', dtype=tf.float32, shape=[None, 64])
y = tf.placeholder(name='y', dtype=tf.float32, shape=[None])
loss_type  = 'log_gaussian' #'Binary_crossentropy'
datapoints = 320000
n_temps    = 32
T_vals = np.linspace(0,2,n_temps)
global_step = tf.train.get_or_create_global_step()


with tf.name_scope('noise_sample'):
    z_cat  = tf.placeholder(tf.float32, [None, n_zcat])
    z_cont = tf.placeholder(tf.float32, [None, n_zc])
    z_rand = tf.placeholder(tf.float32, [None, n_zr])
    z = tf.concat([z_cat ,z_cont, z_rand], axis=1)

def generator(z,reuse= None):
    with tf.variable_scope('Generator', reuse=tf.AUTO_REUSE):
        net   = tf.contrib.layers.fully_connected(inputs=z   , num_outputs=10, activation_fn=tf.nn.relu,scope="inp")
        net2  = tf.contrib.layers.fully_connected(inputs=net , num_outputs=32, activation_fn=tf.nn.relu,scope="hid")
        net_mu= tf.contrib.layers.fully_connected(inputs=net2, num_outputs=64, activation_fn=tf.nn.relu,scope="mu")
        net_sg= tf.contrib.layers.fully_connected(inputs=net2, num_outputs=64, activation_fn=tf.nn.relu,scope="sg") #-ve log(sigma_sq)
        sample= tf.random_normal(shape = tf.shape(net_mu),mean = net_mu,stddev = tf.sqrt(tf.exp(-net_sg)), dtype = tf.float32 )
    return sample

def discriminator(g,reuse = None):
    with tf.variable_scope('Discriminator', reuse=tf.AUTO_REUSE):
        net   = tf.contrib.layers.fully_connected(inputs=g   , num_outputs=32,  activation_fn=tf.nn.relu   ,scope="inp")
        net2  = tf.contrib.layers.fully_connected(inputs=net , num_outputs=10,  activation_fn=tf.nn.relu   ,scope="hid")
        d_out = tf.contrib.layers.fully_connected(inputs=net2, num_outputs=1 ,  activation_fn=tf.nn.sigmoid,scope="prob")
        r1    = tf.contrib.layers.fully_connected(inputs=net2, num_outputs=32,  activation_fn=tf.nn.relu   ,scope="hid2")
        r_cat = tf.contrib.layers.fully_connected(inputs=r1  , num_outputs=n_zcat,activation_fn=tf.nn.relu   ,scope="cat")
        r_mu  = tf.contrib.layers.fully_connected(inputs=r1  , num_outputs=n_zc,activation_fn=None ,scope="mu")
        # r_sg  = tf.contrib.layers.fully_connected(inputs=r1  , num_outputs=n_zc,activation_fn=tf.nn.relu   ,scope="sg")
    return d_out,r_cat,r_mu

G = generator(z)
# print(G)
d_real,_,_ = discriminator(g=tf.concat([x,z_cat],axis=1))
d_fake, r_cat, r_mu = discriminator(g=tf.concat([G,z_cat],axis=1),reuse = False)

y_real = tf.ones(tf.shape(d_fake))
y_fake = tf.zeros(tf.shape(d_fake))

# 3. Calculate loss
# -log(D(G(x))) trick
with tf.variable_scope('lossG'):
    g_loss   = -tf.reduce_mean(tf.log(1.0-d_fake+1e-5))
    cat_loss = 0.5*tf.reduce_mean(tf.reduce_sum((r_cat-z_cat)**2, axis=1))#
    cont_loss = 0.5*tf.reduce_mean(tf.reduce_sum((z_cont-r_mu)**2, axis=1))
    train_g_loss = g_loss + cat_loss + cont_loss*0.1

with tf.variable_scope('lossD'):
    d_loss_fake = -tf.reduce_mean(tf.log(1.0-d_fake+1e-5)) #tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake, labels=y_fake))
    d_loss_real = -tf.reduce_mean(tf.log(d_real+1e-5))#tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_real, labels=y_real))
    train_d_loss = d_loss_fake + d_loss_real

# 4. Update weights
g_param = tf.trainable_variables(scope='Generator')
d_param = tf.trainable_variables(scope='Discriminator')
with tf.name_scope('optimizer'):
    d_optim = tf.train.AdamOptimizer(learning_rate= 1e-4 ).minimize(train_d_loss, var_list=d_param)
    g_optim = tf.train.AdamOptimizer(learning_rate= 1e-4 ).minimize(train_g_loss, var_list=[g_param,d_param])
    saver   = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess,'./GANmodel.ckpt')
    writer = tf.summary.FileWriter('./graphs', sess.graph)
    training_data = data_loader.load_data_wrapper()
    tvals = np.repeat(np.linspace(0.1,2.0,32),10000)
    c = list(zip(training_data,tvals))
    # random.shuffle(c)
    training_data, tvals = zip(*c)
    m = tf.placeholder(tf.float32,[datapoints, 64])
    n = tf.placeholder(tf.float32,[datapoints, 1])
    dataset = tf.data.Dataset.from_tensor_slices((m,n))
    dataset = dataset.prefetch(buffer_size=1000)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_initializable_iterator()
    next = iterator.get_next()

    # print("============< WARNING >===============")
    # sess.run(tf.global_variables_initializer())
    # print("==========< Model DELETED >===========")

    sess.run(iterator.initializer,feed_dict = {m:training_data, n:np.array(tvals).reshape(datapoints,1) + 0.01*np.random.randn(datapoints,1)})
    print("Session initialized :)")
    print("Iterator initialized :)")

    for i in range(0000):      #total data 32*5000 = 160000
        if i>0 and i % (datapoints // batch_size) == 0:
            sess.run(iterator.initializer, feed_dict = {m:training_data, n:np.array(tvals).reshape(datapoints,1)+ 0.01*np.random.randn(datapoints,1) })
        g,h = sess.run(next)

        z_c = np.random.uniform(-1, 1, size=[batch_size, n_zc])
        z_r = np.random.uniform(-1, 1, size=[batch_size, n_zr])

        D_loss_curr,_ = sess.run([train_d_loss,d_optim], feed_dict={x:g, z_cat: h,z_cont: z_c,z_rand: z_r})
        G_loss_curr,_ = sess.run([train_g_loss,g_optim], feed_dict={z_cat: h,z_cont: z_c,z_rand: z_r})

        if i % 1000 == 0:
            print('Iter: {}'.format(i),'  D loss: {:.4}'. format(D_loss_curr),'  G_loss: {:.4}'.format(G_loss_curr))

    # save_path = saver.save(sess, "./GANmodel.ckpt")
    # print("Model saved in path: %s" % save_path)

    zsample = np.mgrid[-1:1:0.2, -1:1:0.2].reshape(2,-1).T #np.linspace(-5,3,n_samples).reshape(n_samples,1)
    n_samples = (10,10)
    n = 100

    for i in range(0,30,5):
        print(T_vals[i])
        t_vals = np.repeat(T_vals[i],100)
        z = np.array(list(zip(t_vals.reshape(100), zsample[:, 0], zsample[:, 1])))
        # z_c = np.linspace(-1,1,n_samples).reshape(n_samples,1)
        # z_r = np.linspace(-1,1,n_samples).reshape(n_samples,1)
        # n_samples = (10,20)    # int(np.shape(zsample)[0]**(1/n_z))
        gsample = sess.run(G, feed_dict={z_cat:z[:,0].reshape(n,1),z_cont:z[:,1].reshape(n,1),z_rand:z[:,2].reshape(n,1)})  #np.array(tvals[:n_samples]).reshape(n_samples,1)}
        gsample = gsample.reshape(n,lattice_size,lattice_size)

        print(360*np.mean(gsample),360*np.std(gsample))

        Magnetization           = get_parameters.get_mean_magnetization(gsample)
        Magnetization_direction = get_parameters.get_magnetization_direction(gsample)
        energy                  = get_parameters.get_energy(gsample)

        # for d in range(n_samples[0]):
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

    def inference(self):

        if self.model_dir is None:
            raise ValueError('Need to provide model directory')

        checkpoint = tf.train.latest_checkpoint(self.model_dir)

        if not checkpoint:
            raise FileNotFoundError('Checkpoint is not found in {}'.format(self.model_dir))
        else:
            print('Loading model checkpoint {}...'.format(self.model_dir))
            self.saver.restore(self.sess, checkpoint)

        for q in range(2):
            col = []
            for c in range(10):
                row = []
                for d in range(11):
                    z_cat = [c]
                    z_cont = -np.ones([1, self.args.num_cont])*2 + d*0.4
                    z_cont[0, q] = 0
                    z_rand = np.random.uniform(-1, 1, size=[1, self.args.num_rand])

                    g = self.sess.run([self.g], feed_dict={self.z_cat: z_cat,
                                                           self.z_cont: z_cont,
                                                           self.z_rand: z_rand})
                    g = np.squeeze(g)
                    multiplier = 255.0 / g.max()
                    g = (g * multiplier).astype(np.uint8)
                    row.append(g)

                row = np.concatenate(row, axis=1)
                col.append(row)
            result = np.concatenate(col, axis=0)
            filename = 'continuous_' + str(q) + '_col_cat_row_change.png'
            cv2.imwrite(os.path.join(self.test_dir, filename), result)
