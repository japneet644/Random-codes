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

lattice_size  = 8
learning_rate = 1e-4
batch_size    = 20
n_z           = 1
x = tf.placeholder(name='x', dtype=tf.float32, shape=[None, 64])
y = tf.placeholder(name='y', dtype=tf.float32, shape=[None,1])
loss_type  = 'log_gaussian' #'Binary_crossentropy'
datapoints = 320000
n_temps    = 32
alpha      = 1
LAMBDA     = 1

fully_connected1 = tf.contrib.layers.fully_connected(inputs=x, num_outputs=100, activation_fn=tf.nn.relu,scope="Fully_Conn1")#tf.concat([y,x],axis=1)
fully_connected2 = tf.contrib.layers.fully_connected(inputs=fully_connected1, num_outputs=20, activation_fn=tf.nn.relu,scope="Fully_Conn2")
z_mu             = tf.contrib.layers.fully_connected(inputs=fully_connected2, num_outputs=n_z, activation_fn=None,scope="Fully_Conn2_mu")
z_log_sigma_sq   = tf.contrib.layers.fully_connected(inputs=fully_connected2, num_outputs=n_z, activation_fn=None,scope="Fully_Conn2_sig")

eps = tf.random_normal(shape=tf.shape(z_log_sigma_sq),mean=0, stddev=1, dtype=tf.float32)
z   = z_mu + tf.sqrt(tf.exp(z_log_sigma_sq)) * eps

#decoder
epsilon = 1e-10
fully_connected_decoder1 = tf.contrib.layers.fully_connected(inputs=tf.concat([y,z],axis=1), num_outputs=20, activation_fn=tf.nn.relu,scope="Fully_Conn1_decoder")#
fully_connected_decoder2 = tf.contrib.layers.fully_connected(inputs=fully_connected_decoder1, num_outputs=100, activation_fn=tf.nn.relu,scope="Fully_Conn2_decoder")

def compute_kernel(x, y):
    x_size = tf.shape(x)[0]
    y_size = tf.shape(y)[0]
    dim = tf.shape(x)[1]
    tiled_x = tf.tile(tf.reshape(x, tf.stack([x_size, 1, dim])), tf.stack([1, y_size, 1]))
    tiled_y = tf.tile(tf.reshape(y, tf.stack([1, y_size, dim])), tf.stack([x_size, 1, 1]))
    return tf.exp(-tf.reduce_mean(tf.square(tiled_x - tiled_y), axis=2) / tf.cast(dim, tf.float32))

def compute_mmd(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    return 100*(tf.reduce_mean(x_kernel) + tf.reduce_mean(y_kernel) - 2 * tf.reduce_mean(xy_kernel))

if loss_type == 'Binary_crossentropy':
    x_hat                = tf.contrib.layers.fully_connected(inputs=fully_connected_decoder2, num_outputs=64, activation_fn=tf.sigmoid,scope="Fully_Conn2_decoder_out")
    recon_loss           =-1*tf.reduce_sum(    x * tf.log(epsilon+x_hat) +(1-x) * tf.log(epsilon+1-x_hat),axis=1)#
elif loss_type == 'log_gaussian':
    x_mu                 = tf.contrib.layers.fully_connected(inputs=fully_connected_decoder2, num_outputs=64, activation_fn=tf.sigmoid,scope="Fully_Conn2_decoder_mu")
    x_log_sigma_sq       = tf.contrib.layers.fully_connected(inputs=fully_connected_decoder2, num_outputs=64, activation_fn=tf.nn.relu,scope="Fully_Conn2_decoder_std")
    recon_loss           = 0.5*tf.reduce_sum( ((x-x_mu)**2)/tf.exp(-x_log_sigma_sq)+1.837- x_log_sigma_sq ,axis=1) #
    # x_hat =tf.random_normal(shape = tf.shape(x_mu) ,mean = x_mu, stddev =tf.sqrt(tf.exp(x_log_sigma_sq)), dtype = tf.float32 )

# Reconstruction Loss
recon_loss = tf.reduce_mean(recon_loss)
# MMD loss
true_samples = tf.random_normal(shape = tf.shape(z),mean = 0.0,stddev = 1.0,dtype = tf.float32)
mmd_loss = compute_mmd(true_samples, z)

KL_loss = - 0.5* tf.reduce_sum(    1 + z_log_sigma_sq - tf.square(z_mu) -tf.exp(z_log_sigma_sq), axis=1)
KL_loss = tf.reduce_mean(KL_loss)

total_loss = recon_loss + (alpha + LAMBDA -1)*mmd_loss + (1-alpha)*KL_loss
train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(total_loss)

losses = {
    'recon_loss': recon_loss,
    'total_loss': total_loss,
    'mmd_loss' : mmd_loss,
    'KL_loss':  KL_loss,
}

saver = tf.train.Saver()
with tf.Session() as sess:
    # saver.restore(sess,'./MMD_VAE_xy2.ckpt')
    # writer = tf.summary.FileWriter('./graphs', sess.graph)
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

    sess.run(iterator.initializer,         feed_dict = {m:training_data, n:np.array(tvals).reshape(datapoints,1) + 0.01*np.random.randn(datapoints,1)})
    print("Session initialized :)")
    print("Iterator initialized :)")

    for i in range(0000):
        if i>0 and i % (datapoints // batch_size) == 0:
            sess.run(iterator.initializer, feed_dict = {m:training_data, n:np.array(tvals).reshape(datapoints,1) + 0.01*np.random.randn(datapoints,1) })
        g,h = sess.run(next)
        _, Losses = sess.run([train_op, losses],feed_dict={x: g,y:h })
        if i%1000==0:
            print(Losses)
    # save_path = saver.save(sess, "./MMD_VAE_xy2.ckpt")
    # print("Model saved in path: %s" % save_path)

    if n_z == 1:
        T_vals    = np.linspace(0.1,2.0,20)
        zsample   = np.mgrid[0.0:2.0:0.2, -3.0:3.0:0.2].reshape(2,-1).T #np.linspace(-5,3,n_samples).reshape(n_samples,1)
        n_samples = (10,30)    # int(np.shape(zsample)[0]**(1/n_z))
        # print(zsample[:,0].reshape(zsample.shape[0],1))
        if loss_type == 'Binary_crossentropy':
            Gsample    = sess.run(fully_connected_decoder1, feed_dict={z: zsample[:,1].reshape(zsample.shape[0],1), y:zsample[:,0].reshape(zsample.shape[0],1) + 0.01*np.random.randn(zsample.shape[0],1)})
            Gsample2   = sess.run(fully_connected_decoder2, feed_dict={fully_connected_decoder1: Gsample  })
            gsample    = sess.run(x_hat,                    feed_dict={fully_connected_decoder2: Gsample2 })
        elif loss_type == 'log_gaussian':
            Gsample    = sess.run(fully_connected_decoder1, feed_dict={z: zsample[:,1].reshape(zsample.shape[0],1), y:zsample[:,0].reshape(zsample.shape[0],1) + 0.01*np.random.randn(zsample.shape[0],1)})  #np.array(tvals[:n_samples]).reshape(n_samples,1)}
            Gsample2   = sess.run(fully_connected_decoder2, feed_dict={fully_connected_decoder1: Gsample})
            Gsamplemu  = sess.run(x_mu,                     feed_dict={fully_connected_decoder2: Gsample2})
            Gsamplesig = sess.run(x_log_sigma_sq,           feed_dict={fully_connected_decoder2:Gsample2})
            gsample    = sess.run(tf.random_normal(shape = tf.shape(Gsamplemu),mean = Gsamplemu,stddev = tf.sqrt(tf.exp(-Gsamplesig)), dtype = tf.float32 ))#tf.sqrt(tf.exp(Gsamplesig))

    if n_z == 2:
        zsample =  np.mgrid[-2:2:0.4, -2:2:0.4].reshape(2,-1).T
        n_samples = (10 ,10)
        if loss_type == 'Binary_crossentropy':
            Gsample    = sess.run(fully_connected_decoder1, feed_dict={z: zsample})
            Gsample2   = sess.run(fully_connected_decoder2, feed_dict={fully_connected_decoder1: Gsample  })
            gsample    = sess.run(x_hat,                    feed_dict={fully_connected_decoder2: Gsample2 })
        elif loss_type == 'log_gaussian':
            Gsample    = sess.run(fully_connected_decoder1, feed_dict={z: zsample})  #np.array(tvals[:n_samples]).reshape(n_samples,1)}
            Gsample2   = sess.run(fully_connected_decoder2, feed_dict={fully_connected_decoder1: Gsample})
            Gsamplemu  = sess.run(x_mu,                     feed_dict={fully_connected_decoder2: Gsample2})
            Gsamplesig = sess.run(x_log_sigma_sq,           feed_dict={fully_connected_decoder2:Gsample2})
            gsample    = sess.run(tf.random_normal(shape = tf.shape(Gsamplemu),mean = Gsamplemu,stddev = tf.sqrt(tf.exp(-Gsamplesig)), dtype = tf.float32 ))#tf.sqrt(tf.exp(Gsamplesig))

    gsample = gsample.reshape(zsample.shape[0],lattice_size,lattice_size)

    print("Specific Heat %f")
    print(360*np.mean(Gsamplemu),360*np.std(Gsamplemu))
    print( 360*np.mean(-1.0*Gsamplesig))
    print("Mean magnetization and its Standard Deviation")

    # for i in range(n_samples):
        # print(get_parameters.get_specific_heat(gsample[i],zsample[i]))

    mean_magnetization = []
    Magnetization           = get_parameters.get_mean_magnetization(gsample)
    Magnetization_direction = get_parameters.get_magnetization_direction(gsample)
    energy                  = get_parameters.get_energy(gsample)

    if n_z == 1:
        for d in range(n_samples[0]):
            plt.plot(zsample[:n_samples[1],1],Magnetization[0][0][d*n_samples[1]:(d+1)*n_samples[1]],label = d) #all values
        plt.xlabel('Latent Variable value', fontsize=12)
        plt.ylabel('Magnetization of a single sample generated by the network', fontsize=10)
        plt.legend()
        plt.show()
        for d in range(n_samples[0]):
            plt.plot(zsample[:n_samples[1],1],Magnetization_direction[d*n_samples[1]:(d+1)*n_samples[1]])
        plt.show()
        plt.hist(energy,bins =100)
        plt.legend()
        plt.show()
        n_maps = 2000 #no of mappings per temp
        for u in range(0,20,2):
            sampledz = sess.run(z,feed_dict={x:training_data[10000*u+500:10000*u+500+n_maps],y:np.array(tvals[10000*u+500:10000*u+500+n_maps]).reshape(n_maps,1)-1})
            plt.hist(sampledz,bins=100,label = u)
        plt.legend()
        plt.show()
        for d in range(0,n_samples[1]):
            plt.plot(zsample[d:zsample.shape[0]:n_samples[1],0],Magnetization[0][0][d:zsample.shape[0]:n_samples[1]],label = d)
        plt.legend(loc='best')
        plt.show()

    elif n_z == 2:
        """
        LV[0] Magnetization direction
        LV[1] Temperature
        plot 1 varying Temperature keeping direction constant
        plot 2 varying directio at a fixed Temperature
        plot 3
        plot 4
        """
        for g in range(n_samples[0]):
            plt.plot(zsample[g*n_samples[1]:(g+1)*n_samples[1]][:,1],Magnetization[0][0][g*n_samples[1]:(g+1)*n_samples[1]])#,label = 'LV[0] = %f' %(zsample[g*n_samples[1]][0]),linestyle='dotted')
        for g in range(n_samples[1]):
            mean_magnetization.append(np.mean(Magnetization[0][0][g:zsample.shape[0]:n_samples[1]]))
        plt.plot(zsample[0:n_samples[1]][:,1],mean_magnetization,label = 'Mean ',linestyle = '--')
        plt.xlabel('Latent Variable value', fontsize=12)
        plt.ylabel('Magnetization of a single sample generated by the network', fontsize=10)
        plt.legend(loc='best')
        plt.savefig('mag_vs_temp.png')
        plt.show()
        for g in range(n_samples[1]):
            plt.plot(zsample[0:zsample.shape[0]:n_samples[1],0],Magnetization_direction[g:zsample.shape[0]:n_samples[1]],label='LV[1] = %f' %(zsample[g][1]))
        plt.xlabel('Rotational Latent Variable value', fontsize=12)
        plt.ylabel('Magnetization vector', fontsize=10)
        plt.legend(loc='best')
        plt.savefig('mag_dir_vs_temp.png')
        plt.show()
        for g in range(n_samples[1]):
            plt.hist(energy[g*n_samples[1]:(g+1)*n_samples[1]],bins = 30, label='LV[1] = %f' %(zsample[g][1]))
        plt.xlabel('Energy', fontsize=12)
        plt.ylabel('No of times')
        plt.legend(loc='best')
        plt.savefig('energy_vs_temp.png')
        plt.show()
        # draw_grid(8,gsample[30*i],zsample[30*i])
        #Just do reverse LV[1] Magnetization direction LV[0] Temperature
        for g in range(n_samples[1]):
            plt.plot(zsample[g:zsample.shape[0]:n_samples[1]][:,0],Magnetization[0][0][g:zsample.shape[0]:n_samples[1]])#,label = 'LV[0] = %f' %(zsample[g*n_samples[1]][0]),linestyle='dotted')
        # for g in range(n_samples[]):
        #     mean_magnetization.append(np.mean(Magnetization[0][0][g:zsample.shape[0]:n_samples[1]]))
        # plt.plot(zsample[0:n_samples[1]][:,1],mean_magnetization,label = 'Mean ',linestyle = '--')
        plt.xlabel('Latent Variable value', fontsize=12)
        plt.ylabel('Magnetization of a single sample generated by the network', fontsize=10)
        plt.legend(loc='best')
        plt.savefig('mag_vs_temp.png')
        plt.show()
        for g in range(n_samples[0]):
            plt.plot(zsample[g*n_samples[1]:(g+1)*n_samples[1]][:,1],Magnetization_direction[g*n_samples[0]:(g+1)*n_samples[0]],label='LV[1] = %f' %(zsample[g][1]))
        plt.xlabel('Rotational Latent Variable value', fontsize=12)
        plt.ylabel('Magnetization vector', fontsize=10)
        plt.legend(loc='best')
        plt.savefig('mag_dir_vs_temp.png')
        plt.show()
        for g in range(n_samples[1]):
            plt.hist(energy[g*n_samples[1]:(g+1)*n_samples[1]],bins = 30, label='LV[1] = %f' %(zsample[g][1]))
        plt.xlabel('Energy', fontsize=12)
        plt.ylabel('No of times')
        plt.legend(loc='best')
        plt.savefig('energy_vs_temp.png')
        plt.show()
        n_maps = 2000 #no of mappings per temp
        print(tvals[500:500+10])
        for d in range(0,32,4):
            sampledz = sess.run(z,feed_dict={x:np.array(training_data[10000*d+500:10000*d+500+n_maps]).reshape(n_maps,64), y:np.array(tvals[10000*d+500:10000*d+500+n_maps]).reshape(n_maps,1)})
            plt.plot(sampledz[:,0],sampledz[:,1],linestyle='dotted',label = d)
        plt.legend()
        plt.show()
        # for u in range(5,10):
        #     sampledmu = sess.run(z_mu,feed_dict={x:training_data[10000*u+500:10000*u+500+n_maps]})
        #     sampledsigma = sess.run(z_log_sigma_sq,feed_dict={x:training_data[10000*u+500:10000*u+500+n_maps]})
        #     plt.plot(sampledz[:,0],sampledz[:,1],linestyle='dotted',label = u)
        # plt.legend()
        # plt.show()
#
#
#
# # Convert a numpy array of shape [batch_size, height, width, 1] into a displayable array
# # of shape [height*sqrt(batch_size, width*sqrt(batch_size))] by tiling the images
# def convert_to_display(samples):
#     cnt, height, width = int(math.floor(math.sqrt(samples.shape[0]))), samples.shape[1], samples.shape[2]
#     samples = np.transpose(samples, axes=[1, 0, 2, 3])
#     samples = np.reshape(samples, [height, cnt, cnt, width])
#     samples = np.transpose(samples, axes=[1, 0, 2, 3])
#     samples = np.reshape(samples, [height*cnt, width*cnt])
#     return samples
#
# mnist = input_data.read_data_sets('mnist_data')
# plt.ion()
#
# train_x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
# train_z = encoder(train_x, z_dim)
# train_xr = decoder(train_z)
#
# # Build the computation graph for generating samples
# gen_z = tf.placeholder(tf.float32, shape=[None, z_dim])
# gen_x = decoder(gen_z, reuse=True)
#
# # Compare the generated z with true samples from a standard Gaussian, and compute their MMD distance
# true_samples = tf.random_normal(tf.stack([200, z_dim]))
# loss_mmd = compute_mmd(true_samples, train_z)
# loss_nll = tf.reduce_mean(tf.square(train_xr - train_x))
# loss = loss_nll + loss_mmd
# trainer = tf.train.AdamOptimizer(1e-3).minimize(loss)
#
# batch_size = 200
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
#
# # Start training
# for i in range(10000):
#     batch_x, batch_y = mnist.train.next_batch(batch_size)
#     batch_x = batch_x.reshape(-1, 28, 28, 1)
#     _, nll, mmd = sess.run([trainer, loss_nll, loss_mmd], feed_dict={train_x: batch_x})
#     if i % 100 == 0:
#         print("Negative log likelihood is %f, mmd loss is %f" % (nll, mmd))
#     if i % 500 == 0:
#         samples = sess.run(gen_x, feed_dict={gen_z: np.random.normal(size=(100, z_dim))})
#         plt.imshow(convert_to_display(samples), cmap='Greys_r')
#         plt.show()
#         plt.pause(0.001)
#
# # If latent z is 2-dimensional we visualize it by plotting latent z of different digits in different colors
# if z_dim == 2:
#     z_list, label_list = [], []
#     test_batch_size = 500
#     for i in range(20):
#         batch_x, batch_y = mnist.test.next_batch(test_batch_size)
#         batch_x = batch_x.reshape(-1, 28, 28, 1)
#         z_list.append(sess.run(train_z, feed_dict={train_x: batch_x}))
#         label_list.append(batch_y)
#     z = np.concatenate(z_list, axis=0)
#     label = np.concatenate(label_list)
#     plt.scatter(z[:, 0], z[:, 1], c=label)
#     plt.show()
