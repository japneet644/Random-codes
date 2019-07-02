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
T_vals = np.linspace(0,2,n_temps)
fully_connected1 = tf.contrib.layers.fully_connected(inputs=x, num_outputs=100, activation_fn=tf.nn.relu,scope="Fully_Conn1")
fully_connected2 = tf.contrib.layers.fully_connected(inputs=fully_connected1, num_outputs=20, activation_fn=tf.nn.relu,scope="Fully_Conn2")
z_mu             = tf.contrib.layers.fully_connected(inputs=fully_connected2, num_outputs=n_z, activation_fn=None,scope="Fully_Conn2_mu")
z_log_sigma_sq   = tf.contrib.layers.fully_connected(inputs=fully_connected2, num_outputs=n_z, activation_fn=None,scope="Fully_Conn2_sig")
z_temp           = tf.contrib.layers.fully_connected(inputs=fully_connected2, num_outputs=n_z, activation_fn=tf.nn.relu,scope="Fully_Conn2_temp")
eps = tf.random_normal(shape=tf.shape(z_log_sigma_sq),mean=0, stddev=1, dtype=tf.float32)
z   = z_mu + tf.sqrt(tf.exp(z_log_sigma_sq)) * eps

#decoder
epsilon = 1e-10
fully_connected_decoder1 = tf.contrib.layers.fully_connected(inputs=tf.concat([z_temp,z],axis=1), num_outputs=20, activation_fn=tf.nn.relu,scope="Fully_Conn1_decoder")
fully_connected_decoder2 = tf.contrib.layers.fully_connected(inputs=fully_connected_decoder1, num_outputs=100, activation_fn=tf.nn.relu,scope="Fully_Conn2_decoder")

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
# Latent loss
KL_loss = -1* tf.reduce_sum(    1 + z_log_sigma_sq - tf.square(z_mu) -tf.exp(z_log_sigma_sq), axis=1)
KL_loss = tf.reduce_mean(KL_loss)
temp_loss = 50*tf.reduce_mean((y-z_temp)**2)
total_loss = recon_loss + KL_loss+temp_loss
train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(total_loss)

losses = {
    'recon_loss': recon_loss,
    'total_loss': total_loss,
    'KL_loss':  KL_loss,
    'temp_loss':temp_loss
}


saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess,'./VAE_xy2.ckpt')
    training_data = data_loader.load_data_wrapper()
    tvals = np.repeat(np.linspace(0.1,2.0,n_temps),10000)
    c = list(zip(training_data,tvals))
    training_data, tvals = zip(*c)
    m = tf.placeholder(tf.float32,[datapoints, 64])
    n = tf.placeholder(tf.float32,[datapoints, 1])
    dataset = tf.data.Dataset.from_tensor_slices((m,n))
    dataset = dataset.prefetch(buffer_size=1000)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_initializable_iterator()
    next = iterator.get_next()

    print("Session initialized :)")
    sess.run(iterator.initializer, feed_dict = {m:training_data,n:np.array(tvals).reshape(datapoints,1)})
    print("Iterator initialized :)")

    n_samples = 500
    print(n_samples)

    for j in range(n_temps):
        zsample = (np.random.randn(n_samples,1)) #sess.run(z, feed_dict={x: np.array(training_data[10000*j: 10000*j+500]).reshape(500,64),y:np.array(tvals[10000*j: 10000*j+500]).reshape(500,1)}) #
        if loss_type == 'Binary_crossentropy':
            # not corrected code is wrong here
            Gsample    = sess.run(fully_connected_decoder1, feed_dict={z: zsample, z_temp:zsample[:,0].reshape(zsample.shape[0],1)})
            Gsample2   = sess.run(fully_connected_decoder2, feed_dict={fully_connected_decoder1: Gsample  })
            gsample    = sess.run(x_hat,                    feed_dict={fully_connected_decoder2: Gsample2 })
        elif loss_type== 'log_gaussian':
            Gsample    = sess.run(fully_connected_decoder1, feed_dict={z: zsample, z_temp: np.array(tvals[j*10000:j*10000+n_samples]).reshape(n_samples,1)} )#np.array(tvals[:n_samples]).reshape(n_samples,1)}
            Gsample2   = sess.run(fully_connected_decoder2, feed_dict={fully_connected_decoder1: Gsample})
            Gsamplemu  = sess.run(x_mu,                     feed_dict={fully_connected_decoder2: Gsample2})
            Gsamplesig = sess.run(x_log_sigma_sq,           feed_dict={fully_connected_decoder2:Gsample2})
            gsample    = sess.run(tf.random_normal(shape = tf.shape(Gsamplemu),mean = Gsamplemu,stddev = tf.sqrt(tf.exp(-1*Gsamplesig)), dtype = tf.float32 ))#tf.sqrt(tf.exp(Gsamplesig))


        gsample = gsample.reshape(zsample.shape[0],lattice_size,lattice_size)

        print(tvals[j*10004],360*np.mean(Gsamplemu),360*np.std(Gsamplemu))
        print(360*np.mean(-1*Gsamplesig),360*np.std(-1*Gsamplesig))
        print("Mean magnetization and its Standard Deviation")

        mean_magnetization = []
        Magnetization = get_parameters.get_mean_magnetization(gsample)
        Magnetization_direction = get_parameters.get_magnetization_direction(gsample)
        energy = get_parameters.get_energy(gsample)
        # print(Magnetization)
        temp = T_vals[j]
        lattices = np.array(training_data[j*10000:j*10000+n_samples]).reshape(n_samples,lattice_size,lattice_size)
        energy_data = (get_parameters.get_energy(lattices))
        thetas_data = (get_parameters.get_magnetization_direction(lattices))
        [mag_data,mag_mean,mag_std]=get_parameters.get_mean_magnetization(lattices)
        # sp_heat.append(get_specific_heat(lattices,temp))

        plt.subplot(3,1,1)
        plt.hist(Magnetization[0][0],bins =20,color='b',range=[0, 1],alpha=0.5)
        plt.hist(mag_data           ,bins =20,color='g',range=[0, 1],alpha=0.5)
        plt.ylabel('Magnetization ')

        plt.subplot(3,1,2)
        plt.plot(Magnetization_direction,linestyle='dotted',color='b')
        plt.plot(thetas_data,            linestyle='dotted',color='g')
        plt.ylabel('Magnetization direction')
        plt.ylim((-360,0))
        # plt.title('Magnetization direction')

        plt.subplot(3,1,3)
        plt.hist(energy     ,bins =300,color='b',range=[-130, 20],alpha =0.5)
        plt.hist(energy_data,bins =300,color='g',range=[-130, 20],alpha=0.5)
        plt.title('Energy')

        # plt.show()
        plt.savefig('./out/combined@ %f.png'%T_vals[j], bbox_inches='tight')
        plt.close()

####################################
#|||||||||||||||||||||||||||||||||||
###===Evaluation OF DATA ===========
#|||||||||||||||||||||||||||||||||||
####################################
l=8
lattice_shape=(l,l)
nsamples=300
index_set=range(0,32,1)

energy = []
S=[]
sp_heat=[]
mag=[]
mag_err=[]
mag_std =[]
thetas=[]

######################################
#########Opening saved data###########
######################################


# # #################################
# # #####Observing vortices#########
# # ###############################
# # data=(get_vorticity_configuration(allTlattices[20][9999])) #first index indicates the temperature index, second index is a no between 1-10000
# # im = plt.imshow(data, interpolation='none')
# # plt.figure(figsize=(8,4))
# # values=range(-7,8)
# # colors = [ im.cmap(im.norm(value)) for value in values]
# # patches = [ mpatches.Patch(color=colors[i], label="Level {l}".format(l=values[i]) ) for i in range(len(values)) ]
# # plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
# # plt.grid(True)
# # plt.show()

###########################
######Specific Heat########
###########################
# print(sp_heat)
# plt.plot(T_vals,sp_heat)
# plt.xlabel('Temperature')
# plt.ylabel('Specific Heat')
# plt.show()
################################
#########Magnetization##########
################################
""""
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
n_z           = 2
x = tf.placeholder(name='x', dtype=tf.float32, shape=[None, 64])
y = tf.placeholder(name='y', dtype=tf.float32, shape=[None,1])
loss_type  = 'log_gaussian' #'Binary_crossentropy'
datapoints = 320000
n_temps    = 32
alpha      = -1
LAMBDA     = 5
T_vals = np.linspace(0,2,n_temps)

fully_connected1 = tf.contrib.layers.fully_connected(inputs=x, num_outputs=100, activation_fn=tf.nn.relu,scope="Fully_Conn1")
fully_connected2 = tf.contrib.layers.fully_connected(inputs=fully_connected1, num_outputs=20, activation_fn=tf.nn.relu,scope="Fully_Conn2")
z_mu             = tf.contrib.layers.fully_connected(inputs=fully_connected2, num_outputs=n_z, activation_fn=None,scope="Fully_Conn2_mu")
z_log_sigma_sq   = tf.contrib.layers.fully_connected(inputs=fully_connected2, num_outputs=n_z, activation_fn=None,scope="Fully_Conn2_sig")

eps = tf.random_normal(shape=tf.shape(z_log_sigma_sq),mean=0, stddev=1, dtype=tf.float32)
z   = z_mu + tf.sqrt(tf.exp(z_log_sigma_sq)) * eps

#decoder
epsilon = 1e-10
fully_connected_decoder1 = tf.contrib.layers.fully_connected(inputs=z, num_outputs=20, activation_fn=tf.nn.relu,scope="Fully_Conn1_decoder")#tf.concat([y,z],axis=1)
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
    return tf.reduce_mean(x_kernel) + tf.reduce_mean(y_kernel) - 2 * tf.reduce_mean(xy_kernel)

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
"""
