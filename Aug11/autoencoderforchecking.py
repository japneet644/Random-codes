import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import data_loader
import pickle, pprint
# import PIL.Image as pil
# from PIL import Image, ImageDraw, ImageFont
import random
import math
import get_parameters

lattice_size = 16
learning_rate=1e-4
batch_size=20
n_z=2
x = tf.placeholder(name='x', dtype=tf.float32, shape=[None, 256])
#Encoder
Ehidden1 = tf.contrib.layers.fully_connected(inputs=x, num_outputs=120, activation_fn=tf.nn.relu,scope="Fully_Conn1")
Ehidden2 = tf.contrib.layers.fully_connected(inputs=Ehidden1, num_outputs=20, activation_fn=tf.nn.relu,scope="Fully_Conn2")
z             = tf.contrib.layers.fully_connected(inputs=Ehidden2, num_outputs=n_z, activation_fn=None,scope="Fully_Conn3_mu")
# z_log_sigma_sq   = tf.contrib.layers.fully_connected(inputs=Ehidden2, num_outputs=n_z, activation_fn=None,scope="Fully_Conn3_sig")

# eps = tf.random_normal(shape=tf.shape(z_log_sigma_sq),mean=0, stddev=1, dtype=tf.float32)
# z = z_mu + tf.sqrt(tf.exp(z_log_sigma_sq)) * eps

#decoder
Dhidden1 = tf.contrib.layers.fully_connected(inputs=z, num_outputs=20, activation_fn=tf.nn.relu,scope="Fully_Conn1_decoder")
Dhidden2 = tf.contrib.layers.fully_connected(inputs=Dhidden1, num_outputs=120, activation_fn=tf.nn.relu,scope="Fully_Conn2_decoder")
x_hat = tf.contrib.layers.fully_connected(inputs=Dhidden2, num_outputs=256, activation_fn=tf.sigmoid,scope="Fully_Conn3_decoder")

# Reconstruction Loss

epsilon = 1e-10
recon_loss = tf.reduce_sum((x-x_hat)**2)#-1*tf.reduce_sum(    x * tf.log(epsilon+x_hat) +(1-x) * tf.log(epsilon+1-x_hat),axis=1)
recon_loss = tf.reduce_mean(recon_loss)

# Latent loss

# KL_loss = -0.5 * tf.reduce_sum(    1 + z_log_sigma_sq - tf.square(z_mu) -tf.exp(z_log_sigma_sq), axis=1)
# KL_loss = tf.reduce_mean(KL_loss)

# total_loss = recon_loss + KL_loss
train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(recon_loss)

losses = {
    'recon_loss': recon_loss
}

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

saver = tf.train.Saver()
with tf.Session() as sess:
    # saver.restore(sess,'./AE.ckpt')
    training_data = data_loader.load_data_wrapper()
    tvals = np.repeat(np.linspace(0.1,2.0,32),10000)
    c = list(zip(training_data,tvals))
    random.shuffle(c)
    training_data, tvals = zip(*c)
    a = tf.placeholder(tf.float32,[320000, 256])
    dataset = tf.data.Dataset.from_tensor_slices(a)
    dataset = dataset.prefetch(buffer_size=1000)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_initializable_iterator()
    next = iterator.get_next()

    sess.run(tf.global_variables_initializer())

    print("Session initialized :)")
    sess.run(iterator.initializer, feed_dict = {a:training_data})
    print("Iterator initialized :)")

    for i in range(10000):
            # print(i)
        if i>0 and i % (320000 // batch_size) == 0:
            sess.run(iterator.initializer, feed_dict = {a:training_data})
        b = sess.run(next)
        _, Losses = sess.run([train_op, losses],feed_dict={x: b })
        if i%1000==0:
            print(Losses)
            Gsample  = sess.run(Ehidden1, feed_dict={x: training_data[0:100]})
            Gsample2 = sess.run(Ehidden2, feed_dict={Ehidden1: Gsample})
            gsample  = sess.run(z,    feed_dict={Ehidden2:Gsample2})
            fig = plt.figure()
            ax = fig.add_subplot(111)
            for u,y in zip(gsample,tvals[:100]):
                ax.scatter(u[0],u[1],label='%f' %y)
            plt.legend(loc='best')
            plt.savefig("./out/ae_%d.png" % i)
            plt.close()
            print('fig svaed in ./out/ae_%d.png' % y)
# plot x,y data with c as the color vector, set the line width of the markers to 0
    save_path = saver.save(sess, "./AE.ckpt")
    print("Model saved in path: %s" % save_path)
    # if n_z == 1:
    #     T_vals = np.linspace(0.01,2.5,20)
    #     n_samples = 30      # int(np.shape(zsample)[0]**(1/n_z))
    #     zsample = np.linspace(-2,2,n_samples).reshape(n_samples,1) #np.random.random([10,n_z])
    # if n_z == 2:
    #     zsample =  np.mgrid[-5:5:1, -5:5:1]
    #     n_samples = (10 ,10)
    #     zsample = zsample.reshape(2,-1).T
    #
    #
    # print(gsample.shape)
    #
    # gsample = gsample.reshape(zsample.shape[0],lattice_size,lattice_size)
    # print("Specific Heat %f")
    # # for i in range(n_samples):
    #     # print(get_parameters.get_specific_heat(gsample[i],zsample[i]))
    # print("Mean magnetization and its Standard Deviation")
    # mean_magnetization = []
    # Magnetization = get_parameters.get_mean_magnetization(gsample)
    # Magnetization_direction = get_parameters.get_magnetization_direction(gsample)
    # energy = get_parameters.get_energy(gsample)
    # if n_z == 1:
    #     plt.plot(zsample,Magnetization[0][0]) #all values
    #     plt.xlabel('Latent Variable value', fontsize=12)
    #     plt.ylabel('Magnetization of a single sample generated by the network', fontsize=10)
    #     plt.savefig('mag_vs_temp.png')
    #     plt.show()
    #     plt.plot(zsample,energy)
    #     plt.savefig('energy_vs_LV.png')
    #     plt.show()
    # elif n_z == 2:
    #     # LV[0] Magnetization LV[1] Temperature
    #     for x in range(n_samples[0]):
    #         plt.plot(zsample[x*n_samples[1]:(x+1)*n_samples[1]][:,1],Magnetization[0][0][x*n_samples[1]:(x+1)*n_samples[1]])#,label = 'LV[0] = %f' %(zsample[x*n_samples[1]][0]),linestyle='dotted')
    #     for x in range(n_samples[1]):
    #         mean_magnetization.append(np.mean(Magnetization[0][0][x:zsample.shape[0]:n_samples[1]]))
    #     plt.plot(zsample[0:n_samples[1]][:,1],mean_magnetization,label = 'Mean ',linestyle = '--')
    #     plt.xlabel('Latent Variable value', fontsize=12)
    #     plt.ylabel('Magnetization of a single sample generated by the network', fontsize=10)
    #     plt.legend(loc='best')
    #     plt.savefig('mag_vs_temp.png')
    #     plt.show()
    #     for x in range(n_samples[1]):
    #         plt.plot(zsample[0:zsample.shape[0]:n_samples[1],0],Magnetization_direction[x:zsample.shape[0]:n_samples[1]],label='LV[1] = %f' %(zsample[x][1]))
    #     plt.xlabel('Rotational Latent Variable value', fontsize=12)
    #     plt.ylabel('Magnetization vector', fontsize=10)
    #     plt.legend(loc='best')
    #     plt.savefig('mag_dir_vs_temp.png')
    #     plt.show()
    #     for x in range(n_samples[1]):
    #         plt.hist(energy[x*n_samples[1]:(x+1)*n_samples[1]],bins = 30, label='LV[1] = %f' %(zsample[x][1]))
    #     plt.xlabel('Energy', fontsize=12)
    #     plt.ylabel('No of times')
    #     plt.legend(loc='best')
    #     plt.savefig('energy_vs_temp.png')
    #     plt.show()
        # draw_grid(8,gsample[30*i],zsample[30*i])
