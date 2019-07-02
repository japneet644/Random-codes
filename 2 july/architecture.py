import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import data_loader
import pickle
import gzip
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
Is_train = False
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
KL_loss = -2.5* tf.reduce_sum(    1 + z_log_sigma_sq - tf.square(z_mu) -tf.exp(z_log_sigma_sq), axis=1)
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
def return_intersection(hist_1, hist_2):
    minima = np.minimum(hist_1, hist_2)
    intersection = np.true_divide(np.sum(minima), np.sum(hist_2))
    return intersection

saver = tf.train.Saver()
with tf.Session() as sess:
    if Is_train == False:
        saver.restore(sess,'./VAE_xy2.ckpt')
        # writer = tf.summary.FileWriter('./graphs', sess.graph)
    if Is_train == True:
        training_data = data_loader.load_data_wrapper()
        tvals = np.repeat(np.linspace(0.1,2.0,32),10000)
        c = list(zip(training_data,tvals))
        random.shuffle(c)
        training_data, tvals = zip(*c)
        m = tf.placeholder(tf.float32,[datapoints, 64])
        n = tf.placeholder(tf.float32,[datapoints, 1])
        dataset = tf.data.Dataset.from_tensor_slices((m,n))
        dataset = dataset.prefetch(buffer_size=1000)
        dataset = dataset.batch(batch_size)
        iterator = dataset.make_initializable_iterator()
        next = iterator.get_next()

        print("============< WARNING >===============")
        sess.run(tf.global_variables_initializer())
        print("==========< Model DELETED >===========")

        print("Session initialized :)")
        print("Iterator initialized :)")
        sess.run(iterator.initializer, feed_dict = {m:training_data,n:np.array(tvals).reshape(datapoints,1) + 0.01*np.random.randn(datapoints,1)})

        for i in range(50000):
                # print(i)
            if i>0 and i % (datapoints // batch_size) == 0:
                sess.run(iterator.initializer, feed_dict = {m:training_data, n:np.array(tvals).reshape(datapoints,1)+ 0.01*np.random.randn(datapoints,1) })
            g,h = sess.run(next)
            _, Losses = sess.run([train_op, losses],feed_dict={x: g,y:h })
            if i%1000==0:
                print(Losses)
        save_path = saver.save(sess, "./VAE_xy2.ckpt")
        print("Model saved in path: %s" % save_path)

    if n_z == 1:
        T_vals  = np.linspace(0.1,2.0,32)
        zsample = np.mgrid[0.0:2.0:0.0625, -2.0:2.0:0.01].reshape(2,-1).T #np.linspace(-5,3,n_samples).reshape(n_samples,1)
        n_samples = (32,400)
        if loss_type == 'Binary_crossentropy':
            Gsample    = sess.run(fully_connected_decoder1, feed_dict={z: zsample[:,1].reshape(zsample.shape[0],1), z_temp:zsample[:,0].reshape(zsample.shape[0],1) + 0.01*np.random.randn(zsample.shape[0],1)})
            Gsample2   = sess.run(fully_connected_decoder2, feed_dict={fully_connected_decoder1: Gsample  })
            gsample    = sess.run(x_hat,                    feed_dict={fully_connected_decoder2: Gsample2 })
        elif loss_type == 'log_gaussian':
            Gsample    = sess.run(fully_connected_decoder1, feed_dict={z: zsample[:,1].reshape(zsample.shape[0],1), z_temp:zsample[:,0].reshape(zsample.shape[0],1) + 0.01*np.random.randn(zsample.shape[0],1)})  #np.array(tvals[:n_samples]).reshape(n_samples,1)}
            Gsample2   = sess.run(fully_connected_decoder2, feed_dict={fully_connected_decoder1: Gsample})
            Gsamplemu  = sess.run(x_mu,                     feed_dict={fully_connected_decoder2: Gsample2})
            Gsamplesig = sess.run(x_log_sigma_sq,           feed_dict={fully_connected_decoder2:Gsample2})
            gsample    = sess.run(tf.random_normal(shape = tf.shape(Gsamplemu),mean = Gsamplemu,stddev = tf.sqrt(tf.exp(-Gsamplesig)), dtype = tf.float32 ))#tf.sqrt(tf.exp(Gsamplesig))

    gsample = gsample.reshape(zsample.shape[0],lattice_size,lattice_size)

    Mdist = []
    Edist = []
    mean_magnetization = []
    var_magnetization = []
    mean_magnetization_data = []
    var_magnetization_data = []
    mean_energy = []
    var_energy = []
    mean_energy_data = []
    var_energy_data = []

    n = n_samples[1]
    if Is_train == False:
        f = open('./DATA/8by8lattices.pkl', 'rb')
        if (f.read(2) == '\x1f\x8b'):
            f.seek(0)
            gzip.GzipFile(fileobj=f)
        else:
            f.seek(0)
        training_data = pickle.load(f, encoding="latin1")
        training_data = np.reshape(training_data,(320000, 64))
    for i in range(0,n_samples[0]):
        Magnetization           = get_parameters.get_mean_magnetization(gsample[i*n_samples[1]:(i+1)*n_samples[1]])
        Magnetization_direction = get_parameters.get_magnetization_direction(gsample[i*n_samples[1]:(i+1)*n_samples[1]])
        energy                  = get_parameters.get_energy(gsample[i*n_samples[1]:(i+1)*n_samples[1]])
        print(i)
        if Is_train == True:
            fig1 = plt.figure(1)
            plt.plot(zsample[:n_samples[1],1],Magnetization[0][0],label = (T_vals[i]))
            fig2 = plt.figure(2)
            plt.plot(zsample[:n_samples[1],1],Magnetization_direction,label = (T_vals[i]))
        if Is_train == False:
            lattices = np.array(training_data[i*10000:i*10000+n]).reshape(n,lattice_size,lattice_size)
            energy_data = get_parameters.get_energy(lattices)
            thetas_data = get_parameters.get_magnetization_direction(lattices)
            [mag_data,mag_mean,mag_std] = get_parameters.get_mean_magnetization(lattices)
            plt.subplot(3,1,1)
            plt.hist(Magnetization[0][0][i*n_samples[1]:(i+1)*n_samples[1]],bins =20,color='b',range=[0, 1],alpha=0.5)
            plt.hist(mag_data           ,bins =20,color='g',range=[0, 1],alpha=0.5)
            plt.ylabel('Magnetization ')

            mean_magnetization.append(Magnetization[1])
            var_magnetization.append(Magnetization[2])
            mean_magnetization_data.append(mag_mean)
            var_magnetization_data.append(mag_std)

            mean_energy.append(np.mean(energy))
            var_energy.append(np.std(energy))
            mean_energy_data.append(np.mean(energy_data))
            var_energy_data.append(np.std(energy_data))

            plt.subplot(3,1,2)
            plt.plot(Magnetization_direction,linestyle='dotted',color='b')
            plt.plot(thetas_data,            linestyle='dotted',color='g')
            plt.ylabel('Magnetization direction')
            plt.ylim((-360,0))
            # plt.title('Magnetization direction')

            plt.subplot(3,1,3)
            plt.hist(energy     ,bins =300,color='b',range=[-130, 20],alpha =0.5)
            plt.hist(energy_data,bins =300,color='g',range=[-130, 20],alpha=0.5)
            plt.ylabel('Energy')

            # plt.show()
            plt.savefig('./out/combined@ %f.png'%((T_vals[i])), bbox_inches='tight')
            plt.close()
            Mhist_1,_ = np.histogram(Magnetization[0][0],bins =20,range=[0, 1])
            Mhist_2,_ = np.histogram(mag_data           ,bins =20,range=[0, 1])
            Mdist.append(return_intersection(Mhist_1,Mhist_2))

            Ehist_1,_ = np.histogram(energy     ,bins =300,range=[-130, 20])
            Ehist_2,_ = np.histogram(energy_data,bins =300,range=[-130, 20])
            Edist.append(return_intersection(Ehist_1,Ehist_2))
    if Is_train ==True:
        plt.xlabel('Latent Variable value', fontsize=12)
        plt.ylabel('Magnetization of a single sample generated by the network', fontsize=10)
        plt.legend()
        plt.show()
    if Is_train == False:
        print("Magnetization Accuracy")
        print(Mdist)
        print(mean_magnetization)
        print(mean_magnetization_data)
        print(var_magnetization)
        print(var_magnetization_data)
        plt.errorbar(T_vals,mean_magnetization,var_magnetization,color='b',label='Samples')
        plt.errorbar(T_vals,mean_magnetization_data,var_magnetization_data,color = 'g',label='Data')
        plt.xlabel("Temperature")
        plt.ylabel('Magnetization')
        plt.title('Vanilla VAE')
        plt.legend()
        plt.savefig('../../Desktop/Vanilla_VAE-Magnetization.png', bbox_inches='tight')
        plt.show()

        print("Energy Accuracy")
        print(Edist)
        print(mean_energy)
        print(mean_energy_data)
        print(var_energy)
        print(var_energy_data)
        plt.errorbar(T_vals,mean_energy,var_energy,color='b',label='Samples')
        plt.errorbar(T_vals,mean_energy_data,var_energy_data,color = 'g',label='Data')
        plt.xlabel("Temperature")
        plt.ylabel('Energy')
        plt.title('Vanilla_VAE')
        plt.legend()
        plt.savefig('../../Desktop/Vanilla_VAE-Energy.png', bbox_inches='tight')
        plt.show()

        print("Specfic Heat")
        plt.plot(T_vals,(np.array(var_energy)**2)/(T_vals**2),color='b',label='Samples')
        plt.plot(T_vals,(np.array(var_energy_data)**2)/(T_vals**2),color='g',label='Data')
        plt.ylabel("Specific Heat")
        plt.xlabel('Temperature')
        plt.title('Vanilla_VAE')
        plt.legend()
        plt.savefig('../../Desktop/Vanilla_VAE-Specific Heat.png', bbox_inches='tight')
        plt.show()

        print("Magnetic Susceptibility")
        plt.plot(T_vals,(np.array(var_magnetization)**2)/(T_vals),color='b',label='Samples')
        plt.plot(T_vals,(np.array(var_magnetization_data)**2)/(T_vals),color='g',label='Data')
        plt.ylabel("Magnetic Susceptibility")
        plt.xlabel('Temperature')
        plt.title('Vanilla_VAE')
        plt.legend()
        plt.savefig('../../Desktop/Vanilla_VAE-Magnetic_Susceptibility.png', bbox_inches='tight')
        plt.show()

    # if n_z == 1:
    #     for d in range(n_samples[0]):
    #         plt.plot(zsample[:n_samples[1],1],Magnetization[0][0][d*n_samples[1]:(d+1)*n_samples[1]],label = d) #all values
    #     plt.xlabel('Latent Variable value', fontsize=12)
    #     plt.ylabel('Magnetization of a single sample generated by the network', fontsize=10)
    #     plt.legend()
    #     plt.show()
    #     for d in range(n_samples[0]):
    #         plt.plot(zsample[:n_samples[1],1],Magnetization_direction[d*n_samples[1]:(d+1)*n_samples[1]])
    #     plt.show()
    #     plt.hist(energy,bins =100)
    #     plt.legend()
    #     plt.show()
    #     n_maps = 2000 #no of mappings per temp
    #     for u in range(0,20,2):
    #         sampledz = sess.run(z,feed_dict={x:training_data[10000*u+500:10000*u+500+n_maps],y:np.array(tvals[10000*u+500:10000*u+500+n_maps]).reshape(n_maps,1)-1})
    #         plt.hist(sampledz,bins=100,label = u)
    #     plt.legend()
    #     plt.show()
    #     for d in range(0,n_samples[1]):
    #         plt.plot(zsample[d:zsample.shape[0]:n_samples[1],0],Magnetization[0][0][d:zsample.shape[0]:n_samples[1]],label = d)
    #     plt.legend(loc='best')
    #     plt.show()

    # elif n_z == 2:
    #     # ######## LV[0] Magnetization LV[1] Temperature
    #     for g in range(n_samples[0]):
    #         plt.plot(zsample[g*n_samples[1]:(g+1)*n_samples[1]][:,1],Magnetization[0][0][g*n_samples[1]:(g+1)*n_samples[1]])#,label = 'LV[0] = %f' %(zsample[g*n_samples[1]][0]),linestyle='dotted')
    #     for g in range(n_samples[1]):
    #         mean_magnetization.append(np.mean(Magnetization[0][0][g:zsample.shape[0]:n_samples[1]]))
    #     plt.plot(zsample[0:n_samples[1]][:,1],mean_magnetization,label = 'Mean ',linestyle = '--')
    #     plt.xlabel('Latent Variable value', fontsize=12)
    #     plt.ylabel('Magnetization of a single sample generated by the network', fontsize=10)
    #     plt.legend(loc='best')
    #     plt.savefig('mag_vs_temp.png')
    #     plt.show()
    #     for g in range(n_samples[1]):
    #         plt.plot(zsample[0:zsample.shape[0]:n_samples[1],0],Magnetization_direction[g:zsample.shape[0]:n_samples[1]],label='LV[1] = %f' %(zsample[g][1]))
    #     plt.xlabel('Rotational Latent Variable value', fontsize=12)
    #     plt.ylabel('Magnetization vector', fontsize=10)
    #     plt.legend(loc='best')
    #     plt.savefig('mag_dir_vs_temp.png')
    #     plt.show()
    #     for g in range(n_samples[1]):
    #         plt.hist(energy[g*n_samples[1]:(g+1)*n_samples[1]],bins = 30, label='LV[1] = %f' %(zsample[g][1]))
    #     plt.xlabel('Energy', fontsize=12)
    #     plt.ylabel('No of times')
    #     plt.legend(loc='best')
    #     plt.savefig('energy_vs_temp.png')
    #     plt.show()
    #     # draw_grid(8,gsample[30*i],zsample[30*i])
    #     #Just do reverse LV[1] Magnetization direction LV[0] Temperature
    #     for g in range(n_samples[1]):
    #         plt.plot(zsample[g:zsample.shape[0]:n_samples[1]][:,0],Magnetization[0][0][g:zsample.shape[0]:n_samples[1]])#,label = 'LV[0] = %f' %(zsample[g*n_samples[1]][0]),linestyle='dotted')
    #     # for g in range(n_samples[]):
    #     #     mean_magnetization.append(np.mean(Magnetization[0][0][g:zsample.shape[0]:n_samples[1]]))
    #     # plt.plot(zsample[0:n_samples[1]][:,1],mean_magnetization,label = 'Mean ',linestyle = '--')
    #     plt.xlabel('Latent Variable value', fontsize=12)
    #     plt.ylabel('Magnetization of a single sample generated by the network', fontsize=10)
    #     plt.legend(loc='best')
    #     plt.savefig('mag_vs_temp.png')
    #     plt.show()
    #     for x in range(n_samples[0]):
    #         plt.plot(zsample[g*n_samples[1]:(g+1)*n_samples[1]][:,1],Magnetization_direction[g*n_samples[0]:(g+1)*n_samples[0]],label='LV[1] = %f' %(zsample[g][1]))
    #     plt.xlabel('Rotational Latent Variable value', fontsize=12)
    #     plt.ylabel('Magnetization vector', fontsize=10)
    #     plt.legend(loc='best')
    #     plt.savefig('mag_dir_vs_temp.png')
    #     plt.show()
    #     for g in range(n_samples[1]):
    #         plt.hist(energy[g*n_samples[1]:(g+1)*n_samples[1]],bins = 30, label='LV[1] = %f' %(zsample[g][1]))
    #     plt.xlabel('Energy', fontsize=12)
    #     plt.ylabel('No of times')
    #     plt.legend(loc='best')
    #     plt.savefig('energy_vs_temp.png')
    #     plt.show()
    #     n_maps = 2000 #no of mappings per temp
    #     for u in range(0,32,4):
    #         sampledz = sess.run(z,feed_dict={x:training_data[10000*u+500:10000*u+500+n_maps]})
    #         plt.plot(10*sampledz[:,0],10*sampledz[:,1],linestyle='dotted',label = u)
    #     plt.legend()
    #     plt.show()
        # for u in range(5,10):
        #     sampledmu = sess.run(z_mu,feed_dict={x:training_data[10000*u+500:10000*u+500+n_maps]})
        #     sampledsigma = sess.run(z_log_sigma_sq,feed_dict={x:training_data[10000*u+500:10000*u+500+n_maps]})
        #     plt.plot(sampledz[:,0],sampledz[:,1],linestyle='dotted',label = u)
        # plt.legend()
        # plt.show()
