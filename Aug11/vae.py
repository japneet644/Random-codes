class VariantionalAutoencoder(object):

    def __init__(self, learning_rate=1e-4, batch_size=64, lattice_size=8):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.lattice_size = lattice_size
        self.n_latent = 5


        tf.reset_default_graph()
        self.build()

        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

    # Build the netowrk and the loss functions
    def build(self):
        self.x = tf.placeholder(name='x', dtype=tf.float32, shape=[lattice_size**2, 1])
        n_hidden = (lattice_size**2)//2
        We1 = tf.Variable(tf.random_normal(( lattice_size**2, n_hidden )))
        be1 = tf.Variable(tf.random_normal(( n_hidden , 1)))
        x_e_hidden = tf.nn.relu(tf.matmul(self.x, We1) + be1)


        We2 = tf.Variable(tf.random_normal(( n_hidden, 2*self.n_latent )))
        be2 = tf.Variable(tf.random_normal(( 2*self.n_latent ,1)))
        x_latent  =  tf.matmul(x_e_hidden, We2) + be2

        # f1 = fc(self.x, 256, scope='enc_fc1', activation_fn=tf.nn.relu)
        # f2 = fc(f1, 128, scope='enc_fc2', activation_fn=tf.nn.relu)
        # f3 = fc(f2, 64, scope='enc_fc3', activation_fn=tf.nn.relu)
        # self.z_mu = fc(f3, self.lattice_size, scope='enc_fc4_mu', activation_fn=None)
        # self.z_log_sigma_sq = fc(f3, self.lattice_size, scope='enc_fc4_sigma',activation_fn=None)
        self.latent_mu = x_latent[:n_latent]
        self.latent_sigma_sq = x_latent[n_latent:]
        eps = tf.random_normal(
            shape=tf.shape(self.z_latent_mu),
            mean=0, stddev=1, dtype=tf.float32)
        # self.z = self.z_mu + tf.sqrt(tf.exp(self.z_log_sigma_sq)) * eps
        self.z = self.latent_mu + tf.sqrt(self.latent_sigma_sq) * eps
        # Decode
        # z -> x_hat
        # g1 = fc(self.z, 64, scope='dec_fc1', activation_fn=tf.nn.elu)
        # g2 = fc(g1, 128, scope='dec_fc2', activation_fn=tf.nn.elu)
        # g3 = fc(g2, 256, scope='dec_fc3', activation_fn=tf.nn.elu)
        # self.x_hat = fc(g3, input_dim, scope='dec_fc4',
        #                 activation_fn=tf.sigmoid)
        Wd1 = tf.Variable(tf.random_normal(( n_latent, n_hidden )))
        bd1 = tf.Variable(tf.random_normal(( n_hidden , 1)))
        x_D  = tf.nn.relu(tf.matmul(self.z, Wd1) + bd1)

        Wd2 = tf.Variable(tf.random_normal(( n_hidden, self.lattice_size**2 )))
        bd2 = tf.Variable(tf.random_normal(( self.lattice_size**2 ,1)))
        x_output  =  tf.matmul(x_D, Wd2) + bd2
        # Loss
        # Reconstruction loss
        # Minimize the cross-entropy loss
        # H(x, x_hat) = -\Sigma x*log(x_hat) + (1-x)*log(1-x_hat)





        epsilon = 1e-10



        recon_loss = -tf.reduce_sum(
            self.x * tf.log(epsilon+self.x_output) +
            (1-self.x) * tf.log(epsilon+1-self.x_output),
            axis=1
        )
        self.recon_loss = tf.reduce_mean(recon_loss)

        # Latent loss
        # KL divergence: measure the difference between two distributions
        # Here we measure the divergence between
        # the latent distribution and N(0, 1)
        latent_loss = -0.5 * tf.reduce_sum(
            1 + self.z_log_sigma_sq - tf.square(self.z_mu) -
            tf.exp(self.z_log_sigma_sq), axis=1)
        self.latent_loss = tf.reduce_mean(latent_loss)

        self.total_loss = self.recon_loss + self.latent_loss
        self.train_op = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate).minimize(self.total_loss)

        self.losses = {
            'recon_loss': self.recon_loss,
            'latent_loss': self.latent_loss,
            'total_loss': self.total_loss,
        }
        return

    # Execute the forward and the backward pass
    def run_single_step(self, x):
        _, losses = self.sess.run(
            [self.train_op, self.losses],
            feed_dict={self.x: x}
        )
        return losses

    # x -> x_hat
    def reconstructor(self, x):
        x_hat = self.sess.run(self.x_hat, feed_dict={self.x: x})
        return x_hat

    # z -> x
    def generator(self, z):
        x_hat = self.sess.run(self.x_hat, feed_dict={self.z: z})
        return x_hat

    # x -> z
    def transformer(self, x):
        z = self.sess.run(self.z, feed_dict={self.x: x})
        return z
