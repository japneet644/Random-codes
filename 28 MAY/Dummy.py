# import data_loader
# training_data = data_loader.load_data_wrapper()
###########################################
#=========================================

# import vae_github
# VAE = vae_github.VariantionalAutoencoder()
# VAE.run_single_step()

#===========================================
############################################
# import VAE_CNN
# VAE = VAE_CNN.VAEcnn()
# VAE.run_single_step()

#===========================================
############################################
import VAE_mnist
import mnist_loader
import numpy as np
import random

training_data = mnist_loader.load_data_wrapper()
# VAE = VAE_mnist.VariantionalAutoencoder()
#training_data is (50,000,2) where first element is of shape 784,1  and 2nd is of shape 10,1
print("started")
random.shuffle(training_data)
VAE.run_single_step(training_data)
