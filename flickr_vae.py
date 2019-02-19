# -*- coding: utf-8 -*-

'''
Create a VAE model and train on flickr images.
Train/test data are read from python pickle files (created by
flickr_get_train-test.py). The encoder is saved and used for
dimensionality reduction to analyze/visualize the distribution
of different image labels and suggest labels for new images
(see webapp.py).

Note: if you are running the script on a machine with limited
memory it is recommended to use flickr_vae_gen.py, which loads
the images from the disk using a generator to train the model.
'''

# the following is the directory where the train/test pickle files
# are loacated and to which the model is saved
maindir='/scratch/mkayvan/flickr/normalized2000ma'

import keras.models as km
import keras.layers as kl
import keras.backend as K
import keras.utils as ku
import numpy as np
import sklearn.model_selection as skms
import flickrutils
import imageutils
import vae_model

labels = flickrutils.get_labels(maindir+'/labels.txt')
x_test = np.load(maindir+'/x_test.npy')
x_train = np.load(maindir+'/x_train.npy')
y_test = np.load(maindir+'/y_test.npy')
y_train = np.load(maindir+'/y_train.npy')

#print(labels)

x_train = np.float32(x_train)/255.0
x_test = np.float32(x_test)/255.0

(encoder,decoder,vae,z_log_var,z_mean) = vae_model.get_model(input_shape=(128,128,3), nlayers=2,nfm=20,d_lspace=100)

######################################
def vae_loss(input_image, output_image):
    """The variational autoencoder loss function.  This function will
    calculate the mean squared error for the resconstruction loss, and the
    KL divergence of Q(z|X)."""

    # Calculate the KL divergence loss.
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = -0.5 * K.sum(kl_loss, axis = -1)
    #kl = 0.5 * K.sum(K.exp(log_sigma) + K.square(mu) - 1. - log_sigma, axis=-1)
    
    # Calculate the mean squared error, and use it for the
    # reconstruction loss.
    reconstruction_loss = 128*128*3 * K.mean(K.square(input_image - output_image))
    #recon = K.sum(K.binary_crossentropy(y_pred, y_true), axis=1)
    
    # Return the sum of the two loss functions.
    return(reconstruction_loss + kl_loss)

########################################

# Compile the model.
vae.compile(loss = vae_loss,
            optimizer = 'rmsprop',
            metrics = ['accuracy'])

# And fit.
fit = vae.fit(x_train, x_train, epochs = 100,
              batch_size = 64,
              verbose = 2)

# Check the test data.
score = vae.evaluate(x_test, x_test)
print('score is', score)
                      
# Save the models
vae.save(maindir+'/vae.h5')
encoder.save(maindir+'/vae_encoder.h5')
decoder.save(maindir+'/vae_decoder.h5')


