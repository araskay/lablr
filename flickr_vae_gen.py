# -*- coding: utf-8 -*-

'''
Create a VAE model and train on flickr images.
The list of images, including the path to images and labels,
is read from a csv file.
The encoder is saved and used for dimensionality reduction
to analyze/visualize the distribution of different image labels
and suggest labels for new images (see webapp.py).

Note: if you feel confident about your computational resources,
e.g., if running on HPC clusters, you can use flickr_vae.py,
which loads all the data to the memory for slightly faster training.
'''

# the following is the directory where the model and also the train/test sets are saved
maindir='/scratch/mkayvan/flickr/normalized2000ma_subset'

# the following is the csv file containing the list of images (create using flickr_resize.py)
imagelist = '/scratch/mkayvan/flickr/normalized2000ma_10labels.csv'

import keras.models as km
import keras.layers as kl
import keras.backend as K
import keras.utils as ku
import numpy as np
#import sklearn.model_selection as skms
import flickrutils
#import imageutils
import cv2
import pandas as pd
import vae_gen_lib, vae_model
import fileutils


K.set_image_data_format('channels_last')

# generate train/test sets
np.random.seed(0) # for reproducibility

data = pd.read_csv(imagelist,header=None)

input_shape=(128,128,3)

n = len(data)
i = np.random.permutation(n)

i_split = int(n*0.9)
data_train = data.iloc[i[:i_split]]
data_test = data.iloc[i[i_split:]]

# save train and test sets for independent validation
fileutils.createdir(maindir)
data_train.to_csv(maindir+'/'+'data_train.csv',header=False,index=False)
data_test.to_csv(maindir+'/'+'data_test.csv',header=False,index=False)


###########################################

(encoder,decoder,vae,z_log_var,z_mean) = vae_model.get_model_maxpool(input_shape=input_shape, nlayers=2,nfm=16,d_lspace=100)

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
    reconstruction_loss = input_shape[0]*input_shape[1]*input_shape[2] * K.mean(K.square(input_image - output_image))
    #recon = K.sum(K.binary_crossentropy(y_pred, y_true), axis=1)
    
    # Return the sum of the two loss functions.
    return(reconstruction_loss + kl_loss)

########################################

# Compile the model.
vae.compile(loss = vae_loss,
            optimizer = 'rmsprop',
            metrics = ['accuracy'])

# And fit.
batch_size = 100

training_gen = vae_gen_lib.DataGenerator(data_train,
                             batch_size=batch_size,
                             dim=input_shape)

test_gen = vae_gen_lib.DataGenerator(data_test,
                         batch_size=batch_size,
                         dim=input_shape)


fit = vae.fit_generator(training_gen, epochs = 30, verbose = 2,
                        use_multiprocessing=True,
                        workers=8)

# Check the test data.
score = vae.evaluate_generator(test_gen, verbose = 2,
                        use_multiprocessing=True,
                        workers=8)
print('score is', score)
                      
# Save the models
vae.save(maindir+'/vae_gen_maxpool_16fm.h5')
encoder.save(maindir+'/vae_encoder_gen_maxpool_16fm.h5')
decoder.save(maindir+'/vae_decoder_gen_maxpool_16fm.h5')


