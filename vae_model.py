import keras.models as km
import keras.layers as kl
import keras.backend as K
import keras.utils as ku
import numpy as np

def sampling(args):
    
    """This function reparameterizes the random sampling which is needed
    to feed the decoding network."""

    # Take apart the input arguments.
    z_mean, z_log_var = args

    # Get the dimensions of the problem.
    batch = K.shape(z_mean)[0]
    dim = K.shape(z_mean)[1]

    # By default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))

    # Return the reparameterized result.
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def get_model(input_shape, nlayers=2, nfm=16, d_lspace=100):
    '''
    VAE model.
    inputs:
        input_shape: shape of the input images
        nlayers: number of convolutional layers
        nfm: number of feature maps in the first convolutional layer.
             number of feature maps increaments by a factor of 2
             with each consecutive layer.
        d_lspace: dimensionality of the latent space
    outputs:
        encoder: the encoder model
        decoder: the decoder model
        vae: the entire vae model
        z_log_var: log variance values of the sampling function
        z_mean: mean values of the sampling function
    '''
    # Encoder
    # Image input to the encoder
    input_img = km.Input(shape = input_shape)
    
    # keep track of the shape changing in each layer
    # used to construct the first layer of the decoder
    s = list(input_shape)
    
    # add the first convolutional layer
    x = kl.Conv2D(nfm, kernel_size = (5,5))(input_img)
    s[0]=s[0]-5+1
    s[1]=s[1]-5+1
    x = kl.BatchNormalization()(x)
    x = kl.Activation('relu')(x)
    # Add the first max pooling layer
    #x = kl.MaxPooling2D(pool_size=(2,2), strides=(2,2))(x)
    
    for i in range(1,nlayers):
        # Add a 2D convolutional layer
        x = kl.Conv2D(nfm*(2**i), kernel_size = (3,3))(x)
        s[0]=s[0]-3+1
        s[1]=s[1]-3+1
        x = kl.BatchNormalization()(x)
        x = kl.Activation('relu')(x)       

        # Add a max pooling layer
        #x = kl.MaxPooling2D(pool_size=(2,2), strides=(2,2))(x)

    # flatten output to be fed into the output layer
    x = kl.Flatten()(x)

    z_mean = kl.Dense(d_lspace)(x)
    z_mean = kl.BatchNormalization()(z_mean)
    z_mean = kl.Activation('linear')(z_mean)
    
    z_log_var = kl.Dense(d_lspace)(x)
    z_log_var = kl.BatchNormalization()(z_log_var)
    z_log_var = kl.Activation('linear')(z_log_var)
    
    z = kl.Lambda(sampling)([z_mean, z_log_var])

    #############################################

    # Decoder
    # the decoder input
    decoder_input = km.Input(shape = (d_lspace,))

    # Add a fully-connected layer, to bluk things up to start.
    x = kl.Dense(int(s[0] * s[1] * nfm*(2**(nlayers-1))))(decoder_input)
    x = kl.BatchNormalization()(x)
    x = kl.Activation('relu')(x)
    
    # Reshape to the correct starting shape.
    x = kl.Reshape((s[0], s[1], int(nfm*(2**(nlayers-1)))))(x)

    for i in range(1,nlayers):
               
        # Add a 2D transpose convolutional layer
        x = kl.Conv2DTranspose((2**(nlayers-i-1))*nfm, kernel_size = (3, 3))(x)
        x = kl.BatchNormalization()(x)
        x = kl.Activation('relu')(x)        
        # Add upsampling
        #x = kl.UpSampling2D(size = (2, 2))(x)

    # Add a 2D transpose convolutional layer, with #channels feature map.  This is
    # the decoder output.

    x = kl.Conv2DTranspose(input_shape[2], kernel_size = (5, 5))(x)
    x = kl.BatchNormalization()(x)
    decoded = kl.Activation('sigmoid')(x)
    # build encoder, decoder, and
    # the full variational autoencoder models

    encoder = km.Model(inputs = input_img,
                       outputs = [z_mean, z_log_var, z])

    decoder = km.Model(inputs = decoder_input,
                       outputs = decoded)

    output_img = decoder(encoder(input_img)[2])

    vae = km.Model(inputs = input_img,
                  outputs = output_img)
    
    return(encoder,decoder,vae,z_log_var,z_mean)

def get_model_maxpool(input_shape, nlayers=2, nfm=16, d_lspace=100):
    '''
    VAE model with max pooling.
    inputs:
        input_shape: shape of the input images
        nlayers: number of convolutional layers
        nfm: number of feature maps in the first convolutional layer.
             number of feature maps increaments by a factor of 2
             with each consecutive layer.
        d_lspace: dimensionality of the latent space
    outputs:
        encoder: the encoder model
        decoder: the decoder model
        vae: the entire vae model
        z_log_var: log variance values of the sampling function
        z_mean: mean values of the sampling function
    '''
    # Encoder
    # Image input to the encoder
    input_img = km.Input(shape = input_shape)
    
    # keep track of the shape changing in each layer
    # used to construct the first layer of the decoder
    s = list(input_shape)
    
    # add the first convolutional layer
    x = kl.Conv2D(nfm, kernel_size = (5,5), padding='same')(input_img)
    x = kl.BatchNormalization()(x)
    x = kl.Activation('relu')(x)
    # Add the first max pooling layer
    x = kl.MaxPooling2D(pool_size=(2,2), strides=(2,2))(x)
    s[0]=int(s[0]/2)
    s[1]=int(s[1]/2)

    for i in range(1,nlayers):
        # Add a 2D convolutional layer
        x = kl.Conv2D(nfm*(2**i), kernel_size = (3,3),padding='same')(x)
        x = kl.BatchNormalization()(x)
        x = kl.Activation('relu')(x)       

        # Add a max pooling layer
        x = kl.MaxPooling2D(pool_size=(2,2), strides=(2,2))(x)
        s[0]=int(s[0]/2)
        s[1]=int(s[1]/2)

    # flatten output to be fed into the output layer
    x = kl.Flatten()(x)

    z_mean = kl.Dense(d_lspace)(x)
    z_mean = kl.BatchNormalization()(z_mean)
    z_mean = kl.Activation('linear')(z_mean)
    
    z_log_var = kl.Dense(d_lspace)(x)
    z_log_var = kl.BatchNormalization()(z_log_var)
    z_log_var = kl.Activation('linear')(z_log_var)
    
    z = kl.Lambda(sampling)([z_mean, z_log_var])

    #############################################

    # Decoder
    # the decoder input
    decoder_input = km.Input(shape = (d_lspace,))

    # Add a fully-connected layer, to bluk things up to start.
    x = kl.Dense(int(s[0] * s[1] * nfm*(2**(nlayers-1))))(decoder_input)
    x = kl.BatchNormalization()(x)
    x = kl.Activation('relu')(x)
    
    # Reshape to the correct starting shape.
    x = kl.Reshape((s[0], s[1], int(nfm*(2**(nlayers-1)))))(x)

    for i in range(1,nlayers):
               
        # Add a 2D transpose convolutional layer
        x = kl.Conv2DTranspose((2**(nlayers-i-1))*nfm, kernel_size = (3, 3),padding='same')(x)
        x = kl.BatchNormalization()(x)
        x = kl.Activation('relu')(x)        
        # Add upsampling
        x = kl.UpSampling2D(size = (2, 2))(x)

    # Add a 2D transpose convolutional layer, with #channels feature map.  This is
    # the decoder output.

    x = kl.Conv2DTranspose(input_shape[2], kernel_size = (5, 5),padding='same')(x)
    x = kl.BatchNormalization()(x)
    x = kl.Activation('sigmoid')(x)
    # Add upsampling
    decoded = kl.UpSampling2D(size = (2, 2))(x)

    # build encoder, decoder, and
    # the full variational autoencoder models

    encoder = km.Model(inputs = input_img,
                       outputs = [z_mean, z_log_var, z])

    decoder = km.Model(inputs = decoder_input,
                       outputs = decoded)

    output_img = decoder(encoder(input_img)[2])

    vae = km.Model(inputs = input_img,
                  outputs = output_img)
    
    return(encoder,decoder,vae,z_log_var,z_mean)