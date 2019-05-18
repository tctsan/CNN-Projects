import tensorflow as tf
import numpy as np


def encoder(input_images, z_dim):
    conv1 = conv("conv1", input_images, 32)  # 64x64x3  -> 32x32x32
    conv2 = conv("conv2", conv1, 64)         # 32x32x32 -> 16x16x64
    conv3 = conv("conv3", conv2, 128)        # 16x16x64 -> 8x8x128
    conv4 = conv("conv4", conv3, 256)        # 8x8x128  -> 4x4x256 (4096)

    # mean, logvar (1x4096 -> 1x256)
    mu = tf.reshape(conv4, [-1, 1, 1, 4096])
    mu = tf.layers.dense(mu, z_dim, name="mu")

    logvar = tf.reshape(conv4, [-1, 1, 1, 4096])
    logvar = tf.layers.dense(logvar, z_dim, name="logvar")

    return mu, logvar


def reparameterize(batch_size, z_dim, mu, logvar, on_train):
    # Gaussian Distribution N(0,1)
    epsilon = tf.random_normal([batch_size, z_dim], mean=0.0, stddev=1.0, 
                                dtype=tf.float32, name="epsilon")

    std = tf.exp(logvar * 0.5)
    ep_std = tf.multiply(epsilon, std)

    if on_train == 0:
        z = tf.add(mu, ep_std)
    else:
        z = mu

    return z


def decoder(latent_vector):
    # 1x256 -> 1x4096
    latent_vector = tf.layers.dense(latent_vector, 4096)
    latent_vector = tf.reshape(latent_vector, [-1, 4, 4, 256])
    latent_vector = tf.nn.relu(latent_vector)
    shape0 = tf.shape(latent_vector)[0]

    deconv1 = conv_transpose("deconv1", latent_vector, [shape0, 8, 8, 128])
    deconv1 = tf.nn.relu(deconv1)

    deconv2 = conv_transpose("deconv2", deconv1, [shape0, 16, 16, 64])
    deconv2 = tf.nn.relu(deconv2)

    deconv3 = conv_transpose("deconv3", deconv2, [shape0, 32, 32, 32])
    deconv3 = tf.nn.relu(deconv3)
    
    deconv4 = conv_transpose("deconv4", deconv3, [shape0, 64, 64, 3])
    deconv4 = tf.nn.sigmoid(deconv4, name="recon_images")
    #deconv4 = tf.nn.tanh(deconv4, name="recon_images")
    
    return deconv4


def loss_function(img, recon_img, mu, logvar, lambda_kl, batch_size):
    MSE = tf.reduce_sum(tf.square(img - recon_img)) / batch_size
    KLD = -0.5 * tf.reduce_sum(1 + logvar - tf.square(mu) - tf.exp(logvar)) / batch_size

    VAE_loss = MSE + KLD * lambda_kl

    return MSE, KLD, VAE_loss


def conv(layer_name, x_input, out_channels,
         kernel_size=[5,5], stride=[1,2,2,1]):
    
    in_shape = x_input.get_shape()
    in_channels = in_shape[-1]
    
    with tf.variable_scope(layer_name):
        w = tf.get_variable(name='weights',
                            shape=[kernel_size[0], kernel_size[1],
                                   in_channels, out_channels],
                            initializer=tf.contrib.layers.xavier_initializer())
        
        b = tf.get_variable(name='biases',
                            shape=[out_channels],
                            initializer=tf.constant_initializer(0.0))
        
        x_output = tf.nn.conv2d(x_input, w, stride, padding='SAME', name='conv')
        x_output = tf.nn.bias_add(x_output, b, name='bias_add')
        x_output = tf.nn.relu(x_output, name='relu')
        
        #print(layer_name, "shape : " , x_output.get_shape())

        return x_output


def conv_transpose(layer_name, x_input, out_shape,
                   kernel_size=[5,5], stride=[1,2,2,1]):
    
    in_shape = x_input.get_shape()
    in_channels = in_shape[-1]
    
    with tf.variable_scope(layer_name):
        w = tf.get_variable(name='weights',
                            shape=[kernel_size[0], kernel_size[1],
                                   out_shape[-1], in_channels],
                            initializer=tf.contrib.layers.xavier_initializer())
        
        b = tf.get_variable(name='biases',
                            shape=[out_shape[-1]],
                            initializer=tf.constant_initializer(0.0))
        
        convt = tf.nn.conv2d_transpose(x_input, w, output_shape=out_shape, strides=stride)
        
        #print(layer_name, "shape : " , convt.get_shape())

        return convt


'''
Auto-Encoder

def encoder(input_images):
    conv1 = conv("conv1", input_images, 4)  # 64x64x3 -> 32x32x4
    conv2 = conv("conv2", conv1, 8)         # 32x32x4 -> 16x16x8
    conv3 = conv("conv3", conv2, 16)        # 16x16x8 -> 8x8x16 (1024)
    
    latent_vector = tf.reshape(conv3, [-1, 8*8*16], name="latent_vector")
    return latent_vector

def decoder(latent_vector):
    l_matrix = tf.reshape(latent_vector, [-1, 8, 8, 16])
    l_matrix = tf.nn.relu(l_matrix)
    
    shape0 = tf.shape(l_matrix)[0]

    deconv1 = conv_transpose("deconv1", l_matrix, [shape0, 16, 16, 8])
    deconv1 = tf.nn.relu(deconv1)

    deconv2 = conv_transpose("deconv2", deconv1, [shape0, 32, 32, 4])
    deconv2 = tf.nn.relu(deconv2)
    
    deconv3 = conv_transpose("deconv3", deconv2, [shape0, 64, 64, 3])
    deconv3 = tf.nn.sigmoid(deconv3, name="recon_images")
    
    return deconv3
'''
