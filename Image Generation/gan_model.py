import tensorflow as tf
import numpy as np


def generator(z, batch_size, z_dim):

    # 100-dim vector to 12288-dim (64x64x3)
    g_w1 = tf.get_variable('g_w1', [z_dim, 12288], dtype=tf.float32, 
                            initializer=tf.truncated_normal_initializer(stddev=0.02))

    g_b1 = tf.get_variable('g_b1', [12288], 
                            initializer=tf.truncated_normal_initializer(stddev=0.02))

    g1 = tf.nn.bias_add(tf.matmul(z, g_w1), g_b1)
    g1 = tf.reshape(g1, [-1, 64, 64, 3])

    g1 = tf.contrib.layers.batch_norm(g1, epsilon=1e-5, scope='g_b1')
    g1 = tf.nn.relu(g1)
    #print("g1 shape: ", g1.shape)


    # Conv1: 50 features
    g_w2 = tf.get_variable('g_w2', [3, 3, 3, 50], dtype=tf.float32, 
                            initializer=tf.truncated_normal_initializer(stddev=0.02))
    g_b2 = tf.get_variable('g_b2', [50], 
                            initializer=tf.truncated_normal_initializer(stddev=0.02))

    g2 = tf.nn.conv2d(g1, g_w2, strides=[1, 2, 2, 1], padding='SAME')
    g2 = tf.nn.bias_add(g2, g_b2)

    g2 = tf.contrib.layers.batch_norm(g2, epsilon=1e-5, scope='g_b2')
    g2 = tf.nn.relu(g2)
    g2 = tf.image.resize_images(g2, [128, 128])
    #print("g2 shape: ", g2.shape)


    # Conv2: 25 features
    g_w3 = tf.get_variable('g_w3', [3, 3, 50, 25], dtype=tf.float32, 
                            initializer=tf.truncated_normal_initializer(stddev=0.02))

    g_b3 = tf.get_variable('g_b3', [25], 
                            initializer=tf.truncated_normal_initializer(stddev=0.02))

    g3 = tf.nn.conv2d(g2, g_w3, strides=[1, 2, 2, 1], padding='SAME')
    g3 = tf.nn.bias_add(g3, g_b3)

    g3 = tf.contrib.layers.batch_norm(g3, epsilon=1e-5, scope='g_b3')
    g3 = tf.nn.relu(g3)
    g3 = tf.image.resize_images(g3, [128, 128])
    #print("g3 shape: ", g3.shape)


    # Conv3: 3 output channels
    g_w4 = tf.get_variable('g_w4', [1, 1, 25, 3], dtype=tf.float32, 
                            initializer=tf.truncated_normal_initializer(stddev=0.02))

    g_b4 = tf.get_variable('g_b4', [3], 
                            initializer=tf.truncated_normal_initializer(stddev=0.02))

    g4 = tf.nn.conv2d(g3, g_w4, strides=[1, 2, 2, 1], padding='SAME')
    g4 = tf.nn.bias_add(g4, g_b4)

    g4 = tf.sigmoid(g4)
    #print("g4 shape: ", g4.shape)
    
    return g4  # 64x64x3 image


def discriminator(images, reuse_variables=None):
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables) as scope:

        # Conv1: 64x64x3 -> 32x32x32
        d_w1 = tf.get_variable('d_w1', [5, 5, 3, 32], 
                                initializer=tf.truncated_normal_initializer(stddev=0.02))
        d_b1 = tf.get_variable('d_b1', [32], 
                                initializer=tf.constant_initializer(0))

        d1 = tf.nn.conv2d(input=images, filter=d_w1, strides=[1, 2, 2, 1], padding='SAME')
        d1 = tf.nn.bias_add(d1, d_b1)

        d1 = tf.contrib.layers.batch_norm(d1, epsilon=1e-5, scope='d_b1')
        d1 = tf.nn.leaky_relu(d1)
        #print("d1 shape: ", d1.shape)


        # Conv2: 32x32x32 -> 16x16x64
        d_w2 = tf.get_variable('d_w2', [5, 5, 32, 64], 
                                initializer=tf.truncated_normal_initializer(stddev=0.02))
        d_b2 = tf.get_variable('d_b2', [64], 
                                initializer=tf.constant_initializer(0))

        d2 = tf.nn.conv2d(input=d1, filter=d_w2, strides=[1, 2, 2, 1], padding='SAME')
        d2 = tf.nn.bias_add(d2, d_b2)

        d2 = tf.contrib.layers.batch_norm(d2, epsilon=1e-5, scope='d_b2')
        d2 = tf.nn.leaky_relu(d2)
        #print("d2 shape: ", d2.shape)


        # FC1: 16384 -> 1024
        d_w3 = tf.get_variable('d_w3', [16 * 16 * 64, 1024], 
                                initializer=tf.truncated_normal_initializer(stddev=0.02))
        d_b3 = tf.get_variable('d_b3', [1024], 
                                initializer=tf.constant_initializer(0))

        d3 = tf.reshape(d2, [-1, 16 * 16 * 64])
        d3 = tf.matmul(d3, d_w3)
        d3 = tf.nn.bias_add(d3, d_b3)

        d3 = tf.contrib.layers.batch_norm(d3, epsilon=1e-5, scope='d_b3')
        d3 = tf.nn.leaky_relu(d3)
        #print("d3 shape: ", d3.shape)


        # FC2: 1024 -> 1
        d_w4 = tf.get_variable('d_w4', [1024, 1], 
                                initializer=tf.truncated_normal_initializer(stddev=0.02))
        d_b4 = tf.get_variable('d_b4', [1], 
                                initializer=tf.constant_initializer(0))

        d4 = tf.matmul(d3, d_w4)
        d4 = tf.nn.bias_add(d4, d_b4)
        #print("d4 shape: ", d4.shape)

        return d4
