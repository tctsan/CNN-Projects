import sys
import os
import tensorflow as tf
import numpy as np
import random
import datetime
import matplotlib.pyplot as plt
from PIL import Image

from util import *
from gan_model import *

input_dir  = sys.argv[1]
output_dir = sys.argv[2]
log_dir = "./logs/"

TRAIN_SIZE = 40000

# Parameters
Z_DIMS = 100

BATCH_SIZE = 32
ITERATION = 1250
EPOCH = 50

LEARNING_RATE_G = 0.0001
LEARNING_RATE_D = 0.0003

# Evaluation (fix input)
#fixed_vector = np.random.normal(0, 1, size=[BATCH_SIZE, Z_DIMS])
#np.savetxt('fixed_vector_gan.txt', fixed_vector, fmt='%f')
fixed_vector = np.loadtxt(log_dir+'fixed_vector_gan.txt', dtype=float)


############################################################################

tf.reset_default_graph()

'''
GAN Input / Output
    g_real: the real images
    g_fake: the generated images
'''
g_real = tf.placeholder(tf.float32, [None, 64, 64, 3], name="g_real")
d_real = discriminator(g_real)

z_input = tf.placeholder(tf.float32, [None, Z_DIMS], name="z_input") 
g_fake = generator(z_input, BATCH_SIZE, Z_DIMS)
d_fake = discriminator(g_fake, reuse_variables=True)

'''
Loss Function
    sigmoid_cross_entropy_with_logits: cross entropy for 2 classes
'''
d_real_label = tf.ones_like(d_real)
d_real_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=d_real, labels=d_real_label)
d_real_loss = tf.reduce_mean(d_real_loss)

d_fake_label = tf.zeros_like(d_fake)
d_fake_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake, labels=d_fake_label)
d_fake_loss = tf.reduce_mean(d_fake_loss)

#d_loss = d_real_loss + d_fake_loss

g_label = tf.ones_like(d_fake)
g_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake, labels=g_label)
g_loss = tf.reduce_mean(g_loss)

'''
Accuracy of Discriminator
'''
d_real_accu = tf.equal(tf.cast(d_real>0.5, tf.float32), d_real_label)
d_real_accu = tf.reduce_mean(tf.cast(d_real_accu, tf.float32))

d_fake_accu = tf.equal(tf.cast(d_fake>0.5, tf.float32), d_fake_label)
d_fake_accu = tf.reduce_mean(tf.cast(d_fake_accu, tf.float32))

'''
Training Optimization
    learning rate of G: 0.0001
    learning rate of D: 0.0003
'''
# Get the varaibles for different network
tvars = tf.trainable_variables()
d_vars = [var for var in tvars if 'd_' in var.name]
g_vars = [var for var in tvars if 'g_' in var.name]

#print([v.name for v in d_vars])
#print([v.name for v in g_vars])

d_real_optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE_D)
d_real_optimizer = d_real_optimizer.minimize(d_real_loss, var_list=d_vars)

d_fake_optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE_D)
d_fake_optimizer = d_fake_optimizer.minimize(d_fake_loss, var_list=d_vars)

#d_optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE_D)
#d_optimizer = d_optimizer.minimize(loss=d_loss, var_list=d_vars)

g_optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE_G)
g_optimizer = g_optimizer.minimize(loss=g_loss, var_list=g_vars)


############################################################################

def train():

    # Record
    step_list = []
    d_real_loss_list = []
    d_real_accu_list = []
    d_fake_loss_list = []
    d_fake_accu_list = []
    d_loss_list = []
    g_loss_list = []
    
    # Restore model
    saver = tf.train.Saver()
    checkpoint_dir = ''
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored.\n")


    # Pre-train discriminator
    print("\n", datetime.datetime.now())
    print("Pre-train discriminator...\n")

    rand_id = random.sample(range(TRAIN_SIZE), TRAIN_SIZE)
    for i in range(300):
        bid = rand_id[BATCH_SIZE*i : BATCH_SIZE*(i+1)]
        real_image = read_data(input_dir, "train", bid)

        fake_input = np.random.normal(0, 1, size=[BATCH_SIZE, Z_DIMS])
        _, __, D_real_loss, D_fake_loss = sess.run([d_real_optimizer, d_fake_optimizer, d_real_loss, d_fake_loss],
                                                   {g_real: real_image, z_input: fake_input})

        if((i+1) % 10 == 0):
            print("%2d ==> real loss: %.8f, fake loss: %.8f" % (i, D_real_loss, D_fake_loss))


    # Train generator and discriminator together
    print("\n", datetime.datetime.now())
    print("Training generator and discriminator together...\n")


    # EPOCH = 50, ITERATION = 1250, BATCH = 32
    for epochs in range(EPOCH):
        rand_id = random.sample(range(TRAIN_SIZE), TRAIN_SIZE)

        for iters in range(ITERATION):
            bid = rand_id[BATCH_SIZE*iters : BATCH_SIZE*(iters+1)]

            # train discriminator on both real and fake images
            fake_input = np.random.normal(0, 1, size=[BATCH_SIZE, Z_DIMS])
            real_image = read_data(input_dir, "train", bid)

            _, __, D_real_loss, D_real_accu, D_fake_loss, D_fake_accu = sess.run([d_real_optimizer, d_fake_optimizer, 
                    d_real_loss, d_real_accu, d_fake_loss, d_fake_accu], feed_dict={g_real: real_image, 
                                                                                    z_input: fake_input})
            D_loss = D_fake_loss + D_real_loss
            
            # train generator
            fixed_vector = np.random.normal(0, 1, size=[BATCH_SIZE, Z_DIMS])

            _, G_image, G_loss = sess.run([g_optimizer, g_fake, g_loss], 
                                            feed_dict={z_input: fixed_vector})

            if iters == 0 or (iters+1) % 50 == 0:
                # Show data
                print(datetime.datetime.now())
                print("Epoch %d  Iter %d" % (epochs, iters))
                print("==> Discriminator")
                print("\treal loss: %f, accu: %f" % (D_real_loss, D_real_accu))
                print("\tfake loss: %f, accu: %f" % (D_fake_loss, D_fake_accu))
                print("\ttotal loss: %f" % D_loss)
                print("==> Generator")
                print("\tloss: %f\n" % G_loss)

                # Add data to list
                step_list.append(epochs*ITERATION+iters)
                d_real_loss_list.append(D_real_loss)
                d_real_accu_list.append(D_real_accu)
                d_fake_loss_list.append(D_fake_loss)
                d_fake_accu_list.append(D_fake_accu)
                d_loss_list.append(D_loss)
                g_loss_list.append(G_loss)

                # Save model
                saver.save(sess, log_dir+"gan.ckpt", epochs)
                print("Model Saved.")

                # Save data
                txtfile = open(log_dir+'gan_result.txt', 'a')
                txtfile.write("%d, " % int(epochs*ITERATION+iters))
                txtfile.write("%f, " % D_real_loss)
                txtfile.write("%f, " % D_real_accu)
                txtfile.write("%f, " % D_fake_loss)
                txtfile.write("%f, " % D_fake_accu)
                txtfile.write("%f, " % D_loss)
                txtfile.write("%f\n\n" % G_loss)
                txtfile.close()
                print("Data Saved.")

                # Save 32 generated images (out_dir, file_name, image, epochs, iters, rows, cols)
                save_image(log_dir+"GAN_result_generated/", "gan_image", G_image, epochs, iters, 4, 8)
                print("Generated Image Saved.\n")

    print("\n", datetime.datetime.now())
    print("Training finished.")


def save_gan_result():

    # [fig2_2.jpg] Save learning curve
    # Read data (loss and accu)
    txtfile = open(log_dir+'gan_result.txt', 'r')
    lines = []
    for i, line in enumerate(txtfile): 
        line = line.strip()
        if i != 0 and i % 2 == 0:
            lines.append(line)
    txtfile.close()

    # Add data to list
    step_list = []
    d_real_loss_list = []
    d_real_accu_list = []
    d_fake_loss_list = []
    d_fake_accu_list = []
    d_loss_list = []
    g_loss_list = []

    for i in range(len(lines)):
        data = lines[i].split(", ")
        step_list.append(int(data[0]))
        d_real_loss_list.append(float(data[1]))
        d_real_accu_list.append(float(data[2]))
        d_fake_loss_list.append(float(data[3]))
        d_fake_accu_list.append(float(data[4]))
        d_loss_list.append(float(data[5]))
        g_loss_list.append(float(data[6]))

    # Plot learning curve
    fig, axs = plt.subplots(1, 2, figsize=(20, 6))

    ax1 = axs[0]
    ax1.plot(step_list, d_real_loss_list, label='Real')
    ax1.plot(step_list, d_fake_loss_list, label='Fake')
    ax1.set_xlabel('step')
    ax1.set_title('Training Loss of Discriminator')
    ax1.legend(loc='lower right')

    ax2 = axs[1]
    ax2.plot(step_list, d_real_accu_list, label='Real')
    ax2.plot(step_list, d_fake_accu_list, label='Fake')
    ax2.set_xlabel('step')
    ax2.set_title('Accuracy of Discriminator')
    ax2.legend(loc='lower right')

    fig.savefig(output_dir+'fig2_2.jpg')
    plt.close(fig)
    print("fig2_2.jpg saved.")


    # Restore model
    saver = tf.train.Saver()
    saver.restore(sess, log_dir+"gan.ckpt")

    # [fig2_3.jpg] Save 10 testing images and reconstructed image
    generated_img = sess.run(g_fake, feed_dict={z_input: fixed_vector})
    save_image_final(output_dir, "fig2_3.jpg", generated_img, 4, 8)
    print("fig2_3.jpg saved.")


############################################################################

sess = tf.Session()
sess.run(tf.global_variables_initializer())

#train()
save_gan_result()

sess.close()