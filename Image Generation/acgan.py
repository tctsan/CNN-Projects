import sys
import os
import tensorflow as tf
import numpy as np
import random
import datetime
import matplotlib.pyplot as plt
from PIL import Image

from util import *
from acgan_model import *

input_dir  = sys.argv[1]
output_dir = sys.argv[2]
log_dir = "./logs/"

TRAIN_SIZE = 40000

# Parameters
Z_DIMS = 100

BATCH_SIZE = 32
ITERATION = 1250
EPOCH = 34

LEARNING_RATE_G = 0.0001
LEARNING_RATE_D = 0.0003

# Evaluation (fix input)
#fixed_vector = np.random.normal(0, 1, size=[10, Z_DIMS])
#fixed_vector = np.concatenate([fixed_vector, fixed_vector], axis=0)

#class1 = np.ones(10)
#class0 = np.zeros(10)
#fixed_class  = np.hstack((class1, class0)).reshape(20,1)

fixed_vector = np.loadtxt(log_dir+'fixed_vector_acgan.txt', dtype=float)               # shape: (20,100)
fixed_class = np.loadtxt(log_dir+'fixed_class_acgan.txt', dtype=float).reshape(-1, 1)  # shape: (20,1)
fixed_input = np.concatenate([fixed_vector, fixed_class], axis=1)                      # shape: (20,101)


############################################################################

tf.reset_default_graph()

'''
GAN Input / Output
    g_real: the real images
    g_fake: the generated images
'''
g_real = tf.placeholder(tf.float32, [None, 64, 64, 3], name="g_real")
d_real, d_real_class = discriminator(g_real)

z_input = tf.placeholder(tf.float32, [None, Z_DIMS+1], name="z_input")
g_fake = generator(z_input, BATCH_SIZE, Z_DIMS)
d_fake, d_fake_class = discriminator(g_fake, reuse_variables=True)

'''
Loss Function
    sigmoid_cross_entropy_with_logits: cross entropy for 2 classes
'''

# Discriminator Loss
d_real_label = tf.ones_like(d_real)
d_real_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=d_real, labels=d_real_label)
d_real_loss = tf.reduce_mean(d_real_loss)

# Auxiliary Classifier Loss
d_real_class_label = tf.placeholder(tf.float32, d_real_class.shape)
d_real_class_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=d_real_class, labels=d_real_class_label)
d_real_class_loss = tf.reduce_mean(d_real_class_loss)

d_real_final_loss = (d_real_loss + d_real_class_loss) / 2


# Discriminator Loss
d_fake_label = tf.zeros_like(d_fake)
d_fake_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake, labels=d_fake_label)
d_fake_loss = tf.reduce_mean(d_fake_loss)

# Auxiliary Classifier Loss
d_fake_class_label = tf.placeholder(tf.float32, d_fake_class.shape)
d_fake_class_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake_class, labels=d_fake_class_label)
d_fake_class_loss = tf.reduce_mean(d_fake_class_loss)

d_fake_final_loss = (d_fake_loss + d_fake_class_loss) / 2


# Generator Loss
g_label = tf.ones_like(d_fake)
g_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake, labels=g_label)
g_loss = tf.reduce_mean(g_loss)

# Auxiliary Classifier Loss
g_class_label = tf.placeholder(tf.float32, d_fake_class.shape)
g_class_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake_class, labels=g_class_label)
g_class_loss = tf.reduce_mean(g_class_loss)

g_train_loss = g_loss + g_class_loss

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
d_real_optimizer = d_real_optimizer.minimize(d_real_final_loss, var_list=d_vars)

d_fake_optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE_D)
d_fake_optimizer = d_fake_optimizer.minimize(d_fake_final_loss, var_list=d_vars)

g_optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE_G)
g_optimizer = g_optimizer.minimize(loss=g_train_loss, var_list=g_vars)


############################################################################

def train():

    # Record
    step_list = []

    d_real_loss_list = []
    d_real_class_loss_list = []
    d_real_accu_list = []

    d_fake_loss_list = []
    d_fake_class_loss_list = []
    d_fake_accu_list = []

    g_loss_list = []
    g_class_loss_list = []

    d_train_loss_list = []  # d_real_final_loss + d_fake_final_loss
    g_train_loss_list = []  # g_loss + g_class_loss
    

    # Load attribute - Smiling
    #np.savetxt('smiling_attr.txt', smiling_attr, fmt='%f')
    #smiling_attr = np.loadtxt('smiling_attr.txt', dtype=float)
    smiling_attr = read_attr(input_dir, "Smiling")
    smiling_attr = smiling_attr.reshape(-1, 1)  # shape: (42621, 1)

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
        # Real
        bid = rand_id[BATCH_SIZE*i : BATCH_SIZE*(i+1)]
        real_image = read_data(input_dir, "train", bid)
        real_class = smiling_attr[bid]

        # Fake
        fake_vector = np.random.normal(0, 1, size=[BATCH_SIZE, Z_DIMS])
        fake_class = np.random.randint(2, size=BATCH_SIZE).reshape(BATCH_SIZE, 1)
        fake_input = np.concatenate([fake_vector, fake_class], axis=1)

        # Input
        feed_dict = {g_real: real_image, d_real_class_label: real_class, 
                     z_input: fake_input, d_fake_class_label: fake_class}

        # Output
        fetches_list = [d_real_optimizer, d_fake_optimizer, d_real_loss, d_real_class_loss, 
                                                            d_fake_loss, d_fake_class_loss]

        _, __, D_real_loss, D_real_class_loss, D_fake_loss, D_fake_class_loss = sess.run(fetches_list, feed_dict=feed_dict)

        if((i+1) % 10 == 0):
            print("%3d ==> real loss: %.8f, class loss: %.8f, fake loss: %.8f, class loss: %.8f" % (i, 
                            D_real_loss, D_real_class_loss, D_fake_loss, D_fake_class_loss))

    
    # Train generator and discriminator together
    print("\n", datetime.datetime.now())
    print("Training generator and discriminator together...\n")


    # EPOCH = 50, ITERATION = 1250, BATCH = 32
    for epochs in range(EPOCH):
        rand_id = random.sample(range(TRAIN_SIZE), TRAIN_SIZE)

        for iters in range(ITERATION):
            # train discriminator on both real and fake images
            # Real
            bid = rand_id[BATCH_SIZE*iters : BATCH_SIZE*(iters+1)]
            real_image = read_data(input_dir, "train", bid)
            real_class = smiling_attr[bid]

            # Fake
            fake_vector = np.random.normal(0, 1, size=[BATCH_SIZE, Z_DIMS])
            fake_class = np.random.randint(2, size=BATCH_SIZE).reshape(BATCH_SIZE, 1)
            fake_input = np.concatenate([fake_vector, fake_class], axis=1)

            # Input
            feed_dict = {g_real: real_image, d_real_class_label: real_class, 
                         z_input: fake_input, d_fake_class_label: fake_class}

            # Output
            fetches_list = [d_real_optimizer, d_fake_optimizer, d_real_loss, d_real_class_loss, d_real_accu, 
                                                                d_fake_loss, d_fake_class_loss, d_fake_accu]

            _, __, D_real_loss, D_real_class_loss, D_real_accu, \
                   D_fake_loss, D_fake_class_loss, D_fake_accu = sess.run(fetches_list, feed_dict=feed_dict)

            D_train_loss = (D_real_loss + D_real_class_loss) / 2 + (D_fake_loss + D_fake_class_loss) / 2


            # train generator
            for k in range(2):
                g_fake_vector = np.random.normal(0, 1, size=[BATCH_SIZE, Z_DIMS])
                g_fake_class = np.random.randint(2, size=BATCH_SIZE).reshape(BATCH_SIZE, 1)
                g_fake_input = np.concatenate([g_fake_vector, g_fake_class], axis=1)

                _, G_loss, G_class_loss = sess.run([g_optimizer, g_loss, g_class_loss], 
                                                    feed_dict={z_input: g_fake_input, g_class_label: g_fake_class})

                G_train_loss = G_loss + G_class_loss


            if iters == 0 or (iters+1) % 50 == 0:
                # Show data
                print(datetime.datetime.now())
                print("Epoch %d  Iter %d" % (epochs, iters))
                print("==> Discriminator")
                print("\treal loss: %f, class loss: %f, accu: %f" % (D_real_loss, D_real_class_loss, D_real_accu))
                print("\tfake loss: %f, class loss: %f, accu: %f" % (D_fake_loss, D_fake_class_loss, D_fake_accu))
                print("\ttotal loss: %f" % D_train_loss)
                print("==> Generator")
                print("\tloss: %f, class loss: %f" % (G_loss, G_class_loss))
                print("\ttotal loss: %f\n" % G_train_loss)

                # Add data to list
                step_list.append(epochs*ITERATION+iters)

                d_real_loss_list.append(D_real_loss)
                d_real_class_loss_list.append(D_real_class_loss)
                d_real_accu_list.append(D_real_accu)

                d_fake_loss_list.append(D_fake_loss)
                d_fake_class_loss_list.append(D_fake_class_loss)
                d_fake_accu_list.append(D_fake_accu)

                g_loss_list.append(G_loss)
                g_class_loss_list.append(G_class_loss)

                d_train_loss_list.append(D_train_loss)
                g_train_loss_list.append(G_train_loss)

                # Save model
                saver.save(sess, log_dir+"acgan.ckpt", epochs)
                print("Model Saved.")

                # Save data
                txtfile = open(log_dir+'acgan_result.txt', 'a')
                txtfile.write("%d, " % int(epochs*ITERATION+iters))
                txtfile.write("%f, " % D_real_loss)
                txtfile.write("%f, " % D_real_class_loss)
                txtfile.write("%f, " % D_real_accu)
                txtfile.write("%f, " % D_fake_loss)
                txtfile.write("%f, " % D_fake_class_loss)
                txtfile.write("%f, " % D_fake_accu)
                txtfile.write("%f, " % G_loss)
                txtfile.write("%f, " % G_class_loss)
                txtfile.write("%f, " % D_train_loss)
                txtfile.write("%f\n\n" % G_train_loss)
                txtfile.close()
                print("Data Saved.")

                # Save 32 generated images (out_dir, file_name, image, epochs, iters, rows, cols)
                G_image = sess.run(g_fake, feed_dict={z_input: fixed_input})
                save_image(log_dir+"ACGAN_result_generated/", "acgan_image", G_image, epochs, iters, 2, 10)
                print("Generated Image Saved.\n")

    print("\n", datetime.datetime.now())
    print("Training finished.")


def save_acgan_result():

    # [fig3_2.jpg] Save learning curve
    # Read data (loss and accu)
    txtfile = open(log_dir+'acgan_result.txt', 'r')
    lines = []
    for i, line in enumerate(txtfile): 
        line = line.strip()
        if i != 0 and i % 2 == 0:
            lines.append(line)
    txtfile.close()

    # Add data to list
    step_list = []
    d_real_loss_list = []
    d_real_class_loss_list = []
    d_real_accu_list = []
    d_fake_loss_list = []
    d_fake_class_loss_list = []
    d_fake_accu_list = []
    g_loss_list = []
    g_class_loss_list = []
    d_train_loss_list = []
    g_train_loss_list = []

    for i in range(len(lines)):
        data = lines[i].split(", ")
        step_list.append(int(data[0]))
        d_real_loss_list.append(float(data[1]))
        d_real_class_loss_list.append(float(data[2]))
        d_real_accu_list.append(float(data[3]))
        d_fake_loss_list.append(float(data[4]))
        d_fake_class_loss_list.append(float(data[5]))
        d_fake_accu_list.append(float(data[6]))
        g_loss_list.append(float(data[7]))
        g_class_loss_list.append(float(data[8]))
        d_train_loss_list.append(float(data[9]))
        g_train_loss_list.append(float(data[10]))

    # Plot
    fig, axs = plt.subplots(1, 2, figsize=(20, 6))

    ax1 = axs[0]
    ax1.plot(step_list, d_real_class_loss_list, label='Real')
    ax1.plot(step_list, d_fake_class_loss_list, label='Fake')
    ax1.set_xlabel('step')
    ax1.set_title('Training Loss of Attribute Classification')
    ax1.legend(loc='lower right')

    ax2 = axs[1]
    ax2.plot(step_list, d_real_accu_list, label='Real')
    ax2.plot(step_list, d_fake_accu_list, label='Fake')
    ax2.set_xlabel('step')
    ax2.set_title('Accuracy of Discriminator')
    ax2.legend(loc='lower right')

    fig.savefig(output_dir+'fig3_2.jpg')
    plt.close(fig)
    print("fig3_2.jpg saved.")


    # Restore model
    saver = tf.train.Saver()
    saver.restore(sess, log_dir+"acgan.ckpt")

    # [fig3_3.jpg] Save 10 testing images and reconstructed image
    generated_img = sess.run(g_fake, feed_dict={z_input: fixed_input})
    save_image_final(output_dir, "fig3_3.jpg", generated_img, 2, 10)
    print("fig3_3.jpg saved.")


############################################################################

sess = tf.Session()
sess.run(tf.global_variables_initializer())

#train()
save_acgan_result()

sess.close()
