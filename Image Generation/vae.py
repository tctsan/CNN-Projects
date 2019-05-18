import sys
import os
import tensorflow as tf
import numpy as np
import random
import datetime
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.manifold import TSNE

from util import *
from vae_model import *

input_dir  = sys.argv[1]
output_dir = sys.argv[2]
log_dir = "./logs/"

TRAIN_SIZE = 40000
TEST_SIZE = 2621

# Parameters
z_dim = 512
lambdaKL = 0.00001

BATCH_SIZE = 128
ITERATION = 312
EPOCH = 50

LEARNING_RATE = 0.001

# Evaluation (fix input)
#fixed_vector = np.random.normal(0, 1, [32, z_dim])
#np.savetxt('fixed_vector_vae.txt', fixed_vector, fmt='%f')
fixed_vector = np.loadtxt(log_dir+'fixed_vector_vae.txt', dtype=float).reshape(32, 1, 1, 512)

############################################################################

tf.reset_default_graph()

input_image = tf.placeholder(tf.float32, [None, 64, 64, 3], name="images")
on_train    = tf.Variable(0, name="on_train")  # 0: train, 1: test

mu, logvar    = encoder(input_image, z_dim)
latent_vector = reparameterize(BATCH_SIZE, z_dim, mu, logvar, on_train)
recon_image   = decoder(latent_vector)

# labels and logits
y_true = tf.reshape(input_image, [-1, 64*64*3], name="y_true")
y_pred = tf.reshape(recon_image, [-1, 64*64*3], name="y_pred")

# loss (MSE, KLD)
MSE, KLD, loss = loss_function(y_true, y_pred, mu, logvar, lambdaKL, BATCH_SIZE)

# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
train_op = optimizer.minimize(loss)


############################################################################

def train():

    # Record
    step_list=[]
    mse_list=[]
    kld_list=[]
    loss_list=[]

    # Restore model
    saver = tf.train.Saver()
    checkpoint_dir = ''
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored.\n")

    # Training
    print("\n", datetime.datetime.now())
    print("Training...\n")

    # EPOCH = 50, ITERATION = 312, BATCH = 128
    for epochs in range(EPOCH):
        rand_id = random.sample(range(TRAIN_SIZE), TRAIN_SIZE)

        for iters in range(ITERATION):
            bid = rand_id[BATCH_SIZE*iters : BATCH_SIZE*(iters+1)]
            batch_image = read_data(input_dir, "train", bid)

            _, train_mse, train_kld, train_loss = sess.run([train_op, MSE, KLD, loss],
                                                            feed_dict={input_image: batch_image, 
                                                                        on_train: 0})

            if iters % 39 == 0:
                # Show data
                print(datetime.datetime.now())
                print("Epoch %d  Iter %d" % (epochs, iters))
                print("\tMSE: %f" % train_mse)
                print("\tKLD: %f" % train_kld)
                print("\tloss: %f\n" % train_loss)

                # Add data to list
                step_list.append(epochs*ITERATION+iters)
                mse_list.append(train_mse)
                kld_list.append(train_kld)
                loss_list.append(train_loss)

                # Save model
                saver.save(sess, log_dir + "vae.ckpt", epochs)
                print("Model Saved.")

                # Save data
                txtfile = open(log_dir+'vae_result.txt', 'a')
                txtfile.write("%d, " % int(epochs*ITERATION+iters))
                txtfile.write("%f, " % train_mse)
                txtfile.write("%f, " % train_kld)
                txtfile.write("%f\n\n" % train_loss)
                txtfile.close()
                print("Data Saved.")

                # Save 10 reconstructed images
                tid = np.arange(40000, 40010)
                test_image = read_data(input_dir, "test", tid)
                test_pred = sess.run(recon_image, feed_dict={input_image: test_image, 
                                                             on_train: 1})
                save_image(log_dir+"result_reconstructed/", "vae_reconstructed_image", 
                            test_pred, epochs, iters, 1, 10)
                print("Reconstructed Image Saved.")

                # Save 32 random generated images
                generated_img = sess.run(recon_image, feed_dict={latent_vector: fixed_vector})
                save_image(log_dir+"result_generated/", "vae_generated_image", 
                            generated_img, epochs, iters, 4, 8)
                print("Generated Image Saved.\n")
                
                
    print("\n", datetime.datetime.now())
    print("Training finished.")


def test():
    print("\n", datetime.datetime.now())
    print("Testing...\n")

    # Restore model
    saver = tf.train.Saver()
    saver.restore(sess, log_dir+"vae.ckpt")
    print("Model restored.\n")

    # Read testing images
    tid = np.arange(40000, 40000+TEST_SIZE)
    test_image = read_data(input_dir, "test", tid)
    
    # Run and save reconstructed images
    total_mse = 0
    total_kld = 0
    total_loss = 0
    for i in np.arange(TEST_SIZE):
        image = test_image[i].reshape(1, 64, 64, 3)
        test_pred, test_mse, test_kld = sess.run([recon_image, MSE, KLD],
                                        feed_dict={input_image: image, 
                                                   on_train: 1})
        total_mse += test_mse
        total_kld += test_kld
        total_loss += total_loss
        save_test_image(log_dir+"result_test/", test_pred[0], 40000+i)

    total_mse /= TEST_SIZE
    total_kld /= TEST_SIZE
    total_loss /= TEST_SIZE

    # Show data
    print("Testing MSE: ", total_mse)
    print("Testing KLD: ", total_kld)
    print("Testing loss: ", total_loss)

    # Save data
    txtfile = open(log_dir+'vae_result_test.txt', 'a')
    txtfile.write("%s\n" % datetime.datetime.now())
    txtfile.write("Testing MSE: %f\n" % total_mse)
    txtfile.write("Testing KLD: %f\n" % total_kld)
    txtfile.write("Testing loss: %f\n\n" % total_loss)
    txtfile.close()

    print("\n", datetime.datetime.now())
    print("Testing finished.")


def save_vae_result():

    # [figl_2.jpg] Save learning curve (MSE and KLD)
    # Read data (loss and accu)
    txtfile = open(log_dir+'vae_result.txt', 'r')
    lines = []
    for i, line in enumerate(txtfile): 
        line = line.strip()
        if i != 0 and i % 2 == 0:
            lines.append(line)
    txtfile.close()

    # Add data to list
    step_list=[]
    mse_list=[]
    kld_list=[]
    loss_list=[]

    for i in range(len(lines)):
        data = lines[i].split(", ")
        step_list.append(int(data[0]))
        mse_list.append(float(data[1]))
        kld_list.append(float(data[2]))
        loss_list.append(float(data[3]))

    # Plot learning curve
    fig, axs = plt.subplots(1, 2, figsize=(20, 6))

    ax1 = axs[0]
    ax1.plot(step_list, mse_list)
    ax1.set_xlabel('step')
    ax1.set_title('MSE')

    ax2 = axs[1]
    ax2.plot(step_list, kld_list)
    ax2.set_xlabel('step')
    ax2.set_title('KLD')

    fig.savefig(output_dir+'fig1_2.jpg')
    plt.close(fig)
    print("fig1_2.jpg saved.")


    # Restore model
    saver = tf.train.Saver()
    saver.restore(sess, log_dir+"vae.ckpt")

    # [figl_3.jpg] Save 10 testing images and reconstructed image
    tid = np.arange(40000, 40010)
    test_image  = read_data(input_dir, "test", tid)
    test_pred   = sess.run(recon_image, feed_dict={input_image: test_image, on_train: 1})
    test_output = np.concatenate([test_image, test_pred], axis=0)
    save_image_final(output_dir, "fig1_3.jpg", test_output, 2, 10)
    print("fig1_3.jpg saved.")

    # [figl_4.jpg] Save 32 random generated images
    generated_img = sess.run(recon_image, feed_dict={latent_vector: fixed_vector})
    save_image_final(output_dir, "fig1_4.jpg", generated_img, 4, 8)
    print("fig1_4.jpg saved.")


    # [figl_5.jpg] latent space
    test_mu = np.loadtxt(log_dir+'test_mu.txt', dtype=float)
    latent_embedded = TSNE(n_components=2, perplexity=80, random_state=4).fit_transform(test_mu)

    male_attr = read_attr(input_dir, "Male")
    male_attr = male_attr[40000:]
    fig, axs = plt.subplots(1, 1, figsize=(16,8))

    for i in [0,1]:
        if i==1:
            gender = "Male"
            color  = "lightblue"
        else:
            gender = "Female"
            color  = "pink"
        
        value = latent_embedded[male_attr==i]
        label = male_attr[attr_arr==i]
        axs.scatter(value[:,0], value[:,1], c=color, label=gender)

    axs.legend()
    axs.set_title("Gender")

    plt.savefig(output_dir+'fig1_5.jpg')
    plt.close(fig)
    print("fig1_5.jpg saved.")


############################################################################

sess = tf.Session()
sess.run(tf.global_variables_initializer())

#train()
test()
save_vae_result()

sess.close()
