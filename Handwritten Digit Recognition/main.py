import sys
import os
import glob
import numpy as np
import math
import random
import tensorflow as tf
from PIL import Image

def read_data(input_dir, input_type):
    
    image_name = []
    image_list = []
    label_list = []
    
    # input_type: train or valid
    input_path = os.path.join(input_dir, input_type)
    input_list = sorted(os.listdir(input_path))
    
    for i, classes in enumerate(input_list):
        class_num = classes.split("_")
        file_path = os.path.join(input_path, classes, '*.png')
        file_name = sorted(glob.glob(file_path))
        
        for fname in file_name:
            image_name.append(fname)
            label_list.append(class_num[1])

    image_list = [np.array(Image.open(fname).convert('L')) for fname in image_name]
    
    images = np.array(image_list)
    labels = np.array(label_list)

    return images, labels


def CNN(images, keep_prob, n_classes):
    
    with tf.name_scope('CNN'):
        nf = 32
        
        x = conv_layer('conv1', images, nf)
        x = max_pooling('maxpool1', x)
        x = tf.nn.dropout(x, keep_prob)

        x = conv_layer('conv2', x, nf*2)
        x = max_pooling('maxpool2', x)
        x = tf.nn.dropout(x, keep_prob)

        x = conv_layer('conv3', x, nf*4)
        x = max_pooling('maxpool3', x)
        x = tf.nn.dropout(x, keep_prob)
        
        x = fc_layer('fc1', x, out_nodes=128, is_softmax=False)
        x = tf.nn.dropout(x, keep_prob)
        
        x = fc_layer('fc2', x, out_nodes=n_classes, is_softmax=True)
        
    return x


def conv_layer(layer_name, x_input, out_channels, kernel_size=[3,3], stride=[1,1,1,1], padding='SAME', is_trainable=True):
    
    in_channels = x_input.get_shape()[-1]
    
    with tf.variable_scope(layer_name):
        w = tf.get_variable(name='weights',
                            shape=[kernel_size[0], kernel_size[1],
                                   in_channels, out_channels],
                            initializer=tf.contrib.layers.xavier_initializer())
        
        b = tf.get_variable(name='biases',
                            shape=[out_channels],
                            trainable = is_trainable,
                            initializer=tf.constant_initializer(0.0))
        
        x_output = tf.nn.conv2d(x_input, w, strides=stride, padding=padding, name='conv')
        x_output = tf.nn.bias_add(x_output, b, name='bias_add')
        x_output = tf.nn.relu(x_output, name='relu')
        
        print(layer_name, "   shape : ", x_output.get_shape())
        
    return x_output


def max_pooling(layer_name, x, kernel_size=[1,2,2,1], stride=[1,2,2,1]):
    with tf.name_scope(layer_name):
        x_output = tf.nn.max_pool(x, ksize=kernel_size, strides=stride, padding='SAME')
        print(layer_name, "shape : ", x_output.get_shape())
        
    return x_output


def fc_layer(layer_name, x, out_nodes, is_softmax=False):
    
    shape = x.get_shape()
    if len(shape) == 4:
        # 4D tensor to 1D length
        size = shape[1].value * shape[2].value * shape[3].value
    else:
        size = shape[-1].value
    
    with tf.variable_scope(layer_name):
        w = tf.get_variable('weights',
                            shape=[size, out_nodes],
                            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('biases',
                            shape=[out_nodes],
                            initializer=tf.constant_initializer(0.0))
        
        flat_x = tf.reshape(x, [-1, size]) # flatten into 1D
        
        if is_softmax == False:
            x = tf.nn.relu(tf.matmul(flat_x, w) + b, name=layer_name)
        else:
            x = tf.add(tf.matmul(flat_x, w), b, name=layer_name)
    
        print(layer_name, " shape :" , x.get_shape())
        
    return x


def train(sess):
    
    # saver
    saver = tf.train.Saver()

    # initialize session
    init = tf.global_variables_initializer()
    sess.run(init)

    # Training
    # epoch = 150, batch = 256
    for epochs in range(Epoch):
        
        rand_id = random.sample(range(N_TrainData), N_TrainData)
        
        for iters in range(Iteration):
            bid = rand_id[Batch_Size*iters : Batch_Size*(iters+1)]
            
            if iters == (Iteration-1):
                bid = rand_id[Batch_Size*iters : N_TrainData]
            
            batch_image = train_images[bid].reshape([-1, Image_Size, Image_Size, Image_Channel])
            batch_label = train_labels[bid]
            
            feed = {x_images: batch_image, y_labels: batch_label, keep_prob: 0.8}
            
            _, train_loss, train_accu = sess.run([train_op, loss, accu], feed_dict=feed)
            
        print("Epoch %2d: train loss %f, accu %f" % (epochs, train_loss, train_accu))
    
    # save model
    model_name = './hw2-3_CNN.model'
    saver.save(sess, model_name)
    
    print("Training finished.")


def valid(sess):
    
    # restore model
    saver = tf.train.Saver()
    model_name = './hw2-3_CNN.model'
    saver.restore(sess, model_name)
    
    # Validation
    n_valid       = valid_images.shape[0]  # 10,000
    image_size    = valid_images.shape[1]  # 28
    image_channel = 1

    total_loss = 0
    total_accu = 0
    
    for i in range(N_ValidData):
        images = valid_images[i:i+1].reshape([-1, Image_Size, Image_Size, Image_Channel])
        labels = valid_labels[i:i+1]
        
        feed = {x_images: images, y_labels: labels, keep_prob: 1.0}
        
        valid_loss, valid_accu = sess.run([loss, accu], feed_dict=feed)
        
        total_loss += valid_loss
        total_accu += valid_accu
    
    total_loss /= N_ValidData
    total_accu /= N_ValidData
    
    print("Validation loss %f, accu %f" % (total_loss, total_accu))
    print("Validation finished.")


def test(sess):
    
    # restore model
    saver = tf.train.Saver()
    model_name = './hw2-3_CNN.model'
    saver.restore(sess, model_name)
    
    # Testing
    csvfile = open(output_path, 'w')
    csvfile.write("id,label\n")
    
    for i in range(N_TestData):
        images = test_images[i:i+1].reshape([-1, Image_Size, Image_Size, Image_Channel])
        
        feed = {x_images: images, keep_prob: 1.0}
        test_pred_label = sess.run([pred_label], feed_dict=feed)
        
        csvfile.write("%s,%d\n" % (str(i).zfill(4), int(test_pred_label[0])))
    
    csvfile.close()
    
    print("Testing finished.")

'''
Main Function
    * Read data
    * Set hyperparameters
    * Create graph
    * Train, Valid and Test
'''

# Read data
input_dir = sys.argv[1]
train_images, train_labels = read_data(input_dir, 'train')
valid_images, valid_labels = read_data(input_dir, 'valid')

# data info
Image_Size    = train_images.shape[1]  # 28
Image_Channel = 1
N_Classes     = 10
N_TrainData   = train_images.shape[0]  # 50,000
N_ValidData   = valid_images.shape[0]  # 10,000


# Set hyperparameters
Learning_Rate = 0.0001
Epoch         = 100
Batch_Size    = 256
Iteration     = int(N_TrainData / Batch_Size) + 1


# Create graph
tf.reset_default_graph()

# placeholder
x_images  = tf.placeholder(tf.float32, [None, Image_Size, Image_Size, Image_Channel])
y_labels  = tf.placeholder(tf.int32,   [None, ])
keep_prob = tf.placeholder(tf.float32)

y_logits  = CNN(x_images, keep_prob, N_Classes)

# predicted and true label
y_label = tf.one_hot(y_labels, depth=N_Classes, dtype=tf.float32)
y_logit = y_logits

true_label = tf.argmax(y_label, 1)
pred_label = tf.argmax(y_logit, 1)

# loss
loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_label, logits=y_logit)
loss = tf.reduce_mean(loss)

# accuracy
accu = tf.equal(true_label, pred_label)
accu = tf.reduce_mean(tf.cast(accu, tf.float32))

# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=Learning_Rate)
train_op  = optimizer.minimize(loss)


def main():
    # Train and Valid
    sess = tf.Session()
    train(sess)
    valid(sess)
    test(sess)
    sess.close()


if __name__ == '__main__':
    main()