import sys
import os

import tensorflow as tf
import numpy as np
import math
import datetime

from tensorflow.contrib import rnn

from tools_reader import *
from tools_model import *


video_dir  = sys.argv[1]  # HW5_data/TrimmedVideos/video/valid
label_path = sys.argv[2]  # HW5_data/TrimmedVideos/label/gt_valid.csv
output_dir = sys.argv[3]  # logs
log_dir    = "./logs/"

video_dict = getVideoList(label_path)
n_video = len(video_dict['Video_name'])


############################################################################


n_input   = 25088
n_steps   = 300
n_hidden  = 512
n_classes = 11


def RNN(x, n_x, weights, biases, keep_prob, is_training):
    
    print("X_sequences shape :", x.shape)

    # LSTM - Multi-layer
    LSTM_cell_1 = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    LSTM_cell_2 = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    LSTM_cell_3 = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    LSTM_cells  = rnn.MultiRNNCell([LSTM_cell_1, LSTM_cell_2, LSTM_cell_3], state_is_tuple=True)
    
    h_outputs, final_states = tf.nn.dynamic_rnn(LSTM_cells, x, sequence_length=n_x, dtype=tf.float32)
    
    # final LSTM output
    outputs = final_states[2].h
    print("LSTM_output shape :", outputs.shape)

    _outputs = tf.contrib.layers.batch_norm(outputs, epsilon=1e-5, is_training=is_training)
    _outputs = tf.nn.dropout(_outputs, keep_prob)

    # fc - RNN feature (512-dim)
    fc1 = tf.nn.relu(tf.matmul(_outputs, weights['fc1']) + biases['fc1'])
    fc1 = tf.contrib.layers.batch_norm(fc1, epsilon=1e-5, is_training=is_training)
    fc1 = tf.nn.dropout(fc1, keep_prob)
    print("FC1         shape :", fc1.shape)

    # predicted labels (512 -> 11)
    y_pred = tf.matmul(fc1, weights['out']) + biases['out']
    print("Pred_labels shape :", y_pred.shape)
    
    return fc1, y_pred, outputs


tf.reset_default_graph()

# Feature Extractor
print("Feature Extractor - VGG")
images   = tf.placeholder(tf.float32, shape=[None, 224, 224, 3])
features = VGG16_feature_extractor(images)

# RNN
print("RNN")
x_sequences = tf.placeholder(tf.float32, shape=[None, n_steps, n_input], name="x_sequences")
y_labels    = tf.placeholder(tf.float32, shape=[None, n_classes], name="y_labels")

n_sequences = tf.placeholder(tf.int32, shape=[None,], name="n_sequences")
keep_prob   = tf.placeholder(tf.float32, name="keep_prob")
is_training = tf.placeholder(tf.bool, name="is_training")

weights = {
    'fc1': tf.Variable(tf.random_normal([n_hidden, n_hidden]),  name="w_fc1"),
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]), name="w_out")
}
biases = {
    'fc1': tf.Variable(tf.random_normal([n_hidden]),  name="b_fc1"),
    'out': tf.Variable(tf.random_normal([n_classes]), name="b_out")
}

y_feature, y_logits, lstm_output = RNN(x_sequences, n_sequences, weights, biases, keep_prob, is_training)

# Evaluate
loss, accu, pred_label, true_label = loss_accuracy(y_labels, y_logits)


def extract_features(frames_image):
    frame_num = frames_image.shape[0]

    frames_feature = []
    for i in range(frame_num):
        img = frames_image[i].reshape([-1, 224, 224, 3])
        feature = sess.run(features, feed_dict={images: img})
        frames_feature.append(feature)
    
    frames_feature = np.array(frames_feature)
    return frames_feature


def valid():
    print(datetime.datetime.now())
    print("Validation...\n")

    # load pre-trained weights
    vgg16_pre_trained_weights = 'vgg16_weights.npy'
    load_with_skip(vgg16_pre_trained_weights, sess, ['fc6','fc7','fc8'])
    print("Pre-trained weights loaded.\n")

    video_features = []
    video_steps = []
    for i in np.arange(n_video):
        category = video_dict['Video_category'][i]
        name     = video_dict['Video_name'][i]
        frames   = readShortVideo(video_dir, category, name)

        frames_features = extract_features(frames).reshape([-1, n_input])
        #print(i, frames_features.shape)

        f_steps = frames_features.shape[0]
        frames_features = np.pad(frames_features, ((0, n_steps-f_steps), (0, 0)), 'constant')
        frames_features = frames_features.reshape([-1, n_steps, n_input])

        video_features.append(frames_features)
        video_steps.append(np.array([f_steps]))


    sess.run(tf.global_variables_initializer())

    # Restore model
    tvars = tf.trainable_variables()
    variables_to_restore = [v for v in tvars if v.name.split('/')[0].startswith('conv') == False]

    model_path = log_dir + 'p2_RNN.ckpt'
    saver = tf.train.Saver(variables_to_restore)
    saver.restore(sess, model_path)
    print("\n\nModel restored.\n")

    total_accu = 0
    txtfile = open(output_dir+'/p2_result.txt', 'w')

    for i in np.arange(n_video):
        frames_labels = np.zeros(n_classes, dtype=np.float32)
        frames_labels[int(video_dict['Action_labels'][i])] = 1.0
        frames_labels = frames_labels.reshape([-1, n_classes])

        feed_dict = {x_sequences: video_features[i], y_labels: frames_labels, 
                     n_sequences: video_steps[i], keep_prob: 1.0, is_training: False}
        
        result_accu, result_pred = sess.run([accu, pred_label], feed_dict=feed_dict)

        total_accu += result_accu
        txtfile.write("%d\n" % int(result_pred[0]))

    total_accu /= n_video
    txtfile.close()

    print("\nValid Accuracy: %f\n" % total_accu)

    print(datetime.datetime.now())
    print("Validation finished.\n")


sess = tf.Session()
sess.run(tf.global_variables_initializer())

valid()

sess.close()