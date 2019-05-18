import sys
import os
from os import listdir
import glob

import tensorflow as tf
import numpy as np
import math
import random
import datetime

from PIL import Image
from tensorflow.contrib import rnn

from tools_model import *


video_dir  = sys.argv[1]  # HW5_data/FullLengthVideos/videos/valid
output_dir = sys.argv[2]  # logs
log_dir    = "./logs/"


def read_vedio_data():
    dir_path = video_dir + '/'
    category_list = listdir(dir_path)

    frame_num_list = []
    frame_image_list = []

    for i, category in enumerate(category_list):
        # frame image
        frame = []
        file_path = glob.glob(dir_path + category + "/*.jpg")

        for fname in file_path:
            frame.append(fname)

        frame_temp = [np.array(Image.open(fname).convert('RGB').resize((224,224),Image.BICUBIC)) for fname in frame]

        for j in range(len(frame_temp)):
            frame_image_list.append(frame_temp[j])
            
        frame_num_list.append(len(frame))
        
    return category_list, frame_num_list, frame_image_list


category_list, num_list, image_list = read_vedio_data()
n_category = len(category_list)
n_frame    = sum(int(i) for i in num_list)

n_input   = 25088
n_steps   = 300
n_hidden  = 512
n_classes = 11


def RNN_seq2seq(x, n_x, keep_prob, is_training):
    
    print("X_sequences shape :", x.shape)

    # LSTM - Multi-layer
    LSTM_cell_1 = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    LSTM_cell_2 = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    LSTM_cell_3 = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    LSTM_cells  = rnn.MultiRNNCell([LSTM_cell_1, LSTM_cell_2, LSTM_cell_3], state_is_tuple=True)
    
    h_outputs, final_states = tf.nn.dynamic_rnn(LSTM_cells, x, sequence_length=n_x, dtype=tf.float32)
    
    # final LSTM output
    x_len = n_x[0]
    h_outputs = h_outputs[:, 0:x_len]
    h_outputs = tf.reshape(h_outputs, shape=[-1, x_len, n_hidden])
    outputs = tf.contrib.layers.batch_norm(h_outputs, epsilon=1e-5, is_training=is_training)
    outputs = tf.nn.dropout(outputs, keep_prob)
    print("LSTM_output shape :", outputs.shape)
    
    # fc - RNN feature (512-dim)
    fc1 = tf.layers.dense(outputs, units=n_hidden, activation=tf.nn.relu)
    fc1 = tf.contrib.layers.batch_norm(fc1, epsilon=1e-5, is_training=is_training)
    fc1 = tf.nn.dropout(fc1, keep_prob)
    print("FC1         shape :", fc1.shape)
    
    # predicted labels (512 -> 11)
    y_pred = tf.layers.dense(fc1, units=n_classes, activation=None)
    print("Pred_labels shape :", y_pred.shape)
    
    return y_pred


tf.reset_default_graph()

# Feature Extractor
print("Feature Extractor - VGG")
images   = tf.placeholder(tf.float32, shape=[None, 224, 224, 3])
features = VGG16_feature_extractor(images)

# RNN seq2seq
print("RNN seq2seq")
x_sequences = tf.placeholder(tf.float32, shape=[None, None, n_input], name="x_sequences")

n_sequences = tf.placeholder(tf.int32, shape=[None,], name="n_sequences")
keep_prob   = tf.placeholder(tf.float32, name="keep_prob")
is_training = tf.placeholder(tf.bool, name="is_training")

y_logits   = RNN_seq2seq(x_sequences, n_sequences, keep_prob, is_training)
pred_label = tf.argmax(y_logits, 2)


def extract_features(frame_num, frame_image_list):
    frame_image = np.array(frame_image_list)
    frame_feature = []
    
    for i in range(frame_num):
        if i % 200 == 0:
            print("...", i)
        
        img = frame_image[i].reshape([-1, 224, 224, 3])
        feature = sess.run(features, feed_dict={images: img})
        frame_feature.append(feature)
    
    frame_feature = np.array(frame_feature)
    return frame_feature


def valid(feature_array):
    print(datetime.datetime.now())
    print("Validation...\n")

    # Restore model
    tvars = tf.trainable_variables()
    variables_to_restore = [v for v in tvars if v.name.split('/')[0].startswith('conv') == False]

    model_path = log_dir + 'p3_RNN_seq2seq.ckpt'
    saver = tf.train.Saver(variables_to_restore)
    saver.restore(sess, model_path)
    print("Model restored.\n")

    for i in np.arange(n_category):
        v_start = sum(num_list[0:i])
        v_end   = sum(num_list[0:i+1])
                    
        valid_feature = np.array(feature_array[v_start:v_end]).reshape([1,-1,n_input])
        valid_steps   = np.array([valid_feature.shape[1]])
        
        feed_dict = {x_sequences: valid_feature, n_sequences: valid_steps, 
                       keep_prob: 1.0, is_training: False}
        
        result_pred = sess.run([pred_label], feed_dict=feed_dict)
        
        txtfile = open(output_dir+'/'+category_list[i]+'.txt', 'w')
        for j in range(num_list[i]):
            #print(result_pred)
            print(result_pred[0][0][j])

            txtfile.write("%d\n" % int(result_pred[0][0][j]))
        txtfile.close()

    print(datetime.datetime.now())
    print("Validation finished.\n")


sess = tf.Session()
sess.run(tf.global_variables_initializer())

# load pre-trained weights
vgg16_pre_trained_weights = 'vgg16_weights.npy'
load_with_skip(vgg16_pre_trained_weights, sess, ['fc6','fc7','fc8'])
print("Pre-trained weights loaded.\n")

feature_array = extract_features(n_frame, image_list)

valid(feature_array)

sess.close()
