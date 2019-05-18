import sys
import os

import tensorflow as tf
import numpy as np
import math
import datetime

from tools_reader import *
from tools_model import *


video_dir  = sys.argv[1]  # HW5_data/TrimmedVideos/video/valid
label_path = sys.argv[2]  # HW5_data/TrimmedVideos/label/gt_valid.csv
output_dir = sys.argv[3]  # logs
log_dir    = "./logs/"

video_dict = getVideoList(label_path)
n_video = len(video_dict['Video_name'])

n_feature = 7*7*512
n_classes = 11


tf.reset_default_graph()

# Feature Extractor
images   = tf.placeholder(tf.float32, shape=[None, 224, 224, 3])
features = VGG16_feature_extractor(images)

# Classifier
x_features = tf.placeholder(tf.float32, shape=[None, n_feature])
y_labels   = tf.placeholder(tf.float32, shape=[None, n_classes])
keep_prob  = tf.placeholder(tf.float32)

y_logits = classifier(x_features, n_classes, keep_prob)

loss, accu, pred_label, true_label = loss_accuracy(y_labels, y_logits)


def extract_features(frames_image):
    frame_num = frames_image.shape[0]

    frames_feature = []
    for i in range(frame_num):
        img = frames_image[i].reshape([-1, 224, 224, 3])
        feature = sess.run(features, feed_dict={images: img})
        frames_feature.append(feature)
    
    frames_feature = np.array(frames_feature)
    frames_feature = np.mean(frames_feature, axis=0)

    return frames_feature


def valid():

    # Restore model
    saver = tf.train.Saver()
    saver.restore(sess, log_dir+'p1_classifier.ckpt')
    print("Model restored.\n")

    # load pre trained weights
    vgg16_pre_trained_weights = 'vgg16_weights.npy'
    load_with_skip(vgg16_pre_trained_weights, sess, ['fc6','fc7','fc8'])

    total_accu = 0
    txtfile = open(output_dir+'/p1_valid.txt', 'w')
    
    for i in np.arange(n_video):
        category = video_dict['Video_category'][i]
        name     = video_dict['Video_name'][i]
        frames   = readShortVideo(video_dir, category, name)

        frames_features = extract_features(frames).reshape([-1, n_feature])

        frames_labels = np.zeros(n_classes, dtype=np.float32)
        frames_labels[int(video_dict['Action_labels'][i])] = 1.0
        frames_labels = frames_labels.reshape([-1, n_classes])
        
        feed_dict = {x_features: frames_features, y_labels: frames_labels, keep_prob: 1.0}
        result_accu, result_pred = sess.run([accu, pred_label], feed_dict=feed_dict)
        
        total_accu += result_accu
        txtfile.write("%d\n" % int(result_pred))
        
    total_accu /= n_video
    txtfile.close()

    print("Valid Accuracy: %f\n" % total_accu)


sess = tf.Session()
sess.run(tf.global_variables_initializer())

valid()

sess.close()