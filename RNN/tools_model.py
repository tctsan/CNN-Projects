import tensorflow as tf
import numpy as np


def VGG16_feature_extractor(images):
    
    with tf.name_scope('VGG'):
        
        x = conv_layer('conv1_1', images, 64)
        x = conv_layer('conv1_2', x, 64)
        x = max_pooling('maxpool1', x)
        #x = tf.nn.dropout(x, keep_prob)

        x = conv_layer('conv2_1', x, 128)
        x = conv_layer('conv2_2', x, 128)
        x = max_pooling('maxpool2', x)
        #x = tf.nn.dropout(x, keep_prob)

        x = conv_layer('conv3_1', x, 256)
        x = conv_layer('conv3_2', x, 256)
        x = conv_layer('conv3_3', x, 256)
        x = max_pooling('maxpool3', x)
        #x = tf.nn.dropout(x, keep_prob)

        x = conv_layer('conv4_1', x, 512)
        x = conv_layer('conv4_2', x, 512)
        x = conv_layer('conv4_3', x, 512)
        x = max_pooling('maxpool4', x)
        #x = tf.nn.dropout(x, keep_prob)

        x = conv_layer('conv5_1', x, 512)
        x = conv_layer('conv5_2', x, 512)
        x = conv_layer('conv5_3', x, 512)
        x = max_pooling('maxpool5', x)
        #x = tf.nn.dropout(x, keep_prob)
        
        x = tf.reshape(x, [-1, 7*7*512])
        #print("feature  shape : ", x.shape)
        
        #x = fc_layer('fc6', x, out_nodes=4096, is_softmax=False)

        return x


def classifier(x, n_classes, keep_prob):
    x = fc_layer('fc1', x, out_nodes=4096, is_softmax=False)
    x = tf.nn.dropout(x, keep_prob)
    
    x = fc_layer('fc2', x, out_nodes=4096, is_softmax=False)
    x = tf.nn.dropout(x, keep_prob)
    
    x = fc_layer('fc3', x, out_nodes=n_classes, is_softmax=True)
    
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
        
        #print(layer_name, " shape : ", x_output.get_shape())
        
    return x_output


def max_pooling(layer_name, x, kernel_size=[1,2,2,1], stride=[1,2,2,1]):
    with tf.name_scope(layer_name):
        x_output = tf.nn.max_pool(x, ksize=kernel_size, strides=stride, padding='SAME')
        #print(layer_name, "shape : ", x_output.get_shape())
        
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
    
        #print(layer_name, " shape :" , x.get_shape())
        
    return x


def load_with_skip(data_path, session, skip_layer):
    #load pre-trained parameter
    data_dict = np.load(data_path, encoding='latin1').item()
    for key in data_dict:
        if key not in skip_layer:
            with tf.variable_scope(key, reuse=True):
                for subkey, data in zip(('weights', 'biases'), data_dict[key]):
                    session.run(tf.get_variable(subkey).assign(data))


def loss_accuracy(y_labels, y_logits):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_labels, logits=y_logits)
    cross_entropy = tf.reduce_mean(cross_entropy, name='cross_entropy')
    
    predict_class = tf.argmax(y_logits, 1)
    actual_class  = tf.argmax(y_labels, 1)

    correct_prediction = tf.equal(predict_class, actual_class)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    return cross_entropy, accuracy, predict_class, actual_class

