import tensorflow as tf

def CNN(images, keep_prob, n_embeddings):
	
	with tf.name_scope('CNN'):
		nf = 128

		x = conv_layer('conv1_1', images, nf)
		x = conv_layer('conv1_2', x, nf)
		x = max_pooling('maxpool1', x)
		x = tf.nn.dropout(x, keep_prob)

		x = conv_layer('conv2_1', x, nf*2)
		x = conv_layer('conv2_2', x, nf*2)
		x = max_pooling('maxpool2', x)
		x = tf.nn.dropout(x, keep_prob)

		x = conv_layer('conv3_1', x, nf*4)
		x = conv_layer('conv3_2', x, nf*4)

		k_size = x.get_shape()[1].value
		x = global_avg_pooling('global_avgpool', x, kernel_size=[1,k_size,k_size,1], stride=[1,k_size,k_size,1])

		x = fc_layer('fc1', x, out_nodes=n_embeddings, is_softmax=True)
		
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
		
		print(layer_name, " shape : ", x_output.get_shape())
		
	return x_output


def max_pooling(layer_name, x, kernel_size=[1,2,2,1], stride=[1,2,2,1]):
	with tf.name_scope(layer_name):
		x_output = tf.nn.max_pool(x, ksize=kernel_size, strides=stride, padding='SAME')
		print(layer_name, "shape : ", x_output.get_shape())
		
	return x_output


def global_avg_pooling(layer_name, x, kernel_size, stride):
	with tf.name_scope(layer_name):
		x_output = tf.nn.avg_pool(x, ksize=kernel_size, strides=stride, padding='SAME')
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
