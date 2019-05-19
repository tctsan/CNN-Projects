import sys
import os
import numpy as np
import tensorflow as tf
import random
import datetime

#import matplotlib as matplt
#matplt.use('agg')
#import matplotlib.pyplot as plt

from module_CNN import CNN
from util import *

class task1(object):
	def __init__(self,sess,args):
		self.sess			= sess

		self.phase			= args.phase			# train or test
		self.continue_train	= args.continue_train	# False
		self.continue_epoch	= args.continue_epoch	# 0

		self.input_dir		= args.input_dir		# ./dataset/Fashion_MNIST_student
		self.output_dir		= args.output_dir		# .
		self.log_dir		= args.log_dir			# ./logs
		self.ckpt_dir		= args.ckpt_dir			# ./logs/checkpoint

		self.image_size		= args.image_size		# 28
		self.image_channel	= args.image_channel	# 1

		self.n_train		= args.n_train			# 2K
		self.n_test			= args.n_test			# 10K
		self.n_classes		= args.n_classes		# 10

		self.learning_rate	= args.learning_rate	# 0.0001
		self.batch_size		= args.batch_size		# 128
		self.iteration		= args.iteration		# 16
		self.epoch			= args.epoch			# 150

		# build model & make checkpoint saver
		self.build_model()
		self.saver = tf.train.Saver()

		print("\n === Args =================================")
		for arg in sorted(vars(args)):
			print("\t", arg, ":", getattr(args, arg))
		print(" ==========================================\n")


	def build_model(self):
		# placeholder
		self.x_images	= tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.image_channel])
		self.y_labels	= tf.placeholder(tf.int64,   [None, ])
		self.keep_prob	= tf.placeholder(tf.float32)

		self.y_logits	= CNN(self.x_images, self.keep_prob, self.n_classes)

		# loss
		y_label = tf.one_hot(self.y_labels, depth=self.n_classes, dtype=tf.float32)
		y_logit = self.y_logits

		self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_label, logits=y_logit))

		# accuracy
		self.true_label	= tf.argmax(y_label, 1)
		self.pred_label	= tf.argmax(y_logit, 1)

		self.accu = tf.reduce_mean(tf.cast(tf.equal(self.true_label, self.pred_label), tf.float32))

		# optimizer
		self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)


	def train(self):
		# read data
		train_images, train_labels = read_train_data(self.input_dir)

		# variable initialize
		self.sess.run(tf.global_variables_initializer())

		# load or not checkpoint
		if self.continue_train and self.checkpoint_load():
			print(" [*] before training, Load SUCCESS ")
		else:
			print(" [!] before training, no need to Load ")

		best_loss = 100
		best_accu = 0

		# epoch = 150, iteration = 15+1, batch = 128
		for epochs in range(self.continue_epoch, self.continue_epoch+self.epoch):
			rand_id = random.sample(range(self.n_train), self.n_train)

			for iters in range(self.iteration):
				bid = rand_id[self.batch_size*iters : self.batch_size*(iters+1)]

				if iters == (self.iteration-1):
					bid = rand_id[self.batch_size*iters : self.n_train]
			
				batch_image = train_images[bid].reshape([-1, self.image_size, self.image_size, self.image_channel])
				batch_label = train_labels[bid]
				
				feed = {self.x_images: batch_image, self.y_labels: batch_label, self.keep_prob: 0.5}

				_, train_loss, train_accu = self.sess.run([self.train_op, self.loss, self.accu], 
											feed_dict=feed)

			# Train Result
			total_loss = 0
			total_accu = 0

			for i in range(self.n_train):
				images = train_images[i:i+1].reshape([-1, self.image_size, self.image_size, self.image_channel])
				labels = train_labels[i:i+1]

				feed = {self.x_images: images, self.y_labels: labels, self.keep_prob: 1.0}

				train_loss, train_accu, train_pred = self.sess.run([self.loss, self.accu, self.pred_label], 
																	feed_dict=feed)

				total_loss += train_loss
				total_accu += train_accu

			total_loss /= self.n_train
			total_accu /= self.n_train

			print(datetime.datetime.now())
			print("Epoch %2d: train loss %f, accu %f\n" % (epochs, total_loss, total_accu))

			txtfile = os.path.join(self.log_dir, 'task1_train_learning_curve.txt')
			txtfile = open(txtfile, 'a')
			txtfile.write("%d, %f, %f\n" % (int(epochs), total_loss, total_accu))
			txtfile.close()

			# Save model
			if total_accu >= best_accu and total_loss < best_loss:
				self.checkpoint_save(epochs)
				best_loss = total_loss
				best_accu = total_accu


	def test(self):
		# load or not checkpoint
		if self.phase=="test" and self.checkpoint_load():
			print(" [*] before testing, Load SUCCESS ")
		else:
			print(" [!] before testing, no need to Load ")

		txtfile = os.path.join(self.output_dir, 'task1_test_pred.csv')
		txtfile = open(txtfile, 'w')
		txtfile.write("image_id,predicted_label\n")

		for i in range(self.n_test):
			images = read_test_data(self.input_dir, i)
			images = images.reshape([-1, self.image_size, self.image_size, self.image_channel])

			feed = {self.x_images: images, self.keep_prob: 1.0}
			test_pred = self.sess.run([self.pred_label], feed_dict=feed)

			txtfile.write("%d,%d\n" % (i, int(test_pred[0])))

		txtfile.close()


	def checkpoint_load(self):
		print(" [*] Reading checkpoint...")
		
		ckpt_name = 'task1_CNN.model'
		self.saver.restore(self.sess, os.path.join(self.ckpt_dir, ckpt_name))
		print(" [*] Reading", os.path.join(self.ckpt_dir, ckpt_name))
		return True
		'''
		ckpt = tf.train.get_checkpoint_state(self.ckpt_dir)
		if ckpt and ckpt.model_checkpoint_path:
			ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
			self.saver.restore(self.sess, os.path.join(self.ckpt_dir, ckpt_name))
			print(" [*] Reading", os.path.join(self.ckpt_dir, ckpt_name))
			return True
		else:
			return False
		'''


	def checkpoint_save(self, step):
		model_name = "task1_CNN.model"
		self.saver.save(self.sess,
						os.path.join(self.ckpt_dir, model_name),
						global_step=step)


	def save_fig(self):
		txtfile = os.path.join(self.log_dir, 'task1_train_learning_curve.txt')
		txtfile = open(txtfile, 'r')

		lines = []
		for i, line in enumerate(txtfile): 
			line = line.strip()
			lines.append(line)
		txtfile.close()

		# Add data to list
		step_list = []
		loss_list = []
		accu_list = []

		for i in range(len(lines)):
			data = lines[i].split(",")
			step_list.append(int(data[0]))
			loss_list.append(float(data[1]))
			accu_list.append(float(data[2]))

		# Plot learning curve
		matplt.rcParams.update({'font.size': 14})
		fig, axs = plt.subplots(1, 2, figsize=(20, 6))

		ax1 = axs[0]
		ax1.plot(step_list, loss_list, 'b', linewidth=2)
		ax1.set_xlabel('epoch')
		ax1.set_title('Training Loss')

		ax2 = axs[1]
		ax2.plot(step_list, accu_list, 'g', linewidth=2)
		ax2.set_xlabel('epoch')
		ax2.set_title('Training Accuracy')

		figname = os.path.join(self.log_dir, 'task1_train_learning_curve.jpg')
		fig.savefig(figname)
		plt.close(fig)
		print(" [Save] task1_train_learning_curve.jpg SUCCESS.")
