import sys
import os
import numpy as np
import tensorflow as tf
import random
import datetime

#import matplotlib as matplt
#matplt.use('agg')
#import matplotlib.pyplot as plt

from sklearn import neighbors # KNN
#from sklearn.manifold import TSNE

from module_CNN import CNN
from triplet_loss import batch_all_triplet_loss
from util import *

class task2(object):
	def __init__(self,sess,args):
		self.sess				= sess

		self.phase				= args.phase			# train or test
		self.pre_train			= args.pre_train		# False
		self.continue_train		= args.continue_train	# False
		self.continue_epoch		= args.continue_epoch	# 0

		self.input_train_dir	= args.input_train_dir	# ./dataset/task2-train-dataset
		self.input_test_dir		= args.input_test_dir	# ./dataset/task2-test-dataset
		self.output_dir			= args.output_dir		# .
		self.log_dir			= args.log_dir			# ./logs
		self.ckpt_dir			= args.ckpt_dir			# ./logs/checkpoint
		self.pre_dir			= args.pre_dir			# ./logs/pre-train

		self.image_size			= args.image_size		# 32
		self.image_channel		= args.image_channel	# 3

		self.n_train			= args.n_train			# 40K + 20/100/200
		self.n_valid			= args.n_valid			# 10K - 20/100/200
		self.n_test				= args.n_test			# 2K
		self.n_classes			= args.n_classes		# 100
		self.k_shot				= args.k_shot			# 1/5/10
		self.random_seed		= args.random_seed		# 666/666/888

		self.learning_rate		= args.learning_rate	# 0.0001
		self.batch_size			= args.batch_size		# 120
		self.iteration			= args.iteration		# 334/335/335
		self.epoch				= args.epoch			# 600

		# triplet parameters
		self.n_embeddings		= args.n_embeddings		# 128
		self.margin				= args.margin			# 0.5

		# KNN
		self.n_neighbor			= args.n_neighbor		# 1
		
		# build model & make checkpoint saver
		self.build_model()
		self.saver = tf.train.Saver()

		print("\n === Args =================================")
		for arg in sorted(vars(args)):
			print("\t", arg, ":", getattr(args, arg))
		print(" ==========================================\n")

		# read data
		self.read_data()


	def build_model(self):
		# placeholder
		self.x_images	= tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.image_channel], name='image')
		self.y_labels	= tf.placeholder(tf.int64,   [None, ], name='label')

		self.keep_prob	= tf.placeholder(tf.float32, name='keep_prob')
		#self.is_train	= tf.placeholder(tf.bool, name='is_train')

		self.embeddings	= CNN(self.x_images, self.keep_prob, self.n_embeddings)

		# triplet loss
		self.loss, _ = batch_all_triplet_loss(self.y_labels, self.embeddings, margin=self.margin)

		# optimizer
		self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

		# accuracy
		self.true_label	= tf.placeholder(tf.int64, [None, ], name='true_label')
		self.pred_label	= tf.placeholder(tf.int64, [None, ], name='pred_label')
		self.accu = tf.reduce_mean(tf.cast(tf.equal(self.true_label, self.pred_label), tf.float32))


	def read_data(self):
		# read data
		if self.phase == "pretrain":
			self.train_images, self.train_labels, self.valid_images, self.valid_labels = \
				read_base_data(self.input_train_dir)
		else:
			self.train_images, self.train_labels, self.valid_images, self.valid_labels = \
				read_novel_data(self.input_train_dir, self.k_shot, self.random_seed)


	def train(self):
		# variable initialize
		self.sess.run(tf.global_variables_initializer())

		# load or not checkpoint
		if self.phase == "pretrain" and self.continue_train and self.pretrain_load():
			print(" [*] before pretraining, Load SUCCESS \n")
		elif self.continue_train and self.checkpoint_load():
			print(" [*] before training, Load SUCCESS \n")
		elif self.pre_train and self.pretrain_load():
			print(" [*] before training, Load pretrain model SUCCESS \n")
		else:
			print(" [!] before training, no need to Load \n")

		# epoch = 600, batch = 120
		for epochs in range(self.continue_epoch, self.continue_epoch+self.epoch):
			rand_id = random.sample(range(self.n_train), self.n_train)

			for iters in range(self.iteration):
				bid = rand_id[self.batch_size*iters : self.batch_size*(iters+1)]

				if iters == (self.iteration-1):
					bid = rand_id[self.batch_size*iters : self.n_train]
			
				batch_image = self.train_images[bid].reshape([-1, self.image_size, self.image_size, self.image_channel])
				batch_label = self.train_labels[bid]
				
				feed = {self.x_images: batch_image, self.y_labels: batch_label, self.keep_prob: 0.5}

				_, train_loss = self.sess.run([self.train_op, self.loss], feed_dict=feed)

			print("[%s] Epoch %3d: train loss %f" % (datetime.datetime.now(), epochs, train_loss))

			if self.phase == "pretrain":
				txtfile = os.path.join(self.log_dir, 'task2_train_learning_curve_pre.txt')
			else:
				txtfile = os.path.join(self.log_dir, 'task2_train_learning_curve_%d.txt'%self.k_shot)

			txtfile = open(txtfile, 'a')
			txtfile.write("%d, %f\n" % (int(epochs), train_loss))
			txtfile.close()
			
		# Save model
		self.checkpoint_save(self.continue_epoch+self.epoch)


	def valid(self):
		print(datetime.datetime.now(), "Validation ...")

		# load or not checkpoint
		if self.phase=="valid" and self.checkpoint_load():
			print(" [*] before validation, Load SUCCESS ")
		else:
			print(" [!] before validation, no need to Load ")
		
		# Train data
		train_images = self.train_images
		train_labels = self.train_labels

		train_embeddings = []
		for i in range(20 * self.k_shot):
			images = train_images[i:i+1].reshape([-1, self.image_size, self.image_size, self.image_channel])

			feed = {self.x_images: images, self.keep_prob: 1.0}
			embeddings = self.sess.run(self.embeddings, feed_dict=feed)
			train_embeddings.append(embeddings)

		train_embeddings = np.array(train_embeddings)
		train_embeddings = train_embeddings.reshape([-1, self.n_embeddings])

		self.train_embeddings = train_embeddings

		# Valid
		valid_pred = []

		txtfile = os.path.join(self.log_dir, 'task2_valid_pred_%d.txt'%self.k_shot)
		txtfile = open(txtfile, 'w')
		txtfile.write("image_id,predicted_label\n")

		print(" [predict] KNN...")
		valid_embeddings = []
		for i in range(self.n_valid):
			images = self.valid_images[i:i+1].reshape([-1, self.image_size, self.image_size, self.image_channel])

			feed = {self.x_images: images, self.keep_prob: 1.0}
			embeddings = self.sess.run(self.embeddings, feed_dict=feed)
			valid_embeddings.append(embeddings)

		valid_embeddings = np.array(valid_embeddings)
		valid_embeddings = valid_embeddings.reshape([-1, self.n_embeddings])

		knn_clf = neighbors.KNeighborsClassifier(self.n_neighbor)
		knn_clf.fit(self.train_embeddings, train_labels)

		valid_pred = knn_clf.predict(valid_embeddings)

		for i in range(self.n_valid):
			txtfile.write("%d,%d\n" % (i, valid_pred[i]))

		txtfile.close()

		'''
		txtfile = os.path.join(self.log_dir, 'task2_valid_pred_%d.txt'%self.k_shot)
		txtfile = open(txtfile, 'r')

		lines = []
		for i, line in enumerate(txtfile): 
			line = line.strip()
			if i != 0:
				lines.append(line)
		txtfile.close()

		# Add data to list
		valid_pred = []
		for i in range(len(lines)):
			data = lines[i].split(",")
			valid_pred.append(int(data[1]))
		valid_pred = np.array(valid_pred)
		'''

		valid_accu = self.sess.run(self.accu, feed_dict={self.true_label: self.valid_labels, 
														 self.pred_label: valid_pred})

		txtfile = os.path.join(self.log_dir, 'task2_train_learning_curve_%d.txt'%self.k_shot)
		txtfile = open(txtfile, 'a')
		txtfile.write("Valid Accu: %f\n" % valid_accu)
		txtfile.close()

		print("%s Valid Accu: %f\n" % (datetime.datetime.now(), valid_accu))


	def test(self):
		print(datetime.datetime.now(), "Test ...")

		# load or not checkpoint
		if self.phase=="test" and self.checkpoint_load():
			print(" [*] before testing, Load SUCCESS ")
		else:
			print(" [!] before testing, no need to Load ")

		# Train data
		train_images = self.train_images
		train_labels = self.train_labels

		if self.phase=="test":
			train_embeddings = []
			for i in range(20 * self.k_shot):
				images = train_images[i:i+1].reshape([-1, self.image_size, self.image_size, self.image_channel])

				feed = {self.x_images: images, self.keep_prob: 1.0}
				embeddings = self.sess.run(self.embeddings, feed_dict=feed)
				train_embeddings.append(embeddings)

			train_embeddings = np.array(train_embeddings)
			train_embeddings = train_embeddings.reshape([-1, self.n_embeddings])

			self.train_embeddings = train_embeddings

		# Test
		test_embeddings = []
		for i in range(self.n_test):
			images = read_test_data(self.input_test_dir, i)
			images = images.reshape([-1, self.image_size, self.image_size, self.image_channel])

			feed = {self.x_images: images, self.keep_prob: 1.0}
			embeddings = self.sess.run(self.embeddings, feed_dict=feed)
			test_embeddings.append(embeddings)

		test_embeddings = np.array(test_embeddings)
		test_embeddings = test_embeddings.reshape([-1, self.n_embeddings])
		self.test_embeddings = test_embeddings

		knn_clf = neighbors.KNeighborsClassifier(self.n_neighbor)
		knn_clf.fit(self.train_embeddings, train_labels)

		test_pred = knn_clf.predict(test_embeddings)
		self.test_pred = test_pred

		txtfile = os.path.join(self.output_dir, 'task2_test_pred_%d.csv'%self.k_shot)
		txtfile = open(txtfile, 'w')
		txtfile.write("image_id,predicted_label\n")

		for i in range(self.n_test):
			txtfile.write("%d,%s\n" % (i, str(test_pred[i]).zfill(2)))

		txtfile.close()

		print(datetime.datetime.now(), "Test Finished.\n")


	def valid_pretrain(self):
		print(datetime.datetime.now(), "Validation Pre-train ...")
		
		# load or not checkpoint
		if self.phase=="pretrain" and self.pretrain_load():
			print(" [*] before validation pre-train, Load pretrain model SUCCESS ")
		else:
			print(" [!] before validation pre-train, no need to Load ")
		
		# Train data
		train_images = self.train_images
		train_labels = self.train_labels

		train_embeddings = []
		for i in range(self.n_train):
			images = train_images[i:i+1].reshape([-1, self.image_size, self.image_size, self.image_channel])

			feed = {self.x_images: images, self.keep_prob: 1.0}
			embeddings = self.sess.run(self.embeddings, feed_dict=feed)
			train_embeddings.append(embeddings)

		train_embeddings = np.array(train_embeddings)
		train_embeddings = train_embeddings.reshape([-1, self.n_embeddings])

		self.train_embeddings = train_embeddings

		print("train embeddings :", train_embeddings.shape)

		# Valid
		valid_pred = []

		txtfile = os.path.join(self.log_dir, 'task2_valid_pred_pretrain.txt')
		txtfile = open(txtfile, 'w')
		txtfile.write("image_id,predicted_label\n")

		valid_embeddings = []
		for i in range(8000):
			images = self.valid_images[i:i+1].reshape([-1, self.image_size, self.image_size, self.image_channel])

			feed = {self.x_images: images, self.keep_prob: 1.0}
			embeddings = self.sess.run(self.embeddings, feed_dict=feed)
			valid_embeddings.append(embeddings)

		valid_embeddings = np.array(valid_embeddings)
		valid_embeddings = valid_embeddings.reshape([-1, self.n_embeddings])

		print("valid embeddings :", valid_embeddings.shape)

		knn_clf = neighbors.KNeighborsClassifier(50)
		knn_clf.fit(self.train_embeddings, train_labels)

		valid_pred = knn_clf.predict(valid_embeddings)

		for i in range(8000):
			txtfile.write("%d,%d\n" % (i, valid_pred[i]))

		txtfile.close()

		valid_accu = self.sess.run(self.accu, feed_dict={self.true_label: self.valid_labels, 
														 self.pred_label: valid_pred})

		txtfile = os.path.join(self.log_dir, 'task2_train_learning_curve_pre.txt')
		txtfile = open(txtfile, 'a')
		txtfile.write("Valid Accu: %f\n" % valid_accu)
		txtfile.close()

		print("%s Valid Accu: %f\n" % (datetime.datetime.now(), valid_accu))


	def pretrain_load(self):
		print(" [*] Reading pre-train model...")
		'''
		ckpt_name = 'task2_pretrain.model-900'
		self.saver.restore(self.sess, os.path.join(self.pre_dir, ckpt_name))
		print(" [*] Reading", os.path.join(self.pre_dir, ckpt_name))
		return True
		'''
		ckpt = tf.train.get_checkpoint_state(self.pre_dir)
		if ckpt and ckpt.model_checkpoint_path:
			ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
			self.saver.restore(self.sess, os.path.join(self.pre_dir, ckpt_name))
			print(" [*] Reading", os.path.join(self.pre_dir, ckpt_name))
			return True
		else:
			return False
		

	def checkpoint_load(self):
		print(" [*] Reading checkpoint...")
		
		ckpt_name = 'task2_' + str(self.k_shot) + '_shot.model'
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
		if self.phase == "pretrain":
			model_name = 'task2_pretrain.model'
			self.saver.save(self.sess,
							os.path.join(self.pre_dir, model_name),
							global_step=step)

		else:
			model_name = 'task2_' + str(self.k_shot) + '_shot.model'
			self.saver.save(self.sess,
							os.path.join(self.ckpt_dir, model_name),
							global_step=step)


	def read_learning_curve_txt(self, filename):
		step_list = []
		loss_list = []

		txtfile = os.path.join(self.log_dir, 'task2_train_learning_curve_%s.txt' % filename)
		txtfile = open(txtfile, 'r')

		lines = []
		for i, line in enumerate(txtfile): 
			line = line.strip()
			if i % 2 == 0 and line.startswith('Valid Accu') == False:
				lines.append(line)
		txtfile.close()

		for i in range(len(lines)):
			data = lines[i].split(",")
			step_list.append(int(data[0]))
			loss_list.append(float(data[1]))

		return step_list, loss_list


	def save_fig(self):
		# Read data to list
		step_list_pre, loss_list_pre = read_learning_curve_txt('pre')
		step_list_1,   loss_list_1   = read_learning_curve_txt('1')
		step_list_5,   loss_list_5   = read_learning_curve_txt('5')
		step_list_10,  loss_list_10  = read_learning_curve_txt('10')

		# Plot learning curve
		matplt.rcParams.update({'font.size': 14})
		fig, axs = plt.subplots(1, 2, figsize=(20, 6))

		ax1 = axs[0]
		ax1.plot(step_list_pre, loss_list_pre, 'b', linewidth=2)
		ax1.set_xlabel('epoch')
		ax1.set_title('Pre-training Loss')

		ax2 = axs[1]
		ax2.plot(step_list_1,  loss_list_1,  linewidth=3, label='1-shot')
		ax2.plot(step_list_5,  loss_list_5,  linewidth=2, label='5-shot')
		ax2.plot(step_list_10, loss_list_10, linewidth=1, label='10-shot')
		ax2.set_xlabel('epoch')
		ax2.set_title('Training Loss')
		ax2.legend(loc='upper right')

		figname = os.path.join(self.log_dir, 'task2_train_learning_curve.jpg')
		fig.savefig(figname)
		plt.close(fig)
		print(" [Save] task2_train_learning_curve.jpg SUCCESS")


	def visualization_tsne(self):
		train_embeddings = self.train_embeddings
		test_embeddings  = self.test_embeddings

		train_labels = self.train_labels
		test_labels  = self.test_pred

		train_size = train_embeddings.shape[0]
		test_size  = test_embeddings.shape[0]

		embeddings = np.concatenate((train_embeddings, test_embeddings))

		embeddings_tsne = TSNE(n_components=2, perplexity=40, random_state=10).fit_transform(embeddings)
		print("TSNE SUCCESS")

		classes = [0, 10, 23, 30, 32, 35, 48, 54, 57, 59, 60, 64, 66, 69, 71, 82, 91, 92, 93, 95]
		
		# colormap
		cm1 = plt.cm.get_cmap("tab10")
		cm2 = plt.cm.get_cmap("Set3", 10)

		cm = []
		for i in range(10):
			cm.append(cm1(i))
		for i in range(10):
			cm.append(cm2(i))

		# fig
		fig, axs = plt.subplots(1, 1, figsize=(20, 12))

		# training embeddings
		train_embeddings_tsne = embeddings_tsne[0:train_size]
		for i, c in enumerate(classes):
			label_name  = "class_" + str(c).zfill(2)
			label_color = cm[i]
			area = (12)**2

			value = train_embeddings_tsne[train_labels==c]
			axs.scatter(value[:,0], value[:,1], s=area, c=label_color, label=label_name)

		# testing embeddings
		test_embeddings_tsne = embeddings_tsne[train_size:]
		for i, c in enumerate(classes):
			label_name  = "class_" + str(c).zfill(2)
			label_color = cm[i]
			area = (4)**2

			value = test_embeddings_tsne[test_labels==c]
			axs.scatter(value[:,0], value[:,1], s=area, c=label_color, label=label_name)

		axs.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
		axs.set_title(str(self.k_shot) + " shot")

		figname = os.path.join(self.log_dir, 'task2_visualization_tsne_%d.jpg' % self.k_shot)
		fig.savefig(figname)
		plt.close(fig)
		print(" [Save] task2_visualization_tsne_%d.jpg SUCCESS\n" % self.k_shot)

