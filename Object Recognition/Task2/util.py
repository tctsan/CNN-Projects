import sys
import os
import glob
import numpy as np
from PIL import Image

def read_novel_data(input_dir, k_shot, random_seed):
	train_image_name = []
	train_image_list = []
	train_label_list = []

	valid_image_name = []
	valid_image_list = []
	valid_label_list = []

	# novel data
	novel_size = 20 * k_shot

	np.random.seed(seed=random_seed)
	rand_id = np.random.randint(500, size=novel_size)

	novel_dir = os.path.join(input_dir, 'novel')
	novel_class_list = os.listdir(novel_dir)
	for i, classes in enumerate(novel_class_list):
		class_num = classes.split("_")
		filepath  = os.path.join(novel_dir, classes, 'train')

		id_list = rand_id[k_shot*i: k_shot*(i+1)]

		for j in range(500):
			# tain (k_shot)
			if j in id_list:
				fname = os.path.join(filepath, '%s.png'%(str(j+1).zfill(3)))
				train_image_name.append(fname)
				train_label_list.append(int(class_num[1]))
			# valid
			else:
				fname = os.path.join(filepath, '%s.png'%(str(j+1).zfill(3)))
				valid_image_name.append(fname)
				valid_label_list.append(int(class_num[1]))
	
	# image
	image_temp = [np.array(Image.open(fname).convert('RGB')) for fname in train_image_name]
	
	for i in range(len(image_temp)):
		# normalize [0, 255] to [0, 1]
		normal_image = np.interp(image_temp[i], (0, 255), (0, 1))
		train_image_list.append(normal_image)
	
	train_images = np.array(train_image_list)
	train_labels = np.array(train_label_list)


	valid_image_temp = [np.array(Image.open(fname).convert('RGB')) for fname in valid_image_name]
	
	for i in range(len(valid_image_temp)):
		# normalize [0, 255] to [0, 1]
		normal_image = np.interp(valid_image_temp[i], (0, 255), (0, 1))
		valid_image_list.append(normal_image)
	
	valid_images = np.array(valid_image_list)
	valid_labels = np.array(valid_label_list)

	return train_images, train_labels, valid_images, valid_labels


def read_base_data(input_dir):
	train_image_name = []
	train_image_list = []
	train_label_list = []

	valid_image_name = []
	valid_image_list = []
	valid_label_list = []

	# base data
	base_dir = os.path.join(input_dir, 'base')
	base_class_list = os.listdir(base_dir)
	for classes in base_class_list:
		class_num = classes.split("_")

		filepath = os.path.join(base_dir, classes, 'train')
		filepath = glob.glob(filepath+'/*.png')

		for fname in filepath:
			train_image_name.append(fname)
			train_label_list.append(int(class_num[1]))

		filepath2 = os.path.join(base_dir, classes, 'test')
		filepath2 = glob.glob(filepath2+'/*.png')

		for fname in filepath2:
			valid_image_name.append(fname)
			valid_label_list.append(int(class_num[1]))

	# image
	image_temp = [np.array(Image.open(fname).convert('RGB')) for fname in train_image_name]
	
	for i in range(len(image_temp)):
		# normalize [0, 255] to [0, 1]
		normal_image = np.interp(image_temp[i], (0, 255), (0, 1))
		train_image_list.append(normal_image)
	
	train_images = np.array(train_image_list)
	train_labels = np.array(train_label_list)


	valid_image_temp = [np.array(Image.open(fname).convert('RGB')) for fname in valid_image_name]
	
	for i in range(len(valid_image_temp)):
		# normalize [0, 255] to [0, 1]
		normal_image = np.interp(valid_image_temp[i], (0, 255), (0, 1))
		valid_image_list.append(normal_image)
	
	valid_images = np.array(valid_image_list)
	valid_labels = np.array(valid_label_list)

	return train_images, train_labels, valid_images, valid_labels


def read_test_data(input_dir, fid):
	filepath = os.path.join(input_dir, 'test')
	fname = os.path.join(filepath, '%d.png'%fid)

	images = np.array(Image.open(fname).convert('RGB'))
	# normalize [0, 255] to [0, 1]
	images = np.interp(images, (0, 255), (0, 1))

	return images

