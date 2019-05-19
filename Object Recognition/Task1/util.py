import sys
import os
import numpy as np
from PIL import Image

def read_train_data(input_dir):
	image_name = []
	image_list = []
	label_list = []
	
	filepath = os.path.join(input_dir, 'train')

	for i in range(10):
		for j in range(200):
			fname = os.path.join(filepath, 'class_%d'%(i), '%s.png'%(str(j+1).zfill(4)))
			image_name.append(fname)
			label_list.append(i)

	image_temp = [np.array(Image.open(fname).convert('L')) for fname in image_name]

	for i in range(len(image_temp)):
		# normalize [0, 255] to [0, 1]
		normal_image = np.interp(image_temp[i], (0, 255), (0, 1))
		image_list.append(normal_image)
	
	images = np.array(image_list)
	labels = np.array(label_list)

	return images, labels


def read_test_data(input_dir, fid):
	filepath = os.path.join(input_dir, 'test')
	fname = os.path.join(filepath, '%d.png'%fid)

	images = np.array(Image.open(fname).convert('L'))
	# normalize [0, 255] to [0, 1]
	images = np.interp(images, (0, 255), (0, 1))

	return images
