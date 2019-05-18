import glob
import numpy as np
import pandas as pd
from PIL import Image


# VAE, GAN, ACGAN
def read_data(input_dir, mode, fid):
	# face images
	face = []
	face_list = []
	
	filepath = input_dir + mode + "/"
	for i in range(np.array(fid).shape[0]):
		fname = filepath + str(fid[i]).zfill(5) + ".png"
		face.append(fname)
	'''
	filepath = glob.glob(filepath + "*.png")
	for fname in filepath:
		face.append(fname)
	'''
	face_temp = [np.array(Image.open(fname).convert('RGB')) for fname in face]

	for i in range(np.array(face_temp).shape[0]):
		# normalize [0, 255] to [0, 1]
		normal_img = np.interp(face_temp[i], (0, 255), (0, 1))
		face_list.append(normal_img)
	
	face_image = np.array(face_list)
	return face_image


# VAE, ACGAN
def read_attr(input_dir, attr):
	train_attr = pd.read_csv(input_dir + "train.csv")
	test_attr  = pd.read_csv(input_dir + "test.csv")

	attr_class = np.hstack((np.array(train_attr[attr]), 
							  np.array(test_attr[attr])))

	return attr_class


# VAE, GAN, ACGAN
def save_image(out_dir, file_name, image, epochs, iters, rows, cols):
	for i in range(rows):
		img_row = image[i*cols]
		for j in range(1, cols):
			img_row = np.concatenate([img_row, image[i*cols+j]], axis=1)
				
		if i == 0:
			img_merge = img_row
		else:
			img_merge = np.concatenate([img_merge, img_row], axis=0)

	normal_img = np.interp(img_merge, (0, 1), (0, 255))
	img = np.uint8(normal_img)
	img = Image.fromarray(img)
	img.save(out_dir + file_name + "_epoch" + str(epochs).zfill(3)
					 + "_iter"  + str(iters).zfill(3) + ".png")


# VAE
def save_test_image(out_dir, image, tid):
	normal_img = np.interp(image, (0, 1), (0, 255))
	img = np.uint8(normal_img)
	img = Image.fromarray(img)
	img.save(out_dir + str(tid).zfill(5) + ".png")


def save_image_final(out_dir, file_name, image, rows, cols):
	for i in range(rows):
		img_row = image[i*cols]
		for j in range(1, cols):
			img_row = np.concatenate([img_row, image[i*cols+j]], axis=1)
				
		if i == 0:
			img_merge = img_row
		else:
			img_merge = np.concatenate([img_merge, img_row], axis=0)

	normal_img = np.interp(img_merge, (0, 1), (0, 255))
	img = np.uint8(normal_img)
	img = Image.fromarray(img)
	img.save(out_dir + file_name)
