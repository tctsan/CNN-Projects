import argparse
import os
import tensorflow as tf
from model import task1

# argument parser
parser = argparse.ArgumentParser(description='')
parser.add_argument('--phase',			type=str,	default='test',	help='train or test')
parser.add_argument('--continue_train',	type=bool,	default=False)
parser.add_argument('--continue_epoch',	type=int,	default=0)

parser.add_argument('--input_dir',		type=str,	default=os.path.join('.', 'dataset', 'Fashion_MNIST_student'))
parser.add_argument('--output_dir',		type=str,	default='.')
parser.add_argument('--log_dir',		type=str,	default=os.path.join('.', 'logs'))
parser.add_argument('--ckpt_dir',		type=str,	default=os.path.join('.', 'logs', 'checkpoint'))

parser.add_argument('--image_size',		type=int,	default=28)
parser.add_argument('--image_channel',	type=int,	default=1)

parser.add_argument('--n_train',		type=int,	default=2000)
parser.add_argument('--n_test',			type=int,	default=10000)
parser.add_argument('--n_classes',		type=int,	default=10)

parser.add_argument('--learning_rate',	type=float,	default=0.0001)
parser.add_argument('--batch_size',		type=int,	default=128)
parser.add_argument('--iteration',		type=int,	default=16)
parser.add_argument('--epoch',			type=int,	default=150)

args = parser.parse_args()

def main(_):
	# make directory if not exist
	try: os.makedirs(args.log_dir)
	except: pass
	try: os.makedirs(args.ckpt_dir)
	except: pass

	# run session
	tf.reset_default_graph()

	with tf.Session() as sess:
		model = task1(sess,args)
		
		if args.phase == 'train':
			model.train()
			model.test()
			#model.save_fig()
		else:
			model.test()

# run main function
if __name__ == '__main__':
	tf.app.run()
