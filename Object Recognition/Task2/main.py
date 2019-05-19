import argparse
import os
import tensorflow as tf
from model import task2

# argument parser
parser = argparse.ArgumentParser(description='')
parser.add_argument('--phase',				type=str,	default='test',	help='pretrain, train, valid or test')
parser.add_argument('--pre_train',			type=bool,	default=False)
parser.add_argument('--continue_train',		type=bool,	default=False)
parser.add_argument('--continue_epoch',		type=int,	default=0)

parser.add_argument('--input_train_dir',	type=str,	default=os.path.join('.', 'dataset', 'task2-train-dataset'))
parser.add_argument('--input_test_dir',		type=str,	default=os.path.join('.', 'dataset', 'task2-test-dataset'))
parser.add_argument('--output_dir',			type=str,	default='.')
parser.add_argument('--log_dir',			type=str,	default=os.path.join('.', 'logs'))
#parser.add_argument('--ckpt_dir',			type=str,	default=os.path.join('.', 'logs', 'checkpoint'))

parser.add_argument('--image_size',			type=int,	default=32)
parser.add_argument('--image_channel',		type=int,	default=3)

parser.add_argument('--n_train',			type=int,	default=100)
parser.add_argument('--n_valid',			type=int,	default=10000)
parser.add_argument('--n_test',				type=int,	default=2000)
parser.add_argument('--n_classes',			type=int,	default=100)
parser.add_argument('--k_shot',				type=int,	default=1,		help='1, 5, or 10')

parser.add_argument('--learning_rate',		type=float,	default=0.0001)
parser.add_argument('--batch_size',			type=int,	default=120)
parser.add_argument('--epoch',				type=int,	default=600)

parser.add_argument('--n_embeddings',		type=int,	default=128,
											help='Dimensionality of the embedding.')
parser.add_argument('--margin',				type=float,	default=0.5,
											help='Positive to negative triplet distance margin.')
parser.add_argument('--triplet_strategy',	type=str,	default='batch_all',
											help='Triplet Strategy: batch_all or batch_hard.')
#parser.add_argument('--n_neighbor',		type=int,	default=1,
#											help='KNN: Number of neighbors')

args = parser.parse_args()

def main(_):

	if args.phase == "pretrain":
		# base data
		args.n_train = 40000
		args.n_valid = 8000
	else:
		# novel data
		args.n_train = 20 * args.k_shot
		args.n_valid = 10000 - 20 * args.k_shot

	# k_shot
	if args.k_shot == 1:
		args.random_seed = 666
		args.n_neighbor  = 1

	elif args.k_shot == 5:
		args.random_seed = 666
		args.n_neighbor  = 5

	elif args.k_shot == 10:
		args.random_seed = 888
		args.n_neighbor  = 1

	args.iteration = int(args.n_train / args.batch_size) + 1

	args.ckpt_dir = args.log_dir
	args.pre_dir  = args.log_dir
	#args.ckpt_dir = os.path.join(args.log_dir, 'checkpoint_%d'%args.k_shot)
	#args.pre_dir  = os.path.join(args.log_dir, 'pre-train')

	# make directory if not exist
	try: os.makedirs(args.log_dir)
	except: pass
	#try: os.makedirs(args.ckpt_dir)
	#except: pass
	#try: os.makedirs(args.pre_dir)
	#except: pass

	# run session
	tf.reset_default_graph()

	with tf.Session() as sess:
		model = task2(sess,args)
		
		if args.phase == 'pretrain':
			model.train()
			#model.valid_pretrain()
		elif args.phase == 'train':
			model.train()
			model.valid()
			model.test()
			#model.save_fig()
		elif args.phase == 'valid':
			model.valid()
			model.test()
		else:
			model.test()
			#model.visualization_tsne()
		

# run main function
if __name__ == '__main__':
	tf.app.run()
