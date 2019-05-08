#
# Source based on: https://github.com/aditya-grover/uae
#


import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets import mnist
import numpy as np
import time
import sys
import os
from scipy.special import expit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.python.platform import flags

outdir = 'outputs/uae/'
datadir = 'datasets/'
latent_dim = 3
batch_size = 64
num_epochs = 50
noise_std = 0.1

dataset = 'mnist'

intermediate_dim = 100
dec_arch = [ intermediate_dim ]
enc_arch = [ intermediate_dim ]

use_vae_loss = False
beta = 1

transfer = 0
transfer_outdir = 'outputs/uae/'
transfer_logdir = 'outputs/uae/models'
logdir = 'outputs/uae/models'
activation = tf.nn.relu
optimizer = tf.train.AdamOptimizer
lr = 0.001
reg_param = 0.
learn_A = True
seed = 1234

convert_to_records = False


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert_to(dataset, name):
  images = dataset.images
  labels = dataset.labels
  num_examples = labels.shape[0]

  if images.shape[0] != num_examples:
    raise ValueError('Images size %d does not match label size %d.' %
                     (images.shape[0], num_examples))

  filename = os.path.join('./datasets', 'mnist', name + '.tfrecords')
  print('Writing', filename)
  writer = tf.python_io.TFRecordWriter(filename)
  for index in range(num_examples):
    image_raw = images[index].tostring()
    example = tf.train.Example(features=tf.train.Features(feature={ 'label': _int64_feature(int(labels[index])), 'features': _bytes_feature(image_raw)}))
    writer.write(example.SerializeToString())
  writer.close()

def main_conv(unused_argv):
  datasets = mnist.read_data_sets('./datasets', dtype=tf.uint8, reshape=False, validation_size=10000)
  convert_to(datasets.train, 'mnist_train')
  convert_to(datasets.validation, 'mnist_valid')
  convert_to(datasets.test, 'mnist_test')

def sigmoid(x, gamma=1):
	"""
	Sigmoid function (numerically stable).
	"""
	u = gamma * x
	return expit(u)

def provide_unlabelled_data(data, batch_size=10):
	"""
	Provide batches of data; data = X
	"""
	N = len(data)
	# Create indices for data
	X_indexed = list(zip(range(N), np.split(data, N)))
	def data_iterator():
		while True:
			idxs = np.arange(0, N)
			np.random.shuffle(idxs)
			X_shuf = [X_indexed[idx] for idx in idxs]
			for batch_idx in range(0, N, batch_size):
				X_shuf_batch = X_shuf[batch_idx:batch_idx+batch_size]
				indices, X_batch = zip(*X_shuf_batch)
				X_batch = np.vstack(X_batch)
				yield indices, X_batch
	return data_iterator()


def provide_data(data, batch_size=10):
	"""
	Provide batches of data; data = (X, y).
	"""
	N = len(data[0])
	X_mean = np.mean(data[0], axis=0)
	X_std = np.std(data[0], axis=0)
	y_mean = np.mean(data[1], axis=0)
	y_std = np.std(data[1], axis=0)
	# Create indices for data
	X_indexed = list(zip(range(N), np.split(data[0], N)))
	y_indexed = list(zip(range(N), np.split(data[1], N)))
	def data_iterator():
		while True:
			idxs = np.arange(0, N)
			np.random.shuffle(idxs)
			X_shuf = [X_indexed[idx] for idx in idxs]
			y_shuf = [y_indexed[idx] for idx in idxs]
			for batch_idx in range(0, N, batch_size):
				X_shuf_batch = X_shuf[batch_idx:batch_idx+batch_size]
				y_shuf_batch = y_shuf[batch_idx:batch_idx+batch_size]
				indices, X_batch = zip(*X_shuf_batch)
				_, y_batch = zip(*y_shuf_batch)
				X_batch = np.vstack(X_batch)
				y_batch = np.vstack(y_batch)
				yield indices, X_batch, y_batch
	return data_iterator(), X_mean, X_std, y_mean, y_std


def plot(samples, m=4, n=None, px=28, title=None):
	"""
	Plots samples.
		n: Number of rows and columns; n^2 samples
		px: Pixels per side for each sample
	"""
	if n is None:
		n = m
	fig = plt.figure(figsize=(m, n))
	gs = gridspec.GridSpec(n, m)
	gs.update(wspace=0.05, hspace=0.05)
	for i, sample in enumerate(samples):
		ax = plt.subplot(gs[i])
		plt.axis('off')
		ax.set_xticklabels([])
		ax.set_yticklabels([])
		ax.set_aspect('equal')
		plt.imshow(sample.reshape(px, px), cmap='Greys')
	if title is None:
		title = 'samples'
	fig.savefig(os.path.join(outdir, title))
	# fig.show()
	plt.close()
	return fig


def plot_colored(samples, m=4, n=None, px=64, title=None):
	"""
	Plots samples.
		n: Number of rows and columns; n^2 samples
		px: Pixels per side for each sample
	"""
	import skimage.io as io

	if n is None:
		n = m
	fig = plt.figure(figsize=(m, n))
	gs = gridspec.GridSpec(n, m)
	gs.update(wspace=0.05, hspace=0.05)
	for i, sample in enumerate(samples):
		ax = plt.subplot(gs[i])
		plt.axis('off')
		ax.set_xticklabels([])
		ax.set_yticklabels([])
		ax.set_aspect('equal')
		# plt.imshow(sample, cmap='Greys')
		io.imshow(sample)
		io.show()
	if title is None:
		title = 'samples'
	fig.savefig(os.path.join(outdir, title))
	# fig.show()
	plt.close()
	return fig

class Datasource(object):
	def __init__(self, sess, datasource='mnist'):
		self.sess = sess
		self.seed = seed
		tf.set_random_seed(self.seed)
		np.random.seed(self.seed)

		self.batch_size = batch_size

		if datasource == 'mnist' or datasource == 'omniglot2mnist':

			self.target_dataset = 'mnist'
			self.TRAIN_FILE = 'mnist_train.tfrecords'
			self.VALID_FILE = 'mnist_valid.tfrecords'
			self.TEST_FILE = 'mnist_test.tfrecords'

			self.input_dim = 784
			self.num_classes = 10
			self.dtype = tf.float32
			self.preprocess = self._preprocess_mnist
			self.get_dataset = self.get_tf_dataset

		elif datasource == 'omniglot' or datasource == 'mnist2omniglot':

			self.target_dataset = 'omniglot'
			self.TRAIN_FILE = 'omniglot_train.tfrecords'
			self.VALID_FILE = 'omniglot_valid.tfrecords'
			self.TEST_FILE = 'omniglot_test.tfrecords'

			self.input_dim = 784
			self.num_classes = 50
			self.dtype = tf.float32
			self.preprocess = self._preprocess_omniglot
			self.get_dataset = self.get_tf_dataset

		elif datasource == 'celeba':

			self.target_dataset = 'celeba'
			self.TRAIN_FILE = 'celeba_train.tfrecords'
			self.VALID_FILE = 'celeba_valid.tfrecords'
			self.TEST_FILE = 'celeba_test.tfrecords'

			self.input_dim = 64 * 64 * 3
			self.input_height = 64
			self.input_width = 64
			self.input_channels = 3
			self.dtype = tf.float32
			self.preprocess = self._preprocess_celebA
			self.get_dataset = self.get_binary_tf_dataset
		

		else:
			raise NotImplementedError

		train_dataset = self.get_dataset('train')

		return

	def _preprocess_omniglot(self, parsed_example):

		image = tf.decode_raw(parsed_example['features'], tf.float32)
		image.set_shape([self.input_dim])
		label = tf.cast(parsed_example['label'], tf.int32)

		return image, label

	def _preprocess_mnist(self, parsed_example):

		image = tf.decode_raw(parsed_example['features'], tf.uint8)
		image.set_shape([self.input_dim])
		image = tf.cast(image, tf.float32) * (1. / 255)
		label = tf.cast(parsed_example['label'], tf.int32)

		return image, label

	def _preprocess_celebA(self, parsed_example):

		image = tf.decode_raw(parsed_example['features'], tf.uint8)
		image = tf.reshape(image, (self.input_height, self.input_width, self.input_channels))
		# image.set_shape([self.input_dim])
		# convert from bytes to data
		image = tf.divide(tf.to_float(image), 127.5) - 1.0
		# convert back to [0, 1] pixels
		image = tf.clip_by_value(tf.divide(image + 1., 2.), 0., 1.)
		return image		

	def get_tf_dataset_celeba(self, split):

		def _parse_function(example_proto):
			example = {'image_raw': tf.FixedLenFeature((), tf.string, default_value=''),
						'height': tf.FixedLenFeature((), tf.int64, default_value=218),
						'width': tf.FixedLenFeature((), tf.int64, default_value=178),
						'channels': tf.FixedLenFeature((), tf.int64, default_value=3)}
			parsed_example = tf.parse_single_example(example_proto, example)
			preprocessed_features = self.preprocess(parsed_example)
			return preprocessed_features

		filename = os.path.join(datadir, self.target_dataset, self.TRAIN_FILE if split=='train' 
			else self.VALID_FILE if split=='valid' else self.TEST_FILE)
		tf_dataset = tf.data.TFRecordDataset(filename)
		return tf_dataset.map(_parse_function)

	def get_tf_dataset(self, split):

		def _parse_function(example_proto):
			example = {'features': tf.FixedLenFeature((), tf.string, default_value=''),
						  'label': tf.FixedLenFeature((), tf.int64, default_value=0)}
			parsed_example = tf.parse_single_example(example_proto, example)
			preprocessed_features, preprocessed_label = self.preprocess(parsed_example)
			return preprocessed_features, preprocessed_label

		filename = os.path.join(datadir, self.target_dataset, self.TRAIN_FILE if split=='train' 
			else self.VALID_FILE if split=='valid' else self.TEST_FILE)
		tf_dataset = tf.data.TFRecordDataset(filename)
		return tf_dataset.map(_parse_function)

	def get_binary_tf_dataset(self, split):

		def _parse_function(example_proto):
			# no labels available for binary MNIST
			example = {'features': tf.FixedLenFeature((), tf.string, default_value='')}
			parsed_example = tf.parse_single_example(example_proto, example)
			preprocessed_features = self.preprocess(parsed_example)
			return preprocessed_features

		filename = os.path.join(datadir, self.target_dataset, self.TRAIN_FILE if split=='train' 
			else self.VALID_FILE if split=='valid' else self.TEST_FILE)
		tf_dataset = tf.data.TFRecordDataset(filename)
		return tf_dataset.map(_parse_function)


class UAE():
	# uncertainty autoencoder

	def __init__(self, sess, datasource, vae=False):

		self.seed = seed
		tf.set_random_seed(self.seed)
		np.random.seed(self.seed)

		self.sess = sess
		self.datasource = datasource
		self.input_dim = self.datasource.input_dim
		self.z_dim = latent_dim
		self.dec_layers = [self.input_dim] + dec_arch
		self.enc_layers = enc_arch + [self.z_dim]
		self.transfer = transfer

		self.learn_A = learn_A
		self.activation = activation
		self.optimizer = optimizer
		self.lr = lr

		# graph ops+variables
		self.x = tf.placeholder(self.datasource.dtype, shape=[None, self.input_dim], name='vae_input')
		self.noise_std = tf.placeholder_with_default(noise_std, shape=(), name='noise_std')
		self.reg_param = tf.placeholder_with_default(reg_param, shape=(), name='reg_param')
		self.mean, self.z, self.x_reconstr_logits = self.create_computation_graph(self.x, std=self.noise_std)

		if vae:
			z_mean_logcov = tf.concat([self.mean, tf.log(self.noise_std)*tf.ones_like(self.mean)], axis=-1)
			self.loss, self.reconstr_loss = self.get_vae_loss(self.x, z_mean_logcov, self.x_reconstr_logits)
		else:
			self.loss, self.reconstr_loss = self.get_loss(self.x, self.x_reconstr_logits)

		# session ops
		self.global_step = tf.Variable(0, name='global_step', trainable=False)

		if self.transfer == 1:
			train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='model/dec')
		elif self.transfer == 2:
			train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='model/enc')
		elif self.learn_A:
			train_vars = tf.trainable_variables()
		else:
			A, A_val = self.get_A()
			self.assign_A_op = tf.assign(A, A_val)
			train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='model/dec')
		

		self.train_op = self.optimizer(learning_rate=self.lr).minimize(self.loss, 
			global_step=self.global_step, var_list=train_vars)

		# summary ops
		self.summary_op = tf.summary.merge_all()

		# session ops
		self.init_op = tf.global_variables_initializer()
		self.saver = tf.train.Saver(max_to_keep=None)


	def encoder(self, x, reuse=True):
		"""
		Specifies the parameters for the mean and variance of p(y|x)
		"""

		e = x
		enc_layers = self.enc_layers
		regularizer = tf.contrib.layers.l2_regularizer(scale=self.reg_param)
		with tf.variable_scope('model', reuse=reuse):
			with tf.variable_scope('encoder', reuse=reuse):
				for layer_idx, layer_dim in enumerate(enc_layers[:-1]):
					e = tf.layers.dense(e, layer_dim, activation=self.activation, kernel_regularizer=regularizer, reuse=reuse, name='fc-'+str(layer_idx))
				z_mean = tf.layers.dense(e, enc_layers[-1], activation=None, use_bias=False, kernel_regularizer=regularizer, reuse=reuse, name='fc-'+str(len(enc_layers)))
		
		return z_mean


	def decoder(self, z, reuse=True, use_bias=False):

		d = tf.convert_to_tensor(z)
		dec_layers = self.dec_layers

		with tf.variable_scope('model', reuse=reuse):
			with tf.variable_scope('decoder', reuse=reuse):
				for layer_idx, layer_dim in list(reversed(list(enumerate(dec_layers))))[:-1]:
					d = tf.layers.dense(d, layer_dim, activation=self.activation, reuse=reuse, name='fc-' + str(layer_idx), use_bias=use_bias)
				x_reconstr_logits = tf.layers.dense(d, dec_layers[0], activation=tf.nn.sigmoid, reuse=reuse, name='fc-0', use_bias=use_bias) # clip values between 0 and 1

		return x_reconstr_logits


	def create_computation_graph(self, x, std=0.1, reuse=False):

		mean = self.encoder(x, reuse=reuse)
		eps = tf.random_normal(tf.shape(mean), 0, 1, dtype=tf.float32)
		z = tf.add(mean, tf.multiply(std, eps))
		x_reconstr_logits = self.decoder(z, reuse=reuse)

		return mean, z, x_reconstr_logits


	def get_vae_loss(self, x, z_mean_logcov, x_reconstr_logits):

		reg_loss = tf.losses.get_regularization_loss() 
		reconstr_loss = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(x, x_reconstr_logits), axis=1))

		z_dim = self.z_dim
		latent_loss = -0.5 * beta * tf.reduce_sum(1 + z_mean_logcov[:, z_dim:]
						   - tf.square(z_mean_logcov[:, :z_dim]) 
						   - tf.exp(z_mean_logcov[:, z_dim:]), 1)

		tf.summary.scalar('reconstruction loss', reconstr_loss)

		total_loss = reconstr_loss + reg_loss + latent_loss

		return total_loss, reconstr_loss


	def get_loss(self, x, x_reconstr_logits):

		reg_loss = tf.losses.get_regularization_loss() 
		reconstr_loss = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(x, x_reconstr_logits), axis=1))

		tf.summary.scalar('reconstruction loss', reconstr_loss)

		total_loss = reconstr_loss + reg_loss

		return total_loss, reconstr_loss

	def train(self, ckpt=None, verbose=True):
		"""
		Trains VAE for specified number of epochs.
		"""
		
		sess = self.sess
		datasource = self.datasource

		if self.transfer > 0:
			log_file = os.path.join(transfer_outdir, 'log.txt')
			if os.path.exists(log_file):
				for line in open(log_file):
					if "Restoring ckpt at epoch" in line:
						ckpt = line.split()[-1]
						break
			if ckpt is None:
				ckpt = tf.train.latest_checkpoint(transfer_logdir)
			self.saver.restore(sess, ckpt)
		else:
			sess.run(self.init_op)
			if not self.learn_A:
				sess.run(self.assign_A_op)

		t0 = time.time()
		train_dataset = datasource.get_dataset('train')
		train_dataset = train_dataset.batch(batch_size)
		train_dataset = train_dataset.shuffle(buffer_size=10000)
		train_iterator = train_dataset.make_initializable_iterator()
		next_train_batch = train_iterator.get_next()

		valid_dataset = datasource.get_dataset('valid')
		valid_dataset = valid_dataset.batch(batch_size*10)
		valid_iterator = valid_dataset.make_initializable_iterator()
		next_valid_batch = valid_iterator.get_next()

		epoch_train_losses = []
		epoch_valid_losses = []
		epoch_save_paths = []

		for epoch in range(num_epochs):
			sess.run(train_iterator.initializer)
			sess.run(valid_iterator.initializer)
			epoch_train_loss = 0.
			num_batches = 0.
			while True:
				try:
					x = sess.run(next_train_batch)[0]
					feed_dict = {self.x: x}
					sess.run(self.train_op, feed_dict)
					batch_loss, gs = sess.run([self.reconstr_loss, self.global_step], feed_dict)
					epoch_train_loss += batch_loss
					num_batches += 1
				except tf.errors.OutOfRangeError:
					break
			if verbose:
				epoch_train_loss /= num_batches
				x = sess.run(next_valid_batch)[0]
				epoch_valid_loss, gs = sess.run([self.reconstr_loss, self.global_step], feed_dict={self.x: x})
				print('Epoch {}, l2 train loss: {:0.6f}, l2 valid loss: {:0.6f}, time: {}s'. \
						format(epoch+1, np.sqrt(epoch_train_loss), np.sqrt(epoch_valid_loss), int(time.time()-t0)))
				sys.stdout.flush()
				save_path = self.saver.save(sess, os.path.join(logdir, 'model.ckpt'), global_step=gs)
				epoch_train_losses.append(epoch_train_loss)
				epoch_valid_losses.append(epoch_valid_loss)
				epoch_save_paths.append(save_path)
		best_ckpt = None
		if verbose:
			min_idx = epoch_valid_losses.index(min(epoch_valid_losses))
			print('Restoring ckpt at epoch', min_idx+1,'with lowest validation error:', epoch_save_paths[min_idx])
			best_ckpt = epoch_save_paths[min_idx]

		return (epoch_train_losses, epoch_valid_losses), best_ckpt

	def test(self, ckpt=None):

		sess = self.sess
		datasource = self.datasource

		if ckpt is None:
			ckpt = tf.train.latest_checkpoint(logdir)
		
		self.saver.restore(sess, ckpt)

		test_dataset = datasource.get_dataset('test')
		test_dataset = test_dataset.batch(batch_size*10)
		test_iterator = test_dataset.make_initializable_iterator()
		next_test_batch = test_iterator.get_next()

		test_loss = 0.
		num_batches = 0.
		sess.run(test_iterator.initializer)
		while True:
			try:
				x = sess.run(next_test_batch)[0]
				batch_test_loss = sess.run(self.reconstr_loss, feed_dict={self.x: x})
				test_loss += batch_test_loss
				
				num_batches += 1.
			except tf.errors.OutOfRangeError:
				break
		test_loss /= num_batches
		print('L2 squared test loss (per image): {:0.6f}'.format(test_loss))
		print('L2 squared test loss (per pixel): {:0.6f}'.format(test_loss/self.input_dim))

		print('L2 test loss (per image): {:0.6f}'.format(np.sqrt(test_loss)))
		print('L2 test loss (per pixel): {:0.6f}'.format(np.sqrt(test_loss)/self.input_dim))

		return test_loss

	def reconstruct(self, ckpt=None, pkl_file=None):

		import pickle

		sess = self.sess
		datasource = self.datasource

		if ckpt is None:
			ckpt = tf.train.latest_checkpoint(logdir)
		self.saver.restore(sess, ckpt)

		if pkl_file is None:
			test_dataset = datasource.get_dataset('test')
			test_dataset = test_dataset.batch(51)
			test_iterator = test_dataset.make_initializable_iterator()
			next_test_batch = test_iterator.get_next()

			sess.run(test_iterator.initializer)
			x = sess.run(next_test_batch)[0]
		else:
			with open(pkl_file, 'rb') as f:
				images = pickle.load(f)
			x = np.vstack([images[i] for i in range(51)])

		x_reconstr_logits = sess.run(self.x_reconstr_logits, feed_dict={self.x: x})
		print(np.max(x_reconstr_logits), np.min(x_reconstr_logits))
		print(np.max(x), np.min(x))
		plot(np.vstack((x[1:11], x_reconstr_logits[1:11],
						x[11:21], x_reconstr_logits[11:21],
						x[21:31], x_reconstr_logits[21:31],
						x[31:41], x_reconstr_logits[31:41],
						x[41:51], x_reconstr_logits[41:51],
						x[51:61], x_reconstr_logits[51:61],
						x[61:71], x_reconstr_logits[61:71],
						x[71:81], x_reconstr_logits[71:81],
						x[81:91], x_reconstr_logits[81:91],
						x[91:101], x_reconstr_logits[91:101])), m=10, n=10, title='reconstructions')
		
		with open(os.path.join(outdir, 'reconstr.pkl'), 'wb') as f:
			pickle.dump(x_reconstr_logits, f, pickle.HIGHEST_PROTOCOL)
		return x_reconstr_logits

	def encode(self, ckpt=None, pkl_file=None, bach_size=10):
		import pickle

		sess = self.sess
		datasource = self.datasource

		if ckpt is None:
			ckpt = tf.train.latest_checkpoint(logdir)
		self.saver.restore(sess, ckpt)

		if pkl_file is None:
			test_dataset = datasource.get_dataset('test')
			test_dataset = test_dataset.batch(bach_size)
			test_iterator = test_dataset.make_initializable_iterator()
			next_test_batch = test_iterator.get_next()

			sess.run(test_iterator.initializer)
			x = sess.run(next_test_batch)[0]
		else:
			with open(pkl_file, 'rb') as f:
				images = pickle.load(f)
			x = np.vstack([images[i] for i in range(bach_size)])

		z_test = sess.run(self.z, feed_dict={self.x: x})
		return z_test


if convert_to_records:
  tf.app.run(main=main_conv)

gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))

datasource = Datasource(sess, dataset)

model = UAE(sess, datasource, use_vae_loss)
learning_curves, best_ckpt = model.train()

model.test(ckpt=best_ckpt)
model.reconstruct(ckpt=best_ckpt, pkl_file=None)

z_test = model.encode(bach_size=10000)
np.savetxt("outputs/uae_mnist_activations-" + str(beta) + ".csv", z_test, delimiter=",", fmt='%f')
