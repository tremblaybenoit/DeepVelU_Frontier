import numpy as np
import platform
import os
import time
import argparse
from astropy.io import fits

os.environ["KERAS_BACKEND"] = "tensorflow"

if (platform.node() != 'vena'):
	os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf

import keras.backend.tensorflow_backend as ktf
from keras.layers import Input, Conv2D, Activation, BatchNormalization, add
from keras.models import Model
# IDL
import idlsave


class deepvel(object):
	
	def __init__(self, observations, output, border_x1=0, border_x2=0, border_y1=0, border_y2=0, same_as_training=0, network_path='network'):
		"""
		---------
		Keywords:
		---------
		
		observations: Input array of shape (nx, ny, n_times*n_inputs) where
						nx & ny: Image dimensions
						n_times: Number of consecutive timesteps
						n_inputs: Number of types of inputs
		
		output: Output array of dimensions (nx, ny, n_depths*n_comp) where
						nx & ny: Image dimensions
						n_depths: Number of optical/geometrical depths to infer
						n_comp: Number of components of the velocity vector to infer
		
		border: Number of pixels to crop from the image in each direction
						border_x1: Number of pixels to remove from the left of the image
						border_x2: Number of pixels to remove from the right of the image
						border_y1: Number of pixels to remove from the bottom of the image
						border_y2: Number of pixels to remove from the top of the image
						
		same_as_training: Set to 1 if using data from the same simulation as the one used for training.
							-> The inputs will be normalized using the same values as the inputs in the
								training set because the values are known.
								
		network: Provide path to the network weights and normalization values
		
		"""
		
		# Only allocate needed memory with Tensorflow
		config = tf.compat.v1.ConfigProto()
		config.gpu_options.allow_growth = True
		session = tf.compat.v1.Session(config=config)
		# ktf.set_session(session)
		
		# -----------------
		# Input properties:
		# -----------------
		# Read
		self.observations = observations
		n_timesteps, nx, ny = observations.shape
		# Number of types of inputs
		self.n_inputs = 1
		# Number of consecutive frames of a given input
		self.n_times = 2
		# Number of images to generate
		self.n_frames = n_timesteps - 1
		# Image dimensions
		self.border_x1 = border_x1
		self.border_x2 = border_x2
		self.border_y1 = border_y1
		self.border_y2 = border_y2
		self.nx = nx - self.border_x1 - self.border_x2
		self.ny = ny - self.border_y1 - self.border_y2
		
		# ------------------
		# Output properties:
		# ------------------
		# Filename
		self.output = output
		# Number of inferred depths
		self.n_depths = 3
		# Number of inferred velocity components
		self.n_comp = 2
		self.n_outputs = self.n_depths*self.n_comp
		
		# -----------------
		# Network properties:
		# -----------------
		# Load training weights
		self.network_path = network_path
		self.weights_filename = self.network_path+'/deepvel_weights.hdf5'
		self.n_filters = 64
		self.kernel_size = 3
		self.n_conv_layers = 20
		self.batch_size = 1
		
		# ------------------------
		# Training set properties:
		# ------------------------
		# Load simulation min/max/mean/median/stddev
		tmp = np.load(self.network_path+'/Stagger_normalization.npz')
		self.ic1_min = tmp['min_ic']
		self.ic1_max = tmp['max_ic']
		self.ic1_mean = tmp['mean_ic']
		self.ic1_median = tmp['median_ic']
		self.ic1_stddev = tmp['stddev_ic']
		self.vv_min = tmp['min_v']
		self.vv_max = tmp['max_v']
		self.vv_mean = tmp['mean_v']
		self.vv_median = tmp['median_v']
		self.vv_stddev = tmp['stddev_v']
		
		# --------------------
		# Test set properties:
		# --------------------
		# Use same normalization values as for the training and validation sets
		self.same_as_training = same_as_training
	
	def residual(self, inputs):
		x = Conv2D(self.n_filters, (self.kernel_size, self.kernel_size), border_mode='same', init='he_normal')(inputs)
		x = BatchNormalization()(x)
		x = Activation('relu')(x)
		x = Conv2D(self.n_filters, (self.kernel_size, self.kernel_size), border_mode='same', init='he_normal')(x)
		x = BatchNormalization()(x)
		x = add([x, inputs])
		
		return x
	
	def define_network(self):
		print("Setting up network...")
		
		inputs = Input(shape=(self.nx, self.ny, self.n_inputs*self.n_times))
		x = inputs
		
		conv = Conv2D(self.n_filters, (self.kernel_size, self.kernel_size), activation='relu', border_mode='same', init='he_normal')(inputs)
		
		x = self.residual(conv)
		for i in range(self.n_conv_layers):
			x = self.residual(x)
		
		x = Conv2D(self.n_filters, (self.kernel_size, self.kernel_size), border_mode='same', init='he_normal')(x)
		x = BatchNormalization()(x)
		x = add([x, conv])
		
		final = Conv2D(self.n_outputs, (1, 1), strides=(1, 1), activation='linear', padding='same', init='he_normal')(x)
		
		self.model = Model(input=inputs, output=final)
		self.model.load_weights(self.weights_filename)
	
	def validation_generator(self):
		
		input_validation = np.zeros((self.batch_size, self.nx, self.ny, self.n_inputs*self.n_times), dtype='float32')
		
		if(self.same_as_training == 0):
			self.ic1_median = np.median(self.observations[:, self.border_x1:self.border_x1+self.nx, self.border_y1:self.border_y1+self.ny])
		
		while 1:
			for i in range(self.n_frames):
				input_validation[:, :, :, 0] = self.observations[i*self.batch_size:(i + 1) * self.batch_size,
													self.border_x1:self.border_x1 + self.nx,
													self.border_y1:self.border_y1 + self.ny] / self.ic1_median
				input_validation[:, :, :, 1] = self.observations[i*self.batch_size + 1:(i + 1) * self.batch_size + 1,
													self.border_x1:self.border_x1 + self.nx,
													self.border_y1:self.border_y1 + self.ny] / self.ic1_median
				
				yield input_validation
		
		f.close()
	
	def predict(self):
		print("Predicting velocities with DeepVel...")
		
		start = time.time()
		out = self.model.predict_generator(self.validation_generator(), self.n_frames, max_q_size=1)
		end = time.time()
		
		print("Prediction took {0} seconds...".format(end - start))
		
		for i in range(self.n_outputs):
			out[:, :, :, i] = out[:, :, :, i] * (self.vv_max[i] - self.vv_min[i]) + self.vv_min[i]
		
		hdu = fits.PrimaryHDU(out)
		hdulist = fits.HDUList([hdu])
		hdulist.writeto(self.output, overwrite=True)


if (__name__ == '__main__'):
	parser = argparse.ArgumentParser(description='DeepVel prediction')
	parser.add_argument('-o', '--out', help='Output file')
	parser.add_argument('-i', '--in', help='Input file')
	parser.add_argument('-bx1', '--border_x1', help='Border size in pixels', default=0)
	parser.add_argument('-bx2', '--border_x2', help='Border size in pixels', default=0)
	parser.add_argument('-by1', '--border_y1', help='Border size in pixels', default=0)
	parser.add_argument('-by2', '--border_y2', help='Border size in pixels', default=0)
	parser.add_argument('-sim', '--simulation', help='Set to 1 if data is from the same simulation as the training set', default=0)
	parser.add_argument('-n', '--network', help='Path to network weights and normalization values', default='network')
	parsed = vars(parser.parse_args())
	
	# Open file with observations and read them
	f = fits.open(parsed['in'])
	imgs = f[0].data
	
	out = deepvel(imgs, parsed['out'],
								border_x1=int(parsed['border_x1']),
								border_x2=int(parsed['border_x2']),
								border_y1=int(parsed['border_y1']),
								border_y2=int(parsed['border_y2']),
								same_as_training=int(parsed['simulation']),
								network_path=parsed['network'])
	out.define_network()
	out.predict()
