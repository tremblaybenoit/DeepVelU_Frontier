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
from keras.layers import Input, Conv2D, Activation, BatchNormalization, Concatenate, UpSampling2D
from keras.models import Model


class deepvel(object):

	def __init__(self, observations1, observations2, output, border_x1=0, border_x2=0, border_y1=0, border_y2=0):
		"""

		Parameters
		----------
		observations : array
			Array of size (n_times, nx, ny) with the n_times consecutive images of size nx x ny
		output : string
			Filename were the output is saved
		border : int (optional)
			Portion of the borders to be removed during computations. This is useful if images are
			apodized
		"""

		# Only allocate needed memory with Tensorflow
		config = tf.compat.v1.ConfigProto()
		config.gpu_options.allow_growth = True
		session = tf.compat.v1.Session(config=config)
		#ktf.set_session(session)

		# Neural network properties
		self.n_filters = 64
		self.kernel_size = 3
		self.batch_size = 1
		self.observations1 = observations1
		self.observations2 = observations2
		self.output = output

		# Data properties
		self.border_x1 = border_x1
		self.border_x2 = border_x2
		self.border_y1 = border_y1
		self.border_y2 = border_y2
		n_timesteps, nx, ny = observations1.shape
		self.nx = nx - self.border_x1-self.border_x2
		self.ny = ny - self.border_y1-self.border_y2
		self.n_frames = n_timesteps - 1
		self.n_times = 2
		self.n_components = 6
		
		# Normalization
		tmp = np.load(self.directory + 'properties/SteinSDO_properties.npz')
		self.ic1_min = tmp['ic1_min']
		self.ic1_max = tmp['ic1_max']
		self.ic1_mean = tmp['ic1_mean']
		self.ic1_median = tmp['ic1_median']
		self.ic1_stddev = tmp['ic1_stddev']
		self.vx1_min = tmp['vx1_min']
		self.vx1_max = tmp['vx1_max']
		self.vx1_mean = tmp['vx1_mean']
		self.vx1_median = tmp['vx1_median']
		self.vx1_stddev = tmp['vx1_stddev']
		self.vy1_min = tmp['vy1_min']
		self.vy1_max = tmp['vy1_max']
		self.vy1_mean = tmp['vy1_mean']
		self.vy1_median = tmp['vy1_median']
		self.vy1_stddev = tmp['vy1_stddev']
		self.vx01_min = tmp['vx01_min']
		self.vx01_max = tmp['vx01_max']
		self.vx01_mean = tmp['vx01_mean']
		self.vx01_median = tmp['vx01_median']
		self.vx01_stddev = tmp['vx01_stddev']
		self.vy01_min = tmp['vy01_min']
		self.vy01_max = tmp['vy01_max']
		self.vy01_mean = tmp['vy01_mean']
		self.vy01_median = tmp['vy01_median']
		self.vy01_stddev = tmp['vy01_stddev']
		self.vx001_min = tmp['vx001_min']
		self.vx001_max = tmp['vx001_max']
		self.vx001_mean = tmp['vx001_mean']
		self.vx001_median = tmp['vx001_median']
		self.vx001_stddev = tmp['vx001_stddev']
		self.vy001_min = tmp['vy001_min']
		self.vy001_max = tmp['vy001_max']
		self.vy001_mean = tmp['vy001_mean']
		self.vy001_median = tmp['vy001_median']
		self.vy001_stddev = tmp['vy001_stddev']

	def define_network(self):
		
		print("Setting up network...")
		
		inputs_ic = Input(shape=(self.nx, self.ny, self.n_times))
		
		conv1 = Conv2D(self.n_filters, (self.kernel_size, self.kernel_size), strides=(1, 1), padding='same',
		               init='he_normal')(inputs_ic)
		conv1 = BatchNormalization()(conv1)
		conv1 = Activation('relu')(conv1)
		# 128 by 128
		stri1 = Conv2D(self.n_filters, (self.kernel_size, self.kernel_size), strides=(2, 2), padding='same',
		               init='he_normal')(conv1)
		stri1 = BatchNormalization()(stri1)
		stri1 = Activation('relu')(stri1)
		
		conv2 = Conv2D(2 * self.n_filters, (self.kernel_size, self.kernel_size), strides=(1, 1), padding='same',
		               init='he_normal')(stri1)
		conv2 = BatchNormalization()(conv2)
		conv2 = Activation('relu')(conv2)
		# 64 by 64
		stri2 = Conv2D(2 * self.n_filters, (self.kernel_size, self.kernel_size), strides=(2, 2), padding='same',
		               init='he_normal')(conv2)
		stri2 = BatchNormalization()(stri2)
		stri2 = Activation('relu')(stri2)
		
		conv3 = Conv2D(4 * self.n_filters, (self.kernel_size, self.kernel_size), strides=(1, 1), padding='same',
		               init='he_normal')(stri2)
		conv3 = BatchNormalization()(conv3)
		conv3 = Activation('relu')(conv3)
		# 32 by 32
		stri3 = Conv2D(4 * self.n_filters, (self.kernel_size, self.kernel_size), strides=(2, 2), padding='same',
		               init='he_normal')(conv3)
		stri3 = BatchNormalization()(stri3)
		stri3 = Activation('relu')(stri3)
		
		conv4 = Conv2D(8 * self.n_filters, (self.kernel_size, self.kernel_size), strides=(1, 1), padding='same',
		               init='he_normal')(stri3)
		conv4 = BatchNormalization()(conv4)
		conv4 = Activation('relu')(conv4)
		# 16 by 16
		stri4 = Conv2D(8 * self.n_filters, (self.kernel_size, self.kernel_size), strides=(2, 2), padding='same',
		               init='he_normal')(conv4)
		stri4 = BatchNormalization()(stri4)
		stri4 = Activation('relu')(stri4)
		
		conv5 = Conv2D(16 * self.n_filters, (self.kernel_size, self.kernel_size), strides=(1, 1), padding='same',
		               init='he_normal')(stri4)
		conv5 = BatchNormalization()(conv5)
		conv5 = Activation('relu')(conv5)
		# 8 by 8
		stri5 = Conv2D(16 * self.n_filters, (self.kernel_size, self.kernel_size), strides=(2, 2), padding='same',
		               init='he_normal')(conv5)
		stri5 = BatchNormalization()(stri5)
		stri5 = Activation('relu')(stri5)
		
		conv6 = Conv2D(32 * self.n_filters, (self.kernel_size, self.kernel_size), strides=(1, 1), padding='same',
		               init='he_normal')(stri5)
		conv6 = BatchNormalization()(conv6)
		conv6 = Activation('relu')(conv6)
		# 4 by 4
		stri6 = Conv2D(32 * self.n_filters, (self.kernel_size, self.kernel_size), strides=(2, 2), padding='same',
		               init='he_normal')(conv6)
		stri6 = BatchNormalization()(stri6)
		stri6 = Activation('relu')(stri6)
		
		conv7 = Conv2D(64 * self.n_filters, (self.kernel_size, self.kernel_size), strides=(1, 1), padding='same',
		               init='he_normal')(stri6)
		conv7 = BatchNormalization()(conv7)
		conv7 = Activation('relu')(conv7)
		# 2 by 2
		stri7 = Conv2D(64 * self.n_filters, (self.kernel_size, self.kernel_size), strides=(2, 2), padding='same',
		               init='he_normal')(conv7)
		stri7 = BatchNormalization()(stri7)
		stri7 = Activation('relu')(stri7)
		
		conv8 = Conv2D(128 * self.n_filters, (self.kernel_size, self.kernel_size), strides=(1, 1), padding='same',
		               init='he_normal')(stri7)
		conv8 = BatchNormalization()(conv8)
		conv8 = Activation('relu')(conv8)
		# 2 by 2
		stri8 = Conv2D(128 * self.n_filters, (self.kernel_size, self.kernel_size), strides=(2, 2), padding='same',
		               init='he_normal')(conv8)
		stri8 = BatchNormalization()(stri8)
		stri8 = Activation('relu')(stri8)
		
		# Bottleneck
		convc = Conv2D(256 * self.n_filters, (self.kernel_size, self.kernel_size), strides=(1, 1), padding='same',
		               init='he_normal')(stri8)
		convc = BatchNormalization()(convc)
		convc = Activation('relu')(convc)
		convc = Conv2D(256 * self.n_filters, (self.kernel_size, self.kernel_size), strides=(1, 1), padding='same',
		               init='he_normal')(convc)
		convc = BatchNormalization()(convc)
		convc = Activation('relu')(convc)
		
		upconv8 = Conv2D(128 * self.n_filters, (self.kernel_size, self.kernel_size), strides=(1, 1), activation='relu',
		                 padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(convc))
		upconv8 = Concatenate(axis=3)([conv8, upconv8])
		upconv8 = Conv2D(128 * self.n_filters, (self.kernel_size, self.kernel_size), padding='same', init='he_normal')(
			upconv8)
		upconv8 = BatchNormalization()(upconv8)
		upconv8 = Activation('relu')(upconv8)
		upconv8 = Conv2D(128 * self.n_filters, (self.kernel_size, self.kernel_size), strides=(1, 1), padding='same',
		                 init='he_normal')(upconv8)
		upconv8 = BatchNormalization()(upconv8)
		upconv8 = Activation('relu')(upconv8)
		
		upconv7 = Conv2D(64 * self.n_filters, (self.kernel_size, self.kernel_size), strides=(1, 1), activation='relu',
		                 padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(upconv8))
		upconv7 = Concatenate(axis=3)([conv7, upconv7])
		upconv7 = Conv2D(64 * self.n_filters, (self.kernel_size, self.kernel_size), padding='same', init='he_normal')(
			upconv7)
		upconv7 = BatchNormalization()(upconv7)
		upconv7 = Activation('relu')(upconv7)
		upconv7 = Conv2D(64 * self.n_filters, (self.kernel_size, self.kernel_size), strides=(1, 1), padding='same',
		                 init='he_normal')(upconv7)
		upconv7 = BatchNormalization()(upconv7)
		upconv7 = Activation('relu')(upconv7)
		
		upconv6 = Conv2D(32 * self.n_filters, (self.kernel_size, self.kernel_size), strides=(1, 1), activation='relu',
		                 padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(upconv7))
		upconv6 = Concatenate(axis=3)([conv6, upconv6])
		upconv6 = Conv2D(32 * self.n_filters, (self.kernel_size, self.kernel_size), padding='same', init='he_normal')(
			upconv6)
		upconv6 = BatchNormalization()(upconv6)
		upconv6 = Activation('relu')(upconv6)
		upconv6 = Conv2D(32 * self.n_filters, (self.kernel_size, self.kernel_size), strides=(1, 1), padding='same',
		                 init='he_normal')(upconv6)
		upconv6 = BatchNormalization()(upconv6)
		upconv6 = Activation('relu')(upconv6)
		
		upconv5 = Conv2D(16 * self.n_filters, (self.kernel_size, self.kernel_size), strides=(1, 1), activation='relu',
		                 padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(upconv6))
		upconv5 = Concatenate(axis=3)([conv5, upconv5])
		upconv5 = Conv2D(16 * self.n_filters, (self.kernel_size, self.kernel_size), padding='same', init='he_normal')(
			upconv5)
		upconv5 = BatchNormalization()(upconv5)
		upconv5 = Activation('relu')(upconv5)
		upconv5 = Conv2D(16 * self.n_filters, (self.kernel_size, self.kernel_size), strides=(1, 1), padding='same',
		                 init='he_normal')(upconv5)
		upconv5 = BatchNormalization()(upconv5)
		upconv5 = Activation('relu')(upconv5)
		
		upconv4 = Conv2D(8 * self.n_filters, (self.kernel_size, self.kernel_size), strides=(1, 1), activation='relu',
		                 padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(upconv5))
		upconv4 = Concatenate(axis=3)([conv4, upconv4])
		upconv4 = Conv2D(8 * self.n_filters, (self.kernel_size, self.kernel_size), padding='same', init='he_normal')(
			upconv4)
		upconv4 = BatchNormalization()(upconv4)
		upconv4 = Activation('relu')(upconv4)
		upconv4 = Conv2D(8 * self.n_filters, (self.kernel_size, self.kernel_size), strides=(1, 1), padding='same',
		                 init='he_normal')(upconv4)
		upconv4 = BatchNormalization()(upconv4)
		upconv4 = Activation('relu')(upconv4)
		
		upconv3 = Conv2D(4 * self.n_filters, (self.kernel_size, self.kernel_size), strides=(1, 1), activation='relu',
		                 padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(upconv4))
		upconv3 = Concatenate(axis=3)([conv3, upconv3])
		upconv3 = Conv2D(4 * self.n_filters, (self.kernel_size, self.kernel_size), padding='same', init='he_normal')(
			upconv3)
		upconv3 = BatchNormalization()(upconv3)
		upconv3 = Activation('relu')(upconv3)
		upconv3 = Conv2D(4 * self.n_filters, (self.kernel_size, self.kernel_size), strides=(1, 1), padding='same',
		                 init='he_normal')(upconv3)
		upconv3 = BatchNormalization()(upconv3)
		upconv3 = Activation('relu')(upconv3)
		
		upconv2 = Conv2D(2 * self.n_filters, (self.kernel_size, self.kernel_size), strides=(1, 1), activation='relu',
		                 padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(upconv3))
		upconv2 = Concatenate(axis=3)([conv2, upconv2])
		upconv2 = Conv2D(2 * self.n_filters, (self.kernel_size, self.kernel_size), padding='same', init='he_normal')(
			upconv2)
		upconv2 = BatchNormalization()(upconv2)
		upconv2 = Activation('relu')(upconv2)
		upconv2 = Conv2D(2 * self.n_filters, (self.kernel_size, self.kernel_size), strides=(1, 1), padding='same',
		                 init='he_normal')(upconv2)
		upconv2 = BatchNormalization()(upconv2)
		upconv2 = Activation('relu')(upconv2)
		
		upconv1 = Conv2D(self.n_filters, (self.kernel_size, self.kernel_size), strides=(1, 1), activation='relu',
		                 padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(upconv2))
		upconv1 = Concatenate(axis=3)([conv1, upconv1])
		upconv1 = Conv2D(self.n_filters, (self.kernel_size, self.kernel_size), padding='same', init='he_normal')(
			upconv1)
		upconv1 = BatchNormalization()(upconv1)
		upconv1 = Activation('relu')(upconv1)
		upconv1 = Conv2D(self.n_filters, (self.kernel_size, self.kernel_size), strides=(1, 1), padding='same',
		                 init='he_normal')(upconv1)
		upconv1 = BatchNormalization()(upconv1)
		upconv1 = Activation('relu')(upconv1)
		
		final = Conv2D(self.n_components, (1, 1), strides=(1, 1), activation='linear', padding='same',
		               init='he_normal')(upconv1)
		
		self.model = Model(inputs=inputs_ic, output=final)

		self.model.load_weights('network/deepvel_weights.hdf5')

	def validation_generator(self):

		input_validation = np.zeros((self.batch_size, self.nx, self.ny, 2), dtype='float32')

		while 1:
			for i in range(self.n_frames):

				input_validation[:, :, :, 0] = (self.observations1[i*self.batch_size:(i+1)*self.batch_size,self.border_x1:self.border_x1+self.nx,self.border_y1:self.border_y1+self.ny])/self.ic1_median
				input_validation[:, :, :, 1] = (self.observations2[i*self.batch_size:(i+1)*self.batch_size,self.border_x1:self.border_x1+self.nx,self.border_y1:self.border_y1+self.ny])/self.ic1_median

				yield input_validation

		f.close()

	def predict(self):
		print("Predicting velocities with DeepVel...")

		start = time.time()
		out = self.model.predict_generator(self.validation_generator(), self.n_frames, max_q_size=1)
		end = time.time()

		print("Prediction took {0} seconds...".format(end-start))

		out[:, :, :,0] = out[:, :, :, 0]*(self.vx1_max-self.vx1_min) + self.vx1_min
		out[:, :, :,1] = out[:, :, :, 1]*(self.vy1_max-self.vy1_min) + self.vy1_min
		out[:, :, :,2] = out[:, :, :, 2]*(self.vx01_max-self.vx01_min) + self.vx01_min
		out[:, :, :,3] = out[:, :, :, 3]*(self.vy01_max-self.vy01_min) + self.vy01_min
		out[:, :, :,4] = out[:, :, :, 4]*(self.vx001_max-self.vx001_min) + self.vx001_min
		out[:, :, :,5] = out[:, :, :, 5]*(self.vy001_max-self.vy001_min) + self.vy001_min

		hdu = fits.PrimaryHDU(out)
		hdulist = fits.HDUList([hdu])
		hdulist.writeto(self.output, overwrite=True)


if (__name__ == '__main__'):

	parser = argparse.ArgumentParser(description='DeepVel prediction')
	parser.add_argument('-o', '--out', help='Output file')
	parser.add_argument('-i1', '--in1', help='Input file')
	parser.add_argument('-i2', '--in2', help='Input file')
	parser.add_argument('-bx1', '--border_x1', help='Border size in pixels', default=0)
	parser.add_argument('-bx2', '--border_x2', help='Border size in pixels', default=0)
	parser.add_argument('-by1', '--border_y1', help='Border size in pixels', default=0)
	parser.add_argument('-by2', '--border_y2', help='Border size in pixels', default=0)
	parsed = vars(parser.parse_args())

	# Open file with observations and read them. We use FITS in our case
	f = fits.open(parsed['in1'])
	imgs1 = f[0].data
	f = fits.open(parsed['in2'])
	imgs2 = f[0].data

	out = deepvel(imgs1, imgs2, parsed['out'], border_x1=int(parsed['border_x1']), border_x2=int(parsed['border_x2']), border_y1=int(parsed['border_y1']), border_y2=int(parsed['border_y2']))
	out.define_network()
	out.predict()