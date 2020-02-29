import os
import json
import argparse
import numpy as np
from astropy.io import fits
# For file searches:
import glob
import sys

os.environ["KERAS_BACKEND"] = "tensorflow"

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from keras.layers import Input, Conv2D, Activation, BatchNormalization, Concatenate, Dropout, UpSampling2D
from keras.callbacks import ModelCheckpoint, Callback
from keras.models import Model, model_from_json
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model as kerasPlot
import tensorflow as tf
import keras.backend.tensorflow_backend as ktf


class LossHistory(Callback):
	def __init__(self, root, losses):
		self.root = root
		self.losses = losses

	def on_epoch_end(self, batch, logs={}):
		self.losses.append(logs)
		with open("{0}_loss.json".format(self.root), 'w') as f:
			json.dump(self.losses, f)

	def finalize(self):
		pass


class train_deepvel(object):

	def __init__(self, root, noise, option):
		"""
		Class used to train DeepVel

		Parameters
		----------
		root : string
			Name of the output files. Some extensions will be added for different files (weights, configuration, etc.)
		noise : float
			Noise standard deviation to be added during training. This helps avoid overfitting and
			makes the training more robust
		option : string
			Indicates what needs to be done
		"""

		# Only allocate needed memory
		config = tf.compat.v1.ConfigProto()
		config.gpu_options.allow_growth = True
		session = tf.compat.v1.Session(config=config)
		# ktf.set_session(session)
		self.root = root
		self.option = option

		# Neural network properties
		self.n_filters = 64
		self.kernel_size = 3
		self.batch_size = 32

		# Data properties
		self.n_training = 1024
		self.n_validation = 224
		self.nx = 256
		self.ny = 256
		self.n_times = 2
		self.n_components = 6

		# Filenames
		self.directory = ''
		filenames_ic = sorted(glob.glob(self.directory+"input/SDO_int*"))
		filenames_vv = sorted(glob.glob(self.directory+"input/SDO_vv*"))
		self.input_file_images_training = filenames_ic[0:self.n_training+1]
		self.input_file_velocity_training = filenames_vv[0:self.n_training+1]
		self.input_file_images_validation = filenames_ic[self.n_training+1:self.n_training+1+self.n_validation+1]
		self.input_file_velocity_validation = filenames_vv[self.n_training+1:self.n_training+1+self.n_validation+1]
		self.batchs_per_epoch_training = int(self.n_training / self.batch_size)
		self.batchs_per_epoch_validation = int(self.n_validation / self.batch_size)

		# Normalization
		tmp = np.load(self.directory+'properties/SteinSDO_properties.npz')
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

	def training_generator(self):

		# Shuffle
		rnd = np.linspace(0, self.n_training-1, num=self.n_training, dtype='int')
		np.random.shuffle(rnd)

		while 1:
			for i in range(self.batchs_per_epoch_training):

				input_train = np.zeros((self.batch_size, self.nx, self.ny, self.n_times))
				output_train = np.zeros((self.batch_size, self.nx, self.ny, self.n_components))

				for j in range(self.batch_size):

					rnd_index = rnd[i*self.batch_size+j]
					f = fits.open(self.input_file_images_training[rnd_index])
					img_tmp = f[0].data
					input_train[j, :, :, 0]=img_tmp[0:self.nx, 0:self.ny].astype('float32')/self.ic1_median
					f.close()
					f = fits.open(self.input_file_images_training[rnd_index+1])
					img_tmp = f[0].data
					input_train[j, :, :, 1]=img_tmp[0:self.nx, 0:self.ny].astype('float32')/self.ic1_median
					f.close()
					f = fits.open(self.input_file_velocity_training[rnd_index])
					img_tmp = f[0].data
					output_train[j, :, :, 0] = (img_tmp[0:self.nx, 0:self.ny, 0, 0].astype('float32')-self.vx1_min)/(self.vx1_max-self.vx1_min)
					output_train[j, :, :, 1] = (img_tmp[0:self.nx, 0:self.ny, 0, 1].astype('float32')-self.vy1_min)/(self.vy1_max-self.vy1_min)
					output_train[j, :, :, 2] = (img_tmp[0:self.nx, 0:self.ny, 1, 0].astype('float32')-self.vx01_min)/(self.vx01_max-self.vx01_min)
					output_train[j, :, :, 3] = (img_tmp[0:self.nx, 0:self.ny, 1, 1].astype('float32')-self.vy01_min)/(self.vy01_max-self.vy01_min)
					output_train[j, :, :, 4] = (img_tmp[0:self.nx, 0:self.ny, 2, 0].astype('float32')-self.vx001_min)/(self.vx001_max-self.vx001_min)
					output_train[j, :, :, 5] = (img_tmp[0:self.nx, 0:self.ny, 2, 1].astype('float32')-self.vy001_min)/(self.vy001_max-self.vy001_min)
					f.close()

				yield input_train, output_train

	def validation_generator(self):

		while 1:
			for i in range(self.batchs_per_epoch_validation):

				input_validation = np.zeros((self.batch_size, self.nx, self.ny, self.n_times))
				output_validation = np.zeros((self.batch_size, self.nx, self.ny, self.n_components))

				for j in range(self.batch_size):

					f = fits.open(self.input_file_images_validation[j])
					img_tmp = f[0].data
					input_validation[j, :, :, 0] = img_tmp[0:self.nx, 0:self.ny].astype('float32')/self.ic1_median
					f.close()
					f = fits.open(self.input_file_images_validation[j+1])
					img_tmp = f[0].data
					input_validation[j, :, :, 1] = img_tmp[0:self.nx, 0:self.ny].astype('float32')/self.ic1_median
					f.close()
					f = fits.open(self.input_file_velocity_validation[j])
					img_tmp = f[0].data
					output_validation[j, :, :, 0] = (img_tmp[0:self.nx, 0:self.ny, 0, 0].astype('float32')-self.vx1_min)/(self.vx1_max-self.vx1_min)
					output_validation[j, :, :, 1] = (img_tmp[0:self.nx, 0:self.ny, 0, 1].astype('float32')-self.vy1_min)/(self.vy1_max-self.vy1_min)
					output_validation[j, :, :, 2] = (img_tmp[0:self.nx, 0:self.ny, 1, 0].astype('float32')-self.vx01_min)/(self.vx01_max-self.vx01_min)
					output_validation[j, :, :, 3] = (img_tmp[0:self.nx, 0:self.ny, 1, 1].astype('float32')-self.vy01_min)/(self.vy01_max-self.vy01_min)
					output_validation[j, :, :, 4] = (img_tmp[0:self.nx, 0:self.ny, 2, 0].astype('float32')-self.vx001_min)/(self.vx001_max-self.vx001_min)
					output_validation[j, :, :, 5] = (img_tmp[0:self.nx, 0:self.ny, 2, 1].astype('float32')-self.vy001_min)/(self.vy001_max-self.vy001_min)
					f.close()

				yield input_validation, output_validation

	def define_network(self):
		print("Setting up network...")

		inputs_ic = Input(shape=(self.nx, self.ny, self.n_times))

		conv1 = Conv2D(self.n_filters, (self.kernel_size, self.kernel_size), strides=(1, 1), padding='same', init='he_normal')(inputs_ic)
		conv1 = BatchNormalization()(conv1)
		conv1 = Activation('relu')(conv1)
		conv1 = Dropout(0.5)(conv1)
		# 128 by 128
		stri1 = Conv2D(self.n_filters, (self.kernel_size, self.kernel_size), strides=(2, 2), padding='same', init='he_normal')(conv1)
		stri1 = BatchNormalization()(stri1)
		stri1 = Activation('relu')(stri1)

		conv2 = Conv2D(2*self.n_filters, (self.kernel_size, self.kernel_size), strides=(1, 1), padding='same', init='he_normal')(stri1)
		conv2 = BatchNormalization()(conv2)
		conv2 = Activation('relu')(conv2)
		conv2 = Dropout(0.5)(conv2)
		# 64 by 64
		stri2 = Conv2D(2*self.n_filters, (self.kernel_size, self.kernel_size), strides=(2, 2), padding='same', init='he_normal')(conv2)
		stri2 = BatchNormalization()(stri2)
		stri2 = Activation('relu')(stri2)

		conv3 = Conv2D(4*self.n_filters, (self.kernel_size, self.kernel_size), strides=(1, 1), padding='same', init='he_normal')(stri2)
		conv3 = BatchNormalization()(conv3)
		conv3 = Activation('relu')(conv3)
		conv3 = Dropout(0.5)(conv3)
		# 32 by 32
		stri3 = Conv2D(4*self.n_filters, (self.kernel_size, self.kernel_size), strides=(2, 2), padding='same', init='he_normal')(conv3)
		stri3 = BatchNormalization()(stri3)
		stri3 = Activation('relu')(stri3)

		conv4 = Conv2D(8*self.n_filters, (self.kernel_size, self.kernel_size), strides=(1, 1), padding='same', init='he_normal')(stri3)
		conv4 = BatchNormalization()(conv4)
		conv4 = Activation('relu')(conv4)
		conv4 = Dropout(0.5)(conv4)
		# 16 by 16
		stri4 = Conv2D(8*self.n_filters, (self.kernel_size, self.kernel_size), strides=(2, 2), padding='same', init='he_normal')(conv4)
		stri4 = BatchNormalization()(stri4)
		stri4 = Activation('relu')(stri4)

		conv5 = Conv2D(16*self.n_filters, (self.kernel_size, self.kernel_size), strides=(1, 1), padding='same', init='he_normal')(stri4)
		conv5 = BatchNormalization()(conv5)
		conv5 = Activation('relu')(conv5)
		conv5 = Dropout(0.5)(conv5)
		# 8 by 8
		stri5 = Conv2D(16*self.n_filters, (self.kernel_size, self.kernel_size), strides=(2, 2), padding='same', init='he_normal')(conv5)
		stri5 = BatchNormalization()(stri5)
		stri5 = Activation('relu')(stri5)

		conv6 = Conv2D(32*self.n_filters, (self.kernel_size, self.kernel_size), strides=(1, 1), padding='same', init='he_normal')(stri5)
		conv6 = BatchNormalization()(conv6)
		conv6 = Activation('relu')(conv6)
		conv6 = Dropout(0.5)(conv6)
		# 4 by 4
		stri6 = Conv2D(32*self.n_filters, (self.kernel_size, self.kernel_size), strides=(2, 2), padding='same', init='he_normal')(conv6)
		stri6 = BatchNormalization()(stri6)
		stri6 = Activation('relu')(stri6)

		conv7 = Conv2D(64*self.n_filters, (self.kernel_size, self.kernel_size), strides=(1, 1), padding='same', init='he_normal')(stri6)
		conv7 = BatchNormalization()(conv7)
		conv7 = Activation('relu')(conv7)
		conv7 = Dropout(0.5)(conv7)
		# 2 by 2
		stri7 = Conv2D(64*self.n_filters, (self.kernel_size, self.kernel_size), strides=(2, 2), padding='same', init='he_normal')(conv7)
		stri7 = BatchNormalization()(stri7)
		stri7 = Activation('relu')(stri7)

		conv8 = Conv2D(128*self.n_filters, (self.kernel_size, self.kernel_size), strides=(1, 1), padding='same', init='he_normal')(stri7)
		conv8 = BatchNormalization()(conv8)
		conv8 = Activation('relu')(conv8)
		conv8 = Dropout(0.5)(conv8)
		# 2 by 2
		stri8 = Conv2D(128*self.n_filters, (self.kernel_size, self.kernel_size), strides=(2, 2), padding='same', init='he_normal')(conv8)
		stri8 = BatchNormalization()(stri8)
		stri8 = Activation('relu')(stri8)

		# Bottleneck
		convc = Conv2D(256*self.n_filters, (self.kernel_size, self.kernel_size), strides=(1, 1), padding='same', init='he_normal')(stri8)
		convc = BatchNormalization()(convc)
		convc = Activation('relu')(convc)
		#convc = Dropout(0.5)(convc)
		convc = Conv2D(256*self.n_filters, (self.kernel_size, self.kernel_size), strides=(1, 1), padding='same', init='he_normal')(convc)
		convc = BatchNormalization()(convc)
		convc = Activation('relu')(convc)
		convc = Dropout(0.5)(convc)

		upconv8 = Conv2D(128*self.n_filters, (self.kernel_size, self.kernel_size), strides=(1, 1), activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(convc))
		upconv8 = Concatenate(axis=3)([conv8,upconv8])
		upconv8 = Conv2D(128*self.n_filters, (self.kernel_size, self.kernel_size), padding='same', init='he_normal')(upconv8)
		upconv8 = BatchNormalization()(upconv8)
		upconv8 = Activation('relu')(upconv8)
		upconv8 = Conv2D(128*self.n_filters, (self.kernel_size, self.kernel_size), strides=(1, 1), padding='same', init='he_normal')(upconv8)
		upconv8 = BatchNormalization()(upconv8)
		upconv8 = Activation('relu')(upconv8)
		upconv8 = Dropout(0.5)(upconv8)

		upconv7 = Conv2D(64*self.n_filters, (self.kernel_size, self.kernel_size), strides=(1, 1), activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(upconv8))
		upconv7 = Concatenate(axis=3)([conv7, upconv7])
		upconv7 = Conv2D(64*self.n_filters, (self.kernel_size, self.kernel_size), padding='same', init='he_normal')(upconv7)
		upconv7 = BatchNormalization()(upconv7)
		upconv7 = Activation('relu')(upconv7)
		upconv7 = Conv2D(64*self.n_filters, (self.kernel_size, self.kernel_size), strides=(1, 1), padding='same', init='he_normal')(upconv7)
		upconv7 = BatchNormalization()(upconv7)
		upconv7 = Activation('relu')(upconv7)
		upconv7 = Dropout(0.5)(upconv7)

		upconv6 = Conv2D(32*self.n_filters, (self.kernel_size, self.kernel_size), strides=(1, 1), activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(upconv7))
		upconv6 = Concatenate(axis=3)([conv6, upconv6])
		upconv6 = Conv2D(32*self.n_filters, (self.kernel_size, self.kernel_size), padding='same', init='he_normal')(upconv6)
		upconv6 = BatchNormalization()(upconv6)
		upconv6 = Activation('relu')(upconv6)
		upconv6 = Conv2D(32*self.n_filters, (self.kernel_size, self.kernel_size), strides=(1, 1), padding='same', init='he_normal')(upconv6)
		upconv6 = BatchNormalization()(upconv6)
		upconv6 = Activation('relu')(upconv6)
		upconv6 = Dropout(0.5)(upconv6)

		upconv5 = Conv2D(16*self.n_filters, (self.kernel_size, self.kernel_size), strides=(1, 1), activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(upconv6))
		upconv5 = Concatenate(axis=3)([conv5, upconv5])
		upconv5 = Conv2D(16*self.n_filters, (self.kernel_size, self.kernel_size), padding='same', init='he_normal')(upconv5)
		upconv5 = BatchNormalization()(upconv5)
		upconv5 = Activation('relu')(upconv5)
		upconv5 = Conv2D(16*self.n_filters, (self.kernel_size, self.kernel_size), strides=(1, 1), padding='same', init='he_normal')(upconv5)
		upconv5 = BatchNormalization()(upconv5)
		upconv5 = Activation('relu')(upconv5)
		upconv5 = Dropout(0.5)(upconv5)

		upconv4 = Conv2D(8*self.n_filters, (self.kernel_size, self.kernel_size), strides=(1, 1), activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(upconv5))
		upconv4 = Concatenate(axis=3)([conv4, upconv4])
		upconv4 = Conv2D(8*self.n_filters, (self.kernel_size, self.kernel_size), padding='same', init='he_normal')(upconv4)
		upconv4 = BatchNormalization()(upconv4)
		upconv4 = Activation('relu')(upconv4)
		upconv4 = Conv2D(8*self.n_filters, (self.kernel_size, self.kernel_size), strides=(1, 1), padding='same', init='he_normal')(upconv4)
		upconv4 = BatchNormalization()(upconv4)
		upconv4 = Activation('relu')(upconv4)
		upconv4 = Dropout(0.5)(upconv4)

		upconv3 = Conv2D(4*self.n_filters, (self.kernel_size, self.kernel_size), strides=(1, 1), activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(upconv4))
		upconv3 = Concatenate(axis=3)([conv3, upconv3])
		upconv3 = Conv2D(4*self.n_filters, (self.kernel_size, self.kernel_size), padding='same', init='he_normal')(upconv3)
		upconv3 = BatchNormalization()(upconv3)
		upconv3 = Activation('relu')(upconv3)
		upconv3 = Conv2D(4*self.n_filters, (self.kernel_size, self.kernel_size), strides=(1, 1), padding='same', init='he_normal')(upconv3)
		upconv3 = BatchNormalization()(upconv3)
		upconv3 = Activation('relu')(upconv3)
		upconv3 = Dropout(0.5)(upconv3)

		upconv2 = Conv2D(2*self.n_filters, (self.kernel_size, self.kernel_size), strides=(1, 1), activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(upconv3))
		upconv2 = Concatenate(axis=3)([conv2, upconv2])
		upconv2 = Conv2D(2*self.n_filters, (self.kernel_size, self.kernel_size), padding='same', init='he_normal')(upconv2)
		upconv2 = BatchNormalization()(upconv2)
		upconv2 = Activation('relu')(upconv2)
		upconv2 = Conv2D(2*self.n_filters, (self.kernel_size, self.kernel_size), strides=(1, 1), padding='same', init='he_normal')(upconv2)
		upconv2 = BatchNormalization()(upconv2)
		upconv2 = Activation('relu')(upconv2)
		upconv2 = Dropout(0.5)(upconv2)

		upconv1 = Conv2D(self.n_filters, (self.kernel_size, self.kernel_size), strides=(1, 1), activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(upconv2))
		upconv1 = Concatenate(axis=3)([conv1, upconv1])
		upconv1 = Conv2D(self.n_filters, (self.kernel_size, self.kernel_size), padding='same', init='he_normal')(upconv1)
		upconv1 = BatchNormalization()(upconv1)
		upconv1 = Activation('relu')(upconv1)
		upconv1 = Conv2D(self.n_filters, (self.kernel_size, self.kernel_size), strides=(1, 1), padding='same', init='he_normal')(upconv1)
		upconv1 = BatchNormalization()(upconv1)
		upconv1 = Activation('relu')(upconv1)
		upconv1 = Dropout(0.5)(upconv1)

		final = Conv2D(self.n_components, (1, 1), strides=(1,1), activation='linear', padding='same', init='he_normal')(upconv1)

		self.model = Model(inputs=inputs_ic, output=final)

		json_string = self.model.to_json()
		f = open('{0}_model.json'.format(self.root), 'w')
		f.write(json_string)
		f.close()

		#kerasPlot(self.model, to_file='{0}_model.png'.format(self.root), show_shapes=True)

	def compile_network(self):
		self.model.compile(loss='mse', optimizer=Adam(lr=1e-4))

	def read_network(self):
		print("Reading previous network...")

		f = open('{0}_model.json'.format(self.root), 'r')
		json_string = f.read()
		f.close()

		self.model = model_from_json(json_string)
		self.model.load_weights("{0}_weights.hdf5".format(self.root))

	def train(self, n_iterations):
		print("Training network...")
		# Show a summary of the model. Check the number of trainable parameters
		#self.summary()
		# Recover losses from previous run
		if (self.option == 'continue'):
			with open("{0}_loss.json".format(self.root), 'r') as f:
				losses = json.load(f)
		else:
			losses = []
		print("********* Initial losses: {}".format(losses))

		self.checkpointer = ModelCheckpoint(filepath="{0}_weights.hdf5".format(self.root), verbose=1, save_best_only=True)
		if (self.option == 'continue'):
			n_val_loss=len(losses)
			val_tmp = np.zeros((n_val_loss))
			for j in range(n_val_loss):
				val_tmp[j] = losses[j]['val_loss']
			self.checkpointer.best=np.amin(val_tmp, axis=None)
			print('Best val_loss: {0}'.format(self.checkpointer.best))
		self.history = LossHistory(self.root, losses)
		for i in range(n_iterations):
			self.metrics = self.model.fit_generator(self.training_generator(), self.n_training, nb_epoch=1,
			  callbacks=[self.checkpointer, self.history], validation_data=(self.validation_generator()), nb_val_samples=self.n_validation)
			#self.metrics = self.model.fit([input1_training,input2_training],output_training, batch_size=self.batch_size, epochs=1,
			#  callbacks=[self.checkpointer, self.history], validation_data=([input1_validation,input2_validation],output_validation))
		#self.metrics = self.model.fit_generator(self.training_generator(), self.n_training, nb_epoch=n_iterations,
		#	callbacks=[self.checkpointer, self.history], validation_data=self.validation_generator(), nb_val_samples=self.n_validation)

		n_val_loss=len(losses)
		list_val_loss=np.zeros(n_val_loss)
		cnt=1
		for i in range(n_val_loss):
			list_val_loss[i]=losses[i]['val_loss']
			if((i > 0) and (list_val_loss[i] < np.amin(list_val_loss[0:i], axis=None))):
				cnt=cnt+1
		unique_val_loss=np.unique(list_val_loss)
		n_unique_val_loss=cnt
		print("Total number of epochs performed: {0}; number of epochs with improvements: {1}".format(n_val_loss,n_unique_val_loss))
		hdu = fits.PrimaryHDU(n_unique_val_loss)
		hdulist = fits.HDUList([hdu])
		hdulist.writeto("{0}_epochs.json".format(self.root), overwrite=True)


		self.history.finalize()

if (__name__ == '__main__'):

	parser = argparse.ArgumentParser(description='Train DeepVel')
	parser.add_argument('-o','--out', help='Output files')
	parser.add_argument('-e','--epochs', help='Number of epochs', default=10)
	parser.add_argument('-n','--noise', help='Noise to add during training', default=0.0)
	parser.add_argument('-a','--action', help='Action', choices=['start', 'continue'], required=True)
	parsed = vars(parser.parse_args())

	root = parsed['out']
	nEpochs = int(parsed['epochs'])
	option = parsed['action']
	noise = parsed['noise']

	out = train_deepvel(root, noise, option)

	if (option == 'start'):
		out.define_network()

	if (option == 'continue'):
		out.read_network()

	if (option == 'start' or option == 'continue'):
		out.compile_network()
		out.train(nEpochs)
