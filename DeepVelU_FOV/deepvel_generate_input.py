import argparse
import numpy as np
import os
from astropy.io import fits
import glob
import sys

################################################################################
# Main
################################################################################

if (__name__ == '__main__'):

	parser = argparse.ArgumentParser(description='DeepVel input generator')
	parser.add_argument('-output1', '--output1', help='Output file')
	parser.add_argument('-output2', '--output2', help='Output file')
	parser.add_argument('-directory', '--directory', help='Input files')
	parser.add_argument('-nb_frames', '--nb_frames', help='Number of frames', default=2)
	parser.add_argument('-first_frame', '--first_frame', help='First frame', default=0)
	parsed = vars(parser.parse_args())

	# Data
	nb_frames = int(parsed['nb_frames'])
	first_frame = int(parsed['first_frame'])
	output1 = parsed['output1']
	output2 = parsed['output2']
	directory = parsed['directory']
	if not os.path.isdir(directory): 
		print('Error: Directory does not exist.')
		sys.exit('Error: Directory does not exist.')
	print('Reading directory: {0}'.format(directory))
	# List files
	filenames = sorted(glob.glob(directory+'/SDO_int*'))
	nb_files = len(filenames)

	# Read first frame
	f = fits.open(filenames[first_frame])
	img_tmp = f[0].data   
	nx, ny = img_tmp.shape

	# Read remaining frames
	input1_data = np.zeros((nb_frames, nx, ny))
	input2_data = np.zeros((nb_frames, nx, ny))
	for i in range(first_frame, first_frame+nb_frames):
		if(i == 0):
			f = fits.open(filenames[i])
			input1_data[i-first_frame, :, :] = f[0].data
			f.close()
		else:
			input1_data[i-first_frame, :, :] = input2_data[(i-1)-first_frame, :, :]
		f = fits.open(filenames[i+1])
		input2_data[i-first_frame, :, :] = f[0].data
		f.close()

	# Save input file
	hdu = fits.PrimaryHDU(input1_data)
	hdulist = fits.HDUList([hdu])
	hdulist.writeto(output1, overwrite=True)
	# Save input file
	hdu = fits.PrimaryHDU(input2_data)
	hdulist = fits.HDUList([hdu])
	hdulist.writeto(output2, overwrite=True)
