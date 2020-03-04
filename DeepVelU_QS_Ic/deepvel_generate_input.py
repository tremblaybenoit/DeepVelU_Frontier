import argparse
import numpy as np
from astropy.io import fits
import glob

################################################################################
# Main
################################################################################

if (__name__ == '__main__'):

    parser = argparse.ArgumentParser(description='DeepVel input generator')
    parser.add_argument('-output', '--output', help='Output file')
    parser.add_argument('-prefix', '--prefix', help='Input files')
    parser.add_argument('-nb_frames', '--nb_frames', help='Number of frames', default=2)
    parser.add_argument('-first_frame', '--first_frame', help='First frame', default=0)
    parsed = vars(parser.parse_args())

    # Data
    nb_frames = int(parsed['nb_frames'])
    first_frame = int(parsed['first_frame'])
    output = parsed['output']
    prefix = parsed['prefix']
    # List files
    filenames = sorted(glob.glob(prefix+'*'))
    nb_files = len(filenames)

    # Read first frame
    f = fits.open(filenames[first_frame])
    img_tmp = f[0].data
    nx, ny = img_tmp.shape

    # Read remaining frames
    input_data = np.zeros((nb_frames, nx, ny))
    for i in range(first_frame, first_frame+nb_frames):
        f = fits.open(filenames[i])
        input_data[i-first_frame, :, :] = f[0].data

    # Save input file
    hdu = fits.PrimaryHDU(input_data)
    hdulist = fits.HDUList([hdu])
    hdulist.writeto(output, overwrite=True)
