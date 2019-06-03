"""
demonstrates how to open an image as an array in Python
StackOverflow:
    [https://stackoverflow.com/questions/7762948/how-to-convert-an-rgb-image-to-numpy-array]
"""
import argparse
# cv2 == Computer Vision (library) ... easy interface for opening up images in Python;
# there is a c++ backend you may be able to access as well ...
import cv2     # you may not have this installed ... run `pip install opencv-python`
import numpy as np
import sys, os

def parse_args():
    parser = argparse.ArgumentParser(description='open an image file')
    parser.add_argument('pngfile',
                        help='the pngfile to read in',
                        type=str)

    parser.add_argument('--normalize',
                        help='import image file with decimal pixel vals rather than pixels in [0,255]',
                        default=False,
                        action='store_true')
    return parser

# grab the path
args = parse_args().parse_args()
pngfile = args.pngfile

if not os.path.exists(pngfile):
    sys.exit('[!!] {} not found'.format(pngfile))

im = cv2.imread(pngfile)

if args.normalize:
    # converts pixel values from [0,255] --> [0,1]
    print('raw image (normalized):',im/255., sep='\n')
else:
    print('raw image (0-255):',im,sep='\n')

print('stored as:', type(im))
print('with shape:',im.shape)


