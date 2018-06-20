"""
redscale an image
"""

import sys
import os
import argparse
import numpy as np
from PIL import Image
from scipy.misc import toimage

def parse_args():

    parser = argparse.ArgumentParser(description='redscale an image')

    parser.add_argument('input_image',
                        help='the image to redscale',
                        type=str)

    return parser


if __name__ == '__main__':

    args = parse_args().parse_args()
    
    if not os.path.exists(args.input_image):
        raise FileNotFoundError('cannot locate %s' % args.input_image)

    
    im = Image.open(args.input_image)
    tens = np.array(im)
    tens[:,:,0] = 255*np.zeros(tens[:,:,2].shape)
    tens[:,:,1] = 255*np.zeros(tens[:,:,2].shape)
    name = args.input_image.split('/')[-1].split('.')[0] + '_red.png'

    toimage(tens, cmin=0.0, cmax=...).save(name)


