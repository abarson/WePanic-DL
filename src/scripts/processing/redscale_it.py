"""
redscale an image
"""

import sys
import os
import argparse
import numpy as np
import cv2

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

    
    im = cv2.imread(args.input_image)
    name = args.input_image.split('/')[-1].split('.')[0] + '_red.png'

    r = im.copy()
    r[:, :, 0] = 0
    r[:, :, 1] = 0

    #cv2.imshow('R-RGB',r)
    #cv2.waitKey(0)
    cv2.imwrite(name, r)


