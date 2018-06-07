#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 20:42:08 2017

@author: danberenberg
"""

import sys
import argparse
import warnings

try:
    import imageio
except ImportError:
    sys.exit("idiotically, you chose the wrong python")

VALID_EXTENSIONS = ('png', 'jpg')

def create_gif(filenames,out_dir,name):
    print (name)
    images = []
    for i,filename in enumerate(filenames):
        image = imageio.imread(filename)
        print (i,image)
        images.append(image)

    gif_name = "cool.gif"
    #gif_name = out_dir.strip("/") + "/" +name.strip(".gif") + ".gif"
    print("done")
    imageio.mimsave(gif_name,images)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("images",help="images that will be compiled into a gif"
                        ,nargs="+",type=str)

    parser.add_argument("gif_name",help="name of gif",type=str)

    return parser


if __name__ == '__main__': 
    
    #script = sys.argv[0]
    
    #if len(sys.argv) < 3:
    #    print('Usage: python {} <p> <path to images separated by space>'.format(script))
    #    sys.exit(1)
    
    #p = float(sys.argv[1])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore','UserWarning')
        args = parse_args().parse_args()

        filenames = args.images
        out_dir   = args.out_dir
        name      = args.gif_name

        
        if not all(f.lower().endswith(VALID_EXTENSIONS) for f in filenames):
            print('Only png and jpg files allowed')
            sys.exit(1)
  
        print (filenames,out_dir,name)
        create_gif(filenames,out_dir,name)
