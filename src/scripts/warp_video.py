import argparse
import we_panic_utils.basic_utils.video_core as vc
import we_panic_utils.nn.processing as fp
import we_panic_utils.basic_utils.basics as base

import numpy as np
import sys
import os
import cv2
import warnings

try:
    import imageio
except ImportError:
    sys.exit("idiotically, you chose the wrong python or need to install imageio")

VALID_EXTENSIONS = ('png', 'jpg')


def parse_input():
    """
    parse the input to the to the app

    returns:
        parser : argparse.ArgumentParser
    """

    parser = argparse.ArgumentParser(description="a command line app for warping videos")
    
    parser.add_argument("movie_file",
                        help="video file with a [.MOV | .mov] extension",
                        type=str)
    
    parser.add_argument("--rotation",
                        help="degree to rotate",
                        type=int,
                        default=0.)

    parser.add_argument("--shear",
                        help="shear intensity",
                        type=float,
                        default=0.)

    parser.add_argument("--width_shift",
                        help="fraction of width to shift by",
                        type=float,
                        default=0.)

    parser.add_argument("--height_shift",
                        help="fraction of height to shift by",
                        type=float,
                        default=0.)

    parser.add_argument("--zoom",
                        help="the amount to zoom",
                        type=float,
                        default=0.)

    parser.add_argument("--horizontal_flip",
                        help="flip horizontally or not",
                        default=False,
                        action='store_true')

    parser.add_argument("--vertical_flip",
                        help="flip verticaally or not",
                        default=False,
                        action='store_true')

    parser.add_argument("-q",
                        help="quiet down messages",
                        default=False,
                        action='store_true')
    
    return parser


def within_interval(i, lowerbd, upperbd):
    """
    return whether or not i is in [lowerbd, upperbd]
    """

    return i >= lowerbd and i <= upperbd


def check_inputs(args):
    """
    make sure the inputs are valid

    args:
        args - Namespace of arguments passed

    returns:
        None, exit with errors
    """

    assert os.path.exists(args.movie_file), "%s doesn't exist" % args.movie_file
    assert args.movie_file.endswith(".MOV") or args.movie_file.endswith(".mov"), "%s doesn't have a [.MOV | .mov] extension" % args.movie_file

    assert args.rotation >= 0, "rotation range should be > 0, got %d" % args.rotation
    assert args.shear >= 0, "shear intensity should be > 0, got %d" % args.shear
    assert args.zoom >= 0, "zoom range should be > 0, got %0.2f" % args.zoom

    assert within_interval(args.width_shift, 0, 1), "width shift range should be in [0,1], got %0.2f" % args.width_shift_range
    assert within_interval(args.height_shift, 0, 1), "height shift range should be in [0,1], got %0.2f" % args.height_shift_range


if __name__ == '__main__':
    args = parse_input().parse_args()
    check_inputs(args)
    
    output_dir = "_frames/"
    frames = vc.video_file_to_frames(args.movie_file, output_dir=output_dir, suppress=args.q)
    
    output_dir = "/".join(frames[0].split("/")[:-1])

    rsz_dir = os.path.join("_rsz/", "".join(frames[0].split("/")[1:-1]))

    frames = ["/".join(f.split("/")[1:]) for f in frames]
    
    vc.resize_frame_dir(output_dir, rsz_dir)
    frames = [os.path.join("_rsz/", f) for f in frames]
    
    sequence = fp.build_image_sequence(frames)
    
    if args.rotation > 0.0:
        sequence = fp.random_sequence_rotation(sequence, args.rotation)
    
    if args.width_shift > 0.0 or args.height_shift > 0.0:
        sequence = fp.random_sequence_shift(sequence, args.width_shift, args.height_shift)
    
    if args.shear > 0.0:
        sequence = fp.random_sequence_shear(sequence, args.shear)
    
    if args.zoom > 0.0:
        sequence = fp.random_sequence_zoom(sequence, args.zoom)
    
    if args.vertical_flip:
        # with probability 0.5, flip vertical axis
        coin_flip = np.random.random_sample() > 0.5
        
        sequence = fp.sequence_flip_axis(sequence, 0)   # flip on the row axis
    
    if args.horizontal_flip:
        # with probability 0.5, flip horizontal axis (cols)
        coin_flip - np.random.random_sample() > 0.5
    
        sequence = fp.sequence_flip_axis(sequence, 1)   # flip on the column axis
    
    final = "_warped/"
    base.check_exists_create_if_not(final)
    sequence = [img * 255. for img in sequence]
     
    for i, img in enumerate(sequence):
        pth = os.path.join(final, "frame%04d.png" % i)
        cv2.imwrite(pth, img)
    
    gif_name = args.movie_file[:-4] + "_warped.gif"   
    imageio.mimsave(gif_name, sequence)
    
    print("cleaning ...")
    os.system("rm -r _frames/")
    os.system("rm -r _rsz/")

    print("wrote a .gif : %s" % gif_name)
