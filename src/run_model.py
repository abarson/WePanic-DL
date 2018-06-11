"""
command line app for train/testing models.
"""

import argparse
import sys
import os
import time
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

import we_panic_utils.basic_utils as basic_utils
from we_panic_utils.nn import Engine
from we_panic_utils.nn.processing import FrameProcessor


def parse_input():
    """
    parse the input to the script
    
    ### should add --
        -- test_percent : float -- percentage testing set
        -- val_percent : float -- percentage validation set
    
    args:
        None

    returns:
        parser : argparse.ArgumentParser - the namespace containing 
                 all of the relevant arguments
    """

    parser = argparse.ArgumentParser(description="a suite of commands for running a model")

    parser.add_argument("model_type",
                        help="the type of model to run",
                        type=str,
                        choices=["C3D", "3D-CNN", "CNN_3D_small"])
    
    parser.add_argument("data",
                        help="director[y|ies] to draw data from",
                        type=str)
     
    parser.add_argument("--csv",
                        help="csv containing labels subject -- trial -- heart rate -- resp rate",
                        type=str,
                        default="wepanic_collated_catalogue.csv")

    parser.add_argument("--train",
                        help="states whether the model should be trained",
                        # type=bool,
                        default=False,
                        action="store_true")

    parser.add_argument("--test",
                        help="states whether the model should be tested",
                        # type=bool,
                        default=False,
                        action="store_true")

    parser.add_argument("--batch_size",
                        help="size of batch",
                        type=int,
                        default=4)

    parser.add_argument("--epochs",
                        help="if training, the number of epochs to train for",
                        type=int,
                        default=100)

    parser.add_argument("--output_dir",
                        help="the output directory",
                        type=str,
                        default="outputs")
    
    parser.add_argument("--input_dir",
                        help="the input directory when testing model",
                        type=str,
                        default=None)
    
    parser.add_argument("--rotation_range",
                        help="the range to rotate the sequences",
                        type=int,
                        default=0)

    parser.add_argument("--width_shift_range",
                        help="the range to shift the width",
                        type=float,
                        default=0.0)

    parser.add_argument("--height_shift_range",
                        help="the range to shift the height",
                        type=float,
                        default=0.0)

    parser.add_argument("--zoom_range",
                        help="the range to zoom",
                        type=float,
                        default=0.0)
    
    parser.add_argument("--shear_range",
                        help="the range to shear",
                        type=float,
                        default=0.0)

    parser.add_argument("--vertical_flip",
                        help="flip the vertical axis",
                        default=False,
                        action="store_true")
    
    parser.add_argument("--horizontal_flip",
                        help="flip the horizontal axis",
                        default=False,
                        action="store_true")
    
    parser.add_argument("--greyscale_on",
                        help="convert images to greyscale at runtime",
                        default=False,
                        action="store_true")
    
    parser.add_argument("--steps_per_epoch",
                        help="steps per epoch during training",
                        default=100,
                        type=int)
    
    parser.add_argument("--dimensions",
                        help="frame dims",
                        type=int,
                        nargs=2,
                        default=(32,32))

    return parser


def validate_arguments(args):
    """
    --validate the arguments passed into this app,
    --handle incomplete/bad input,
    --save the world
    
    args:
        namespace - the argparse namespace provide into this application

    returns:
        the same namespace with possibly modified inputs
    --------------------------------------------------------------------
    """
        
    if not os.path.exists(args.csv):
        raise FileNotFoundError("can't locate %s" % args.csv)

    if not os.path.isdir(args.data):
        raise ArgumentError("Bad data directory : %s" % args.data)


    # if batch_size was provided to be negative, exit for bad input
    if args.batch_size < 0:
        raise ArgumentError("The --batch_size should be > 0; " +
                            "got %d" % args.batch_size)

    # if epochs was provided to be negative, exit for bad input
    if args.epochs < 0:
        raise ArgumentError("The --epochs should be > 0; " +
                            "got %d" % args.epochs)
    
    # if --test=False and --train=False, exit because there's nothing to do
    if (not args.train) and (not args.test):
        raise ArgumentError("Both --train and --test were provided as False " +
                            "exiting because there's nothing to do ...")
    
    # if --test was provided only
    if args.test and not args.train:
        # if no input directory specified, exit for bad input
        if args.input_dir is None:
            raise ArgumentError("--test was specified but found no input " +
                                "directory, provide an input directory to " +
                                "test a model only")
        
        # an input directory was specified but if doesn't exist
        if not os.path.isdir(args.input_dir):
            raise ArgumentError("Cannot find input directory %s" % args.input_dir)

        # verify input directory structure
        if not verify_directory_structure(args.input_dir):
            raise ArgumentError("Problems with directory structure of %s" % args.input_dir)
        
    # if --test=True and --train=True, then we need only an output directory
    if args.train and args.test:
        args.input_dir = generate_output_directory(args.output_dir)
    
    bad_augs = []
    for arg_name in ['batch_size', 'steps_per_epoch', 'epochs',
                     'shear_range', 'width_shift_range', 'height_shift_range',
                     'rotation_range', 'zoom_range']:
        
        if vars(args)[arg_name] < 0:
            bad_augs.append("%s can't be < 0" % arg_name)

        if arg_name in ['shear_range', 'width_shift_range', 'height_shift_range'] and vars(args)[arg_name] > 1:
            bad_augs.append("%s is a value in [0, 1]" % arg_name)


    if len(bad_augs) > 0:
        raise ArgumentError(",".join(bad_augs))

    return args
        

def summarize_arguments(args):
    """
    report the arguments passed into the app 
    """
    formatter = "[%s] %+15s"
    
    keys = vars(args).keys()
    max_arglen = max([len(k) for k in keys])
    
    formatter = "[%{}s] %+15s".format(max_arglen)

    for k in keys:
        print(formatter % (k, vars(args)[k]))

    
    
class ArgumentError(Exception):
    """
    custom exception to thrown due to bad parameter input
    """
    pass


def generate_output_directory(output_dirname):
    """
    create an output directory that contains looks like this:
    
    output_dirname/
        models/
    
    args:
        output_dirname : string - the ouput directory name
    
    returns:
        output_dir : string - name of output_directory
    """

    model_dir = os.path.join(output_dirname, "models")

    basic_utils.basics.check_exists_create_if_not(output_dirname)
    basic_utils.basics.check_exists_create_if_not(model_dir)

    return output_dirname


def verify_directory_structure(dirname):
    """
    verify that the directory provided conforms to a structure
    that looks like this:

    dir_name/
        models/
        train.csv
        validation.csv
        test.csv
    
    args:
        dirname : string - directory in question

    returns:
        verified : bool - whether or not this directory structure
                          is satisfactory 
    """
    
    verified = True

    if os.path.isdir(dirname):
        if not os.path.isdir(os.path.join(dirname, "models")):
            print("[verify_directory_structure] - no %s/models/ directory" % dirname)
            verified = False
        
        if not os.path.exists(os.path.join(dirname, "train.csv")):
            print("[verify_directory_structure] - no %s/train.csv" % dirname)
            verified = False

        if not os.path.exists(os.path.join(dirname, "val.csv")):
            print("[verify_directory_structure] - no %s/validation.csv" % dirname)
            verified = False

        if not os.path.exists(os.path.join(dirname, "test.csv")):
            print("[verify_directory_structure] - no %s/test.csv" % dirname)
            verified = False

        return verified

    else:
        return False
    

if __name__ == "__main__":
    
    args = parse_input().parse_args()
    args = validate_arguments(args)
    
    summarize_arguments(args)

    fp = FrameProcessor(rotation_range=args.rotation_range,
                        width_shift_range=args.width_shift_range,
                        height_shift_range=args.height_shift_range,
                        shear_range=args.shear_range,
                        zoom_range=args.zoom_range,
                        vertical_flip=args.vertical_flip,
                        horizontal_flip=args.horizontal_flip,
                        batch_size=args.batch_size,
                        greyscale_on=args.greyscale_on)

    input_shape = None
    x, y = args.dimensions

    if args.greyscale_on:
        input_shape = (60, x, y, 1)
    else:
        input_shape = (60, x, y, 3)

    engine = Engine(data=args.data,
                    model_type=args.model_type,
                    csv=args.csv,
                    batch_size=args.batch_size,
                    epochs=args.epochs,
                    train=args.train,
                    test=args.test,
                    inputs=args.input_dir,
                    outputs=args.output_dir,
                    frameproc=fp,
                    input_shape=input_shape,
                    output_shape=2,
                    steps_per_epoch=args.steps_per_epoch)

    print("starting ... ")
    start = time.time()
    engine.run()
    end = time.time()
    total = (end - start) / 60

    if args.train:
        with open(os.path.join(args.output_dir, "time.txt"), 'w') as t:
            t.write(str(total))
