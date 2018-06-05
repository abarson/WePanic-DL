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
from we_panic_utils.nn.data_load.train_test_split_csv import train_test_split_with_csv_support
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
                        choices=["C3D", "CNN+LSTM", "3D-CNN", "CNN_3D_small", "CNN_Stacked_GRU", "ResidualLSTM_v01", "ResidualLSTM_v02", "OpticalFlowCNN"])
    
    parser.add_argument("data",
                        help="director[y|ies] to draw data from",
                        type=str,
                        nargs="+")
     
    parser.add_argument("--ignore_augmented",
                        help="specify which phases of running model should ignore augmented data",
                        type=str,
                        nargs="+",
                        default=[],
                        choices=["train", "validation", "test"])

    parser.add_argument("--partition_csv",
                        help="csv containing the mapping from partition paths to heart rate/resp rates",
                        type=str,
                        default="reg_part_out.csv")

    parser.add_argument("--csv",
                        help="csv containing labels subject -- trial -- heart rate -- resp rate",
                        type=str,
                        default="NextStartingPoint.csv")

    parser.add_argument("--train",
                        help="states whether the model should be trained",
                        # type=bool,
                        default=False,
                        action="store_true")

    parser.add_argument("--load",
                        help="states whether an existing model should be loaded for training",
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
    
    parser.add_argument("--cyclic_learning_rate",
                        help="enable cyclic learning rate",
                        nargs=2,
                        default=[])

    parser.add_argument("--alt_opt_flow",
                        help="use alternative optical flow method",
                        default=False,
                        action="store_true")
    
    parser.add_argument("--normalize",
                        help="squash labels down to -1 to 1 range, good for LSTM",
                        default=False,
                        action="store_true")
    
    parser.add_argument("--dimensions",
                        help="frame dims",
                        type=int,
                        nargs=2,
                        default=(32,32))

    parser.add_argument("--opt_flow",
                        help="compute optical flow",
                        default=False,
                        action="store_true")
    return parser


def summarize_arguments(args):
    """
    report the arguments passed into the app 
    """
    formatter = "[%s] %s"

    print(formatter % ("model_type", args.model_type))
    print(formatter % ("data", args.data))
    print(formatter % ("partition_csv", args.partition_csv))
    print(formatter % ("csv", args.csv))
    
    print(formatter % ("ignore_augmented", str(args.ignore_augmented)))
    formatter = "[%s] %r"

    print(formatter % ("train", args.train))
    print(formatter % ("load", args.load))
    print(formatter % ("test", args.test))

    formatter = "[%s] %d"

    print(formatter % ("batch_size", args.batch_size))
    print(formatter % ("epochs", args.epochs)) 
    print(formatter % ("greyscale_on", args.greyscale_on)) 
    print(formatter % ("alt_opt_flow", args.alt_opt_flow)) 
    print(formatter % ("normalize", args.normalize)) 
    print(formatter % ("optical_flow", args.opt_flow)) 
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
    

def validate_arguments(args):
    """
    validate the arguments provided, handle bad/incomplete input

    args:
        args - the arguments provided to the script throught parse_input

    returns:
        regular : str - path to the "regular" data directory
        augmented : str - path to the "augmented" data directory
        csv : str - path to filtered_csv
        partitions_csv : str - the path to to the label data
        batch_size : int - batch size
        epochs : int - epochs to train for, if necessary
        train : bool - whether or not to train
        test : bool - whether or not to test
        input_dir : str - the path to the directory containing post experiment/training meta data and results
        output_dir : str - the path to the directory where we will place experimental meta data and results
    """
    for data_dir in args.data:
        if not os.path.isdir(data_dir):
            raise ArgumentError("Bad data directory : %s" % data_dir)
    
    regular, augmented = None, None

    if len(args.data) > 1:
        assert len(args.data) == 2, "Expected maximum two directories, regular and augmented; got %d" % len(args.data)
        print("[validate_arguments] : taking %s to be `regular`, %s to be `augmented`" % (args.data[0], args.data[1]))
        augmented = args.data[1]
    
    if args.ignore_augmented is not None and len(args.ignore_augmented) > 1:
        assert len(args.ignore_augmented) < 4, "Expected maximum three phases to ignore, got %d" % len(args.ignore_augmented)
        
        assert args.ignore_augmented[0] != args.ignore_augmented[1], "Why would you pass in two \
                of the same argument? Are you dumb? %s and %s" % (args.ignore_augmented[0], args.ignore_augmented[1])
        
        if len(args.ignore_augmented) > 2:
            
            assert args.ignore_augmented[0] != args.ignore_augmented[2], "Why would you pass in two \
                    of the same argument? Are you dumb? %s and %s" % (args.ignore_augmented[0], args.ignore_augmented[2])
            
            assert args.ignore_augmented[1] != args.ignore_augmented[2], "Why would you pass in two \
                    of the same argument? Are you dumb? %s and %s" % (args.ignore_augmented[1],
                                                                      args.ignore_augmented[2])
    
    regular = args.data[0]
    if args.opt_flow:
        assert args.model_type in ["OpticalFlowCNN", "3D-CNN"]

    # if --test=False and --train=False, exit because there's nothing to do
    if (not args.train) and (not args.test):
        raise ArgumentError("Both --train and --test were provided as False " +
                            "exiting because there's nothing to do ...")

    # if batch_size was provided to be negative, exit for bad input
    if args.batch_size < 0:
        raise ArgumentError("The --batch_size should be > 0; " +
                            "got %d" % args.batch_size)
    
    batch_size = args.batch_size

    # if epochs was provided to be negative, exit for bad input
    if args.epochs < 0:
        raise ArgumentError("The --epochs should be > 0; " +
                            "got %d" % args.epochs)
   
    epochs = args.epochs

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
        
        # verified everythign works, now erase the output directory because the input is the output
        print("[validate_arguments] : overwriting output directory from %s to %s" % (args.output_dir, args.input_dir))
        args.output_dir = args.input_dir

    # if --test=True and --train=True, then we need only an output directory
    if args.train and args.test:
        generate_output_directory(args.output_dir)
        if not args.load:
            print("[validate_arguments] : overwriting input directory from %s to %s" % (args.input_dir, args.output_dir))
            args.input_dir = args.output_dir
        
    input_dir, output_dir = args.input_dir, args.output_dir
    
    assert os.path.exists(args.csv), "%s not found" % args.csv
    assert os.path.exists(args.partition_csv), "%s not found" % args.partition_csv
 
    return regular, augmented, args.csv, args.partition_csv, batch_size, epochs, args.train, args.load, args.test, input_dir, output_dir, args.greyscale_on 


if __name__ == "__main__":
    
    args = parse_input().parse_args()
    regular, augmented, filtered_csv, partition_csv, batch_size, epochs, train, load, test, inputs, outputs, greyscale_on = validate_arguments(args)
    
    summarize_arguments(args)
    scaler = None
    if args.normalize:
        scaler = MinMaxScaler(feature_range=(-1,1))
        sd = pd.read_csv(partition_csv)
        values = np.array(list(sd['Heart Rate'])).reshape(-1,1)
        scaler.fit(values)
    fp = FrameProcessor(scaler,
                        rotation_range=args.rotation_range,
                        width_shift_range=args.width_shift_range,
                        height_shift_range=args.height_shift_range,
                        shear_range=args.shear_range,
                        zoom_range=args.zoom_range,
                        vertical_flip=args.vertical_flip,
                        horizontal_flip=args.horizontal_flip,
                        batch_size=batch_size,
                        greyscale_on=greyscale_on)

    input_shape = None
    x, y = args.dimensions
    if args.opt_flow:
        input_shape = (60, x, y, 2)
    elif greyscale_on:
        input_shape = (60, y, x, 1)
    else:
        input_shape = (60, x, y, 3)
    print(input_shape)
    cyclic_lr = [float(i) for i in args.cyclic_learning_rate]

    engine = Engine(data=regular,
                    model_type=args.model_type,
                    filtered_csv=partition_csv,
                    batch_size=batch_size,
                    epochs=epochs,
                    train=train,
                    load=load,
                    test=test,
                    inputs=inputs,
                    outputs=outputs,
                    frameproc=fp,
                    ignore_augmented=args.ignore_augmented,
                    input_shape=input_shape,
                    steps_per_epoch=args.steps_per_epoch,
                    cyclic_lr=cyclic_lr,
                    alt_opt_flow=args.alt_opt_flow,
                    opt_flow=args.opt_flow)

    print("starting ... ")
    start = time.time()
    engine.run2()
    end = time.time()
    total = (end - start) / 60
    if train:
        with open(os.path.join(outputs, "time.txt"), 'w') as t:
            t.write(str(total))
