"""
command line app for train/testing models.
"""

# inter library imports
import argparse
import sys
import os
import time
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from functools import partial
from glob import glob

# haahahaahahah remoe stupid LOGS!!!!

#os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
#stderr = sys.stderr

# goodbye Using Tensorflow backend.
#sys.stderr = open('/dev/null', 'w')

# intra library imports
import we_panic_utils.basic_utils.basics as B
import we_panic_utils.nn.models as models
import we_panic_utils.nn.functions as funcs

from we_panic_utils.nn import Engine
from we_panic_utils.nn.processing import FrameProcessor
from we_panic_utils.nn.callbacks import CyclicLRScheduler

# fix it up
#sys.stdout = stderr


# get available models, loss functions
MODEL_CHOICES = B.get_module_attributes(models, exclude_set=['RegressionModel'])
LOSS_FUNCTIONS = list(filter(lambda fun: fun.split('_')[-1] == 'loss', B.basics.get_module_attributes(funcs))) + funcs.get_keras_losses()
LEARNING_RATES = list(filter(lambda fun: fun.endswith('lr'), B.basics.get_module_attributes(funcs)))
FEATURE_TRANSLATE = {'hr' : 'HEART_RATE_BPM', 'rr' : 'RESP_RATE_BR_PM'}

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
                        choices=MODEL_CHOICES)
    
    parser.add_argument("data",
                        help="director[y|ies] to draw data from",
                        type=str)

    parser.add_argument("--features",
                        help="the features to use for model training",
                        nargs='+', 
                        type=str,
                        default='hr',
                        choices=['hr', 'rr'])

    parser.add_argument("--csv",
                        help="csv containing labels subject -- trial -- heart rate -- resp rate",
                        type=str,
                        default="wepanic_collated_catalogue.csv")

    parser.add_argument("--qbc",
                        help="perform a query by committee test (specify a directory of .h5 files)",
                        type=str,
                        default=None)

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
    
    parser.add_argument("--redscale_on",
                        help="convert images to redscale at runtime",
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
    
    parser.add_argument('--kfold',
                        help='folds for CV',
                        type=int,
                        default=None)

    parser.add_argument('--num_val_clips',
                        help='number of validation clips to sample from validation set',
                        type=int,
                        default=10)

    parser.add_argument('--cyclic_lr',
                        help='cyclic learning rate function to apply; a tuple (func_name, min_lr,  max_lr, stepsize)',
                        nargs=4)

    parser.add_argument('--loss',
                        help='loss function inn [keras.losses, we_panic_utils.functions]',
                        default=None,
                        choices=LOSS_FUNCTIONS)

    parser.add_argument('--early_stopping',
                        help='tolerance for # of iterations without improvement to start',
                        default=None,
                        type=int)

    parser.add_argument('--sequence_length',
                        help='sequence size for feeding the DNN',
                        default=60,
                        type=int)
    
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
        raise NotADirectoryError("Bad data directory : %s" % args.data)


    # if batch_size was provided to be negative, exit for bad input
    if args.batch_size < 0:
        raise ValueError("The --batch_size should be > 0; " +
                         "got %d" % args.batch_size)

    # if epochs was provided to be negative, exit for bad input
    if args.epochs < 0:
        raise ValueError("The --epochs should be > 0; " +
                         "got %d" % args.epochs)
    
    # if --test=False --train=False, --kfold=None, exit because there's nothing to do
    if args.qbc is None and args.kfold is None:
        sys.exit("QBC, Kfold not activated. Exiting because there's nothing to do")
    
    # if kfold we need only an output directory
    if args.kfold is not None:
        args.input_dir = generate_output_directory(args.output_dir)
        if args.qbc is not None and args.qbc != os.path.join(args.input_dir, 'models'): 
            print(">>> QBC ({}) is not input_dir despite kfold={} being specified. That's ok, " + 
                  "just checking {} contains .h5's ... ".format(args.qbc, args.kfold, args.qbc))

            verify_h5_exists(args.qbc)
    else:
        verify_h5_exists(args.qbc)
        args.output_dir = args.input_dir

    bad_augs = []
    for arg_name in ['steps_per_epoch', 'shear_range', 'width_shift_range', 
                     'height_shift_range', 'rotation_range', 'zoom_range', 
                     'early_stopping', 'sequence_length']:
        
        try:
            if vars(args)[arg_name] < 0:
                bad_augs.append("%s can't be < 0" % arg_name)

            if arg_name in ['shear_range', 'width_shift_range', 'height_shift_range'] and vars(args)[arg_name] > 1:
                bad_augs.append("%s is a value in [0, 1]" % arg_name)

        except TypeError:
            #sys.exit('arg_name : %s ' % arg_name)
            pass

    if len(bad_augs) > 0:
        raise ValueError(",".join(bad_augs))

    
    if args.cyclic_lr is not None:
        try:
            func_name, min_lr, max_lr, stepsize = args.cyclic_lr 
            min_lr, max_lr = float(min_lr), float(max_lr)
            stepsize = int(stepsize)
            assert isinstance(min_lr, float) and isinstance(max_lr, float), 'min_lr, max_lr expected to be floats got %f and %f' % (min_lr, max_lr)
            assert isinstance(stepsize, int), 'expected stepsize to be an int, got {}'.format(stepsize)
            assert 0 < stepsize, 'stepsize must be > 0, not {}'.format(stepsize)
            assert 0. <= min_lr < max_lr, 'need this relation ship : 0. <= min_lr={} < max_lr={} <= 1.'.format(min_lr, max_lr) 

            inner_func = getattr(funcs, func_name)
            lr_func = partial(inner_func,
                              base_lr=min_lr,
                              max_lr=max_lr,
                              stepsize=stepsize)
            
            args.cylic_LR_schedule = args.cyclic_lr
            args.cyclic_lr = CyclicLRScheduler(output_dir=args.output_dir,
                                               schedule=lr_func,
                                               steps_per_epoch=args.steps_per_epoch,
                                               base_lr=min_lr,
                                               max_lr=max_lr)
        except AttributeError as e:
            sys.exit(e)

    if len(args.features) > 1:
        assert args.features[0] != args.features[1], "The same feature cannot be chosen twice for training"

    if args.greyscale_on and args.redscale_on:
        raise ValueError("both redscale and greyscale cannot be activated")

    return args
        

def summarize_arguments(args):
    """
    report the arguments passed into the app 
    """
    formatter = "[%s] %+15s"
    
    keys = vars(args).keys()
    max_arglen = max([len(k) for k in keys])
    
    formatter = "[%{}s] %40s".format(max_arglen)
    
    summ_str = ""
    for k in keys:
        if k in ['cyclic_lr']:
            continue

        summ_str += formatter % (k, vars(args)[k]) + '\n'
    
    print(summ_str)
    
    arg_summary = os.path.join(args.output_dir, 'arg_summary.txt')

    with open(arg_summary, 'w') as summary:
        summary.write(summ_str)
    
    print('Wrote input args to %s' % arg_summary)
    
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

    B.check_exists_create_if_not(output_dirname)
    B.check_exists_create_if_not(model_dir)

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
        
        return verified

    else:
        return False

def verify_h5_exists(directory):
    """
    --- check directory exists
    --- check file is a directory
    --- verify .h5 files are in it
    --- save the world
    """

    if not os.path.exists(directory):
        raise FileNotFoundError('No such directory: {}'.format(directory))

    if not os.path.isdir(directory):
        raise NotADirectoryError('Not a directory: {}'.format(directory))
    
    
    contents = glob(os.path.join(directory, '*.h5'))
    
    if not contents:
        raise ValueError('{} contains no .h5 files. QBC requires .h5 files'.format(directory)) 

if __name__ == "__main__":
    args = parse_input().parse_args()
    args = validate_arguments(args)

    feat_trans = [FEATURE_TRANSLATE[feat] for feat in args.features]
    summarize_arguments(args)

    input_shape = None
    x, y = args.dimensions

    if args.greyscale_on or args.redscale_on:
        input_shape = (args.sequence_length, x, y, 1)

    else:
        input_shape = (args.sequence_length, x, y, 3)

    fp = FrameProcessor(features=feat_trans,
                        input_shape=input_shape[1:],
                        rotation_range=args.rotation_range,
                        width_shift_range=args.width_shift_range,
                        height_shift_range=args.height_shift_range,
                        shear_range=args.shear_range,
                        zoom_range=args.zoom_range,
                        vertical_flip=args.vertical_flip,
                        horizontal_flip=args.horizontal_flip,
                        batch_size=args.batch_size,
                        greyscale_on=args.greyscale_on,
                        redscale_on=args.redscale_on,
                        num_val_clips=args.num_val_clips,
                        sequence_length=args.sequence_length)


    engine = Engine(data=args.data,
                    model_type=args.model_type,
                    features=feat_trans,
                    csv=args.csv,
                    batch_size=args.batch_size,
                    epochs=args.epochs,
                    qbc=args.qbc,
                    inputs=args.input_dir,
                    outputs=args.output_dir,
                    frameproc=fp,
                    input_shape=input_shape,
                    steps_per_epoch=args.steps_per_epoch,
                    kfold=args.kfold,
                    cyclic_lr=args.cyclic_lr,
                    loss_fun=args.loss)

    start_banner =  """\t\t\t_________________________________
                       |   ______________________  __   | 
                       |   |    \  __| _   |  |  \|  |  |
                       |   | :  /  __|\__  |  |      |  |
                       |   | :  \____|__/  /  |      |  |
                       |   |____/_________/|__|__|\__|  |
                       |________________________________|"""
    
    end_banner   = """\t\t\t__________________________________
                      |
                      |   ______
                      |   |  ___|
                  
                          |   |__ 
                      |   |_____|
                      |__________________________________"""	
    print(start_banner,sep='')
    start = time.time()
    engine.run()
    end = time.time()
    total = (end - start) / 60

    if args.kfold:
        with open(os.path.join(args.output_dir, "time.txt"), 'w') as t:
            t.write(str(total))
