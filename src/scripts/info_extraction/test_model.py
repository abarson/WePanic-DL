"""
test a specific model on the testing set and
observe its output
"""

import argparse
import os
import pandas as pd

def parse_args():

    parser = argparse.ArgumentParser(description='test a model on the test set')

    parser.add_argument('model_dir',
                        help='model directory; if no model provided then infer the best one' +
                        'from cvresults.csv',
                        type=str)

    parser.add_argument('--model',
                        help='relative path of a specific model',
                        type=str,
                        default=None)

    parser.add_argument('--metadata',
                        help='the csv data to use as a catalogue to find the test set',
                        type=str,
                        default='wepanic_collated_catalogue.csv')
    return parser



def validate(args):
    """
    validate the arguments passed in
    """
    if not os.path.exists(args.model_dir):
        raise OSError('No such directory exists: %s' % args.model_dir)

    if not os.path.isdir(args.model_dir):
        raise NotADirectoryError('Not a directory: %s' % args.model_dir)

    # args.model_dir is a directory ...
    if args.model is not None: 
        if not os.path.exists(os.path.join(args.model_dir, args.model)):
            raise FileNotFoundError('No such file: %s' % os.path.join(args.model_dir, args.model))

        if not args.model.endswith('.h5'):
            raise ValueError('Malformed weights file: %s' % args.model)
   
    if not os.path.exists(args.metadata):
        raise FileNotFoundError('No such metadata file: %s' % args.metadata)

    return args


def infer_model(args):
    """
    infer the model to test
     - if no model is provided, infer from $model_dir/cvresults.csv
     - otherwise return the model provided

    args:
        args : (Namespace) - the validated argument list passed to the script    

    returns:
        (string) the relative path of the filename of the model in question 
    """
    
    if args.model is not None:
        return args.model

    cvresults = pd.read_csv(os.path.join(args.model_dir, 'cvresults.csv')) 
    minimum_loss = cvresults['loss'].min()
    top_row = cvresults[cvresults['loss'] == minimum_loss]

    model_name = 'CV_%d_%s.h5' % (top_row['model_idx'], top_row['model_type'])

    return model_name 

if __name__ == '__main__':
    args = validate(parse_args().parse_args())
    args.model = infer_model(args) 

    fp = FrameProcessor(redscale_on=True)
