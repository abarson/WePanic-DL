"""       __________________________
perform a [[[ query by committee ]]] on a directory containing 
          --------------------------

at least one .h5 file in its top level
---------------------------------------------------------------
Querying By Committee (QBC) is pooling the prediction of several
networks and somehow combining (average, weighted average, etc)
their answers into one prediction
"""

import argparse
import os
import pandas as pd
from we_panic_utils.nn.processing import FrameProcessor

from keras.models import load_model

def parse_args():
    parser = argparse.ArgumentParser(description='pool the opinion of a committee of nets')

    parser.add_argument('committee',
                        help='directory containing >= 1 .h5 file in its top level',
                        type=str)
    
    parser.add_argument('--features',
                        help='features to predict on',
                        nargs='+',
                        choices=['hr','rr'])

    parser.add_argument('-f',
                        help='location of test set description (def: wepanic_collated_catalogue.csv',
                        default='wepanic_collated_catalogue.csv',
                        type=str)

    return parser

def verify(args):

    if not os.path.exists(args.committee):
        raise FileNotFoundError('No such directory: {}'.format(args.committee))

    if not os.path.isdir(args.committee):
        raise NotADirectoryError('Not a directory: {}'.format(args.committee))
    
    contents = os.listdir(args.committee)

    if not any([os.path.join(args.committee, c).endswith('.h5') for c in contents]):
        raise ValueError('need at least one member (.h5) of committee')

    if not args.features:
        raise ValueError('need at least one of {hr, rr} as features')

    if not os.path.exists(args.f):
        raise FileNotFoundError('No such file {}'.format(args.f))
    
if __name__ == '__main__':
    args = parse_args().parse_args()
    verify(args)

    metadata = pd.read_csv(args.f)
    test_data = metadata[metadata['GOOD'] == 1]    
    
    contents = os.listdir(args.committee)
    contents = [os.path.join(args.committee, c) for c in contents]
    members  = list(filter(lambda c: c.endswith('.h5'), contents))
