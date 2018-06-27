"""
add a row to DeepLearningClassData.csv
"""

import pandas as pd
import argparse
import sys, os

def parse_args():

    parser = argparse.ArgumentParser(description='add a row to DeepLearningClassData.csv')

    parser.add_argument('dlcd_loc',
                        help='DeepLearningClassData.csv location',
                        type=str)

    parser.add_argument('-t1','--trial1',
                        nargs=2, type=int,
                        help='the heart rate, respiratory rate pair for trial 1')

    parser.add_argument('-t2','--trial2',
                        nargs=2, type=int,
                        help='the heart rate, respiratory rate pair for trial 2')
                        
    
    return parser

def check_inputs(args):

    if not os.path.exists(args.dlcd_loc):
        raise FileNotFoundError('No such file: {}'.format(args.dlcd_loc))

    if not args.dlcd_loc.endswith('.csv'):
        raise ValueError('Pass in DeepLearningClassData.csv, bad extension: {}'.format(args.dlcd_loc[args.dlcd_loc.find('.'):]))
    
    if not args.dlcd_loc.split('/')[-1] == 'DeepLearningClassData.csv':
        raise ValueError('Specifically looking for DeepLearningClassData.csv, not {}'.format(args.dlcd_loc.split('/')[-1]))

    arg_errs = []
    if args.trial1 is None:
        arg_errs.append('Trial 1')

    if args.trial2 is None:
        arg_errs.append('Trial 2')

    if len(arg_errs) > 0:
        raise ValueError('Need ' + ' AND '.join(arg_errs) + ' statistics')

    return args


if __name__ == '__main__':

    args = check_inputs(parse_args().parse_args())

    dlcd_df = pd.read_csv(args.dlcd_loc)
    
    nrows = len(dlcd_df)
    print('Got a DeepLearningClassData csv with {} entries.'.format(nrows))

    max_subj = max(dlcd_df['SUBJECT'].values.tolist())
    next_subj = max_subj + 1

    print('Adding Subject {}'.format(next_subj))
    t1_hr, t1_rr = args.trial1
    t2_hr, t2_rr = args.trial2
    #print(dlcd_df) 
    print('S{:04d} --[ T1: [hr={}, rr={}] -- T2: [hr={}, rr={}] ]--'.format(next_subj, t1_hr, t1_rr, t2_hr, t2_rr))
    dlcd_df.loc[nrows] = [next_subj, t1_hr, t1_rr, t2_hr, t2_rr]

    dlcd_df.to_csv(args.dlcd_loc, index=False)
        

