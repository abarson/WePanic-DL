"""
------------------
| collate_data.py |
------------------

a (single use) script for collating all of the data necessary to
run the run_model.py script

OBJECTIVE
---------
--> build a tidy data csv containing the following fields:
    -SUBJECT---------| the subject
    -TRIAL-----------| the trial
    -FRAME_PTH-------| path to the frame directory of this (subject, trial) pair
    -HEART_RATE_BPM--| heart rate (beats per minute)
    -RESP_RATE_BR_PM-| respiratory rate (breaths per minue)
    -GOOD------------| whether this subject, trial pair is useable
"""

import pandas as pd
import sys, os
import argparse
import random
from sklearn.model_selection import train_test_split

#random.seed(42) # why not
seed = 42 # why not

# some invariants

NEW_BATCH = [
            (2, 1), (6, 1), (13, 1), (13, 2),
            (25, 1), (31, 1), (38, 1), (38, 2), (39, 1), (48, 1),
            (48, 2), (101, 1), (101, 2), (102, 1), (105, 1), (105, 2),
            (106, 1), (106, 2), (107, 1), (107, 2), (108, 1), (108, 2),
            (109, 1), (109, 2), (110, 1), (110, 2), (111, 1), (111, 2),
            (112, 1), (112, 2), (113, 1), (113, 2), (114, 1), (114, 2),
            (115, 1), (115, 2), (116, 1), (116, 2), (117, 1), (117, 2),
            (118, 1), (118, 2)
            ]

GOOD_PAIRS = list(set(NEW_BATCH))

# our tidy data columns
COLUMNS = 'SUBJECT,TRIAL,FRAME_PTH,HEART_RATE_BPM,RESP_RATE_BR_PM,GOOD'.split(',')

OUTPUT_CSV = 'wepanic_collated_catalogue.csv'

def parse_args():
    parser = argparse.ArgumentParser(description='collate ALL the data!')

    parser.add_argument('dlcd_loc',
                        type=str,
                        help='the location of DeepLearningClassData.csv on your computer')
    
    parser.add_argument('--frame_dir',
                        type=str,
                        help='the frame directory',
                        default='picky_32_32')
    return parser


def verify(filename):
    """
    verify that dlcd_loc is in fact an instance of
    DeepLearningClassData.csv (at least at the filename level)
    and that the file itself is valid

    args:
        filename: string - the filename in question

    returns:
        boolean stating whether or not this is indeed a correct input
    """
    verified = True
    errs = []

    file_title = filename.split('/')[-1]  # should be DeepLearningClassData.csv 

    if not file_title == 'DeepLearningClassData.csv':
        verified = False
        errs.append('No seriously, pass in the DeepLearningClassData.csv on your computer, not %s' % file_title)
        

    if not os.path.exists(filename):
        verified = False
        errs.append('No such file: %s' % filename)
   
    errs = ['[!] ' + e for e in errs]

    if len(errs) > 0:
        print('\n'.join(errs))

    return verified

    
if __name__ == '__main__':
    args = parse_args().parse_args()
    dlcd_loc = parse_args().parse_args().dlcd_loc
    print('assuming this script was called from the top level of this directory ...')

    if not verify(dlcd_loc):
        sys.exit('An error occurred')
    
    dlcd_df = pd.read_csv(dlcd_loc)

    new_cols = ['SUBJECT',
                'Trial 1 Heart Rate (beats per minute)',
                'Trial 1 Respiratory Rate (breaths per minute)',
                'Trial 2 Heart Rate (beats per minute)',
                'Trial 2 Respiratory Rate (breaths per minute)']

    old_cols = list(dlcd_df.columns)
    col_rename = dict(zip(old_cols, new_cols))
    dlcd_df.rename(columns=col_rename, inplace=True)

    collated_df = pd.DataFrame(columns=COLUMNS)
    tf_dict = {True:1, False:0} 
    subj_fmt = 'S%04d'
    trial_fmt = 'Trial%d_frames'
    
    i = 0

    Xs, ys = [], []

    for row in dlcd_df.iterrows():
        idx, data = row
        
        subject = data['SUBJECT']
        hrate_col = 'Trial {} Heart Rate (beats per minute)'
        resp_rate_col = 'Trial {} Respiratory Rate (breaths per minute)'
        for trial in [1, 2]:
            
            good = tf_dict[(subject, trial) in GOOD_PAIRS]

            if good == 1:
                Xs.append((subject, trial))            
                ys.append(1)

            hrate = data[hrate_col.format(trial)]
            resp_rate = data[resp_rate_col.format(trial)]
            frame_pth = os.path.join(args.frame_dir, subj_fmt % subject, trial_fmt % trial) 
            
            collated_row = [subject, trial, frame_pth, hrate, resp_rate, good]
            collated_df.loc[i] = collated_row

            i += 1
    
    
    # get a test set
    Xtrain, Xtest, _, _, = train_test_split(Xs, ys, test_size=0.2, random_state=seed)
    print('Train : ', Xtrain)
    print('Test: ', Xtest)
    for subj, tri in Xtest:
        row_idx = collated_df.loc[(collated_df['SUBJECT'] == subj) & (collated_df['TRIAL'] == tri)].index
        collated_df.loc[row_idx, 'GOOD'] = 2
        
        #collated_df[(collated_df['SUBJECT'] == subj) & (collated_df['TRIAL'] == tri)]['GOOD'] = 2  # add to test set


    collated_df.to_csv(OUTPUT_CSV, index=False)
    print('wrote collated data to %s' % OUTPUT_CSV)


