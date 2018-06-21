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

# some invariants
FRAME_DIR = 'frames/'

# good subject trial pairs
GOOD_STARTING_POINT_DATA = [(2, 1), (2, 2), (4, 1), (4, 2),
                            (12, 1), (12, 2), (13, 2), (23, 1),
                            (23, 2), (31, 1), (31, 2)]

NEXT_STARTING_POINT_DATA = [(6, 1), (6, 2), (8, 2), (13, 1), 
                            (25, 1), (30, 1)]

NEW_BATCH = [(38, 1), (38, 2), (39, 1), (39, 2),
             (47, 1), (47, 2), (48, 1), (48, 2)]

OUR_RECORDS = [(101, 1), (101, 2), (102, 1), (103, 2),
               (104, 1), (104, 2), (105, 1), (105, 2),
               (106, 1), (106, 2), (107, 1), (107, 2)]

GOOD_PAIRS = list(set(GOOD_STARTING_POINT_DATA + NEXT_STARTING_POINT_DATA + NEW_BATCH))

# our tidy data columns
COLUMNS = 'SUBJECT,TRIAL,FRAME_PTH,HEART_RATE_BPM,RESP_RATE_BR_PM,GOOD'.split(',')

OUTPUT_CSV = 'wepanic_collated_catalogue.csv'

def parse_args():
    parser = argparse.ArgumentParser(description='collate ALL the data!')

    parser.add_argument('dlcd_loc',
                        type=str,
                        help='the location of DeepLearningClassData.csv on your computer')

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
    for row in dlcd_df.iterrows():
        idx, data = row
        
        subject = data['SUBJECT']
        hrate_col = 'Trial {} Heart Rate (beats per minute)'
        resp_rate_col = 'Trial {} Respiratory Rate (breaths per minute)'
        for trial in [1, 2]:
            
            good = tf_dict[(subject, trial) in GOOD_PAIRS]
            if (subject, trial) in OUR_RECORDS:
                good = 2  # throw this pair into the untouched test set

            hrate = data[hrate_col.format(trial)]
            resp_rate = data[resp_rate_col.format(trial)]
            frame_pth = os.path.join(FRAME_DIR, subj_fmt % subject, trial_fmt % trial) 
            
            collated_row = [subject, trial, frame_pth, hrate, resp_rate, good]
            collated_df.loc[i] = collated_row

            i += 1

    collated_df.to_csv(OUTPUT_CSV)
    print('wrote collated data to %s' % OUTPUT_CSV)


