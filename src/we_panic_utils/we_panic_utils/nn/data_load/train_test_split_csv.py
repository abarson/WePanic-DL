from . import split_utils as util
import we_panic_utils.basic_utils.basics as base

import csv
import os
import sys
import pandas as pd
import random
random.seed(7)
import numpy as np

"""
Public API
"""


def data_set_from_csv(csv_path, augmented_dir=None):
    ignore = False

    if augmented_dir is not None:
        ignore = True

    if not os.path.exists(csv_path):
        raise FileNotFoundError("{} was not found".format(csv_path))
    data = {}
    with open(csv_path, 'r') as csv_in:
        reader = csv.reader(csv_in)
        next(reader)
        for path, hr, rr in reader:
            if not ignore or augmented_dir not in path:
                # data[path] = (hr, rr)

                data[path] = (int(hr), int(rr))
    return data


def data_set_to_csv(data_set, csv_path, verbose=True):
    if not csv_path.endswith('.csv'):
        csv_path += '.csv'
    if verbose:
        print('[data_set_to_csv]: creating csv at -> {}'.format(csv_path))
    with open(csv_path, 'w') as csv_out:
        writer = csv.writer(csv_out)
        header = ['PATH', 'HEART RATE', 'RESPIRATORY RATE']
        writer.writerow(header)
        for key in data_set:
            data = data_set[key]

            # writer.writerow([key, data[0], data[1]])
            writer.writerow([key, str(data[0]), str(data[1])])


def generate_paths2labels(df, data_path):
    """
    create the sought after and beloved `paths2labels`
    also know as `filtered_#$@%ing_paths` from a dataframe
    
    format assuming "S%4d/Trial%d_frames" is the name of every
    directory inside of data_path

    args:
        df : DataFrame
        data_path : path to data
    """
    
    filtered_paths = dict()
   
    subj_fmt = "S%04d"
    trial_fmt = "Trial%d_frames"

    for row in df.iterrows():
        subject, trial, h_rate, resp_rate = row['Subject'], row['Trial'], row['Heart Rate'], row['Respiratory Rate']
    
        subject = subj_fmt % subject
        trial = trial_fmt % trial
    
        P = os.path.join(data_path, subject, trial)
        filtered_paths[P] = (h_rate, resp_rate)

    return filtered_paths

def __choose_rand_test_set(all_subs, set_size):
    """
    Simple utility method that creates a subset of data to be used for testing.
    Reduces the size of the dataset passed in.
    """

    testing_set = []
    while len(testing_set) < set_size:
        rand_index = random.randint(0, len(all_subs)-1)
        testing_set.append(all_subs[rand_index])
        all_subs.pop(rand_index)
    return testing_set

def __dataframe_from_subject_info(metadf, subjects):
    """
    Utility method that, given a list of subject-trial pairs, extracts those pairs from
    an existing data frame, and returns a data frame containing those subject-trial pairs,
    along with their accompanying augmentations.
    """

    dfs = []
    for sub, trial in subjects:
        dfs.append(metadf[(metadf['Subject'] == sub) & (metadf['Trial'] == trial)])
        #extract augmented videos associated with this particular subject-trial pair.
        dfs.append(metadf[metadf['Subject'].apply(lambda x: not x.isdigit() and int(x[-2:]) == int(sub)) 
            & (metadf['Trial'] == trial)])
    return pd.concat(dfs)

def create_train_test_split_dataframes(data_path, metadata, output_dir,
            test_split=0.2, val_split=0.15, verbose=True):
    """
    Description coming soon!
    """

    metadf = pd.read_csv(metadata)
    metadf['Path'] = metadf.apply (lambda row: os.path.join(data_path, "S" + str(row["Subject"]).zfill(4), 
        "Trial%d_frames" % row["Trial"]), axis=1)
    
    real_subjects_df = metadf[metadf['Subject'].apply(lambda x: x.isdigit())]
    
    real_subs = list(zip(real_subjects_df['Subject'], real_subjects_df['Trial']))
    num_test, num_val = int(len(real_subjects_df) * test_split), int(len(real_subjects_df) * val_split)
    
    test_subs = __choose_rand_test_set(real_subs, num_test)
    val_subs = __choose_rand_test_set(real_subs, num_val)
    train_subs = real_subs

    test_df    = __dataframe_from_subject_info(metadf, test_subs)
    val_df     = __dataframe_from_subject_info(metadf, val_subs)
    train_df   = __dataframe_from_subject_info(metadf, train_subs)
    assert len(test_df) + len(val_df) + len(train_df) == len(metadf)
    base.check_exists_create_if_not(output_dir)
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)

    return train_df, test_df, val_df

def ttswcvs3(data_path, metadata, output_dir,
             test_split=0.2, val_split=0.2, verbose=True):

    """
    DEPRECATED (use create_train_test_split_dataframes)

    Train test split with CSV support version 3.
    Currently no support for ignoring augmented data!

    This function is similar to the previous variants, except that it creates and returns data frames,
    instead of directly working with CSV files.
    This version works with "buckets", which each data point belongs to. This is to ensure that all data
    has roughly the same chance of being seen.
    """
    metadf = pd.read_csv(metadata)
    
    metadf['Path'] = metadf.apply (lambda row: os.path.join(data_path, "S%04d" % row["Subject"], 
        "Trial%d_frames" % row["Trial"]), axis=1)
        
    base.check_exists_create_if_not(output_dir)    
    
    test_len = int(test_split * len(metadf))
    val_len = int(val_split * len(metadf))
    metadf, test_df = util.get_testing_set(metadf, test_len)
    metadf, val_df = util.get_testing_set(metadf, val_len) 

    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
    metadf.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    return metadf, test_df, val_df
