from . import split_utils as util
import we_panic_utils.basic_utils.basics as base

import csv
import os
import sys
import pandas as pd
import random
random.seed(7)
import numpy as np
import math

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
    metadf['Path'] = metadf.apply(lambda row: os.path.join(data_path, "S" + str(row["Subject"]).zfill(4), 
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


def fold(df, k=4):
    """
    a generator for serving up train/validation splits

    args:
        df - the dataframe that describes all of the valid data
        k - the number of folds

    yields:
        training and validation dataframes
    """
    df = df[df['GOOD'] == 1]  # just to make sure
    uniques = list(set(df['SUBJECT'].values.tolist()))
    n_val = int(np.ceil(len(uniques) / k))
    n_train = len(uniques) - n_val

    validated = []
    for _ in range(k):
        
        try:
            train = validated + random.sample(uniques, n_train - len(validated))
        except ValueError:
            train = validated

        val = list(set(uniques) - set(train))
        train_df = df[df['SUBJECT'].isin(train)]
        val_df = df[df['SUBJECT'].isin(val)]

        validated.extend(val)
        uniques = list(set(uniques) - set(validated))

        yield train_df.drop('Unnamed: 0', axis=1), val_df.drop('Unnamed: 0', axis=1)

def fold_v2(df, k=4):
    """
    a generator for serving up train/validation splits

    args:
        df - the dataframe that describes all of the valid data
        k - the number of folds

    yields:
        training and validation dataframes
    """
    df = df[df['GOOD'] == 1]  # just to make sure
    df['SUBJECT'] = df['SUBJECT'].apply(lambda row : int(row))
    subs = list(df['SUBJECT'].values.tolist())
    tris = list(df['TRIAL'].values.tolist())
    
    indexes = [i for i in range(len(subs))]

    tot = len(df)
    completed = 0
    current_index = 0

    for _ in range(k):
       
        take_out = math.ceil(tot / (k - completed))
        tot -= take_out
        completed += 1
        train_in = indexes[:current_index] + indexes[current_index+take_out:]
        test_in = indexes[current_index:current_index+take_out]
        current_index += take_out
        
        #filter out rows of the original df that contain subj/tri pairs that are in the train set
        train_df = df[df[['SUBJECT', 'TRIAL']].apply(lambda x : 
            __filter_column(*x, zipped=[(subs[index], tris[index]) for index in train_in]), axis=1)]

        #filter out rows of the original df that contain subj/tri pairs that are in the val set
        val_df = df[df[['SUBJECT', 'TRIAL']].apply(lambda x : 
            __filter_column(*x, zipped=[(subs[index], tris[index]) for index in test_in]), axis=1)]
        
        try:
            yield train_df.drop('Unnamed: 0', axis=1), val_df.drop('Unnamed: 0', axis=1)
        
        except ValueError:
            yield train_df, val_df

        except KeyError:
            yield train_df, val_df


def tiers_by_magnitude(sorted_list, n_tier=5):
    """
    private helper function for building a list
    of lists sorted by magnitudes

    args:
        n_tier : number of tiers
        sorted_list : list to split
    
    returns:
        (list) sorted list of lists representing tiers
    """
    tiers = []

    for i in range(n_tier):
        take_out = math.ceil(len(sorted_list) / (n_tier - i))
        tiers.append(sorted_list[:take_out])
        sorted_list = sorted_list[take_out:]

    return tiers

def sorted_stratified_kfold(df, features, k=5):
    """
    generate a K-fold cross validation using sorted stratification;
        - sort cross fold by appropriate feature
        - no subject spans train and validation sets
    
    args:
        :df (pd.DataFrame) - the structure containing the label and metainfo about each subject, trial pair
        :features (list) - the list of features to sort by
        :k (optional, int) - the # of folds

    yields:
        :train_df (pd.DataFrame) - training dataframe
        :val_df (pd.DataFrame) - the validation dataframe
    """
    df              = df[df['GOOD'] == 1]                         # just to make sure
    df['SUBJECT']   = df['SUBJECT'].apply(lambda row: int(row))   # make subjects float --> int

    #subs, tris, hrs = zip(*df[['SUBJECT', 'TRIAL', 'HEART_RATE_BPM']].values.tolist())
    #compiled        = sorted(zip(subs, tris, hrs), key=lambda tup: tup[2])
    
    # sort subjects by multiplicative magnitude
    df = df[df['GOOD'] == 1]

    df['MUL'] = df['HEART_RATE_BPM'] * df['RESP_RATE_BR_PM']
    if len(features) > 1:
        df = df.sort_values('MUL', ascending=[False])
    else:
        df = df.sort_values(features, ascending=[False])

    rows = df.values.tolist() 

    # keep order; filter duplicates
    delegates = []                  # list of subjects kept in relevant sorted order 
    subject_set = set()
    for row in rows:
        subject = row[0]
        if subject not in subject_set:
            delegates.append(subject)
            subject_set |= set([subject])
    
    # break into tiers
    tiers = tiers_by_magnitude(delegates, n_tier=k) 
    
    # generate folds
    folds = [[] for _ in range(k)]
    i = 0
    for T in tiers:
        # distribute T among the folds
        while T:
            # choose a subject by popping a random index from the tier
            chosen_subject = T.pop(random.randint(0, len(T) - 1)) 
            # derive the sample by filtering all r in rows s.t r[0] == the chosen subject;
            #                              r[0] in this case is the subject
            sample = list(filter(lambda r: r[0] == chosen_subject, rows))
            # add this to the fold
            folds[i].extend(sample)
            
            # increment the fold index
            i = (i + 1) % k
    
    # by this point we have a set of folds represented by a list of lists
    # convert each list of lists to 2 dataframes: training and validation 

    ## TODO -- make folds dataframes from the outset 
    for f in folds:
        val_df = pd.DataFrame(columns=df.columns)
        train_df = df.copy()

        for i, fi in enumerate(f):
            subject, trial = fi[0], fi[1]  # extract subject
            rowdf = df[(df['SUBJECT'] == subject) & (df['TRIAL'] == trial)]
            train_df.drop(rowdf.index, inplace=True)
            row  = rowdf.values.tolist()[0]
            val_df.loc[i] = row 
        
        try:
            yield train_df.drop('Unnamed: 0', axis=1), val_df.drop('Unnamed: 0', axis=1)
        
        except ValueError:
            yield train_df, val_df

        except KeyError:
            yield train_df, val_df
    
def __filter_column(*x, zipped):
    return (x[0], x[1]) in zipped


def ttswcsv(data_path, metadata, output_dir,
             test_split=0.2, val_split=0.2, verbose=True):

    """
    Train test split with CSV support version 3.
    Currently no support for ignoring augmented data!

    This function is similar to the previous variants, except that it creates and returns data frames,
    instead of directly working with CSV files.
    This version works with "buckets", which each data point belongs to. This is to ensure that all data
    has roughly the same chance of being seen.
    """
    metadf = pd.read_csv(metadata)
    
    #metadf['FRAME_PTH'] = metadf.apply (lambda row: os.path.join(data_path, "S%04d" % row["SUBJECT"], 
    #    "Trial%d_frames" % row["TRIAL"]), axis=1)
    
    metadf = metadf[metadf.GOOD == 1]

    base.check_exists_create_if_not(output_dir)    
    
    test_len = int(test_split * len(metadf))
    val_len = int(val_split * len(metadf))
    metadf, test_df = util.get_random_test_set(metadf, test_len)
    metadf, val_df = util.get_random_test_set(metadf, val_len) 

    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
    metadf.to_csv(os.path.join(output_dir, 'train.csv'), index=False)

    print('Test DF\n', list(test_df['HEART_RATE_BPM']))
    print('Val DF\n', list(val_df['HEART_RATE_BPM']))
    print('Train DF\n', list(metadf['HEART_RATE_BPM']))

    return metadf, test_df, val_df
