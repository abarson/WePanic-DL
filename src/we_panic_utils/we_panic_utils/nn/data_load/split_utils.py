import os
import csv
import random
import pandas as pd

"""
Implementation details
"""

def get_testing_set(df_in, size):
    """
    Extracts a test set given a dataframe of all of the input data.
    Each subject contained by the testing set will belong to a unique bucket.

    args:
        df_in: the dataframe of input data
        size: the number of subjects that the testing set will contain
    return:
        -> a dataframe containing the testing subjects
    """
    df_out = pd.DataFrame()

    l = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9]

    #just to make sure that the testing sets don't exhaust any of the buckets
    #or choose from the same bucket twice, added some additional checks
    selected = []
    while(len(selected) < size):
        chosen_bucket = l[random.randint(0, len(l)-1)]
        
        bucket = df_in[buckets(df_in, chosen_bucket)]
        if (len(bucket) > 1 and chosen_bucket not in selected):
            selected.append(chosen_bucket)
    
    for s in selected:
        df = df_in[buckets(df_in, s)] 
        index = random.randint(0, len(df)-1)
        df_index = df[index:index+1]
        df_out = df_out.append(df_index)
        subject = list(df_index["Subject"])[0]
        trial = list(df_index["Trial"])[0]
        df_in = df_in[(df_in.Subject != subject) | (df_in.Trial != trial)] 
    
    return df_in, df_out

def buckets(df, val):
    if val < 0.1:
        return (df['Heart Rate'] < 45)
    if val < 0.2:
        return (df['Heart Rate'] >= 45) & (df['Heart Rate'] < 60)
    if val < 0.3:
        return (df['Heart Rate'] >= 60) & (df['Heart Rate'] < 75)
    if val < 0.4:
        return (df['Heart Rate'] >= 75) & (df['Heart Rate'] < 90)
    if val < 0.5:
        return (df['Heart Rate'] >= 90) & (df['Heart Rate'] < 105)
    if val < 0.6:
        return (df['Heart Rate'] >= 105) & (df['Heart Rate'] < 120)
    if val < 0.7:
        return (df['Heart Rate'] >= 120) & (df['Heart Rate'] < 135)
    if val < 0.8:
        return (df['Heart Rate'] >= 135) & (df['Heart Rate'] < 150)
    if val < 0.9:
        return (df['Heart Rate'] >= 150) & (df['Heart Rate'] < 175)
    if val <= 1.0:
        return (df['Heart Rate'] >= 175)

def filter_path_with_set(filter_set, all_paths, augment_path=None, verbose=True):
    
    filtered_set_paths = {}
    for subject, trial in filter_set:
        subj_tri = "S{}_t{}".format("0" * (4 - len(subject)) + subject, trial)
       
        # ¯\_﹙ツ﹚_/¯
        # filter out only the keys of the dictionary the belong to this subject and trial
        filtered_keys = [key for key in all_paths.keys() 
                if key.split('/')[1].split('_')[0] == subj_tri.split('_')[0] 
                and key.split('/')[1].split('_')[1] == subj_tri.split('_')[1]]
        
        for key in filtered_keys:
            # filtered_set_paths[key] = all_paths[key] 
            filtered_set_paths[key] = (int(all_paths[key][0]), int(all_paths[key][1]))
            if augment_path is not None:
                try:
                    augmented = fetch_augmented(key, augment_path)
                except FileNotFoundError as e:
                    if verbose:
                        print(e)
                else:
                    heart_rate, resp_rate = filtered_set_paths[key]
                    heart_rate = int(heart_rate)
                    resp_rate = int(resp_rate)
                    if augmented.split('_')[2] == "t1":
                        heart_rate *= 2
                    else:
                        heart_rate //= 2
                    resp_rate = int(round(heart_rate / 4))
                    filtered_set_paths[augmented] = (heart_rate, resp_rate)
            del(all_paths[key])

    return filtered_set_paths


def fetch_augmented(path, augmented_path):
    without_dir = path.split('/')[1]
    with_augmented = os.path.join(augmented_path, without_dir)
    if not os.path.exists(with_augmented):
        raise FileNotFoundError("{} augmented path does not exist, skipping...".format(with_augmented))
    return with_augmented


def set_to_str(title, data_set):
    tit = '{title: ^15}\n'.format(title=title)
    out = tit
    fmt = '{subject: ^10} | {trial: ^10}\n'
    out += fmt.format(subject='Subject', trial='Trial') 
    out += fmt.format(subject='----', trial='----')
    
    for subj, trial in data_set:
        out += fmt.format(subject=subj, trial=trial) 

    fmt = '{bord: ^20}\n'.format(bord='-' * (len(tit) - 1))
    out += fmt
    return out


def fetch_paths_with_labels(consolidated_csv, data_path): 
    with open(consolidated_csv, 'r') as p_csv:
        reader = csv.reader(p_csv)
        next(reader)  # skip the header
        paths = {}
        for path in reader:
            full_path = os.path.join(data_path, path[0])
            if not os.path.exists(full_path):
                raise FileNotFoundError("{} does not exist!".format(full_path))
            paths[full_path] = (path[1], path[2])
    return paths
        

def split_subjects(trial1, trial2, split_p=0.2):
    smallest = min(len(trial1), len(trial2))
    
    num_split = int((smallest * split_p))
    num_split = 1 if num_split == 0 else num_split
    split_set = []
    while len(split_set) < num_split * 2:
        trial1_index = random.randint(0, len(trial1) - 1)
        trial2_index = random.randint(0, len(trial2) - 1)
        split_set.append(trial1[trial1_index])
        split_set.append(trial2[trial2_index])
        trial1.pop(trial1_index)
        trial2.pop(trial2_index)   
    return split_set


def all_subjects(filtered_csv):
    with open(filtered_csv, 'r') as starting_point:
        reader = csv.reader(starting_point)
        next(reader)
        
        all_subj = []
        trial1 = []
        trial2 = []
        for subj in reader:
            if subj[0] not in all_subj:
                all_subj.append(subj[0])
            if subj[1] == "1":
                trial1.append(subj)
            else:
                trial2.append(subj)
        return trial1, trial2
