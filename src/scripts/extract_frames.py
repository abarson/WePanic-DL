"""
Extract frames by passing a csv file
and extracting the frames from the subjects listed in that csv

"""

import sys
import os
import pandas as pd
import argparse

import we_panic_utils.basic_utils.basics as base
import we_panic_utils.basic_utils.video_core as vc


def usage(with_help=True):
    print("[Usage]: %s <csv_file> <data_dir> <output_directory>" % sys.argv[0])
    if with_help:
        print("         %s HELP|help|h for more info" % sys.argv[0])
    sys.exit()


def help_msg():
    print("[%s]" % sys.argv[0])
    print("-" * len("[%s]" % sys.argv[0]))

    print("\t| script meant to extract video files from a specified data_dir ")
    print("\t| by reading a .csv file and convert the videos to frame data\n")
    usage(with_help=False)


def parse_input():
    parser = argparse.ArgumentParser("Extract frames from video files in a specified movie directory")
    parser.add_argument("selects",
                        help="csv of selected subjects",
                        type=str)

    parser.add_argument("data_csv",
                        help="the csv containing all of the subject data",
                        type=str)

    parser.add_argument("movie_directory",
                        help="video file directory",
                        type=str)

    parser.add_argument("output_directory",
                        help="the directory to save the frame directories",
                        type=str)

    return parser


if __name__ == "__main__":
    
    args = parse_input().parse_args()
    
    selects = args.selects
    metadata = args.data_csv
    movie_dir = args.movie_directory
    output_directory = args.output_directory
    
    # validate input
    base.check_exists_create_if_not(output_directory)
    
    if not os.path.exists(selects):
        raise FileNotFoundError("[selects] -- %s not found" % selects)
    
    if not os.path.exists(metadata):
        raise FileNotFoundError("[data_csv] -- %s not found" % metadata)

    if not os.path.isdir(movie_dir):
        raise FileNotFoundError("[movie_dir] -- %s not found" % movie_dir)
    
    #  we're going to assume the csvs specified are properly formatted and the columns are correct
    
    selects_df = pd.read_csv(selects)
    metadf = pd.read_csv(metadata)

    frame_df = pd.DataFrame(columns=["Subject", "Trial", "Path", "Heart Rate", "Respiratory Rate"])

    fmt_file = "Trial%d.MOV"
    
    imgs_captured = []
    
    columns = metadf.columns
    
    i = 0
    for index, row in selects_df.iterrows(): 
        subject, trial = row['Subject'], int(row['Trial'])
        fmt_dir = 'S' + subject.zfill(4)
        target_dir = os.path.join(movie_dir, fmt_dir)
        target_file = fmt_file % trial

        target = os.path.join(target_dir, target_file)
        
        data = metadf[metadf['SUBJECT'] == str(subject)]
        
        subj, t1_hrate, t1_resprate, t2_hrate, t2_resprate = [list(data[col])[0] for col in columns]
        
        if trial == 1:
            r = [subject, trial, os.path.join(output_directory, str(subject), str(trial)), t1_hrate, t1_resprate]

        else:
            r = [subject, trial, os.path.join(output_directory, str(subject), str(trial)), t2_hrate, t2_resprate]
        
        frame_df.loc[i] = r
        if not os.path.exists(target.replace(movie_dir, output_directory).split('.')[0]+'_frames'):
            imgs = vc.video_file_to_frames(target, output_dir=output_directory, suppress=False)
            print("-" * 78)
            imgs_captured.extend(imgs)
        else:
            print("{} already exists, skipping".format(target.replace(movie_dir, output_directory)))
        i += 1
    frame_df.to_csv("subject_data.csv", index=False)
    print("[*] Extracted %d images from %d different video files" % (len(imgs_captured), i + 1))
