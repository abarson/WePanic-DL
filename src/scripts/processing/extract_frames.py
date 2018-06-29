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

    parser.add_argument("movie_directory",
                        help="video file directory",
                        type=str)

    parser.add_argument("--collated_data_csv",
                        help="the csv containing all of the subject data",
                        type=str,
                        default='wepanic_collated_catalogue.csv')

    parser.add_argument("--output_directory",
                        help="the directory to save the frame directories",
                        type=str,
                        default='frames/')

    parser.add_argument("--resize_dims",
                        help="resize dimensions",
                        nargs=2,
                        type=int)
    return parser


if __name__ == "__main__":
    
    args = parse_input().parse_args()
    
    metadata = args.collated_data_csv
    movie_dir = args.movie_directory
    output_directory = args.output_directory
    resize_dims = args.resize_dims
    
    if resize_dims == (0, 0):
        resize_dims = None

    # validate input
    base.check_exists_create_if_not(output_directory)
    
    if not os.path.exists(metadata):
        raise FileNotFoundError("[data_csv] -- %s not found" % metadata)

    if not os.path.isdir(movie_dir):
        raise FileNotFoundError("[movie_dir] -- %s not found" % movie_dir)
    
    #  we're going to assume the csvs specified are properly formatted and the columns are correct
    
    metadf = pd.read_csv(metadata)
    selects_df = metadf[metadf['GOOD'] > 0]
    fmt_file = "Trial%d.MOV"
    
    imgs_captured = []
    
    i = 0
    for index, row in selects_df.iterrows(): 
        subject, trial = row['SUBJECT'], int(row['TRIAL'])
        fmt_dir = 'S%04d' % subject
        target_dir = os.path.join(movie_dir, fmt_dir)
        target_file = fmt_file % trial
        
        target = os.path.join(target_dir, target_file)
        
        if not os.path.exists(target.replace(movie_dir, output_directory).split('.')[0]+'_frames'):
            imgs = vc.video_file_to_frames(target, output_dir=output_directory, resize=resize_dims, suppress=False)
            print("-" * 78)
            imgs_captured.append(len(imgs))

        else:
            print("{} already exists, skipping".format(target.replace(movie_dir, output_directory)))

        i += 1

    print("[*] Extracted %d images from %d different video files" % (sum(imgs_captured), i))
