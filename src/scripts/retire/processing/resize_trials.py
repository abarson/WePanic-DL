import sys
import os
import argparse

import we_panic_utils.basic_utils.video_core as vc
import we_panic_utils.basic_utils.basics as base

def usage(with_help=True): 
    print("[Usage]: %s <partitions_dir> <resized_out>" % sys.argv[0])
    if with_help:
        print("         %s HELP|help|h for more info" % sys.argv[0])
    sys.exit()


def help_msg():
    print("[%s]" % sys.argv[0])
    print("-"*len("[%s]" % sys.argv[0]))

    print("\t| script meant to take a directory of")
    print("\t| frame partitions and resize them")
    usage(with_help=False)


def extract_subject_name(subj_dir):
    subj_dir = subj_dir[1:]
    return str(int(subj_dir))


def parse_input():
    parser = argparse.ArgumentParser("resize frame directories")
    parser.add_argument("frame_dir",
                        help="directory to resize",
                        type=str)

    parser.add_argument("--output_dir",
                        help="resized directory name",
                        type=str,
                        default="rsz")

    parser.add_argument("--xdim", "-x",
                        help="x dimension to resize",
                        type=int,
                        default=100)

    parser.add_argument("--ydim", "-y",
                        help="y dimension to resize",
                        type=int,
                        default=100)

    return parser


if __name__ == "__main__":
    args = parse_input().parse_args()

    if not os.path.isdir(args.frame_dir):
        raise IOError("Error: path {} is not a directory".format(args.frame_dir))

    if args.xdim <= 0:
        raise ValueError("Error: xdim should be > 0, got {}".format(args.xdim))

    if args.ydim <= 0:
        raise ValueError("Error: ydim should be > 0, got {}".format(args.ydim))

    base.check_exists_create_if_not(args.output_dir)

    for subject in os.listdir(args.frame_dir):
        
        trials = [t for t in os.listdir(os.path.join(args.frame_dir, subject))]

        trial_paths = [os.path.join(args.frame_dir, subject, t) for t in trials]
        
        for trial, trial_path in zip(trials, trial_paths):
            if not os.path.isdir(trial_path):
                continue
            
            os.makedirs(os.path.join(args.output_dir, subject, trial), exist_ok=True)

            # for frame in os.listdir(trial_path):
            frame_path = trial_path
                
            output_path = os.path.join(args.output_dir, subject, trial)
            vc.resize_frame_dir(frame_path, output_path, width=args.xdim, height=args.ydim)
    
    print("Done.")

