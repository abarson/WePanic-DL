"""
Iterate through a set of frames, predict the heart rate and respiratory rate of each frame, save to csv
"""

import sys, os
import argparse
from glob import glob
import pandas as pd

from we_panic_utils.nn.processing import build_image_sequence
from keras.models import load_model

HR_PREDICTOR = "TOPMODS/hr_greyscale/greyscale__Fold_2__Loss_44.177.h5" 
RR_PREDICTOR = "TOPMODS/rr_greyscale/greyscale__Fold_2__Loss_94.278.h5"
SEQLEN = 90

def parse_args():
    parser = argparse.ArgumentParser(description="generate the demo video")
    parser.add_argument("-v",
                        dest="pred_frames",
                        help="path to resized frames for prediction",
                        type=dirvalidate,
                        default="demo_vid/S0107/Trial2_frames_small")
    
    parser.add_argument("-hr",
                        help="the actual heart rate of the passed video",
                        type=int,
                        default=102)

    parser.add_argument("-rr",
                        help="the actual respiratory rate of the passed video",
                        type=int,
                        default=20)

    return parser

def dirvalidate(d):
    """
    verify that a directory exists

    args:
        :d (str) - the directory to validate
    returns:
        :(str) validated directory
    raises:
        :argparse.ArgumentError if validation fails
    """
    d = str(d)
    if not os.path.isdir(d):
        raise argparse.ArgumentError("Cannot find {}".format(d))

    return d


if __name__ == "__main__":
    args = parse_args().parse_args()
    pred_frames  = sorted(glob(os.path.join(args.pred_frames, "*.png")))
    HR_PREDICTOR, RR_PREDICTOR = map(load_model, (HR_PREDICTOR, RR_PREDICTOR)) 
    
    prediction_df = pd.DataFrame(columns=["frame", "hr_pred","hr","rr_pred","rr"])
    
    for i, f in enumerate(pred_frames[:-SEQLEN*2]):
        frames = pred_frames[i:i+SEQLEN*2:2]  
        frames = build_image_sequence(frames, greyscale_on=True)

        phr = HR_PREDICTOR.predict(frames)
        prr = RR_PREDICTOR.predict(frames)

        print(phr, prr)
        break
