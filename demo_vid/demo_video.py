"""
generate the demo video, this script is meant to be called
after using the predict_video.py script to generate HR, RR predictions
for each frame in a video
"""

import argparse
import cv2
import sys, os
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.transforms as tr
import numpy as np
from scipy import ndimage
import imageio
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser(description="generate the demo video")
    parser.add_argument("-v",
                        dest="video_frames",
                        help="actual frames (normal size)",
                        type=dirvalidate,
                        default="demo_vid/S0107/Trial2_frames")

    parser.add_argument("-n",
                        dest="nn_frames",
                        help="frame inputs to neural network (32 x 32)",
                        type=dirvalidate,
                        default="demo_vid/S0107/Trial2_frames_small")


    parser.add_argument("-c",
                        dest="preds",
                        help="the csv output of predict_video.py (predictions)",
                        type=csvvalidate,
                        default="demo_vid/predictions.csv")
    

    parser.add_argument("-o",
                        dest="frameloc",
                        help="output directory of the new frames",
                        type=str,
                        default="demo_vid/demo")
    return parser

def csvvalidate(c):
    """
    generate dataframe from csv path input

    args:
        :c (str) - path to alleged csv
    return:
        :(pd.DataFrame) - df from csv
    raises:
        :argparse.ArgumentError - if invalid input
    """

    c = str(c)

    try:
        return pd.read_csv(c)
    
    except Exception as e:
        raise argparse.ArgumentError("Something went wrong in resolving {} to a dataframe::-->>{}".format(c,e))

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


def add_subplot_axes(ax,rect,axisbg='w'):
    fig = plt.gcf()
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position  = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]  # <= Typo was here
    subax = fig.add_axes([x,y,width,height])
    x_labelsize = subax.get_xticklabels()[0].get_size()
    y_labelsize = subax.get_yticklabels()[0].get_size()
    x_labelsize *= rect[2]**0.5
    y_labelsize *= rect[3]**0.5
    subax.xaxis.set_tick_params(labelsize=x_labelsize)
    subax.yaxis.set_tick_params(labelsize=y_labelsize)

    return subax

def rgb2gray(f):
    img = Image.open(f).convert("L")
    return np.asarray(img)
    

if __name__ == "__main__":
    
    args = parse_args().parse_args()
    output = args.frameloc
    os.makedirs(output, exist_ok=True)

    vid_frames = map(plt.imread, sorted(glob(os.path.join(args.video_frames, "*.png"))))
    pred_frames  = map(rgb2gray, sorted(glob(os.path.join(args.nn_frames, "*.png"))))
    predictions = args.preds 

    rect = [0.2, 0.2,0.7, 0.7]
    imgs = []
    #with imageio.get_writer(os.path.join(args.frameloc,"demo.gif"), mode='I') as writer:
    for i, (vframe, pframe, predrow) in enumerate(zip(vid_frames, pred_frames, predictions.iterrows())):
        _, predrow = predrow
        lvid, wvid, _ = vframe.shape
        lprd, wprd = pframe.shape
        lrect, wrect = lvid/lprd, wvid/wprd
        hr_pred = round(predrow['hr_pred'],2) 
        rr_pred = round(predrow['rr_pred'],2)
        rr = predrow['rr']
        hr = predrow['hr']

        fig, ax = plt.subplots()
        subax = add_subplot_axes(ax, [0.66, 0.2, 0.33, 0.33])
        #subax_small = add_subplot_axes(ax, [0.2, 0.2, lrect, wrect])

        ax.set_xticks([])
        ax.set_yticks([])
        subax.set_xticks([])
        subax.set_yticks([])
        #subax_small.set_xticks([])
        #subax_small.set_yticks([])
    
        #subax_small.imshow(pframe, cmap="gray")
        ax.imshow(vframe)
        subax.imshow(pframe, cmap="gray")
        subax.text(0,-1,"NN Input")
        line = 30
        ax.text(2*line, 4*line, "HR")# fontsize=18)
        ax.text(2*line, 7*line, "RR")# fontsize=18)

        ax.text(6*line,60, "Actual")
        ax.text(6*line,120, hr)
        ax.text(6*line,210, rr)
        
        ax.text(13*line,60, "Prediction")
        ax.text(13*line,120, hr_pred)
        ax.text(13*line,210, rr_pred)

        ax.text(25*line,60, "$Actual-Prediction$")
        ax.text(30*line,120, round(hr - hr_pred,2))
        ax.text(30*line,210, round(rr - rr_pred,2))
        
        filename = os.path.join(args.frameloc, "DemoFrame__{:04d}.png".format(i))
        imgs.append(filename)
        plt.savefig(filename,transparent=True, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        print("\r generating images :--> [[{:3d}]]".format(i), end="")
    
    images = []
    for i, frame in enumerate(sorted(glob(os.path.join(args.frameloc,"*.png")))):
        print("\r writing image to gif :--> [[{:3d}]]".format(i), end="")
        images.append(imageio.imread(frame))

    imageio.mimsave(os.path.join(args.frameloc,"demo.gif"), images)


