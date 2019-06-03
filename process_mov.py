#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
condense a sequence of K frames down to a single one
"""

from keras.models import load_model

from argparse import ArgumentParser
from itertools import count
from pathlib import Path
import numpy as np
import sys, os
import cv2

CLEAR = 80 * ' '

def arguments():
    parser = ArgumentParser(description='condense a sequence')
    parser.add_argument('mov',
                        help='.mov file to make predictions',
                        type=mov)

    parser.add_argument('model',
                        help='Model used to make predictions',
                        type=mod)
    parser.add_argument('--output-file','-o',
                        dest='output',
                        help='path to output file (default to stdout)',
                        type=csv,
                        default='default')
    
    return parser

def _ft(x, endings=[], exceptions=[], assert_exists=True):
    exists = os.path.exists(x) if assert_exists else True
    if (exists and os.path.splitext(x)[1].lower() in endings) or x in exceptions:
        return x
    raise IOError

def csv(x):
    return _ft(x, endings=['.csv'], assert_exists=False)

def mov(x):
    return _ft(x, endings=['.mov'])

def mod(x):
    loader = load_model
    return loader(_ft(x,endings=['.h5']))

def chunks(items,chunksize, reorg=lambda x: x):
    """
    Step through `items`, generating `chunksize` chunks of it. Throw out the last bit
    if there is `chunksize` does not evenly divide `len(item)`
    args:
        :items     (iterable)
        :chunksize (int > 0) - the size of the chunks
                :ints < 0 do not raise an error, the chunksize just defaults to 1
        :reorg     (callable) - function to apply to each sub chunk of the chunks  
    yields:
        :chunks of items of size `chunksize`
    raises:
        :ValueError for chunksize == 0
        :TypeError for items not iterable
        :AssertionError for noncallable reorgs
    usage:
    
    >>> mystr = "CHUNKME"
    >>> chunkifier = chunks(mystr, 3)
    >>> type(chunkifier)
        generator
    >>> [chunk for chunk in chunkifier]
       ["CHU","HUN","UNK","NKM","KME"] 

    >>> chunks_processed = chunks(mystr, 3, reorg=lambda x: reversed(x))
    >>> [cp for cp in chunks_processed]
       ["UHC","NUH","KNU","MKN","EMK"] 

    """
    chunksize = int(chunksize)
    if not chunksize:
        raise ValueError("Expected `chunksize`≠ 0")

    if not hasattr(items, '__iter__'):
        raise TypeError("`item` is not iterable")

    assert callable(reorg)
    
    chunkerator = iter(items)  # do generator chunks
    try:
        window = [next(chunkerator) for _ in range(chunksize)]
    except StopIteration:
        return

    while True:
        try:
            yield reorg(window)
            window = window[1:] + [next(chunkerator)]
        except StopIteration:
            return

def resize(frame, xdim=32, ydim=32):
    """
    downsample a frame from its current shape to (xdim, ydim)

    args:
        :frame (np.array)
        :xdim (int)
        :ydim (int)
    returns:
        :np.array 
    """

    xold, yold = frame.shape 
    assert xold >= xdim and yold >= ydim

    frame = cv2.resize(frame, (xdim,ydim))
    return frame

def greyscale(frame):
    """
    convert image to greyscale

    args:
        :frame (np.array)
    returns:
        :(np.array) - greyscaled frame
    """
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 

def normalize(frame):
    """
    normalize frame to all pixels in [0,1]

    args:
        :frame (np.array)
    returns:
        :(np.array)
    """
    return frame / 255.


def frame_generator(path, 
                    nosubsamp=True,
                    colorshift=greyscale,
                    preprocs=[resize, normalize],
                    rep_tol=5):
    """
    generate a sequence of frames until it is no longer possible
    from a movie file

    [--adapted from--]
    https://stackoverflow.com/questions/50661942/
       grayscale-image-using-opencv-from-numpy-array-failed/50663174
    [==============--]

    args:
        :path      (str) ------------------ path to the movie file
        :nosubsamp (bool) ----------------- take every (if True) or every other (if False)
        :colorshift (callable) ------------ redscale? greyscale?
        :preprocs  (iterable of callable) - preprocessing routines to apply 
        :rep_tol   (int) ------------------ number of times to repeat frame position
    returns:
        :single frame
    """

    assert hasattr(preprocs, '__iter__'),          "Preprocessing routines not iterable"
    assert all(callable(pre) for pre in preprocs + [colorshift]), "One or more preprocs not callable"
    preprocs = [colorshift] + list(preprocs)

    cap = cv2.VideoCapture(path)
    # open the video
    while not cap.isOpened():
        cap = cv2.VideoCapture(path)
        cv2.waitKey(1000)
        print("Waiting for the header...", file=sys.stderr)
    
    subsamp = lambda pos: True if nosubsamp else not (pos % 2)
    reps = 0
    last_position = -1
    while True:
        flag, frame = cap.read() # get frame, flag whether succesfful
        if flag:
            # The frame is ready and already captured
            frame_position = cap.get(cv2.CAP_PROP_POS_FRAMES)

            for preproc in preprocs:
                frame = preproc(frame)

            if subsamp(frame_position):
                yield frame
            
            else:
                continue
        else:
            # The next frame is not ready, so we try to read it again
            cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) - 1)
            print ("frame is not ready",file=sys.stderr)
            # It is better to wait for a while for the next frame to be ready
            cv2.waitKey(1000)

        #if frame_position >= cap.get(cv2.CAP_PROP_FRAME_COUNT):
        if reps >= rep_tol:
            # If the number of captured frames is equal to the total number of frames,
            # we stop
            return

        reps = reps + 1 if last_position == frame_position else 0
        last_position = frame_position

if __name__ == '__main__':
    args = arguments().parse_args()
    gen = frame_generator(args.mov, nosubsamp=False)
    # (batch_size, sequence length, x dim, y dim, color channel)
    sequence_length = args.model.layers[0].input_shape[1]
    input_shape     = tuple([1] + list(args.model.layers[0].input_shape[1:])) 
    reorganizer     = lambda x: np.reshape(np.stack(x), input_shape) 
    chunks = chunks(gen, sequence_length, reorg=reorganizer)
    
    with open(args.output, 'w') as f:
        print('frame_position','prediction',sep=',', file=f)
        for hpos, chunk in zip(count(0, step=2), chunks):
            prediction = np.squeeze(args.model.predict(chunk))
            print(hpos, f'{prediction:0.4f}', sep=',', file=f)
            print(f"\r{CLEAR}\rframe at head: {hpos}, pred: {prediction:0.4f}", 
                  end='',
                  flush=True)
    
    print()
    
    

