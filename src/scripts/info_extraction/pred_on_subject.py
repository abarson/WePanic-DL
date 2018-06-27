import os
import numpy as np
from keras import models

from we_panic_utils.nn import processing

import random
def choose_random_frames(subj_dir):
    chosen_frames = []
    all_frames = os.listdir(subj_dir)
    starting_point = random.randint(0, len(all_frames)-60-1)
    print(starting_point)
    print(all_frames[starting_point:starting_point+60])
    return [os.path.join(subj_dir, frame) for frame in all_frames[starting_point:starting_point+60]]

def prepare_frames(frames):
    seq = processing.build_image_sequence(frames, (32, 32, 3), greyscale_on=True)
    return seq

def main():
    SUB = 'S0105'
    TRI = 'Trial2_frames'
    model = models.load_model('run_history/p_32_32_grey/models/CV_2_CNN_3D.h5')

    complete_vid = os.path.join('picky_32_32', SUB, TRI)

    frame_seqs = np.array([prepare_frames(choose_random_frames(complete_vid)) for _ in range(5)])
    print('Predicting...') 
    print(model.predict(frame_seqs))

main()
