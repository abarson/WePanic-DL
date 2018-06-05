import os
import cv2

data_dir = "data"
for subject in os.listdir(data_dir):
    trial_path = os.path.join(data_dir, subject)
    for trial in os.listdir(trial_path):
        movie = os.path.join(trial_path, trial)
        vidcap = cv2.VideoCapture(movie)
        FPS = int(round(vidcap.get(cv2.CAP_PROP_FPS)))
        print(movie, ': ', FPS, sep='')
