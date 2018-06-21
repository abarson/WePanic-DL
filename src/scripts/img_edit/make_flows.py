from we_panic_utils.basic_utils.video_core import write_optical_flow
import os
import sys


path = 'rsz32/'
subjects = os.listdir(path)
subjects = [os.path.join(path, subject) for subject in subjects]

for subject in subjects:
    print("\n-=-=-=-=-=-=-=--- %s  ---=-=-=-=-=-=-=-=-" % subject)
    trials = os.listdir(subject)
    trials = [os.path.join(subject, trial) for trial in trials]
    for trial in trials:
        pth = trial
        write_optical_flow(pth, 2)


