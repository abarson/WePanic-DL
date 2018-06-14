# WePanic-DL
* Heart rate and respiratory rate prediction with deep video regression

## Overview

There is promising evidence to suggest that pulse is detectable by smartphone cameras: one can lightly press their finger up to a smartphone camera with flash on, and produce a video that clearly shows their pulse. This project aims to leverage deep learning techniques to predict a patient’s heart rate and respiratory rate, given short videos of this nature. 

To run this project, you will need to collect samples yourself, or be granted permission to see the existing samples. Contact repo owners for more information. This project expects each clip to be around 30 seconds long, and for the directory structure of the video clips to be organized as follows:

- movie_data <br />
  - S0001/ <br />
    - Trial1.MOV <br />
    - Trial2.MOV <br />
  - S0002/ <br />
    - Trial1.MOV <br />
    - Trial2.MOV <br />
  ... <br />
  - S[N]/ <br />
    - Trial1.MOV <br />
    - Trial2.MOV <br />

Where each subject directory contains two video trials: one at resting heart rate, and one after 60 seconds following an intense workout. This project also assumes the existence a csv file, denoted `WEPANIC_CSVDATA`. The first one contains trial data for each subject in the following format:

| Subject  | Trial1_Heart_Rate| Trial2_Heart_Rate | Trial1_Respiratory_Rate| Trial2_Respiratory_Rate |
|:--------:|:----------------:|:-----------------:|:----------------------:|:-----------------------:|
| 1        | 60               | 120               | 30                     | 45                      |
| ..       | ..               | ..                | ..                     | ..                      |

## Install Requirements

This project assumes that Anaconda3 is installed on your home computer and your path variable
has been exported to include it.

First, install pip if you do not already have it.
```{r, engine='bash'}
conda install pip
```

Change directories to the repository.
```{r, engine='bash'}
cd WePanic-DL/
```
Install the required packages for data and video utilities. Run the following command:
```{r, engine='bash'}
pip install src/we_panic_utils/
```

## Preprocess Data

Once you have the above steps completed, you can begin preprocessing data. Run the following command:
```{r, engine='bash'}
./setup $WEPANIC_CSVDATA $movie_data
```

This does the following:
* convert each .mov file into a directory of frames and convert each .mov file into 30 fps
* resize all frames to 32x32x3


## TODO items (As of 6/14/18) (ideas and more):
1. Find a more state of the art architecture for action recognition
    * possibly reduce depth
    * could be too little filters to learn augmentation invariance
    * global average pooling + 1D-convolutions?

2. Fast fourier transform on data
    * converts to frequency

3. Increase number of validation sequences
    * report average over these sequences

4. Plotting submodule
5. Snapshot ensembles with cyclic learning rate
6. Loss function
    * square root of Euclidean distance between (observed_hr, observed_rr) (predicted_hr, predicted_rr)

7. Reduce or increase the sequence length


