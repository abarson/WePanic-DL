"""
A module for processing frames as a sequence
"""

from .data_load import buckets
import threading 
import os
import random
#random.seed(7)
import numpy as np

from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.image import apply_transform, transform_matrix_offset_center
import keras.backend as K
from skimage.color import rgb2grey
from PIL import ImageEnhance
from PIL import Image as pil_image
import scipy as sp
from sklearn.preprocessing import MinMaxScaler

class threadsafe_iterator:
    """
    A class for threadsafe iterator
    """
    def __init__(self, iterator):
        self.iterator = iterator
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.iterator)


def threadsafe_generator(func):
    """ decorator """

    def gen(*a, **kw):
        return threadsafe_iterator(func(*a, **kw))

    return gen


def random_sequence_rotation(seq, rotation_range, row_axis=0, col_axis=1, channel_axis=2,
                             fill_mode='nearest', cval=0):
    """
    apply a rotation to an entire sequence of frames
    
    args:
        seq : list (or array-like) - list of 3D input tensors
        rotation_range : int - amount to rotate in degrees
        row_axis : int - index of row axis on each tensor
        col_axis : int - index of cols axis on each tensor
        channel_axis : int - index of channel ax on each tensor
        fill_mode : string - points outside the boundaries of input are filled
                             according to the given input mode, one of 
                             {'nearest', 'constant', 'reflect', 'wrap'}

        cval : float - constant value used for fill_mode constant
    
    returns:
        rotated : the rotated sequence of tensors
    """

    theta = np.deg2rad(np.random.uniform(-rotation_range, rotation_range))
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                               [np.sin(theta), np.cos(theta), 0],
                               [0, 0, 1]])

    h, w = seq[0].shape[row_axis], seq[0].shape[col_axis]
    transform_matrix = transform_matrix_offset_center(rotation_matrix, h, w) 
    return [apply_transform(x, transform_matrix, channel_axis, fill_mode, cval) for x in seq] 
        

def random_sequence_shift(seq, height_shift_range, width_shift_range, row_axis=0, col_axis=1, channel_axis=2, 
                          fill_mode="nearest", cval=0):
    """
    apply a height/width shift to an entire sequence of frames
    
    args:
        seq : list - the list of 3D input tensors
        height_shift_range : float - amount to shift height (fraction of total)
        width_shift_range : float - amount to shift width (fraction of total)
        row_axis : int - the index of row axis on each tensor
        col_axis : int - the index of col axis on each tensor
        channel_axis : int - the index of channel ax on each tensor
        fill_mode : string - points outside the boundaries of input are filled
                             according to the given input mode, one of 
                             {'nearest', 'constant', 'reflect', 'wrap'}

        cval : float - the constant value used for fill_mode constant 
    """

    h, w = seq[0].shape[row_axis], seq[0].shape[col_axis]
    tx = np.random.uniform(-height_shift_range, height_shift_range) * h
    ty = np.random.uniform(-width_shift_range, width_shift_range) * w

    translation_matrix = np.array([[1, 0, tx],
                                  [0, 1, ty],
                                  [0, 0, 1]])

    transform_matrix = translation_matrix  # no offset necessary
    
    return [apply_transform(x, transform_matrix, channel_axis, fill_mode, cval) for x in seq]


def random_sequence_shear(seq, shear_range, row_axis=0, col_axis=1, channel_axis=2,
                          fill_mode='nearest', cval=0):
    """
    apply a random shear to an entire sequence of frames

    args:
        seq : list - the list of 3D input tensors
        shear_range : float - the amount of shear to apply
        row_axis : int - the index of row axis on each tensor
        col_axis : int - the index of col axis on each tensor
        channel_axis : int - the index of channel ax on each tensor
        fill_mode : string - points outside the boundaries of input are filled
                             according to the given input mode, one of 
                             {'nearest', 'constant', 'reflect', 'wrap'}

        cval : float - the constant value used for fill_mode constant 
    
    returns:
        the sequence of sheared frames
    """

    shear = np.deg2rad(np.random.uniform(-shear_range, shear_range))
    shear_matrix = np.array([[1, -np.sin(shear), 0],
                             [0, np.cos(shear), 0],
                             [0, 0, 1]])

    h, w = seq[0].shape[row_axis], seq[0].shape[col_axis]
    transform_matrix = transform_matrix_offset_center(shear_matrix, h, w)
    
    return [apply_transform(x, transform_matrix, channel_axis, fill_mode, cval) for x in seq]


def random_sequence_zoom(seq, zoom_range, row_axis=0, col_axis=1, channel_axis=1,
                         fill_mode='nearest', cval=0): 
    """
    apply a random zoom on an entire sequence of frames

    args:
        seq : list - the list of 3D input tensors
        zoom_range : center of range to zoom/unzoom
        row_axis : int - the index of row axis on each tensor
        col_axis : int - the index of col axis on each tensor
        channel_axis : int - the index of channel ax on each tensor
        fill_mode : string - points outside the boundaries of input are filled
                             according to the given input mode, one of 
                             {'nearest', 'constant', 'reflect', 'wrap'}

        cval : float - the constant value used for fill_mode constant 
    
    returns:
        the sequence of zoomed frames
    """

    zlower, zupper = 1 - zoom_range, 1 + zoom_range
    
    if zlower == 1 and zupper == 1:
        zx, zy = 1, 1
    
    else:
        zx, zy = np.random.uniform(zlower, zupper, 2)

    zoom_matrix = np.array([[zx, 0, 0],
                            [0, zy, 0],
                            [0, 0, 1]])

    h, w = seq[0].shape[row_axis], seq[0].shape[col_axis]

    transform_matrix = transform_matrix_offset_center(zoom_matrix, h, w)

    return [apply_transform(x, transform_matrix, channel_axis, fill_mode, cval) for x in seq]


def sequence_flip_axis(seq, axis):
    
    """
    flip a sequence of images to a different axis (vertical, horizontal) 
    
    args:
        seq : list - the list of 3D image tensors
        axis : axis on which to rotate

    returns
        rotated image sequence
    """

    seq = [np.asarray(x).swapaxes(axis, 0) for x in seq]
    seq = [x[::-1, ...] for x in seq]
    seq = [x.swapaxes(0, axis) for x in seq]

    return seq


def get_sample_frames(sample):
    """
    return the sorted list of absolute image paths for this sample
    """

    contents = os.listdir(sample)
    max_frame = len(contents)
    
    filenames = [os.path.join(sample, "frame%d.png" % i) for i in range(max_frame)]

    return filenames
  

def build_image_sequence(frames, input_shape=(32, 32, 3), greyscale_on=False, redscale_on=False):
    """
    return a list of images from filenames
    """
    return [process_img(frame, input_shape, greyscale_on=greyscale_on, redscale_on=redscale_on) for frame in frames]


def just_greyscale(arr):
    x = (arr / 255.).astype(np.float32)
    x = (0.21 * x[:, :, :1]) + (0.72 * x[:, :, 1:2]) + (0.07 * x[:, :, -1:])
    return x

def process_img(frame, input_shape, greyscale_on=False, redscale_on=False):
    """
    load up an image as a numpy array

    args:
        frame : str - image path
        input_shape : tuple (h, w, nchannels)

    returns
        x : the loaded image
    """
    h_, w_, _ = input_shape
    image = load_img(frame, target_size=(h_, w_))
    img_arr = img_to_array(image)
    
    x = (img_arr / 255.).astype(np.float32)

    if greyscale_on:
        x = (0.21 * x[:, :, :1]) + (0.72 * x[:, :, 1:2]) + (0.07 * x[:, :, -1:])
    
    if redscale_on:
        x = x[:,:,-1]
        x = np.reshape(x, x.shape + tuple([1])) 
    
    return x


def DFT_img_sequence(imgs, x_dim, y_dim):
    imgs = np.array(imgs)
    for x in range(x_dim):
        for y in range(y_dim):
            pix = imgs[::,x,y]
            imgs[::,x,y] = sp.fft(pix).real
    return imgs

class FrameProcessor:
    """
    the one stop shop object for data frame sequence augmentation and generation 
    
    usage example:
        train_proc = FrameProcessor(rotation_range=0.5, zoom_range=0.2)
        train_gen  = train_proc.frame_generator(filtered_training_paths, 'train')

    args:
        rotation_range : int - degree range for random rotations
        width_shift_Range : float - fraction of total width for horizontal shifts
        height_shift_range : float - fraction of total height for vertical shifts
        shear_range: Float. Shear Intensity (Shear angle in counter-clockwise direction in degrees)
        zoom_range : float - range for random zoom, lower_bd = 1 - zoom_range, upper_bd = 1 + zoom_range 
        horizontal_flip : bool - whether or not to flip horizontall with prob 0.5
        vertical_flip : bool - whether or not to flip vertically with prob 0.5
        greyscale_on : bool - whether or not the frames are to be converted to greyscale
        test_selections : int - how many random selections to sample from the test set 
        sequence_length : int - the number of frames to be yielded by the generator
        batch_size : int - the batch size
    """
    def __init__(self,
                 rotation_range=0,
                 width_shift_range=0.,
                 height_shift_range=0.,
                 shear_range=0.,
                 zoom_range=0.,
                 horizontal_flip=False,
                 vertical_flip=False,
                 batch_size=4,
                 sequence_length=60,
                 greyscale_on=False,
                 redscale_on=False,
                 num_val_clips=10):

        self.rotation_range = rotation_range
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range
        self.shear_range = shear_range
        self.zoom_range = zoom_range
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.greyscale_on = greyscale_on
        self.redscale_on = redscale_on
        self.num_val_clips = num_val_clips
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.test_iter = 0
        
        assert type(self.rotation_range) == int, "rotation_range should be integer valued"

        assert type(self.width_shift_range) == float, "width_shift_range should be a float"
        assert self.width_shift_range <= 1. and self.width_shift_range >= 0., "width_shift_range should be in [0, 1], got %f" % self.width_shift_range 
        
        assert type(self.height_shift_range) == float, "height_shift_range should be a float"
        assert self.height_shift_range <= 1. and self.height_shift_range >= 0., "width_shift_range should be in [0, 1], got %f" % self.height_shift_range

        assert type(self.zoom_range) == float, "zoom_range should be a float"
        
        assert type(self.horizontal_flip) == bool, "horizontal_flip should be a boolean"
        assert type(self.vertical_flip) == bool, "vertical_flip should be a boolean"
        
        assert type(self.sequence_length) == int, "sequence_length should be an integer"
        assert self.sequence_length > 0, "sequence_length should be > 0"

    @threadsafe_generator    
    def test_generator(self, test_df):
        #print(self.num_val_clips)
        paths, hr, rr = list(test_df["FRAME_PTH"]), list(test_df["HEART_RATE_BPM"]), list(test_df["RESP_RATE_BR_PM"])
        i = 0
        while True:
            X, y = [], []
            current_path = paths[i]
            current_hr = float(hr[i])
            current_rr = float(rr[i]) 
            frame_dir = sorted(os.listdir(current_path))
            
            for _ in range(self.num_val_clips):
                start = random.randint(0, len(frame_dir)-self.sequence_length)
                frames = frame_dir[start:start+self.sequence_length]
                frames = [os.path.join(current_path, frame) for frame in frames]
                X.append(build_image_sequence(frames, greyscale_on=self.greyscale_on, redscale_on=self.redscale_on))
                y.append([current_hr, current_rr])

            i+=1
            if i == len(test_df):
                i = 0
            
            #print(np.array(X).shape, np.array(y).shape, " (test generator)")
            yield np.array(X), np.array(y)

    def train_generator(self, train_df):
        while True:
            X, y = [], []

            for _ in range(self.batch_size):
                
                random_index = random.randint(0, len(train_df)-1)
                path = list(train_df['FRAME_PTH'])[random_index]
                hr = float(list(train_df['HEART_RATE_BPM'])[random_index])
                rr = float(list(train_df['RESP_RATE_BR_PM'])[random_index]) 

                frame_dir = sorted(os.listdir(path))
                start = random.randint(0, len(frame_dir)-self.sequence_length)
                frames = frame_dir[start:start+self.sequence_length]
                frames = [os.path.join(path, frame) for frame in frames]

                sequence = build_image_sequence(frames, greyscale_on=self.greyscale_on, redscale_on=self.redscale_on)
                
                # now we want to apply the augmentation
                if self.rotation_range > 0.0:
                    sequence = random_sequence_rotation(sequence, self.rotation_range)

                if self.width_shift_range > 0.0 or self.height_shift_range > 0.0:
                    sequence = random_sequence_shift(sequence, self.width_shift_range, self.height_shift_range)
                
                if self.shear_range > 0.0:
                    sequence = random_sequence_shear(sequence, self.shear_range)

                if self.zoom_range > 0.0:
                    sequence = random_sequence_zoom(sequence, self.zoom_range)
                
                if self.vertical_flip:
                    # with probability 0.5, flip vertical axis
                    coin_flip = np.random.random_sample() > 0.5
                    if coin_flip:
                        sequence = sequence_flip_axis(sequence, 1)   # flip on the row axis
                
                if self.horizontal_flip:
                    # with probability 0.5, flip horizontal axis (cols)
                    coin_flip - np.random.random_sample() > 0.5

                    if coin_flip:
                        sequence = sequence_flip_axis(sequence, 2)   # flip on the column axis

                X.append(sequence)
                y.append([hr, rr])
            
            #print(np.array(X).shape, np.array(y).shape, " (train generator)")
            yield np.array(X), np.array(y)
