"""
A module for processing frames as a sequence
"""

from .data_load import buckets
import threading 
import os
import random
#random.seed(7)
import numpy as np
import keras
from keras.preprocessing.image import load_img, img_to_array
from keras_preprocessing.image import apply_affine_transform, apply_brightness_shift, transform_matrix_offset_center, random_brightness
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
        channel_axis=channel_axis : int - index of channel ax on each tensor
        fill_mode=fill_mode : string - points outside the boundaries of input are filled
                             according to the given input mode, one of 
                             {'nearest', 'constant', 'reflect', 'wrap'}

        cval=cval : float - constant value used for fill_mode=fill_mode constant
    
    returns:
        rotated : the rotated sequence of tensors
    """

    theta = np.deg2rad(np.random.uniform(-rotation_range, rotation_range))
    return [apply_affine_transform(x, theta=theta, channel_axis=channel_axis, fill_mode=fill_mode, cval=cval) for x in seq] 
        

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
        channel_axis=channel_axis : int - the index of channel ax on each tensor
        fill_mode=fill_mode : string - points outside the boundaries of input are filled
                             according to the given input mode, one of 
                             {'nearest', 'constant', 'reflect', 'wrap'}

        cval=cval : float - the constant value used for fill_mode=fill_mode constant 
    """

    h, w = seq[0].shape[row_axis], seq[0].shape[col_axis]
    tx = np.random.uniform(-height_shift_range, height_shift_range) * h
    ty = np.random.uniform(-width_shift_range, width_shift_range) * w

    return [apply_affine_transform(x, tx=tx, ty=ty, channel_axis=channel_axis, fill_mode=fill_mode, cval=cval) for x in seq]



def random_sequence_shear(seq, shear_range, row_axis=0, col_axis=1, channel_axis=2,
                          fill_mode='nearest', cval=0):
    """
    apply a random shear to an entire sequence of frames

    args:
        seq : list - the list of 3D input tensors
        shear_range : float - the amount of shear to apply
        row_axis : int - the index of row axis on each tensor
        col_axis : int - the index of col axis on each tensor
        channel_axis=channel_axis : int - the index of channel ax on each tensor
        fill_mode=fill_mode : string - points outside the boundaries of input are filled
                             according to the given input mode, one of 
                             {'nearest', 'constant', 'reflect', 'wrap'}

        cval=cval : float - the constant value used for fill_mode=fill_mode constant 
    
    returns:
        the sequence of sheared frames
    """

    shear = np.deg2rad(np.random.uniform(-shear_range, shear_range))
    h, w = seq[0].shape[row_axis], seq[0].shape[col_axis]

    return [apply_affine_transform(x, shear=shear, channel_axis=channel_axis, fill_mode=fill_mode, cval=cval) for x in seq]


def random_sequence_zoom(seq, zoom_range, row_axis=0, col_axis=1, channel_axis=2,
                         fill_mode='nearest', cval=0): 
    """
    apply a random zoom on an entire sequence of frames

    args:
        seq : list - the list of 3D input tensors
        zoom_range : center of range to zoom/unzoom
        row_axis : int - the index of row axis on each tensor
        col_axis : int - the index of col axis on each tensor
        channel_axis=channel_axis : int - the index of channel ax on each tensor
        fill_mode=fill_mode : string - points outside the boundaries of input are filled
                             according to the given input mode, one of 
                             {'nearest', 'constant', 'reflect', 'wrap'}

        cval=cval : float - the constant value used for fill_mode=fill_mode constant 
    
    returns:
        the sequence of zoomed frames
    """

    zlower, zupper = 1 - zoom_range, 1 + zoom_range
    
    if zlower == 1 and zupper == 1:
        zx, zy = 1, 1
    
    else:
        zx, zy = np.random.uniform(zlower, zupper, 2)

    h, w = seq[0].shape[row_axis], seq[0].shape[col_axis]

    return [apply_affine_transform(x, zx=zx, zy=zy, channel_axis=channel_axis, fill_mode=fill_mode, cval=cval) for x in seq]


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

def random_local_brightness_shift(seq, brightness_range):
    """
    perturb local brightness -- the brightness of each image -- differently
    """
    return [random_brightness(frame,brightness_range) for frame in seq]


def random_global_brightness_shift(seq, brightness_range):
    """
    perturb sequence wide brightness -- the brightness of the entire sequence
    """
    brightness_lo, brightness_hi = brightness_range
    u = np.random.uniform(brightness_lo, brightness_hi)

    return [apply_brightness_shift(x, u) for x in seq]
         

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
                 features,
                 input_shape,
                 rotation_range=0,
                 width_shift_range=0.,
                 height_shift_range=0.,
                 shear_range=0.,
                 zoom_range=0.,
                 horizontal_flip=False,
                 vertical_flip=False,
                 brightness_range_local=None,
                 brightness_range_global=None,
                 batch_size=4,
                 sequence_length=60,
                 greyscale_on=False,
                 redscale_on=False,
                 num_val_clips=10):
        
        self.features = features
        self.input_shape = input_shape
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
        self.brightness_range_local = brightness_range_local
        self.brightness_range_global = brightness_range_global
        
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
        paths = list(test_df['FRAME_PTH'])
        feats = [list(test_df[feat]) for feat in self.features]

        i_s = [i for i in range(len(test_df))]
        while True:
            X, y = [], []

            i = i_s.pop(0)

            current_path = paths[i]
            current_feats = [feats[n][i] for n in range(len(feats))]
            frame_dir = sorted(os.listdir(current_path))
            
            for _ in range(self.num_val_clips):
                start = random.randint(0, len(frame_dir)-self.sequence_length*2)
                frames = frame_dir[start:start+self.sequence_length*2:2]
                frames = [os.path.join(current_path, frame) for frame in frames]
                X.append(build_image_sequence(frames, input_shape=self.input_shape, 
                    greyscale_on=self.greyscale_on, redscale_on=self.redscale_on))
                y.append(current_feats)

            if not i_s:
                i_s = [i for i in range(len(test_df))]
            
            yield np.array(X), np.array(y)

    def train_generator(self, train_df):
        while True:
            X, y = [], []

            for _ in range(self.batch_size):
                
                random_index = random.randint(0, len(train_df)-1)
                path = list(train_df['FRAME_PTH'])[random_index]
                current_feats = [float(list(train_df[feat])[random_index]) for feat in self.features]

                frame_dir = sorted(os.listdir(path))
                start = random.randint(0, len(frame_dir)-self.sequence_length*2)
                frames = frame_dir[start:start+self.sequence_length*2:2]
                frames = [os.path.join(path, frame) for frame in frames]

                sequence = build_image_sequence(frames, input_shape=self.input_shape,
                        greyscale_on=self.greyscale_on, redscale_on=self.redscale_on)
                
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
                
                if self.brightness_range_local is not None:
                   sequence = random_local_brightness_shift(sequence, self.brightness_range_local)

                if self.brightness_range_global is not None:
                   sequence = random_global_brightness_shift(sequence, self.brightness_range_global) 

                X.append(sequence)
                y.append(current_feats)
            
            yield np.array(X), np.array(y)
