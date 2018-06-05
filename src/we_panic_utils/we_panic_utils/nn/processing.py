"""
A module for processing frames as a sequence
"""

from .data_load import buckets
from ..basic_utils.video_core import optical_flow_of_first_and_rest
import threading 
import os
import random
random.seed(7)
import numpy as np

from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.image import apply_transform, transform_matrix_offset_center
import keras.backend as K
from skimage.color import rgb2grey
from PIL import ImageEnhance
from PIL import Image as pil_image

from sklearn.preprocessing import MinMaxScaler

class threadsafe_iterator:
    """
    A class for threadsafe iteratio
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
  

def build_image_sequence(frames, input_shape=(32, 32, 3), greyscale_on=False):
    """
    return a list of images from filenames
    """
    return [process_img(frame, input_shape, greyscale_on=greyscale_on) for frame in frames]


def just_greyscale(arr):
    x = (arr / 255.).astype(np.float32)
    x = (0.21 * x[:, :, :1]) + (0.72 * x[:, :, 1:2]) + (0.07 * x[:, :, -1:])
    return x

def process_img(frame, input_shape, greyscale_on=False):
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
    return x


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
        zoom_range : float - range for random zoom, lower_bd = 1 - zoom_range, upper_bd = 1 + zoom_range 
        vertical_flip : bool - whether or not to flip vertically with prob 0.5
        horizontal_flip : bool - whether or not to flip horizontall with prob 0.5
        batch_size : int - the batch size
        shear_range: Float. Shear Intensity (Shear angle in counter-clockwise direction in degrees)
    """
    def __init__(self,
                 scaler=None,
                 rotation_range=0,
                 width_shift_range=0.,
                 height_shift_range=0.,
                 shear_range=0.,
                 zoom_range=0.,
                 horizontal_flip=False,
                 vertical_flip=False,
                 batch_size=4,
                 sequence_length=60,
                 greyscale_on=False):
        self.scaler = scaler
        self.rotation_range = rotation_range
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range
        self.shear_range = shear_range
        self.zoom_range = zoom_range
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.greyscale_on = greyscale_on 
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
    def testing_generator(self, paths2labels, generator_type):
        print("[*] creating a %s testing generator with %d samples" % (generator_type, len(paths2labels)))
        sequence_paths = [pth for pth in paths2labels]
        current = 0
        while 1:
            X, y = [], []
            selected_path = sequence_paths[current]
            y = paths2labels[selected_path]

            y = [y]
            #y = [paths2labels[pth] for pth in selected_paths]
            frames = get_sample_frames(selected_path)
            sequence = build_image_sequence(frames, greyscale_on=self.greyscale_on)
            X.append(sequence)
            print(selected_path)
            
            current += 1
            
            # print(selected_path) 
            if current == len(sequence_paths):   
                 current = 0
            
            yield np.array(X), np.array(y)

    
    @threadsafe_generator
    def testing_generator_v2(self, paths2labels):
        """
        generate sequences of self.batch_size clips length self.sequence_length
        but don't augment the data
        """
    
        sequence_paths = [path for path in paths2labels]
        while True:
            X, y = [], []
            
            selected_paths = random.sample(sequence_paths, self.batch_size)
            content_sizes = [len(os.listdir(p)) for p in selected_paths]
            fps = []

            for sz in content_sizes:
                if sz > 1300:
                    fps.append(60)

                else:
                    fps.append(30)
            
            for path, sz, fps_ in zip(selected_paths, content_sizes, fps):
                
                heart_rate, resp_rate = paths2labels[path]
                
                frames = os.listdir(path)
                frames = sorted(frames)
                frames = [os.path.join(path, frame) for frame in frames]
                
                selected_frames = None

                if fps_ > 30:
                    seq_begin = random.randint(0, sz - 2 * self.sequence_length)
                    selected_frames = frames[seq_begin:seq_begin + self.sequence_length:2]

                else:
                    seq_begin = random.randint(0, sz - self.sequence_length) 
                    selected_frames = frames[seq_begin:seq_begin + self.sequence_length]
                
                sequence = build_image_sequence(selected_frames, greyscale_on=self.greyscale_on)

                X.append(sequence)
                y.append(heart_rate, resp_rate)

            yield np.array(X), np.array(y)

    @threadsafe_generator    
    def testing_generator_v3(self, test_df):
        paths, hr = list(test_df["Path"]), list(test_df["Heart Rate"])
        i = 0
        while True:
            X, y = [], []
            current_path = paths[i]
            current_hr = hr[i]
            
            if self.scaler:
                current_hr = self.scaler.transform(current_hr)[0][0]
            
            frame_dir = sorted(os.listdir(current_path))
            #hard-code to 2 for now, because there are a lot of samples
            for _ in range(2):
                start = random.randint(0, len(frame_dir)-self.sequence_length)
                frames = frame_dir[start:start+self.sequence_length]
                frames = [os.path.join(current_path, frame) for frame in frames]
                X.append(build_image_sequence(frames, greyscale_on=self.greyscale_on))
                y.append(current_hr)

            i+=1
            if i == len(test_df):
                i = 0
            
            #print(np.array(X).shape, np.array(y).shape, " (test generator)")
            yield np.array(X), np.array(y)


    @threadsafe_generator    
    def test_generator_alt_optical_flow(self, test_df):
        paths, hr = list(test_df["Path"]), list(test_df["Heart Rate"])
        i = 0
        self.greyscale_on = True

        while True:
            X, y = [], []
            current_path = paths[i]
            current_hr = hr[i]
            if self.scaler:
                current_hr = self.scaler.transform(current_hr)[0][0]
            #hard-code to 2 for now, because there are a lot of samples
            for _ in range(2):

                all_frames = sorted(os.listdir(current_path))
                start = random.randint(0, len(all_frames)-self.sequence_length-1)
                frames = all_frames[start:start+self.sequence_length+1]
                frames = [os.path.join(current_path, frame) for frame in frames]
                
                flows_x, flows_y = optical_flow_of_first_and_rest(frames)
                sequence_hor = np.array(flows_x)
                sequence_ver = np.array(flows_y)

                sequence_hor = np.expand_dims(np.array(flows_x), axis=3)
                sequence_ver = np.expand_dims(np.array(flows_y), axis=3)
                
                X.append(np.concatenate([sequence_hor, sequence_ver], axis=3))
                y.append(current_hr)
                
            i+=1
            if i == len(test_df):
                i = 0
            self.test_iter = i
 
            #print(np.array(X).shape, np.array(y).shape, " (test generator)")
            yield np.array(X), np.array(y)

    @threadsafe_generator    
    def train_generator_alt_optical_flow(self, train_df):
        bucket_list = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9]

        while True:
            X, y = [], []
            for _ in range(self.batch_size):
                
                rand_bucket = bucket_list[random.randint(0, len(bucket_list)-1)]
                df = train_df[buckets(train_df, rand_bucket)]
                rand_subj_index = random.randint(0, len(df)-1)
                rand_subj_df = df[rand_subj_index:rand_subj_index+1]

                path = list(rand_subj_df["Path"])[0]
                hr = list(rand_subj_df["Heart Rate"])[0]

                if self.scaler:
                    hr = self.scaler.transform(hr)[0][0]
                
                all_frames = sorted(os.listdir(path))
                start = random.randint(0, len(all_frames)-self.sequence_length-1)
                frames = all_frames[start:start+self.sequence_length+1]
                frames = [os.path.join(path, frame) for frame in frames]
                flows_x, flows_y = optical_flow_of_first_and_rest(frames)
                sequence_hor = np.expand_dims(np.array(flows_x), axis=3)
                sequence_ver = np.expand_dims(np.array(flows_y), axis=3)
                            
                # now we want to apply the augmentation
                if self.rotation_range > 0.0:
                    sequence_hor = random_sequence_rotation(sequence_hor, self.rotation_range)
                    sequence_ver = random_sequence_rotation(sequence_ver, self.rotation_range)

                if self.width_shift_range > 0.0 or self.height_shift_range > 0.0:
                    sequence_hor = random_sequence_shift(sequence_hor, self.width_shift_range, self.height_shift_range)
                    sequence_ver = random_sequence_shift(sequence_ver, self.width_shift_range, self.height_shift_range)

                if self.shear_range > 0.0:
                    sequence_hor = random_sequence_shear(sequence_hor, self.shear_range)
                    sequence_ver = random_sequence_shear(sequence_ver, self.shear_range)

                if self.zoom_range > 0.0:
                    sequence_hor = random_sequence_zoom(sequence_hor, self.zoom_range)
                    sequence_ver = random_sequence_zoom(sequence_ver, self.zoom_range)

                if self.vertical_flip:
                    # with probability 0.5, flip vertical axis
                    coin_flip = np.random.random_sample() > 0.5
                    if coin_flip:
                        sequence_hor = sequence_flip_axis(sequence_hor, 1)   # flip on the row axis
                        sequence_ver = sequence_flip_axis(sequence_ver, 1)   # flip on the row axis

                if self.horizontal_flip:
                    # with probability 0.5, flip horizontal axis (cols)
                    coin_flip - np.random.random_sample() > 0.5

                    if coin_flip:
                        sequence_hor = sequence_flip_axis(sequence_hor, 2)   # flip on the column axis
                        sequence_ver = sequence_flip_axis(sequence_ver, 2)   # flip on the column axis

                X.append(np.concatenate([sequence_hor, sequence_ver], axis=3))
                y.append(hr)
            
            #print(np.array(X).shape, np.array(y).shape, " (train generator)")
            yield np.array(X), np.array(y)


    @threadsafe_generator    
    def test_generator_optical_flow(self, test_df):
        paths, hr = list(test_df["Path"]), list(test_df["Heart Rate"])
        i = 0
        self.greyscale_on = True

        while True:
            X, y = [], []
            current_path = paths[i]
            current_hr = hr[i]
            
            if self.scaler:
                current_hr = self.scaler.transform(current_hr)[0][0]

            frame_hor_dir = sorted(os.listdir(os.path.join(current_path, 'flow_h')))
            frame_ver_dir = sorted(os.listdir(os.path.join(current_path, 'flow_v')))

            #frame_hor_dir = [path for path in frame_hor_dir if path != 'flow_h' and path != 'flow_v']
            #frame_ver_dir = [path for path in frame_ver_dir if path != 'flow_h' and path != 'flow_v']
            #hard-code to 2 for now, because there are a lot of samples
            for _ in range(2):
                start = random.randint(0, len(frame_hor_dir)-self.sequence_length)
                frames_hor = frame_hor_dir[start:start+self.sequence_length]
                frames_hor = [os.path.join(os.path.join(current_path, 'flow_h'), frame) for frame in frames_hor]
                frames_ver = frame_ver_dir[start:start+self.sequence_length]
                frames_ver = [os.path.join(os.path.join(current_path, 'flow_v'), frame) for frame in frames_ver]

                sequence_hor = build_image_sequence(frames_hor, greyscale_on=self.greyscale_on)
                sequence_ver = build_image_sequence(frames_ver, greyscale_on=self.greyscale_on)
                #print(sequence_hor.shape)            
                #flowX = np.dstack(sequence_hor)
                #flowY = np.dstack(sequence_ver)
                X.append(np.concatenate([sequence_hor, sequence_ver], axis=3))
                y.append(current_hr)
                
            i+=1
            if i == len(test_df):
                i = 0
            self.test_iter = i
 
            #print(np.array(X).shape, np.array(y).shape, " (test generator)")
            yield np.array(X), np.array(y)

    @threadsafe_generator    
    def train_generator_optical_flow(self, train_df):
        bucket_list = [.1, .2, .3, .4, .5, .6]

        while True:
            X, y = [], []
            for _ in range(self.batch_size):
                
                #rand_bucket = bucket_list[random.randint(0, len(bucket_list)-1)]
                #df = train_df[buckets(train_df, rand_bucket)]
                #rand_subj_index = random.randint(0, len(df)-1)
                #rand_subj_df = df[rand_subj_index:rand_subj_index+1]

                #path = list(rand_subj_df["Path"])[0]
                #hr = list(rand_subj_df["Heart Rate"])[0]
                
                random_index = random.randint(0, len(train_df)-1)
                path = list(train_df['Path'])[random_index]
                hr = list(train_df['Heart Rate'])[random_index]
                
                if self.scaler:
                    hr = self.scaler.transform(hr)[0][0]
                
                frame_hor_dir = sorted(os.listdir(os.path.join(path,'flow_h')))
                frame_ver_dir = sorted(os.listdir(os.path.join(path,'flow_v')))
                
                #frame_hor_dir = [path for path in frame_hor_dir if path != 'flow_h' and path != 'flow_v']
                #frame_ver_dir = [path for path in frame_ver_dir if path != 'flow_h' and path != 'flow_v']
                
                start = random.randint(0, len(frame_hor_dir)-self.sequence_length)
                frames_hor = frame_hor_dir[start:start+self.sequence_length]
                frames_hor = [os.path.join(os.path.join(path, 'flow_h'), frame) for frame in frames_hor]
                frames_ver = frame_ver_dir[start:start+self.sequence_length]
                frames_ver = [os.path.join(os.path.join(path, 'flow_v'), frame) for frame in frames_ver]

                sequence_hor = build_image_sequence(frames_hor, greyscale_on=self.greyscale_on)
                sequence_ver = build_image_sequence(frames_ver, greyscale_on=self.greyscale_on)

                # now we want to apply the augmentation
                if self.rotation_range > 0.0:
                    sequence_hor = random_sequence_rotation(sequence_hor, self.rotation_range)
                    sequence_ver = random_sequence_rotation(sequence_ver, self.rotation_range)

                if self.width_shift_range > 0.0 or self.height_shift_range > 0.0:
                    sequence_hor = random_sequence_shift(sequence_hor, self.width_shift_range, self.height_shift_range)
                    sequence_ver = random_sequence_shift(sequence_ver, self.width_shift_range, self.height_shift_range)

                if self.shear_range > 0.0:
                    sequence_hor = random_sequence_shear(sequence_hor, self.shear_range)
                    sequence_ver = random_sequence_shear(sequence_ver, self.shear_range)

                if self.zoom_range > 0.0:
                    sequence_hor = random_sequence_zoom(sequence_hor, self.zoom_range)
                    sequence_ver = random_sequence_zoom(sequence_ver, self.zoom_range)

                if self.vertical_flip:
                    # with probability 0.5, flip vertical axis
                    coin_flip = np.random.random_sample() > 0.5
                    if coin_flip:
                        sequence_hor = sequence_flip_axis(sequence_hor, 1)   # flip on the row axis
                        sequence_ver = sequence_flip_axis(sequence_ver, 1)   # flip on the row axis

                if self.horizontal_flip:
                    # with probability 0.5, flip horizontal axis (cols)
                    coin_flip - np.random.random_sample() > 0.5

                    if coin_flip:
                        sequence_hor = sequence_flip_axis(sequence_hor, 2)   # flip on the column axis
                        sequence_ver = sequence_flip_axis(sequence_ver, 2)   # flip on the column axis

                X.append(np.concatenate([sequence_hor, sequence_ver], axis=3))
                y.append(hr)
            
            #print(np.array(X).shape, np.array(y).shape, " (train generator)")
            yield np.array(X), np.array(y)

    def train_generator_v3(self, train_df):
        #bucket_list = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9]
        bucket_list = [.1, .2, .3, .4, .5, .6]
        while True:
            X, y = [], []

            for _ in range(self.batch_size):
                #rand_bucket = bucket_list[random.randint(0, len(bucket_list)-1)]
                #df = train_df[buckets(train_df, rand_bucket)]
                #rand_subj_index = random.randint(0, len(df)-1)
                #rand_subj_df = df[rand_subj_index:rand_subj_index+1]
                
                #path = list(rand_subj_df["Path"])[0]
                #hr = list(rand_subj_df["Heart Rate"])[0]
                
                random_index = random.randint(0, len(train_df)-1)
                path = list(train_df['Path'])[random_index]
                hr = list(train_df['Heart Rate'])[random_index]

                if self.scaler:
                    hr = self.scaler.transform(hr)[0][0]
               
                frame_dir = sorted(os.listdir(path))
                start = random.randint(0, len(frame_dir)-self.sequence_length)
                frames = frame_dir[start:start+self.sequence_length]
                frames = [os.path.join(path, frame) for frame in frames]

                sequence = build_image_sequence(frames, greyscale_on=self.greyscale_on)
                
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
                y.append(hr)
            
            #print(np.array(X).shape, np.array(y).shape, " (train generator)")
            yield np.array(X), np.array(y)

    @threadsafe_generator    
    def train_generator(self, paths2labels):
        """
        generate a sequence of self.batch_size clips length self.sequence_length        
        """
        
        sequence_paths = [path for path in paths2labels]
        while True:
            X, y = [], []
            
            selected_paths = random.sample(sequence_paths, self.batch_size)
            content_sizes = [len(os.listdir(p)) for p in selected_paths]
            fps = []

            for sz in content_sizes:
                if sz > 1300:
                    fps.append(60)

                else:
                    fps.append(30)
            
            for path, sz, fps_ in zip(selected_paths, content_sizes, fps):
                
                heart_rate, resp_rate = paths2labels[path]
                
                frames = os.listdir(path)
                frames = sorted(frames)
                frames = [os.path.join(path, frame) for frame in frames]
                
                selected_frames = None

                if fps_ > 30:
                    seq_begin = random.randint(0, sz - 2 * self.sequence_length)
                    selected_frames = frames[seq_begin:seq_begin + self.sequence_length:2]

                else:
                    seq_begin = random.randint(0, sz - self.sequence_length) 
                    selected_frames = frames[seq_begin:seq_begin + self.sequence_length]
                
                sequence = build_image_sequence(selected_frames, greyscale_on=self.greyscale_on)

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
                y.append(heart_rate, resp_rate)

            yield np.array(X), np.array(y)

    @threadsafe_generator
    def frame_generator(self, paths2labels, generator_type):
        """
        a generator for serving up batches of frame sequences 
        
        args:
            paths2labels : dictionary - map from image sequence path names to tuples (heart rate, resp rate)
            generator_type : just a little title string to insert into the below format statement
        """

        print("[*] creating a %s generator with %d samples" % (generator_type, len(paths2labels)))
        
        sequence_paths = [pth for pth in paths2labels]

        while 1:
            X, y = [], []
            selected_paths = random.sample(sequence_paths, self.batch_size)
            
            y = [paths2labels[pth] for pth in selected_paths]
            
            for pth in selected_paths:
               
                frames = get_sample_frames(pth)
                sequence = build_image_sequence(frames, greyscale_on=self.greyscale_on)
                
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

            yield np.array(X), np.array(y)

