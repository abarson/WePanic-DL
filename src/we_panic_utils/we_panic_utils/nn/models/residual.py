"""
A sub-library for recurrent residual net building blocks
=========================================================
noteworthy citations:

    [1] https://github.com/raghakot/keras-resnet/blob/master/resnet.py                                              | resnet implementation
    [2] Keras documentation on github                                                                               | keras dox
    [3] https://arxiv.org/pdf/1706.08807.pdf --- Recurrent Residual Learning for Action Recognition, Iqbalm et al.  | 
    [4] https://stats.stackexchange.com/questions/56950/neural-network-with-skip-layer-connections                  |
    [5] https://arxiv.org/pdf/1512.03385.pdf --- Deep Residual Learning for Image Recognition, He et al.            |
    [6] https://arxiv.org/pdf/1603.05027v2.pdf --- Identity Mappings in Deep Residual Networks, He et al.           |
    [7] https://gist.github.com/bzamecnik/8ed16e361a0a6e80e2a4a259222f101e                                          | residual LSTMs
    [8] https://github.com/broadinstitute/keras-resnet/blob/master/keras_resnet/blocks/_time_distributed_2d.py      | time distributed residual
"""

from __future__ import division
from keras.models import Model
from keras.layers import Input, Activation, Dense, Flatten, Lambda, LSTM, ConvLSTM2D
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.merge import add
from keras.layers.wrappers import TimeDistributed
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K

from keras_resnet.blocks import time_distributed_bottleneck_2d

K.set_image_data_format('channels_last')

def batchnorm_relu(input_lyrs, channel_axis=2):
    """
    helper function for building a BN --> ReLU block, [1]
    """

    norm = BatchNormalization(axis=channel_axis)(input_lyrs)
    return Activation('relu')(norm)


def conv_bn_relu(**conv_params):
    """
    helper function for building a Conv --> BN --> ReLU block, [1]
    """

    filters = conv_params['filters']
    kernel_sz = conv_params['kernel_size']
    strides = conv_params.setdefault('strides', (1, 1))
    kernel_initializer = conv_params.setdefault('kernel_initializer', 'he_normal')
    padding = conv_params.setdefault('padding', 'same')
    kernel_regularizer = conv_params.setdefault('kernel_regularizer', l2(1.e-4))

    def f(inputs):
        conv = Conv2D(filters=filters, kernel_size=kernel_sz,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)(inputs)

        return batchnorm_relu(conv) 
    return f


def bn_relu_conv(**conv_params):
    """
    helper function for building a BN --> ReLU --> Conv block, [1]
    """

    filters = conv_params['filters']
    kernel_sz = conv_params['kernel_size']
    strides = conv_params.setdefault('strides', (1, 1))
    kernel_initializer = conv_params.setdefault('kernel_initializer', 'he_normal')
    padding = conv_params.setdefault('padding', 'same')
    kernel_regularizer = conv_params.setdefault('kernel_regularizer', l2(1.e-4))

    def f(inputs):
        activation = batchnorm_relu(inputs)
        conv = Conv2D(filters=filters, kernel_size=kernel_sz,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)(activation)

        return conv 
    return f


def skipconnect(inputs, residual, row_ax=0, col_ax=1, channel_ax=2):
    """
    add a skip connection between the input and the residual block and merge
    with a 'sum', [1] 
    """
    
    input_shape = K.int_shape(inputs)
    residual_shape = K.int_shape(residual)
    stride_width = int(round(input_shape[row_ax] / residual_shape[row_ax]))
    stride_height = int(round(input_shape[col_ax] / residual_shape[col_ax]))
    equal_channels = input_shape[channel_ax] == residual_shape[channel_ax] 

    skipconn = inputs
    
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        skipconn = Conv2D(filters=residual_shape[channel_ax],
                          kernel_size=(1, 1),
                          strides=(stride_width, stride_height),
                          padding='valid',
                          kernel_initializer='he_normal',
                          kernel_regularizer=l2(0.0001))(inputs)

    return add([skipconn, residual])


def basic_block(filters, strides=(1, 1), fst_blk_fst_lyrs=False):
    """
    Basic residual block implementation for 3x3 convolution for resnets <= 34 layers
    Follows proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf, [1]
    """

    def f(inputs):
        if fst_blk_fst_lyrs:
            conv1 = Conv2D(filters=filters, strides=strides, kernel_size=(3, 3),
                           padding='same', kernel_initializer='he_normal',
                           kernel_regularizer=l2(1e-4))(inputs)

        else:
            conv1 = bn_relu_conv(filters=filters, kernel_size=(3, 3), strides=strides)(inputs)

        residual = bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv1)

        return skipconnect(inputs, residual)
    return f


def bottleneck(filters, strides=(1, 1), fst_blk_fst_lyrs=False):
    """Bottleneck architecture for > 34 layer resnet.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
    Returns:
        A final conv layer of filters * 4, [1]
    """
    def f(inputs):

        if fst_blk_fst_lyrs:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv_1_1 = Conv2D(filters=filters, kernel_size=(1, 1),
                              strides=strides,
                              padding="same",
                              kernel_initializer="he_normal",
                              kernel_regularizer=l2(1e-4))(inputs)
        else:
            conv_1_1 = bn_relu_conv(filters=filters, kernel_size=(1, 1),
                                    strides=strides)(inputs)

        conv_3_3 = bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv_1_1)
        residual = bn_relu_conv(filters=filters * 4, kernel_size=(1, 1))(conv_3_3)
        return skipconnect(inputs, residual)

    return f


def residual_block(filters, repetitions, is_first_layer=False):
    """
    Builds a residual block with repeating bottleneck blocks.
    [1]
    """
    def f(inputs):
        for i in range(repetitions):
            init_strides = (1, 1)
            if i == 0 and not is_first_layer:
                init_strides = (2, 2)
            inputs = bottleneck(filters=filters, init_strides=init_strides,
                                fst_blk_fst_lyrs=(is_first_layer and i == 0))(inputs)
        return inputs

    return f


def make_residual_LSTM_layers(inputs, rnn_width, rnn_depth, rnn_dropout):
    """
    make intermediate residual LSTM layers, [7] 
    """

    x = inputs

    for i in range(rnn_depth):
        return_seqs = i < rnn_depth - 1
        x_rnn = LSTM(rnn_width, recurrent_dropout=rnn_dropout, dropout=rnn_dropout, return_sequences=return_seqs)(x)

        if return_seqs:
            if i > 0 or inputs.shape[-1] == rnn_width:
                x = add([x, x_rnn]) 

            else:
                x = x_rnn

        else:
            def slice_last(x):
                return x[..., -1, :] 

            x = add([Lambda(slice_last), x_rnn])
    
    return x


def residualLSTMblock(inputs, rnn_depth, filters, kernel_size, dropout=0.5, return_sequences=True):
    """
    see [7]
    """
    x = inputs

    for i in range(rnn_depth):
        x_rnn = ConvLSTM2D(filters,
                           kernel_size,
                           recurrent_dropout=dropout,
                           dropout=dropout,
                           padding='same',
                           return_sequences=return_sequences)(x)

        return_sequences = i < rnn_depth - 1

        if return_sequences:
            if i > 0 or inputs.shape[-1] == filters:
                x = add([x, x_rnn])

            else:
                x = x_rnn

        else:

            def slice_last(x):
                return x[-1,:,:,:]
            
            x = add([Lambda(slice_last)(x), x_rnn])

    #print('residual lstm block : ', x.shape)
    return x 


