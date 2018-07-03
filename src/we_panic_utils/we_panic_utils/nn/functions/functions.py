"""
functions submodule, defines some useful learning rates, losses, etc to be used
in conjunction with various functional parts of keras api
"""
import numpy as np
import keras.backend as K

from ...basic_utils.basics import get_module_attributes

def triangular_lr(step, stepsize, base_lr, max_lr):
    """
    defines a triangular learning rate
    
    args:
        step (int) -- the iteration number
        stepsize (int) -- half of a cycle (number of iterations)
        base_lr (float) -- the minimum learning rate
        max_lr (float) -- the maximum learning rate

    returns:
        (float) -- the next learning rate
    """
    cycle = np.floor(1 + step/(2 * stepsize))
    x = np.abs(step / stepsize - 2 * cycle + 1)
    lr = base_lr + (max_lr - base_lr) * np.maximum(0, (1-x))
    return lr

def euclidean_distance_loss(y_true, y_pred):
    """
    compute the distance between points (x1,y1), (x2, y2) 
    """

    return K.sqrt(K.sum(K.square(y_true - y_pred), axis=-1, keepdims=True))

def smooth_l1_loss(y_true, y_pred):
    """
    compute smooth l1 loss; a smoothed MAE
    """
    HUBER_DELTA = 0.5

    x = K.abs(y_true - y_pred)
    if K._BACKEND == 'tensorflow':
        import tensorflow as tf
        x = tf.where(x < HUBER_DELTA, 0.5 * x ** 2, HUBER_DELTA * (x -0.5 * HUBER_DELTA))
        return K.sum(x)

def get_keras_losses():
    import keras.losses
    return get_module_attributes(keras.losses, exclude_set=['absolute_import',
                                                            'print_function',
                                                            'serialize_keras_object',
                                                            'deserialize_keras_object',
                                                            'division',
                                                            'six',
                                                            'K',
                                                            'get',
                                                            'serialize',
                                                            'deserialize'])

    
