"""
functions submodule, defines some useful learning rates, losses, etc to be used
in conjunction with various functional parts of keras api
"""
import numpy as np
import keras.backend as K

def cos_cyclic_lr(step, lr, lr0=0.2, total_steps=400, cycles=8):
    """
         Defines a cyclic learning rate schedule which decays from lr0 to a tiny value
         before starting the next cycle back at lr0.

         Args:
              step -------:
              lr ---------: (float) Previous learning rate. Required by Keras api, but unused.
              lr0 --------: (float) Initial learning rate
              total_steps-: (int) Number of training epochs or epochs * batches per epoch
              cycles -----: (int) Number of cycles to perform.

         Returns:
             (float) Learning rate at the current training step.

    """
    return 0.5 * lr0 * (np.cos(np.pi * (step % np.ceil(total_steps / cycles)) / np.ceil(total_steps / cycles)) + 1)


def euclidean_distance_loss(y_true, y_pred):
    """
    compute the distance between points (x1,y1), (x2, y2) 
    """

    return K.sqrt(K.sum(K.square(y_true - y_pred), axis=-1, keepdims=True))
