"""
Description:
    *-------------* 
    | callbacks.py|, the one stop location for custom callbacks used througout the system
    *-------------*

"""

import keras.backend as K
import numpy as np
from keras.callbacks import Callback
import os

class CyclicLRScheduler(Callback):
    """
    Cylic learning rate scheduler
    """

    def __init__(self, schedule, batches_per_epoch, output_dir, verbose=True):
        super().__init__()
        self.schedule = schedule
        self.batches_per_epoch = batches_per_epoch

        self.verbose = verbose
        self.epoch = 0

        self.output_log = os.path.join(output_dir, 'schedule.log')
        self.output_dir = output_dir
        
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch

    def on_batch_begin(self, batch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute. ')

        lr = float(K.get_value(self.model.optimizer.lr))

        try:  # new API
            lr = self.schedule(self.epoch * self.batches_per_epoch + batch, lr=lr)
        except TypeError:  # new API for backward compatibility
            lr = self.schedule(self.epoch * self.batches_per_epoch + batch)

        if not isinstance(lr, (float, np.float32, np.float64)):
            raise ValueError('The output of the "schedule" function should be a float.')

        K.set_value(self.model.optimizer.lr, lr)

        if self.verbose:
            with open(self.output_log, 'a') as f:
                print(f'\nStep {self.epoch * self.batches_per_epoch + batch}:'
                      f' learning rate = {lr}.', file=f)
                

