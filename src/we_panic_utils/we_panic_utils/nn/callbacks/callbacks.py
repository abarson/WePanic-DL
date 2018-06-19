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
from sklearn.metrics import mean_squared_error

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
                
##This needs a touch up
class TestResultsCallback(Callback):
    """
    a callback for testing the model at certain timesteps
    and viewing its actual output
    """
    def __init__(self, test_gen, test_set, log_file, batch_size, epochs=5):
        self.test_gen = test_gen
        self.test_set = test_set
        self.log_file = log_file
        self.batch_size = batch_size
        self.epochs=epochs

    def on_epoch_end(self, epoch, logs):

        #get the actual mse
        if (epoch+1) % self.epochs == 0:
            print('Logging tests at epoch', epoch)
            with open(self.log_file, 'a') as log:
                gen = self.test_gen.test_generator(self.test_set)
                
                pred = self.model.predict_generator(gen, len(self.test_set))
                
                subjects = list(self.test_set['SUBJECT'])
                trial = list(self.test_set['TRIAL'])
                hr = list(self.test_set['HEART_RATE_BPM'])
                rr = list(self.test_set['RESP_RATE_BR_PM'])
                
                i = 0
                s = 0
                error_hr = mean_squared_error(np.reshape([i for t in zip(hr,hr) for i in t], (-1, 1)), [row[0] for row in pred])
                error_rr = mean_squared_error(np.reshape([i for t in zip(rr,rr) for i in t], (-1, 1)), [row[1] for row in pred])
                error = error_hr + error_rr
                log.write("Epoch: " + str(epoch+1) + ', Error: ' + str(error) + '\n')
                for p in pred:
                    subj = subjects[s]
                    tri = trial[s]
                    h = hr[s]
                    r = rr[s]
                    
                    log.write(str(subj) + ', ' + str(tri) + '| prediction=' + str(p) + ', actual=' + str([h, r]) + '\n')
                    i+=1
                    if i % self.test_gen.num_val_clips == 0:
                        s += 1
                    if s == len(subjects):
                        s = 0

