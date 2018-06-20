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

    def __init__(self, steps_per_epoch, output_dir, schedule, verbose=True):
        super().__init__()
        self.schedule = schedule
        self.steps_per_epoch = steps_per_epoch

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
            lr = self.schedule(self.epoch * self.steps_per_epoch + batch, lr=lr)
        except TypeError:  # new API for backward compatibility
            lr = self.schedule(self.epoch * self.steps_per_epoch + batch)

        if not isinstance(lr, (float, np.float32, np.float64)):
            raise ValueError('The output of the "schedule" function should be a float.')

        K.set_value(self.model.optimizer.lr, lr)

        if self.verbose:
            with open(self.output_log, 'a') as f:
                print(f'\nStep {self.epoch * self.steps_per_epoch + batch}:'
                      f' learning rate = {lr}.', file=f)

##This needs a touch up
class TestResultsCallback(Callback):
    """
    a callback for testing the model at certain timesteps
    and viewing its actual output
    """

    def __init__(self, test_gen, test_set, log_file, epochs=5):
        self.test_gen = test_gen
        self.test_set = test_set
        self.log_file = log_file
        self.epochs=epochs

        self.__remove_file_if_exists(log_file) 

    def __remove_file_if_exists(self, log):
        if os.path.exists(log):
            os.remove(log)

    def on_epoch_end(self, epoch, logs):
        if (epoch+1) % self.epochs == 0:
            print('Logging tests at epoch', epoch+1)
            with open(self.log_file, 'a') as log:
                gen = self.test_gen.test_generator(self.test_set)
                pred = self.model.predict_generator(gen, len(self.test_set))

                subjects = list(self.test_set['SUBJECT'])
                trial = list(self.test_set['TRIAL'])
                hr = list(self.test_set['HEART_RATE_BPM'])
                rr = list(self.test_set['RESP_RATE_BR_PM'])
                log.write("Epoch: " + str(epoch+1) + '\n')
                
                #Divide the list of predictions into a number of partitions equal to the number of validation clips per subject
                preds = [pred[i:i+self.test_gen.num_val_clips] for i in range(0, len(pred), self.test_gen.num_val_clips)]

                for s, t, hr, rr, p in zip(subjects, trial, hr, rr, preds):
                    hr_avg = format(sum(p[0:,0])/len(p), '.3f')
                    rr_avg = format(sum(p[0:,1])/len(p), '.3f')
                    hr_std = format(np.std(p[0:,0]), '.2f')
                    rr_std = format(np.std(p[0:,1]), '.2f')
                    log.write('{:<4} {} | avg_hr={:<10} avg_rr={:<10} | act_hr={:<6} act_rr={:<6} | hr_std={:<6} rr_std={:<6}\n'
                            .format(int(s), t, hr_avg, rr_avg, hr, rr, hr_std, rr_std))
