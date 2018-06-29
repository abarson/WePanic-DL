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
from functools import reduce

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

        #todo: put this somewhere else
        self.translate_dict = {'HEART_RATE_BPM' : 'hr', 'RESP_RATE_BR_PM' : 'rr'}
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
                
                feats = [list(self.test_set[feat]) for feat in self.test_gen.features]
                subjects = list(self.test_set['SUBJECT'])
                trial = list(self.test_set['TRIAL'])

                log.write("Epoch: " + str(epoch+1) + '\n')
                
                #Divide the list of predictions into a number of partitions equal to the number of validation clips per subject
                preds = [pred[i:i+self.test_gen.num_val_clips] for i in range(0, len(pred), self.test_gen.num_val_clips)]
                
                for s, t, *fs, p in zip(subjects, trial, *feats, preds):
                    avgs = [format(sum(p[0:,i])/len(p), '.3f') for i in range(len(fs))] 
                    stds = [format(np.std(p[0:,i]), '.2f') for i in range(len(fs))]
                    
                    concat = lambda x1,x2:x1+x2
                    #I'm so sorry
                    #This formats the log file to look pretty, while being robust to the number of features
                    avg_str = reduce(concat, ['avg_{}={:<10}'.
                        format(self.translate_dict[self.test_gen.features[i]], avgs[i]) for i in range(len(avgs))])
                    act_str = reduce(concat, ['act_{}={:<5}'.
                        format(self.translate_dict[self.test_gen.features[i]], int(fs[i])) for i in range(len(fs))])
                    std_str = reduce(concat, ['std_{}={:<6}'.
                        format(self.translate_dict[self.test_gen.features[i]], stds[i]) for i in range(len(stds))])
                    log.write('{:<4} {} | {} | {} | {}\n'.format(int(s), t, avg_str, act_str, std_str))
