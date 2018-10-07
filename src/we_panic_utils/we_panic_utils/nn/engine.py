# intra-library imports
from .data_load import ttswcsv, sorted_stratified_kfold
from ..basic_utils.basics import check_exists_create_if_not
from .callbacks import TestResultsCallback, CyclicLRScheduler
from .functions import get_keras_losses
from ..basic_utils.graphing import compute_bland_altman

# inter-library imports
from keras import models
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
import tensorflow as tf
import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import numpy as np
import time
from glob import glob
from functools import reduce

import matplotlib.pyplot as plt
plt.switch_backend('agg')

class Engine():
    """
    The engine for training/testing a model

    args:
        data ------------: directory containing subject frame data
        model_type ------: the model to be trained/tested
        csv -------------: the csv containing all of the stats for every sample
        batch_size ------: the batch size
        epochs ----------: number of epochs to train
        train -----------: boolean stating whether or not to train
        test ------------: boolean stating whether or not to test
        inputs ----------: the input directory to be used if testing without training
        outputs ---------: the output directory to save the new model to
        frameproc -------: FrameProcessor object for augmentation
        input_shape -----: shape of the sequence passed, SEQ_LEN separate (Rows, Cols, Channs) frames
        output_shape ----: the number of outputs
        steps_per_epoch -: steps per epoch 
        kfold -----------: not used if None, otherwise this is the number of folds to
                        \: use in a kfold cross validation
        cyclic_lr -------: instance of CyclicLRScheduler
        loss_fun ------------: the loss function that should be used

    """
    def __init__(self, 
                 model_type, 
                 data,
                 features,
                 frameproc,
                 csv, 
                 inputs, 
                 outputs,
                 epochs, 
                 batch_size=14, 
                 qbc=None, 
                 input_shape=(60, 32, 32, 1),
                 steps_per_epoch=500,
                 kfold=None,
                 cyclic_lr=None,
                 early_stopping=None,
                 loss_fun=None):

        #passed in params
        self.data = data
        self.model_type = model_type
        self.features = features
        self.metadata = csv
        self.batch_size = batch_size
        self.epochs = epochs
        self.qbc = qbc
        self.inputs = inputs
        self.outputs = outputs
        self.input_shape = input_shape
        self.processor = frameproc
        self.steps_per_epoch = steps_per_epoch
        self.kfold = kfold
        self.cyclic_lr = cyclic_lr
        self.loss_fun = loss_fun
        self.early_stopping = early_stopping

        #created instance variables
        self.model = None
        self.train_set = None
        self.test_set = None
        self.val_set = None
        
        #if self.early_stopping is not None:
        #    self.early_stopping = EarlyStopping(patience=self.early_stopping)

        if self.loss_fun is None:
            self.loss_fun = 'mean_squared_error'

        self.loss_fun = self.__choose_loss()

    def __infer_top_model(self):
        """
        call this method when we want to check for the best model in 
        a directory as shown by that directory's cvresults.csv
        """
    
        cvresults = pd.read_csv(os.path.join(self.inputs, 'cvresults.csv')) 
        minimum_loss = cvresults['loss'].min()
        top_row = cvresults[cvresults['loss'] == minimum_loss]
        idx = top_row['model_idx'].values.tolist()[0]
        model_type = top_row['model_type'].values.tolist()[0]
        model_name = 'CV_%d_%s.h5' % (idx, model_type)
        
        return os.path.join(self.inputs, 'models', model_name) 
    
    @property
    def __next_model(self):
        """
        return the next model in the output directory for testing
        """

        cvresults = pd.read_csv(os.path.join(self.inputs, 'cvresults.csv'))
        
        for _, row in cvresults.iterrows():
            #row = row.values.tolist()
            idx = row['model_idx']
            model_type = row['model_type']

            print(model_type, idx)
            model_name = 'CV_%d_%s.h5' % (idx, model_type)

            yield os.path.join(self.inputs,'models',model_name), row['loss']

    def callbacks(self,
                  train_results='unnormalized-training_results.log',
                  test_results='test_results.log',
                  csv_log='training.log',
                  chkpt='default'):
        """
        return fresh callbacks for an upcoming procedure

        args:
            train_results: string -- relative path to train logfile
            test_results : string -- relative path to test  logfile
            csv_log      : string -- csvlog name (relative path)
            chkpt        : string -- name of checkpoint file (relative path with a .h5)
        """
        
        csv_logger = CSVLogger(os.path.join(self.outputs, csv_log))

        if chkpt == 'default':
            checkpointer = ModelCheckpoint(filepath=os.path.join(self.outputs, 'models', self.model_type + '.h5'), 
                                           verbose=1, 
                                           save_best_only=True)

        else:
            checkpointer = ModelCheckpoint(filepath=os.path.join(self.outputs, 'models', chkpt), verbose=1, save_best_only=True)
        
        results_dir = check_exists_create_if_not(os.path.join(self.outputs, 'results',))
        test_results = os.path.join(results_dir, test_results)
        train_results = os.path.join(results_dir, train_results)
        
        #this will record the output of the model on the training data at the end of every epoch
        train_callback = TestResultsCallback(self.processor, 
                                             self.train_set, 
                                             train_results, 
                                             epochs=1)

        #this will record the output of the model on the testing data at the end of every 5 epochs
        test_callback = TestResultsCallback(self.processor,
                                            self.test_set,
                                            test_results)
        

        callbacks = [csv_logger, checkpointer, train_callback, test_callback]  # train_callback]    

        if self.cyclic_lr is not None:
            assert isinstance(self.cyclic_lr, CyclicLRScheduler), 'cyclic_lr should be a CyclicLRScheduler'
            self.cyclic_lr.reset()
            callbacks.append(self.cyclic_lr)
        
        if self.early_stopping is not None:
            callbacks.append(EarlyStopping(monitor='val_loss',patience=self.early_stopping))

        return callbacks

    def __QBC(self):
        """
        Internal method to perform QBC (query by comm). 
        -----------------------------------------------
        self.qbc is a directory containing .h5 files, the object of this method
        is to test those models and report their aggregate predictions and their MSEs
        """

        metadf = pd.read_csv(self.metadata)
        self.test_set = metadf[metadf['GOOD'] == 3]
        print('[QBC] built test set') 
        map_me = {'HEART_RATE_BPM' : 'hr', 'RESP_RATE_BR_PM' : 'rr'}
        map_back = dict(map(reversed, map_me.items())) 

        # infer .h5 files
        committee_members = glob(os.path.join(self.qbc, '*.h5'))
        committee_members_noh5 = [m.rstrip('.h5').split('/')[-1] for m in committee_members]
        
        test_generator = self.processor.test_generator(self.test_set)

        actual_answers = ['actual_{}'.format(map_me[f]) for f in self.features]
        predictions = ['predicted_{}_{}'.format(map_me[f], cmemb) for f in self.features for cmemb in committee_members_noh5]
        
        columns = actual_answers + predictions
        QBCdf = pd.DataFrame(columns=columns + ['QBC_{}'.format(map_me[f]) for f in self.features])
        hr_idxs = [i for i in range(len(columns)) if 'hr' in columns[i]]
        rr_idxs = [i for i in range(len(columns)) if 'rr' in columns[i]]
        
        committee = []
        for name in committee_members:
            opt = Adam(lr=1e-5, decay=1e-6)
            memb = models.load_model(name, compile=False)
            memb.compile(loss='mean_squared_error', optimizer=opt)
            committee.append(memb)
        print('[QBC] Got {} committee members'.format(len(committee_members)))

        for i in range(len(self.test_set)):
            Xs, ys = next(test_generator)
            row = {col:None for col in columns}
            for memb, name in zip(committee,committee_members_noh5):
                print("\r[{:2d}] {:>20s}...".format(i, name[:20]), flush=True, end='')
                self.model = memb
                
                # the actual result answers
                feats = zip(*ys)
                mean_feats = [np.mean(feat) for feat in feats]

                # the prediction for this model
                preds = self.model.predict(Xs)   
                feat_preds = zip(*preds)
                mean_preds = [np.mean(pred) for pred in feat_preds]
                
                for p, p_true, f in zip(mean_preds, mean_feats, self.features):
                    row['predicted_{}_{}'.format(map_me[f], name)] = p
                    row['actual_{}'.format(map_me[f])] = p_true
            
            results = [row[col] for col in columns]
    
            preds = [None, None]
            if hr_idxs:
                predictions_hr = [results[i] for i in hr_idxs][1]
                preds[self.features.index(map_back['hr'])] = predictions_hr

            if rr_idxs:
                predictions_rr = [results[i] for i in rr_idxs][1]
                #predictions_rr = predictions_rr[2]
                preds[self.features.index(map_back['rr'])] = predictions_rr
            
            preds = list(filter(lambda p: p is not None, preds))
            QBCdf.loc[i] = results + preds
            print("\r[{:2d}] {} DONE".format(i, " "*20))

        print()
        keys = reduce(lambda l1, l2: l1 + l2, ['actual_{},QBC_{}'.format(map_me[f], map_me[f]).split(',') for f in self.features]) 

        feed = {k:QBCdf[k].values.tolist() for k in keys}
        
        QBCdf.to_csv(os.path.join(self.outputs, 'QBC.csv')) 
        print('Wrote performance to {}'.format(os.path.join(self.outputs, 'QBC.csv')))
        print()
        print(" --- ".join(keys + ['SE', '  SSE '])) 
        print('-'*len(" --- ".join(keys + [' SE ', '  SSE '])))

        SSE = 0.
        
        answers, predictions = [], [] 
        
        try:
            with open(os.path.join(self.outputs,'QBC_performance.txt'), 'w') as qbcperf:
                print(" --- ".join(keys + ['SE', '  SSE ']), file=qbcperf) 
                print('-'*len(" --- ".join(keys + [' SE ', '  SSE '])), file=qbcperf)
                for i in range(len(feed[keys[0]])):
                    row = [feed[k][i] for k in keys]
                    actual, pred = [round(r, 3) for r in row]
                    SE = (actual - pred)**2
                    SSE += SE
                    
                    answers.append(actual)
                    predictions.append(pred)

                    print("{:5.01f} {:13.03f} {:9.03f} {:9.03f}".format(actual, pred, SE, SSE))
                    print("{:5.01f} {:13.03f} {:9.03f} {:9.03f}".format(actual, pred, SE, SSE), file=qbcperf)
                    
                print('-'*len(" --- ".join(keys + [' SE ', '  SSE '])))
                print('-'*len(" --- ".join(keys + [' SE ', '  SSE '])), file=qbcperf)
                MSE = SSE / len(feed[keys[0]])
                print('MSE ::==:: {:.03f}'.format(MSE))
                print('MSE ::==:: {:.03f}'.format(MSE), file=qbcperf) 

        except ValueError:
            print(QBCdf)
        
        #bland, altman = zip(*compute_bland_altman(answers, predictions, log=False))
        #plt.scatter(bland, altman)
        #plt.xlabel('Actual - Prediction')
        #plt.ylabel('(Actual + Prediction)/2')
        #plt.savefig(os.path.join('figs', 'bland_altman_{}.png'.format(len(committee_members))))
        
    def __cross_val(self):
        """
        split the data a la cross-validation and train up some models
        """
            
        print('>>> cross validating on %d folds' % self.kfold)

        # get all the data
        metadf = pd.read_csv(self.metadata)
        
        # only samples deemed good
        good_samps = metadf[metadf['GOOD'] == 1]
        self.the_most_testiest_samps = metadf[metadf['GOOD'] == 3]

        # records for later
        cv_results = pd.DataFrame(columns=['model_type','model_idx','elapsed_time', 'loss']) #'predictive_acc')
        
        # do self.kfold separate training/validation iters
        for idx, (train_set, val_set) in enumerate(sorted_stratified_kfold(good_samps, self.features, k=self.kfold)):
            
            start = time.time()
            # record :)
            self._record_cvsets(train_set, val_set, idx) 
            
            # if we are fine tuning a model or something, load that
            # model up
            #if self.model_path is not None:
            #    self.model = models.load_model(self.model_path)
                
            #else:
            print('>>> new model ...')
            self.model = self.__choose_model().instantiate()
            if self.cyclic_lr is not None:
                self.cyclic_lr.model = self.model
                self.cyclic_lr.fold = idx
                
            if not os.path.exists(os.path.join(self.outputs, 'model_summary.txt')):
                with open(os.path.join(self.outputs,'model_summary.txt'), 'w') as summary:
                    self.model.summary(print_fn=lambda x: summary.write(x + '\n'))
                    print('>>> wrote model summary to {}'.format(os.path.join(self.outputs, 'model_summary.txt')))

            # get the train/val gens
            self.test_set = val_set
            self.train_set = train_set

            tgen = self.processor.train_generator(self.train_set) 
            vgen = self.processor.test_generator(self.test_set) 
            te_gen = self.processor.test_generator(self.the_most_testiest_samps)

            print('>>> validation set size: %d' % len(self.test_set))
            vsteps = len(self.test_set)
            # get some callbacks with custom filepaths
            callbacks = self.callbacks(train_results='unnormalized_training{}.log'.format(idx),
                                       test_results='test_results{}.log'.format(idx),
                                       csv_log='training{}.log'.format(idx),
                                       chkpt='CV_%d_%s.h5' % (idx, self.model_type)) 
            
            # fit
            self.model.fit_generator(generator=tgen,
                                     steps_per_epoch=self.steps_per_epoch,
                                     epochs=self.epochs,
                                     verbose=1,
                                     callbacks=callbacks,
                                     validation_data=vgen,
                                     validation_steps=vsteps,
                                     workers=4)
            
            # record predictive acc and loss
            #pred = self.model.predict_generator(vgen, vsteps)
            loss = None
            self.model = models.load_model(os.path.join(self.outputs, 'models', 'CV_%d_%s.h5' % (idx, self.model_type)),
                                           compile=False, custom_objects={"tf":tf})

            optimizer = Adam(lr=1e-5, decay=1e-6)
            self.model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mse'])
            loss = self.model.evaluate_generator(te_gen, len(self.the_most_testiest_samps))[0]

            end = time.time()
            total = (end - start) / 60
            
            cv_results.loc[idx] = [self.model_type, idx, total, loss]
            cv_results.to_csv(os.path.join(self.outputs,'cvresults.csv'), index=False)
         
        cv_results.to_csv(os.path.join(self.outputs,'cvresults.csv'), index=False)
    
        print('finished a %d-fold cross validation\n\tavg_loss: %0.5f' % (self.kfold,
                                                                          cv_results['loss'].mean()))

    def _record_cvsets(self, train, val, idx):
        """
        stupid helper method to record the cross validation sets chosen and save
        them uniquely

        """
        cvsets = check_exists_create_if_not(os.path.join(self.outputs, 'CVsets'))
        train.to_csv(os.path.join(cvsets, 'train{}.csv'.format(idx)))
        val.to_csv(os.path.join(cvsets, 'val{}.csv'.format(idx)))

    def run(self):
        """
        a general method that computes the 'procedure' to follow based on the
        preferences passed in the constructor and runs that procedure
        """
        
        # do cross val if we've got a kfold
        if self.kfold is not None:
            self.__cross_val()
        
        if self.qbc is not None:
            self.__QBC()
        # other wise train/test as ushe
        #else:
            #if self.train:
            #    self.__train_model()
            #
            #if self.test:
            #    return self.__test_model()
        
    def __choose_model(self):
        """
        choose a model based on preferences
        """
        import importlib  
        module_object = importlib.import_module('.models',package='we_panic_utils.nn')
        target_class = getattr(module_object, self.model_type)

        return target_class(self.input_shape, len(self.features), loss=self.loss_fun)

    def __choose_loss(self):
        """
        choose loss function based on preferences
        """

        if self.loss_fun in get_keras_losses():
            import keras.losses
            target_fun = getattr(keras.losses, self.loss_fun)
        else:

            import importlib
            module_object = importlib.import_module('.functions', package='we_panic_utils.nn')
            target_fun = getattr(module_object, self.loss_fun)

        return target_fun
