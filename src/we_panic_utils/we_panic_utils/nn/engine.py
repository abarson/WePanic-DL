# intra-library imports
from .data_load import ttswcsv, sorted_stratified_kfold
from ..basic_utils.basics import check_exists_create_if_not
from .callbacks import TestResultsCallback, CyclicLRScheduler
from .functions import get_keras_losses

# inter-library imports
from keras import models
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from keras import backend as K

import os
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold
import numpy as np
import time
from glob import glob

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
                 train=False, 
                 test=False, 
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
        self.train = train
        self.test = test
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
        
        if self.early_stopping is not None:
            self.early_stopping = EarlyStopping(patience=self.early_stopping)

        if self.loss_fun is None:
            self.loss_fun = 'mean_squared_error'

        self.loss_fun = self.__choose_loss()

    def __train_model(self):
        """
         _______
        ( TODO  )
       ( SHROOM  ) ==============================================*
      {___________}                                              |
        { ____ }   ** Update for newer iterations of project *** |
        {_||||_} ================================================*

        Internal method to train the model. This method is responsible for instantiating the model
        based on the user's inputs, as well as the train, test, and validation sets. Once instantiated,
        these objects are maintained as instance variables by the engine object for later use.
        """

        if not self.model:  #instantiate the model to be trained
            self.model = self.__choose_model().instantiate()
        print("Training the model")

        self.train_set, self.test_set, self.val_set = ttswcsv(self.data, self.metadata, self.outputs)
        
        train_generator = self.processor.train_generator(self.train_set)
        val_generator = self.processor.test_generator(self.val_set)

        self.model.fit_generator(generator=train_generator,
                                 steps_per_epoch=self.steps_per_epoch,
                                 epochs=self.epochs,
                                 verbose=1,
                                 callbacks=self.callbacks,
                                 validation_data=val_generator,
                                 validation_steps=len(self.val_set), workers=4)

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
            callbacks.append(self.cyclic_lr)
        
        if self.early_stopping is not None:
            callbacks.append(self.early_stopping)

        return callbacks

    def __test_model(self):
        """
        Internal method to test the model. If the model was not trained before this method is called,
        then the model will be loaded from the input directory provided by the user. Otherwise, the model
        instantiated during the training phase will be tested.
        """

        if not self.model:  #load the model if it wasn't created during the training phase
            
            print('>>> No model found, inferring top model given the input directory')
            
            #self.model = models.load_model(self.__infer_top_model(), compile=False)
            metadf = pd.read_csv(self.metadata)
            self.test_set = metadf[metadf['GOOD'] == 2]
            print('>>> Built test set')
            optimizer = Adam(lr=1e-5, decay=1e-6)
            #self.model.compile(loss=self.loss_fun, optimizer=optimizer, metrics=['mse'])
            print('>>> Compiled and ready to go')

        else:  #test the model created during training
            print("Testing model after training.")
        
        map_me = {'HEART_RATE_BPM' : 'hr', 'RESP_RATE_BR_PM' : 'rr'}
        agg_df = pd.DataFrame(columns=['actual_{}'.format(map_me[f]) for f in self.features])

        for model_name, loss in self.__next_model:
            self.model = models.load_model(model_name, compile=False)
            self.model.compile(loss=self.loss_fun, optimizer=optimizer, metrics=['mse'])
            print('>>> {} Compiled and ready to go'.format(model_name))
            
            test_generator = self.processor.test_generator(self.test_set)
            
            model_id = model_name.split('/')[-1]
            model_id = model_id.rstrip('.h5')
            test_slug = 'testSetPerformance_name-%s_loss%0.4f' % (model_id, loss) 
        
            header = sum([['actual_{}'.format(map_me[f]), 'predicted_{}'.format(map_me[f])] for f in self.features], [])
            performance_df = pd.DataFrame(columns=header)

            for i in range(len(self.test_set)):
                print('\r[__test_model]: sample %2d' % i, flush=True, end=' ')
                Xs, ys = next(test_generator) 
            
                feats = zip(*ys)
                mean_feats = [np.mean(feat) for feat in feats]
                
                preds = self.model.predict(Xs, batch_size=len(Xs))   
                feat_preds = zip(*preds)
                mean_preds = [np.mean(pred) for pred in feat_preds]
            
                row = sum([list(value) for value in zip(*[mean_feats, mean_preds])], [])
                performance_df.loc[i] = row
            
            print('\n',performance_df)

            for f in self.features:
                agg_df['preds_{}_{}'.format(model_id, map_me[f])] = performance_df['predicted_{}'.format(map_me[f])]
                agg_df['actual_{}'.format(map_me[f])] = performance_df['actual_{}'.format(map_me[f])] 
        
         
        for f in self.features:
            feature_index = list(agg_df.columns).index('actual_{}'.format(map_me[f]))
            means = []
            for idx, row in agg_df.iterrows():
                row = list(row)
                cols = []
                for i, col in enumerate(agg_df.columns):
                    if map_me[f] in col:
                        cols.append(i)

                actual = row[feature_index]
                cols.pop(feature_index)

                preds = [row[i] for i in cols]
                mean_pred = np.mean(preds)
                means.append(mean_pred)
            
            agg_df['mean_pred_{}'.format(map_me[f])] = pd.Series(means, index=agg_df.index)

        agg_df.to_csv(os.path.join(self.outputs, 'testSetPerformance.csv'))
    
          
    def __cross_val(self):
        """
        split the data a la cross-validation and train up some models
        """
            
        print('>>> cross validating on %d folds' % self.kfold)

        # get all the data
        metadf = pd.read_csv(self.metadata)
        
        # only samples deemed good
        good_samps = metadf[metadf['GOOD'] == 1]
        self.the_most_testiest_samps = metadf[metadf['GOOD'] == 2]

        # records for later
        cv_results = pd.DataFrame(columns=['model_type','model_idx','elapsed_time', 'loss']) #'predictive_acc')
        
        
        # do self.kfold separate training/validation iters
        for idx, (train_set, val_set) in enumerate(sorted_stratified_kfold(good_samps, k=self.kfold)):
            
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
                                           compile=False)

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
        
        # other wise train/test as ushe
        else:
            if self.train:
                self.__train_model()

            if self.test:
                return self.__test_model()
        
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
