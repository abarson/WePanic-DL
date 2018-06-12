# intra-library imports
from .data_load import ttswcsv
from .models import C3D, CNN_3D, CNN_3D_small

# inter-library imports
from keras import models
from keras.callbacks import CSVLogger, ModelCheckpoint, Callback
from keras import backend as K
import os
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold
import numpy as np

class Engine():
    """
    The engine for training/testing a model

    args:
        data ------------ directory containing subject frame data
        model_type ------ the model to be trained/tested
        csv ------------- the csv containing all of the stats for every sample
        batch_size ------ the batch size
        epochs ---------- number of epochs to train
        train ----------- boolean stating whether or not to train
        test ------------ boolean stating whether or not to test
        inputs ---------- the input directory to be used if testing without training
        outputs --------- the output directory to save the new model to
        frameproc ------- FrameProcessor object for augmentation
        input_shape ----- shape of the sequence passed, 60 separate 100x100x3 frames
        output_shape ---- the number of outputs
        steps_per_epoch - steps per epoch 
        kfold ----------- not used if None, otherwise this is the number of folds to
                        \ use in a kfold cross validation

    """
    def __init__(self, 
                 data,
                 model_type, 
                 csv, 
                 batch_size, 
                 epochs, 
                 train, 
                 test, 
                 inputs, 
                 outputs,
                 frameproc,
                 input_shape,
                 output_shape,
                 steps_per_epoch,
                 kfold=None):

        #passed in params
        self.data = data
        self.model_type = model_type
        self.metadata = csv
        self.batch_size = batch_size
        self.epochs = epochs
        self.train = train
        self.test = test
        self.inputs = inputs
        self.outputs = outputs
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.processor = frameproc
        self.steps_per_epoch = steps_per_epoch

        #created instance variables
        self.model = None
        self.train_set = None
        self.test_set = None
        self.val_set = None

    def __train_model(self):
        """
        Internal method to train the model. This method is responsible for instantiating the model
        based on the user's inputs, as well as the train, test, and validation sets. Once instantiated,
        these objects are maintained as instance variables by the engine object for later use.
        """

        if not self.model: #instantiate the model to be trained
            self.model = self.__choose_model().instantiate()
        print("Training the model")

        self.train_set, self.test_set, self.val_set = ttswcsv(self.data, self.metadata, self.outputs)
        
        train_generator = self.processor.train_generator(self.train_set)
        val_generator = self.processor.test_generator(self.val_set)

        csv_logger = CSVLogger(os.path.join(self.outputs, "training.log"))
        checkpointer = ModelCheckpoint(filepath=os.path.join(self.outputs, 'models', self.model_type + '.h5'), 
                                       verbose=1, 
                                       save_best_only=True)

        test_results_file = os.path.join(self.outputs, "test_results.log")
        train_results = os.path.join(self.outputs, "unnormalized_training.log")
        
        #this will record the output of the model on the training data at the end of every epoch
        train_callback = TestResultsCallback(self.processor, 
                                             self.train_set, 
                                             train_results, 
                                             self.batch_size,
                                             epochs=1)

        #this will record the output of the model on the testing data at the end of every 5 epochs
        test_callback = TestResultsCallback(self.processor,
                                            self.test_set,
                                            test_results_file,
                                            self.batch_size)
        
        callbacks = [csv_logger, checkpointer, test_callback, train_callback]    

        self.model.fit_generator(generator=train_generator,
                            steps_per_epoch=self.steps_per_epoch,
                            epochs=self.epochs,
                            verbose=1,
                            callbacks=callbacks,
                            validation_data=val_generator,
                            validation_steps=len(self.val_set), workers=4)
    def __test_model(self):
        """
        Internal method to test the model. If the model was not trained before this method is called,
        then the model will be loaded from the input directory provided by the user. Otherwise, the model
        instantiated during the training phase will be tested.
        """

        if not self.model: #load the model if it wasn't created during the training phase
            model_dir = os.path.join(self.inputs, "models")
            print("Testing model without training. Loading model from {}".format(model_dir))
            model_path = "" 
            for path in os.listdir(model_dir):
                if self.model_type in path and path.endswith(".h5"):
                    model_path = os.path.join(model_dir, path)
                    break

            if model_path == "":
                raise FileNotFoundError("Could not locate model file in {}-- have you trained the model yet?".format(model_dir))

            self.model = models.load_model(model_path)

            test_dir = os.path.join(self.inputs, "test.csv")
            self.test_set = pd.read_csv(test_dir) #load the testing data csv

        else: #test the model created during training
            print("Testing model after training.")

        test_generator = self.processor.test_generator(self.test_set)
        
        pred = self.model.predict_generator(test_generator, len(self.test_set))
        loss = self.model.evaluate_generator(test_generator, len(self.test_set))[0]
        
        with open(os.path.join(self.outputs, "test.log"), 'w') as log:
            log.write(str(loss)) 

        print(loss)
        print(pred) 

    
    def __cross_val(self, nfolds=4):
        pass

    def run(self):
        """
        a general method that computes the 'procedure' to follow based on the
        preferences passed in the constructor and runs that procedure
        """
        
        if self.train:
            self.__train_model()
        if self.test:
            self.__test_model()

    
    def __choose_model(self):
        """
        choose a model based on preferences
        """

        if self.model_type == "C3D": 
            return C3D(self.input_shape, self.output_shape)
        
        if self.model_type == "3D-CNN":
            return CNN_3D(self.input_shape, self.output_shape)
        
        if self.model_type == "CNN_3D_small":
            return CNN_3D_small(self.input_shape, self.output_shape)
        
        raise ValueError("Model type does not exist: {}".format(self.model_type))


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
                    if i % 2 == 0:
                        s += 1
                    if s == len(subjects):
                        s = 0

