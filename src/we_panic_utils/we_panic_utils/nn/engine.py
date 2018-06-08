# intra-library imports
from .data_load import ttswcvs3
from .models import C3D, CNN_LSTM, CNN_3D, CNN_3D_small

# inter-library imports
from keras import models
from keras.callbacks import CSVLogger, ModelCheckpoint, Callback
from keras import backend as K
import os
import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np

class Engine():
    """
    The engine for training/testing a model

    args:
        regular_data - data not generated
        augmented_data - data generated using halve/doubling speed augmentation
        filtered_csv - the csv containing all of the stats for every sample
        partition_csv - the csv that maps individual partititions to their respective labels
        batch_size - the batch size
        epochs - number of epochs to train
        train - boolean stating whether or not to train
        test - boolean stating whether or not to test
        frameproc - FrameProcessor object for augmentation
        ignore_augmented - list containing phases of running the model in which to ignore augmented data
        input_shape - shape of the sequence passed, 60 separate 100x100x3 frames
        output_shape - the number of outputs
    """
    def __init__(self, 
                 data,
                 model_type, 
                 filtered_csv, 
                 batch_size, 
                 epochs, 
                 train, 
                 load,
                 test, 
                 inputs, 
                 outputs,
                 frameproc,
                 steps_per_epoch=100,
                 ignore_augmented=[""], 
                 input_shape=(60, 100, 100, 3),
                 output_shape=1):

        self.data = data
        self.model_type = model_type
        self.metadata = filtered_csv
        self.batch_size = batch_size
        self.epochs = epochs
        self.train = train
        self.load = load
        self.test = test
        self.inputs = inputs
        self.outputs = outputs
        self.ignore_augmented = ignore_augmented 
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.processor = frameproc
        self.steps_per_epoch = steps_per_epoch

        
    def run(self):

        """
        a general method that computes the 'procedure' to follow based on the
        preferences passed in the constructor and runs that procedure
        """
        model = self.__choose_model().instantiate()   
        train_set = test_set = val_set = None

        if self.train and not self.load:
            print("Training the model.")
            train_set, test_set, val_set = ttswcvs3(self.data, self.metadata, self.outputs)

            train_generator = self.processor.train_generator(train_set)
            val_generator = self.processor.test_generator(val_set)
            test_generator = self.processor.test_generator(test_set)
            gen_type = 'regular'

            csv_logger = CSVLogger(os.path.join(self.outputs, "training.log"))
            checkpointer = ModelCheckpoint(filepath=os.path.join(self.outputs, 'models', self.model_type + '.h5'), 
                                           verbose=1, 
                                           save_best_only=True)

                
            test_results_file = os.path.join(self.outputs, "test_results.log")
            train_results = None
            train_callback = None
            if True:
                train_results = os.path.join(self.outputs, "unnormalized_training.log")
                train_callback = TestResultsCallback(self.processor, train_set, 
                                                     train_results, self.batch_size,
                                                     gen_type, epochs=1)

            test_callback = TestResultsCallback(self.processor,
                                                test_set,
                                                test_results_file,
                                                self.batch_size,
                                                gen_type)
            
            callbacks = [csv_logger, checkpointer, test_callback]    

            if train_callback:
                callbacks.append(train_callback)

            model.fit_generator(generator=train_generator,
                                steps_per_epoch=self.steps_per_epoch,
                                epochs=self.epochs,
                                verbose=1,
                                callbacks=callbacks,
                                validation_data=val_generator,
                                validation_steps=len(val_set), workers=4)
        
        if self.test:

            # if the test set doesn't exist yet, it means we are testing without training

            if test_set is None:
                print("Testing model without training.")
                model_dir = os.path.join(self.inputs, "models")
                model_path = "" 
                for path in os.listdir(model_dir):
                    if self.model_type in path and path.endswith(".h5"):
                        model_path = os.path.join(model_dir, path)
                        break

                if model_path == "":
                    raise FileNotFoundError("Could not locate model file in {}-- have you trained the model yet?".format(model_dir))
                
                model = models.load_model(model_path)
                
                test_dir = os.path.join(self.inputs, "test.csv")
                
                test_set = pd.read_csv(test_dir)

                test_generator = self.processor.test_generator(test_set)

                pred = model.predict_generator(test_generator, len(test_set))

                if self.processor.scaler:
                    pred = self.processor.scaler.inverse_transform(pred)
                    hr = list(test_set['Heart Rate'])
                    loss = mean_squared_error(np.reshape([i for t in zip(hr,hr) for i in t], (-1, 1)), pred)
                else:
                    loss = model.evaluate_generator(test_generator, len(test_set))[0]
                
                with open(os.path.join(self.outputs, "test.log"), 'w') as log:
                    log.write(str(loss)) 

                print(loss)
                print(pred) 

            # otherwise, we can use the existing test set that was generated during the training phase
            else:
                print("Testing model after training.")
                pred = model.predict_generator(test_generator, len(test_set))
                
                loss = model.evaluate_generator(test_generator, len(test_set))[0]
                
                with open(os.path.join(self.outputs, "test.log"), 'w') as log:
                    log.write(str(loss)) 

                print(loss)
                print(pred) 

    
    def __choose_model(self):
        """
        choose a model based on preferences
        """
        norm=False
        if self.processor.scaler:
            norm = True

        if self.model_type == "C3D": 
            return C3D(self.input_shape, self.output_shape)
        
        if self.model_type == "CNN+LSTM":
            return CNN_LSTM(self.input_shape, self.output_shape)

        if self.model_type == "3D-CNN":
            return CNN_3D(self.input_shape, self.output_shape, norm=norm)
        
        if self.model_type == "CNN_3D_small":
            return CNN_3D_small(self.input_shape, self.output_shape)
        
        raise ValueError("Model type does not exist: {}".format(self.model_type))


##This needs a touch up
class TestResultsCallback(Callback):
    """
    a callback for testing the model at certain timesteps
    and viewing its actual output
    """
    def __init__(self, test_gen, test_set, log_file, batch_size, gen_type, epochs=5):
        self.test_gen = test_gen
        self.test_set = test_set
        self.log_file = log_file
        self.batch_size = batch_size
        self.gen_type = gen_type
        self.epochs=epochs

    def on_epoch_end(self, epoch, logs):

        #get the actual mse
        if (epoch+1) % self.epochs == 0:
            print('Logging tests at epoch', epoch)
            with open(self.log_file, 'a') as log:
                gen = None
                if self.gen_type == 'alt_opt_flow':
                    gen = self.test_gen.test_generator_alt_optical_flow(self.test_set)
                elif self.gen_type == 'opt_flow':
                    gen = self.test_gen.test_generator_optical_flow(self.test_set)
                elif self.gen_type == 'regular':
                    gen = self.test_gen.test_generator(self.test_set)
                else:
                    raise ValueError("{} is not a valid generator type".format(self.gen_type))
                
                print('Gen type {}'.format(self.gen_type))
                pred = self.model.predict_generator(gen, len(self.test_set))
                
                if self.test_gen.scaler:
                    pred = self.test_gen.scaler.inverse_transform(pred)

                subjects = list(self.test_set['Subject'])
                trial = list(self.test_set['Trial'])
                hr = list(self.test_set['Heart Rate'])
                i = 0
                s = 0
                error = mean_squared_error(np.reshape([i for t in zip(hr,hr) for i in t], (-1, 1)), pred)
                log.write("Epoch: " + str(epoch+1) + ', Error: ' + str(error) + '\n')
                for p in pred:
                    subj = subjects[s]
                    tri = trial[s]
                    h = hr[s]
                    
                    #val = p[0]
                    log.write(str(subj) + ', ' + str(tri) + '| prediction=' + str(p) + ', actual=' + str(h) + '\n')
                    i+=1
                    if i % 2 == 0:
                        s += 1
                    if s == len(subjects):
                        s = 0

