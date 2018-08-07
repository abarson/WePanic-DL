from keras.layers import Dense, Flatten, Dropout, ZeroPadding3D, Input, concatenate, TimeDistributed, Lambda
from keras.models import Sequential  # load_model
from keras.optimizers import Adam,  RMSprop, SGD
from keras.layers.convolutional import Conv3D, MaxPooling3D
from keras.layers import Activation, BatchNormalization
from keras.models import Model
import numpy as np
import tensorflow as tf

class RegressionModel():
    
    def __init__(self, input_shape, output_shape, loss='mean_squared_error'):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.loss = loss
        self.lr=1e-5
        self.optimizer = None    

    def instantiate(self):
        model = self.get_model() 
        if self.optimizer is None:
            self.optimizer = Adam(lr=self.lr,decay=1e-6)#beta_2=1.0)

        metrics = ['mse', 'mape']
        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=metrics)
        #model.summary()

        return model

    def get_model(self):
        raise NotImplementedError 

class C3D(RegressionModel):
    def __init__(self, input_shape, output_shape, loss=None):
        RegressionModel.__init__(self, input_shape, output_shape, loss=loss)

    def instantiate(self):
        return super(C3D, self).instantiate()

    def get_model(self):
        model = Sequential()
        # 1st layer group
        model.add(Conv3D(64, 3, 3, 3, activation='relu',
                         border_mode='same', name='conv1',
                         subsample=(1, 1, 1),
                         input_shape=self.input_shape))
        model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2),
                               border_mode='valid', name='pool1'))
        # 2nd layer group
        model.add(Conv3D(128, 3, 3, 3, activation='relu',
                         border_mode='same', name='conv2',
                         subsample=(1, 1, 1)))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                               border_mode='valid', name='pool2'))
        # 3rd layer group
        model.add(Conv3D(256, 3, 3, 3, activation='relu',
                         border_mode='same', name='conv3a',
                         subsample=(1, 1, 1)))
        model.add(Conv3D(256, 3, 3, 3, activation='relu',
                         border_mode='same', name='conv3b',
                         subsample=(1, 1, 1)))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                               border_mode='valid', name='pool3'))
        # 4th layer group
        model.add(Conv3D(512, 3, 3, 3, activation='relu',
                         border_mode='same', name='conv4a',
                         subsample=(1, 1, 1)))
        model.add(Conv3D(512, 3, 3, 3, activation='relu',
                         border_mode='same', name='conv4b',
                         subsample=(1, 1, 1)))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                               border_mode='valid', name='pool4'))

        # 5th layer group
        model.add(Conv3D(512, 3, 3, 3, activation='relu',
                         border_mode='same', name='conv5a',
                         subsample=(1, 1, 1)))
        model.add(Conv3D(512, 3, 3, 3, activation='relu',
                         border_mode='same', name='conv5b',
                         subsample=(1, 1, 1)))
        model.add(ZeroPadding3D(padding=(0, 1, 1)))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                               border_mode='valid', name='pool5'))
        model.add(Flatten())

        # FC layers group
        model.add(Dense(4096, activation='relu', name='fc6'))
        model.add(Dropout(0.5))
        model.add(Dense(4096, activation='relu', name='fc7'))
        model.add(Dropout(0.5))
        model.add(Dense(self.output_shape, activation='linear'))

        return model

class BN_CNN_3D_DO(RegressionModel):
   
    def __init__(self, input_shape, output_shape, loss=None):
        RegressionModel.__init__(self, input_shape, output_shape, loss=loss)

    def instantiate(self):
        return super(BN_CNN_3D_DO, self).instantiate()

    def get_model(self):
       
        invariants = {'kernel_initializer':'he_normal'}
        model = Sequential()
         
        model.add(Conv3D(64, kernel_size=(15, 5, 5), padding='valid',
                  input_shape=self.input_shape, **invariants))
        model.add(Activation('relu'))
        model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), padding='valid'))
        model.add(BatchNormalization()) 

        model.add(Conv3D(128, kernel_size=5, padding='valid', **invariants))
        model.add(Activation('relu'))
        model.add(MaxPooling3D(pool_size=2, strides=(1, 2, 2), padding='valid'))
        model.add(BatchNormalization()) 

        model.add(Conv3D(256, kernel_size=3, padding='valid', **invariants))
        model.add(Activation('relu'))
        model.add(MaxPooling3D(pool_size=2, strides=(1, 2, 2), padding='valid'))
        model.add(BatchNormalization()) 

        #model.add(Conv3D(256, kernel_size=3))
        #model.add(Activation('relu'))
        #model.add(BatchNormalization()) 
        #model.add(MaxPooling3D(pool_size=2, strides=(2, 2, 2)))

        #model.add(Conv3D(512, kernel_size=1))
        #model.add(Activation('relu'))
        #model.add(BatchNormalization()) 
        #model.add(MaxPooling3D(pool_size=1, strides=1))

        model.add(Flatten()) 
        model.add(Dense(512, activation='relu', **invariants))
        model.add(Dropout(0.15))
        model.add(Dense(512, activation='relu', **invariants))
        model.add(Dropout(0.15))
        model.add(Dense(self.output_shape, activation='linear'))
        
        return model

class DualNet(RegressionModel):
   
    def __init__(self, input_shape, output_shape, loss=None):
        RegressionModel.__init__(self, input_shape, output_shape, loss=loss)

    def instantiate(self):
        return super(DualNet, self).instantiate()

    def get_model(self):
       
        invariants = {'kernel_initializer':'he_normal'}
        #model = Sequential()
        seq = Input(shape=self.input_shape)
        grey = TimeDistributed(Lambda(lambda x: tf.image.rgb_to_grayscale(x)))(seq)
        red = Lambda(lambda x: x[:,:,:,:,0])(seq) 
        red = Lambda(lambda x: x[:,:,:,:,None])(red)
        
        x1 = Conv3D(64, kernel_size=(15,5,5),padding='valid', **invariants)(grey)
        x1 = Activation('relu')(x1)
        x1 = MaxPooling3D(pool_size=(1,2,2),strides=(1,2,2),padding='valid')(x1)
        x1 = BatchNormalization()(x1)
        x1 = Conv3D(128, kernel_size=5,padding='valid', **invariants)(x1)
        x1 = Activation('relu')(x1)
        x1 = MaxPooling3D(pool_size=(1,2,2),strides=(1,2,2),padding='valid')(x1)
        x1 = BatchNormalization()(x1)

        x2 = Conv3D(64, kernel_size=(15,5,5),padding='valid', **invariants)(red)
        x2 = Activation('relu')(x2)
        x2 = MaxPooling3D(pool_size=(1,2,2),strides=(1,2,2),padding='valid')(x2)
        x2 = BatchNormalization()(x2)
        x2 = Conv3D(128, kernel_size=5,padding='valid', **invariants)(x2)
        x2 = Activation('relu')(x2)
        x2 = MaxPooling3D(pool_size=(1,2,2),strides=(1,2,2),padding='valid')(x2)
        x2 = BatchNormalization()(x2)

        x = concatenate([x1, x2])
        x = Conv3D(256, kernel_size=3, padding='valid', **invariants)(x)
        x = Activation('relu')(x)
        x = MaxPooling3D(pool_size=2, strides=(1,2,2),padding='valid')(x)
        x = BatchNormalization()(x)
        x = Flatten()(x)

        x = Dense(512, activation='relu', **invariants)(x)
        x = Dropout(0.15)(x)
        x = Dense(512, activation='relu', **invariants)(x)
        x = Dropout(0.15)(x)

        pred = Dense(self.output_shape, activation='linear')(x)
        model = Model(inputs=seq, outputs=pred) 

        return model

class BN_CNN_3D_DO2(RegressionModel):
   
    def __init__(self, input_shape, output_shape, loss=None):
        RegressionModel.__init__(self, input_shape, output_shape, loss=loss)

    def instantiate(self):
        return super(BN_CNN_3D_DO2, self).instantiate()

    def get_model(self):
       
        invariants = {'kernel_initializer':'he_normal'}
        model = Sequential()
         
        model.add(Conv3D(84, kernel_size=(15, 5, 5), padding='valid',
                  input_shape=self.input_shape, **invariants))
        model.add(Activation('relu'))
        model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), padding='valid'))
        model.add(BatchNormalization()) 

        model.add(Conv3D(150, kernel_size=5, padding='valid', **invariants))
        model.add(Activation('relu'))
        model.add(MaxPooling3D(pool_size=2, strides=(1, 2, 2), padding='valid'))
        model.add(BatchNormalization()) 

        model.add(Conv3D(300, kernel_size=3, padding='valid', **invariants))
        model.add(Activation('relu'))
        model.add(MaxPooling3D(pool_size=2, strides=(1, 2, 2), padding='valid'))
        model.add(BatchNormalization()) 

        #model.add(Conv3D(256, kernel_size=3))
        #model.add(Activation('relu'))
        #model.add(BatchNormalization()) 
        #model.add(MaxPooling3D(pool_size=2, strides=(2, 2, 2)))

        #model.add(Conv3D(512, kernel_size=1))
        #model.add(Activation('relu'))
        #model.add(BatchNormalization()) 
        #model.add(MaxPooling3D(pool_size=1, strides=1))

        model.add(Flatten()) 
        model.add(Dense(512, activation='relu', **invariants))
        model.add(Dropout(0.15))
        model.add(Dense(512, activation='relu', **invariants))
        model.add(Dropout(0.15))
        model.add(Dense(self.output_shape, activation='linear'))
        
        return model

class Deeper(RegressionModel):
   
    def __init__(self, input_shape, output_shape, loss=None):
        RegressionModel.__init__(self, input_shape, output_shape, loss=loss)
        self.lr = 0.01
        #self.optimizer = SGD(lr=self.lr, momentum=0.8, decay=self.lr/30)

    def instantiate(self):
        return super(Deeper, self).instantiate()

    def get_model(self):
       
        invariants = {'kernel_initializer':'he_normal'}
        model = Sequential()
         
        model.add(Conv3D(64, kernel_size=(20, 5, 5), padding='valid',
                  input_shape=self.input_shape, **invariants))
        model.add(Activation('relu'))
        model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), padding='valid'))
        model.add(BatchNormalization()) 

        model.add(Conv3D(128, kernel_size=(15, 3, 3), padding='valid', **invariants))
        model.add(Activation('relu'))
        model.add(MaxPooling3D(pool_size=2, strides=(1, 2, 2), padding='valid'))
        model.add(BatchNormalization()) 

        model.add(Conv3D(256, kernel_size=(10, 3, 3), padding='valid', **invariants))
        model.add(Activation('relu'))
        model.add(MaxPooling3D(pool_size=2, strides=(1, 2, 2), padding='valid'))
        model.add(BatchNormalization()) 

        #model.add(Conv3D(256, kernel_size=3))
        #model.add(Activation('relu'))
        #model.add(BatchNormalization()) 
        #model.add(MaxPooling3D(pool_size=2, strides=(2, 2, 2)))

        #model.add(Conv3D(512, kernel_size=1))
        #model.add(Activation('relu'))
        #model.add(BatchNormalization()) 
        #model.add(MaxPooling3D(pool_size=1, strides=1))

        model.add(Flatten()) 
        model.add(Dense(512, activation='relu', **invariants))
        model.add(Dropout(0.15))
        model.add(Dense(512, activation='relu', **invariants))
        model.add(Dropout(0.15))
        model.add(Dense(self.output_shape, activation='linear'))
        
        return model

class CNN_3D(RegressionModel):
   
    def __init__(self, input_shape, output_shape, loss=None):
        RegressionModel.__init__(self, input_shape, output_shape, loss=loss)

    def instantiate(self):
        return super(CNN_3D, self).instantiate()

    def get_model(self):
       
        model = Sequential()
        
        model.add(Conv3D(64, kernel_size=(3, 3, 3), 
                  input_shape=self.input_shape, activation='relu'))
        model.add(MaxPooling3D(pool_size=2, strides=(1, 2, 2)))

        model.add(Conv3D(64, kernel_size=(3, 2, 2), 
                  input_shape=self.input_shape, activation='relu'))
        
        model.add(Conv3D(128, kernel_size=(3, 2, 2), 
                  activation='relu'))
        model.add(MaxPooling3D(pool_size=2, strides=(1, 2, 2)))

        model.add(Conv3D(128, kernel_size=(3, 2, 2), 
                  activation='relu'))
        
        model.add(MaxPooling3D(pool_size=2, strides=2))
        model.add(Dropout(0.15)) 
        model.add(Conv3D(256, kernel_size=(3, 2, 2), 
                  activation='relu')) 
        model.add(BatchNormalization()) 

        model.add(Flatten()) 
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.15))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.15))
        model.add(Dense(self.output_shape, activation='linear'))
        
        return model


class CNN_3D_small(RegressionModel):
   
    def __init__(self, input_shape, output_shape, loss=None):
        RegressionModel.__init__(self, input_shape, output_shape, loss=loss)

    def instantiate(self):
        return super(CNN_3D_small, self).instantiate()
    
    def get_model(self):
        model = Sequential()
        model.add(Conv3D(32, kernel_size=(3, 3, 3),
                  input_shape=self.input_shape, activation='relu'))
        model.add(Conv3D(32, kernel_size=(3, 3, 3),
                  input_shape=self.input_shape, activation='relu'))
        model.add(Conv3D(64, kernel_size=(3, 3, 3),
                  input_shape=self.input_shape, activation='relu'))
        model.add(MaxPooling3D(pool_size=2, strides=(1, 2, 2)))
        model.add(Conv3D(64, kernel_size=(3, 3, 3),
                  input_shape=self.input_shape, activation='relu'))
        model.add(MaxPooling3D(pool_size=2, strides=(1, 2, 2)))
        model.add(Conv3D(128, kernel_size=(3, 3, 3),
                  activation='relu'))  
        model.add(MaxPooling3D(pool_size=2, strides=2)) 
        model.add(Conv3D(128, kernel_size=(3, 3, 3),
                  activation='relu'))
        model.add(MaxPooling3D(pool_size=2, strides=2)) 

        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.output_shape, activation='linear'))
        return model

class ShallowC3D(RegressionModel):
    def __init__(self, input_shape, output_shape, loss=None):
        RegressionModel.__init__(self, input_shape, output_shape, loss=loss)

    def instantiate(self):
        return super(ShallowC3D, self).instantiate()
    
    def get_model(self):
        model = Sequential()
        # 1st layer group
        model.add(Conv3D(64, 3, 3, 3, activation='relu',
                         border_mode='same', name='conv1',
                         subsample=(1, 1, 1),
                         input_shape=self.input_shape))
        model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2),
                               border_mode='valid', name='pool1'))
        # 2nd layer group
        model.add(Conv3D(128, 3, 3, 3, activation='relu',
                         border_mode='same', name='conv2',
                         subsample=(1, 1, 1)))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                               border_mode='valid', name='pool2'))
        # 3rd layer group
        #model.add(Conv3D(256, 3, 3, 3, activation='relu',
        #                 border_mode='same', name='conv3a',
        #                 subsample=(1, 1, 1)))
        model.add(Conv3D(256, 3, 3, 3, activation='relu',
                         border_mode='same', name='conv3b',
                         subsample=(1, 1, 1)))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                               border_mode='valid', name='pool3'))
        model.add(Flatten())

        # FC layers group
        model.add(Dense(512, activation='relu', name='fc7'))
        model.add(Dropout(0.5))
        model.add(Dense(self.output_shape, activation='linear'))

        return model
