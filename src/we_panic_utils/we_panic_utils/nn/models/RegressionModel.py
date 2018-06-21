from keras.layers import Dense, Flatten, Dropout, ZeroPadding3D, Input
from keras.models import Sequential  # load_model
from keras.optimizers import Adam,  RMSprop, SGD
from keras.layers.convolutional import Conv3D, MaxPooling3D
from keras.layers import BatchNormalization
from keras.models import Model

class RegressionModel():
    
    def __init__(self, input_shape, output_shape, loss='mean_squared_error'):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.loss = loss
    
    def instantiate(self):
        model = self.get_model() 
        optimizer = Adam(lr=1e-5, decay=1e-6)
        metrics = ['mse']
        model.compile(loss=self.loss, optimizer=optimizer, metrics=metrics)
        model.summary()

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
        # batch norm??

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
