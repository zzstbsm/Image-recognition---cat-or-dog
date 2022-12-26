import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd

import sys
import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import keras
from keras import Model
from keras import initializers, regularizers
from keras import losses, optimizers, metrics
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Input
from keras.layers import Flatten, Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator, load_img

from sklearn.model_selection import train_test_split

class network():
    
    def __init__(self,new=True,input_shape=None,filename=""):
        """
        Load a preexisting model or create a new one
        """
        if new and input_shape:
            print("New model")
            # Generate new model and save
            self.model = self.generateModel(input_shape)
            
            self.model.compile(loss="categorical_crossentropy",
                            optimizer=optimizers.Adam(lr=1e-5),
                            metrics=["accuracy"])
                            
            self.save(filename)
            
        elif new and input_shape==None:
            sys.exit("Missing imput shape")
        else:
            print("Loading model from memory")
            # Load saved model from memory
            self.model = keras.models.load_model(filename)
    
    @staticmethod
    def generateModel(input_shape):
        """
        Create the neural network.
        With ReLU activation, the loss does not converge
        """
        
        # model = Sequential()
        
        # # Add layers
        # model.add(Conv2D(32,(5,5),
                            # input_shape=input_shape,
                            # activation="sigmoid"
                            # ))
        # model.add(MaxPooling2D(pool_size=(2,2)))
        # model.add(BatchNormalization())
        
        # model.add(Conv2D(32,(5,5),
                        # activation="sigmoid"
                        # ))
        # model.add(MaxPooling2D(pool_size=(2,2)))
        # model.add(BatchNormalization())
        
        # model.add(Conv2D(32,(5,5),
                        # activation="sigmoid"
                        # ))
        # model.add(MaxPooling2D(pool_size=(2,2)))
        # model.add(BatchNormalization())
        
        # model.add(Flatten())
        # model.add(Dense(300,
                    # activation="sigmoid"
                    # ))
        # model.add(Dropout(.2))
        # model.add(Dense(2,activation="softmax"))
        
        # Input
        input_layer = Input(shape=input_shape)
        
        weight_initializer = initializers.RandomNormal(mean=0,stddev=0.01)
        bias_initializer = initializers.Zeros()
        
        # 1st conv block
        x = Conv2D(filters=64,
                    kernel_size=5,
                    padding="same",
                    activation="relu",
                    kernel_initializer=weight_initializer,
                    kernel_regularizer=regularizers.l2(5e-4),
                    bias_initializer=bias_initializer) (input_layer)
        x = MaxPooling2D(pool_size=2,strides=2) (x)
        # 2nd conv block
        x = Conv2D(filters=64,
                    kernel_size=5,
                    padding="same",
                    activation="relu",
                    kernel_initializer=weight_initializer,
                    kernel_regularizer=regularizers.l2(5e-4),
                    bias_initializer=bias_initializer) (x)
        x = MaxPooling2D(pool_size=2,strides=2) (x)
        
        # Fully connect
        x = Flatten()(x)
        x = Dropout(.5)(x)
        x = Dense(units=512,
                    activation="relu",
                    kernel_initializer=weight_initializer,
                    kernel_regularizer=regularizers.l2(5e-4),
                    bias_initializer=bias_initializer)(x)
        
        # Output
        output_layer = Dense(units=2,
                    activation="softmax")(x)
        
        model = Model(inputs=input_layer,outputs=output_layer)
        
        return model
    
    def save(self,filename):
        """
        Save the current model
        """
        self.model.save(filename)
        return
    
    def load_data(self,folder):
        """
        Prepare the format of the training data
        """
        self.train_folder = folder
        filenames = os.listdir(self.train_folder)
        categories = []
        for name in filenames:
            category = name.split(".")[0]
            if category == "dog":
                categories.append("dog")
            else:
                categories.append("cat")
        
        # Import in dataframe
        df = pd.DataFrame({"filename":filenames,"category":categories})
        
        # Split the dataset in train set and validation set
        train_df,validation_df = train_test_split(df,test_size=.2,shuffle=True,random_state=42)
        train_df = train_df.reset_index(drop=True)
        validation_df = validation_df.reset_index(drop=True)
        
        self.train_df = train_df
        self.validation_df = validation_df
        
        return train_df,validation_df
    
    def set_image_paramenters(self,width,height,channels,batch_size):
        
        self.batch_size = batch_size
        
        self.img_width = width
        self.img_height = height
        self.img_channels = channels
        
        self.img_size = (width,height)

        train_datagen = ImageDataGenerator(rotation_range=15,
                                            rescale=1./255,
                                            shear_range=0.1,
                                            zoom_range=0.2,
                                            horizontal_flip=True,
                                            vertical_flip=True,
                                            width_shift_range=0.1,
                                            height_shift_range=0.1
                                            )
        train_generator = train_datagen.flow_from_dataframe(self.train_df,
                                                            self.train_folder,
                                                            x_col="filename",
                                                            y_col="category",
                                                            target_size=self.img_size,
                                                            class_mode="categorical",
                                                            batch_size=self.batch_size)

        validation_datagen = ImageDataGenerator(rescale=1./255)
        validation_generator = validation_datagen.flow_from_dataframe(self.validation_df,
                                                                        self.train_folder,
                                                                        x_col="filename",
                                                                        y_col="category",
                                                                        target_size=self.img_size,
                                                                        class_mode="categorical",
                                                                        batch_size=self.batch_size)
        
        self.train_generator = train_generator
        self.validation_generator = validation_generator
    
    def train(self,epochs):
        """
        Train model
        """
        n_train = self.train_df.shape[0]
        n_validate = self.validation_df.shape[0]
        self.model.fit_generator(self.train_generator,
                            epochs=epochs,
                            validation_data=self.validation_generator,
                            validation_steps=n_validate//self.batch_size,
                            steps_per_epoch=n_train//self.batch_size
                            #callbacks=[keras.callbacks.EarlyStopping()]
                            )
    def predict(self,x):
    
        return self.model.predict(x)