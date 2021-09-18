
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, applications, optimizers, losses, metrics
from keras.callbacks import *
import pandas
import matplotlib.pyplot as plt


class MalwareDetectionModels:
    TRAINING_DATA_DIRECTORY = '/content/data/dataset/training_set'
    TEST_DATA_DIRECTORY = '/content/data/dataset/test_set'
    COLOR_MODE = 'grayscale'
    IMAGE_HEIGHT = 256
    IMAGE_WIDTH = 256
    BATCH_SIZE = 10
    SEED = 1337

    SEED = 1337

    def __init__(self, img_w, img_h, model_name,training_set, validation_set):
        self.image_width = img_w
        self.image_height = img_h
        self.model = None
        if model_name == "M1":
            self.model = self.bulid_m1()
            self.epochs = 50
            self.rate = 0.000075
        elif model_name == "M2":
            self.model = self.bulid_m2()
            self.epochs = 50
            self.rate = 0.00001
        elif model_name == "M3":
            '''  In M-3, initially, each input image has to go through three convolution layers of 32 neurons each.
             Then it has to go through a max pooling layer and finally through a fully connected layer of 16384 neurons. 
             It executed for 200 epochs, with a batch size of 32, and a learning rate of 0.0001.'''
            self.model = self.bulid_m3()
            self.epochs = 200
            self.batch_size = 32
            self.rate = 0.01
        elif model_name == "M4":
            self.model = self.bulid_m4()
            self.epochs = 120
            self.rate = 0.01
        elif model_name == "M5":
            # M-5 (convolution followed by global pool and a fully connected layer) has again poor performances. T
            self.model = self.bulid_m5()
            self.epochs = 50
            self.rate = 0.01
        elif model_name == "M6":
            self.model = self.bulid_m6()
            self.epochs = 50
            self.rate = 0.01
        elif model_name == "M7":
            self.model = self.bulid_m7()
            self.epochs = 250
            self.rate = 0.01
        elif model_name == "M8":
            self.model = self.bulid_m8()
            self.epochs = 200
            self.rate = 0.01
        elif model_name == "M9":
            self.model = self.bulid_m9()
            self.epochs = 30
            self.rate = 0.0001
        elif model_name == "M10":
            '''  Here, in the case of M-10 (VGG3 with dropout) the network topology is VGG3 
            followed by three fully connected layers with 1024, 2048 and 4096 respectively. 
            Hyperparameters are leraning rate of 0.0001 and batch size of 64.'''
            self.model = self.bulid_m10()
            self.epochs = 30
            self.rate = 0.0001
        elif model_name == "M11":
            '''
            Filter size of convolution layer used is 3×3 for all the models.
      
            Its architecture is composed of VGG3 followed by three fully connected layers of 4096 neurons each.
             In this model dropout rates of 0.2, 0.2, 0.3, and batch normalization were included between 
             the convolutionpooling layers of VGG3. In addition, same dropout rates and batch normalization 
             were added in between fully connected layers.
      
            for M-11 (VGG3 with dropout and batch normalization) were 30 epochs, batch-size of 64 and a learning
            rate of 0.0001 on Dataset-I while the number of epochs were increased to 200 for M-11 (VGG3 with dropout and batch normalization) on Dataset-II
            '''
            '''model M-11 (VGG3 with dropout and batch normalization) reports high performance 
            measures with a rise of 0.89% and 1.83% in F-measure compared to the model M-11 
            (VGG3 with dropout and batch normalization) generated using RGB images of Dataset-I 
            and Dataset-II respectively. The model M-11 (VGG3 with dropout and batch normalization) 
            was executed for 50 epochs with dropout rates 0.1, 0.2, 0.3 and learning rate 0.0001 on Dataset-I. For Dataset-II the dropouts were 0.1, 0.2, 0.3 with a learning rate of 0.000007. However, the epochs remained the same.'''
            self.batch_size = 64
            self.model = self.bulid_m11()
            self.epochs = 5
            self.rate = 0.0001
        elif model_name == "M12":
            ''' ResNet-50 along with three fully connected layers of 1024, 512 and 256 neurons.'''
            self.model = self.bulid_m12()
            self.epochs = 30
            self.rate = 0.0001

        self.define_callbacks()
        self.history = self.train_bb_detector(training_set, validation_set)
        # self.evaluate_bb_detector()
        # self.plot_history()

    def bulid_m1(self):
        m1 = models.Sequential()
        # m1.add(layers.experimental.preprocessing.Rescaling(1./255, input_shape=(self.image_width, self.image_height, 3)))
        # Conv + Conv + Pool + Dense +Dense
        m1.add(layers.Conv2D(16, 3, padding='same', activation='relu',
                             input_shape=(self.image_width, self.image_height, 1)))
        m1.add(layers.Conv2D(32, 3, padding='same', activation='relu'))
        m1.add(layers.MaxPooling2D(pool_size=2))
        m1.add(layers.Flatten())  # if it is not flattened, only the last layer can be fed into the dense
        m1.add(layers.Dense(1024, activation='relu'))
        m1.add(layers.Dense(2, activation='relu'))
        return m1

    def bulid_m2(self):
        m2 = models.Sequential()
        # m2.add(layers.experimental.preprocessing.Rescaling(1./255, input_shape=(self.image_width, self.image_height, 3)))
        # Conv + Pool + Dense + Dense
        m2.add(layers.Conv2D(16, 3, padding='same', activation='relu',
                             input_shape=(self.image_width, self.image_height, 1)))
        m2.add(layers.MaxPooling2D(pool_size=2))
        # m2.add(layers.Flatten())#if it is not flattened, only the last layer can be fed into the dense
        m2.add(layers.Dense(128, activation='relu'))
        m2.add(layers.Dense(128, activation='relu'))
        return m2

    def bulid_m3(self):
        m3 = models.Sequential()
        # m3.add(layers.experimental.preprocessing.Rescaling(1./255, input_shape=(self.image_width, self.image_height, 3))) 
        # Conv + Conv + Conv + Pool + Dense + Dense
        m3.add(layers.Conv2D(16, 3, padding='same', activation='relu',
                             input_shape=(self.image_width, self.image_height, 1)))
        m3.add(layers.Conv2D(16, 3, padding='same', activation='relu'))
        m3.add(layers.Conv2D(16, 3, padding='same', activation='relu'))
        m3.add(layers.MaxPooling2D(pool_size=2))
        # m3.add(layers.Flatten())#if it is not flattened, only the last layer can be fed into the dense
        m3.add(layers.Dense(128, activation='relu'))
        m3.add(layers.Dense(128, activation='relu'))
        return m3

    def bulid_m4(self):
        m4 = models.Sequential()
        # m4.add(layers.experimental.preprocessing.Rescaling(1./255, input_shape=(self.image_width, self.image_height, 3)))
        # onv + Pool + Conv + Dense + Dense
        m4.add(layers.Conv2D(16, 3, padding='same', activation='relu',
                             input_shape=(self.image_width, self.image_height, 3)))
        m4.add(layers.MaxPooling2D(pool_size=2))
        m4.add(layers.Conv2D(16, 3, padding='same', activation='relu'))
        # m4.add(layers.Flatten())#if it is not flattened, only the last layer can be fed into the dense
        m4.add(layers.Dense(128, activation='relu'))
        m4.add(layers.Dense(128, activation='relu'))
        return m4

    def bulid_m5(self):
        m5 = models.Sequential()
        # m5.add(layers.experimental.preprocessing.Rescaling(1./255, input_shape=(self.image_width, self.image_height, 3)))
        # Conv + Globalpool + Dense + Dense
        m5.add(layers.Conv2D(16, 3, padding='same', activation='relu',
                             input_shape=(self.image_width, self.image_height, 3)))
        m5.add(layers.GlobalMaxPooling2D(data_format=None, keepdims=False))
        # m5.add(layers.Flatten())#if it is not flattened, only the last layer can be fed into the dense
        m5.add(layers.Dense(128, activation='relu'))
        m5.add(layers.Dense(128, activation='relu'))
        return m5

    def bulid_m6(self):
        m6 = models.Sequential()
        # m6.add(layers.experimental.preprocessing.Rescaling(1./255, input_shape=(self.image_width, self.image_height, 3)))
        # Conv + Conv + Globalpool + Dense + Dense   
        m6.add(layers.Conv2D(16, 3, padding='same', activation='relu',
                             input_shape=(self.image_width, self.image_height, 3)))
        m6.add(layers.Conv2D(16, 3, padding='same', activation='relu'))
        m6.add(layers.GlobalMaxPooling2D(data_format=None, keepdims=False))
        # m6.add(layers.Flatten())#if it is not flattened, only the last layer can be fed into the dense
        m6.add(layers.Dense(128, activation='relu'))
        m6.add(layers.Dense(128, activation='sigmoid'))
        return m6

    def bulid_m7(self):
        m7 = models.Sequential()
        # m7.add(layers.experimental.preprocessing.Rescaling(1./255, input_shape=(self.image_width, self.image_height, 3)))
        # Conv + Pool + Conv + Pool + Dense + Dens
        m7.add(layers.Conv2D(16, 3, padding='same', activation='relu',
                             input_shape=(self.image_width, self.image_height, 3)))
        m7.add(layers.MaxPooling2D())
        m7.add(layers.Conv2D(16, 3, padding='same', activation='relu'))
        m7.add(layers.MaxPooling2D())
        m7.add(layers.Dense(128, activation='relu'))
        m7.add(layers.Dense(128, activation='sigmoid'))
        return m7

    def bulid_m8(self):
        m8 = models.Sequential()
        # m8.add(layers.experimental.preprocessing.Rescaling(1./255, input_shape=(self.image_width, self.image_height, 3)))
        # Conv + Pool + Conv + Pool + Conv + Pool + Dense + Dense
        m8.add(layers.Conv2D(16, 3, padding='same', activation='relu',
                             input_shape=(self.image_width, self.image_height, 3)))
        m8.add(layers.MaxPooling2D())
        m8.add(layers.Conv2D(16, 3, padding='same', activation='relu'))
        m8.add(layers.MaxPooling2D())
        m8.add(layers.Conv2D(16, 3, padding='same', activation='relu'))
        m8.add(layers.MaxPooling2D())
        m8.add(layers.Dense(128, activation='relu'))
        m8.add(layers.Dense(128, activation='sigmoid'))
        return m8

    def bulid_m9(self):
        m9 = models.Sequential()
        # m9.add(layers.experimental.preprocessing.Rescaling(1./255, input_shape=(self.image_width, self.image_height, 3)))
        m9.add(layers.Conv2D(16, 3, padding='same', activation='relu',
                             input_shape=(self.image_width, self.image_height, 3)))
        m9.add(layers.Conv2D(16, 3, padding='same', activation='relu'))
        m9.add(layers.MaxPooling2D())
        m9.add(layers.Conv2D(16, 3, padding='same', activation='relu'))
        m9.add(layers.Conv2D(16, 3, padding='same', activation='relu'))
        m9.add(layers.MaxPooling2D())
        m9.add(layers.Conv2D(16, 3, padding='same', activation='relu'))
        m9.add(layers.Conv2D(16, 3, padding='same', activation='relu'))
        m9.add(layers.MaxPooling2D())
        m9.add(layers.Dense(128, activation='relu'))
        m9.add(layers.Dense(128, activation='relu'))
        m9.add(layers.Dense(128, activation='relu'))
        m9.add(layers.Dense(128, activation='sigmoid'))
        return m9

    def bulid_m10(self):
        m10 = models.Sequential()
        # m10.add(layers.experimental.preprocessing.Rescaling(1./255, input_shape=(self.image_width, self.image_height, 3)))
        m10.add(layers.Conv2D(16, 3, padding='same', activation='relu',
                              input_shape=(256,256,1)))
        m10.add(layers.Conv2D(16, 3, padding='same', activation='relu'))
        m10.add(layers.MaxPooling2D())
        m10.add(layers.Dropout(.2))
        m10.add(layers.Conv2D(16, 3, padding='same', activation='relu'))
        m10.add(layers.Conv2D(16, 3, padding='same', activation='relu'))
        m10.add(layers.MaxPooling2D())
        m10.add(layers.Dropout(.2))
        m10.add(layers.Conv2D(16, 3, padding='same', activation='relu'))
        m10.add(layers.Conv2D(16, 3, padding='same', activation='relu'))
        m10.add(layers.MaxPooling2D())
        m10.add(layers.Dropout(.2))
        m10.add(layers.Dense(128, activation='relu'))
        m10.add(layers.Dropout(.2))
        m10.add(layers.Dense(128, activation='relu'))
        m10.add(layers.Dropout(.2))
        m10.add(layers.Dense(128, activation='relu'))
        m10.add(layers.Dropout(.2))
        m10.add(layers.Dense(128, activation='sigmoid'))
        return m10

    def bulid_m11(self):
        ''' the dropout rates of 0.2, 0.3, 0.4 and batch normalization assigned between the convolution-pooling layers of VGG3, also the same added between the three fully connected layers of 4096 neurons each '''
        m11 = models.Sequential()
        #m11.add(layers.experimental.preprocessing.Rescaling(1./255, input_shape=(self.image_width, self.image_height, 1)))

        m11.add(layers.Conv2D(32, 3, padding='same', activation='relu',input_shape=(self.image_width, self.image_height, 1)))
        m11.add(layers.BatchNormalization())

        m11.add(layers.Conv2D(32, 3, padding='same', activation='relu'))
        m11.add(layers.BatchNormalization())
        m11.add(layers.MaxPooling2D())
        m11.add(layers.Dropout(.2))

        m11.add(layers.Conv2D(32, 3, padding='same', activation='relu'))
        m11.add(layers.BatchNormalization())
        m11.add(layers.Conv2D(32, 3, padding='same', activation='relu'))
        m11.add(layers.BatchNormalization())
        m11.add(layers.MaxPooling2D())
        m11.add(layers.Dropout(.3, input_shape=(2,)))

        m11.add(layers.Conv2D(32, 3, padding='same', activation='relu'))
        m11.add(layers.BatchNormalization())
        m11.add(layers.Conv2D(32, 3, padding='same', activation='relu'))
        m11.add(layers.BatchNormalization())
        m11.add(layers.MaxPooling2D())
        m11.add(layers.Dropout(.4, input_shape=(2,)))

        m11.add(layers.Flatten())
        m11.add(layers.Dense(4096, activation='relu'))
        m11.add(layers.BatchNormalization())
        m11.add(layers.Dropout(.2))
        m11.add(layers.Dense(4096, activation='relu'))
        m11.add(layers.BatchNormalization())
        m11.add(layers.Dropout(.3, input_shape=(2,)))
        m11.add(layers.BatchNormalization())
        m11.add(layers.Dense(4096, activation='relu'))
        m11.add(layers.BatchNormalization())
        m11.add(layers.Dropout(.4, input_shape=(2,)))
        m11.add(layers.Dense(2, activation='sigmoid'))
        return m11

    def bulid_m12(self):
        m12 = models.Sequential([
            # layers.experimental.preprocessing.Rescaling(1./255, input_shape=(self.image_width, self.image_height, 3)),
            applications.resnet50.ResNet50(include_top=True, weights='imagenet', input_tensor=None, input_shape=None,
                                           pooling=None, classes=1000),
            layers.Dense(128, activation='relu'),
            layers.Dense(1024, activation='relu'),
            layers.Dense(512, activation='relu'),
            layers.Dense(256, activation='sigmoid'),
        ])
        return m12

    def define_callbacks(self):
        checkpoint_cb = ModelCheckpoint("/content/Model.h5", save_freq="epoch", save_best_only=True)
        early_stopping_cb = EarlyStopping(patience=10, restore_best_weights=True)
        logger_cb = CSVLogger('training.log', separator="|")
        return [checkpoint_cb, early_stopping_cb, logger_cb]

    def train_bb_detector(self,training_set,validation_set):
      
        #uploader = UploadData()
        #training_set, validation_set = uploader.upload_set()
        # binary crossentropy or sparse categorical???
      self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
      history = self.model.fit(training_set, epochs=self.epochs, validation_data=validation_set,
                                 callbacks=self.define_callbacks())
      return history

    def plot_history(self):
      data_frame = pandas.DataFrame(self.history.history)
      data_frame.plot(figsize=(7, 3))
      plt.xlabel('Epochs')
      plt.ylabel('Sparse categorical cross-entropy')

    def evaluate_bb_detector(self):
      #uploader = UploadData()
      #test_set = uploader.upload_test_set('/content/dataseett/dataset/test_set')
      validation_set = keras.preprocessing.image_dataset_from_directory(
          self.TEST_DATA_DIRECTORY,
          labels='inferred', 
          label_mode='int',
          class_names=None,
          color_mode=self.COLOR_MODE,
          seed=self.SEED,
          interpolation="area",
          batch_size=self.BATCH_SIZE,
          image_size=(self.IMAGE_HEIGHT, self.IMAGE_WIDTH),
          shuffle=True,
      )
      test_loss, test_acc = self.model.evaluate(test_set, verbose=2)
      print('\nTest accuracy:', test_acc)
