
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import layers, models, applications, optimizers, losses, metrics
from tensorflow.keras.callbacks import *
import pandas
import matplotlib.pyplot as plt
import ImageProcessing
import os
import cv2
from os.path import join as pjoin
import math

class MalwareDetectionModels:

    def __init__(self, img_w, img_h, model_name,training_set, validation_set,test_set,loaded):
        self.image_width = img_w
        self.image_height = img_h
        self.model = None
        self.loaded = loaded
        if not loaded:
            if model_name == "M11":
                self.batch_size = 64
                self.model = self.bulid_m11()
                self.epochs = 15
                self.rate = 0.00001
            
            self.define_callbacks()
            self.history = self.train_bb_detector(training_set, validation_set)
            self.evaluate_bb_detector(test_set)
            #self.plot_history()
        else:
            self.model = tf.keras.models.load_model('/home/qian/Masterproject/PROJECT/saved_model/model_test.h5')


    
    def bulid_m11(self):
        m11 = models.Sequential()

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
        m11.add(layers.Dense(1, activation='sigmoid'))
        return m11


    def define_callbacks(self):

        checkpoint_cb = ModelCheckpoint("/home/qian/Masterproject/PROJECT/saved_model/model_test.h5", save_best_only=True)
        early_stopping_cb = EarlyStopping(patience=10, restore_best_weights=True)
        logger_cb = CSVLogger('training.log', separator="|")
        return [checkpoint_cb, early_stopping_cb, logger_cb]

    def train_bb_detector(self,training_set,validation_set):
      
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'],run_eagerly = True)
        history = self.model.fit(training_set, epochs=self.epochs, validation_data=validation_set,
                                 callbacks=self.define_callbacks())
        self.model.save('/home/qian/Masterproject/PROJECT/saved_model/model_test.h5')
        return history

    def plot_history(self):
      data_frame = pandas.DataFrame(self.history.history)
      data_frame.plot(figsize=(7, 3))
      plt.xlabel('Epochs')
      plt.ylabel('Sparse categorical cross-entropy')

    def evaluate_bb_detector(self,test_set):
      test_loss, test_acc = self.model.evaluate(test_set, verbose=2)
      print('\nTest accuracy:', test_acc)


    def clear_folder(self,dir):
        for f in os.listdir(dir):
            os.remove(os.path.join(dir, f))
    def process_data(self,iterable,epoch):
        colormap = np.load('/home/qian/Masterproject/dataset/colormap/gray_colormap.npy')
        width = 256
        img_to_predict='/home/qian/Masterproject/dataset/mmid_imgs/'
        self.clear_folder('/home/qian/Masterproject/dataset/mmid_imgs/mid_imgs')

        i =0
        for it in iterable:
            img_bin_array = np.array(it).astype(int)
            #img_bin_array = it.reshape(it.shape[0]*it.shape[1]).astype(int)
            flat = img_bin_array.flatten()
            grayscale_array = ImageProcessing.to1DArray_grayscale(flat,colormap)
            height = math.ceil(len(grayscale_array)/width)
            height1 = math.ceil(len(flat)/width)
            if i<=9:
                ImageProcessing.saveImg('/home/qian/Masterproject/dataset/mmid_imgs/mid_imgs/0'+str(i)+'.png', grayscale_array, (width,height),'L')
                if (epoch%20 == 0) and i<=4:
                    ImageProcessing.saveImg('/home/qian/Masterproject/dataset/noise_mal_comb_exe/comb_mid/'+str(epoch)+'Epoch'+str(i)+'.png', flat, (width,height1),'L')
                    ImageProcessing.saveImg('/home/qian/Masterproject/dataset/mmid_imgs/mid_imgs/'+str(i)+'.png', grayscale_array, (width,height),'L')
            elif i>=10:
                ImageProcessing.saveImg('/home/qian/Masterproject/dataset/mmid_imgs/mid_imgs/'+str(i)+'.png', grayscale_array, (width,height),'L')
            i = i+1
        
        mid_img_set = self.upload_mid_set(img_to_predict)
        for only_batch in  mid_img_set:
            return only_batch  

          

    def upload_mid_set(self, DIRECTORY):
        COLOR_MODE = 'grayscale'
        IMAGE_HEIGHT = 256
        IMAGE_WIDTH = 256
        BATCH_SIZE = 64
        SEED = 1337
        mid_img_set= keras.preprocessing.image_dataset_from_directory(
            DIRECTORY,
            label_mode = None,
            color_mode=COLOR_MODE,
            seed=SEED,
            interpolation="area",
            batch_size=BATCH_SIZE,
            image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
            shuffle=False
        )
        return mid_img_set

    def make_prediction(self,samples,epoch):
 
        self.data = self.process_data(samples,epoch)
        if self.loaded:
            return self.model.predict(self.data)
        else:
            return self.model.model(self.data)
