import tensorflow as tf;
from tensorflow import keras;

from tensorflow.keras.models import *
import tensorflow.keras.layers as layers
import tensorflow.keras.applications as applications
from tensorflow.keras.layers  import *
import numpy as np
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics

tf.__version__

# TRAINING_DATA_DIRECTORY = '/content/dataset/Train'
# TEST_DATA_DIRECTORY = '/content/dataset/Test'
# BATCH_SIZE = 128
# IMAGE_WIDTH = 256
# IMAGE_HEIGHT = 256
# SEED = 1337

class Malware_detection_model():
    def __init__(self,img_w, img_h,model_name):
        self.IMAGE_WIDTH = img_w
        self.IMAGE_HEIGHT = img_h
        self.Model=None
        if model_name=="M1":
            self.Model= self.bulid_m1(img_h,img_w)
            self.epochs = 0.000075
            self.rate =50
        elif model_name=="M2":
            self.Model= self.bulid_m2(img_h,img_w)
            self.epochs = 0.00001
            self.rate =50
        elif model_name=="M3":
            self.Model= self.bulid_m3(img_h,img_w)
            self.epochs = 200
            self.rate =0.01
        elif model_name=="M4":
            self.Model= self.bulid_m4(img_h,img_w)
            self.epochs =120
            self.rate =0.01
        elif model_name=="M5":
            self.Model= self.bulid_m5(img_h,img_w)
            self.epochs =50
            self.rate =0.01
        elif model_name=="M6":
            self.Model= self.bulid_m6(img_h,img_w)
            self.epochs =50
            self.rate =0.01
        elif model_name=="M7":
            self.Model= self.bulid_m7(img_h,img_w)
            self.epochs =250
            self.rate =0.01
        elif model_name=="M8":
            self.Model= self.bulid_m8(img_h,img_w)
            self.epochs =200
            self.rate =0.01
        elif model_name=="M9":
            self.Model= self.bulid_m9(img_h,img_w)
            self.epochs =30
            self.rate =0.0001
        elif model_name=="M10":
            self.Model= self.bulid_m10(img_h,img_w)
            self.epochs =30
            self.rate =0.0001
        elif model_name=="M11":
            self.Model = self.build_m11(img_h,img_w)
            self.epochs =30
            self.rate =0.0001
        elif model_name=="M12":
            self.Model= self.bulid_m12(img_h,img_w)
            self.epochs =30
            self.rate =0.0001
    
    def bulid_m1(self, img_h, img_w):
        return Sequential([
        layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_h, img_w, 3)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),#the number of Parmeter:size*size*dim_of_input*number+bias(number)
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(pool_size=2),
        layers.Flatten(),#if it is not flattened, only the last layer can be fed into the dense
        layers.Dense(1024, activation='relu'),
        layers.Dense(2, activation='relu'),#there are two typs of output, malware and benign, so the dense has 2 neurons
        ])

    def build_m2(self, img_h, img_w):
        return Sequential([
        layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_h, img_w, 3)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        ])


    





# **M-3:** Conv + Conv + Conv + Pool + Dense + Dense **|**
# 
# *Epoch=200, LR=0.01*

# page 11 
# 
# "In M-3, initially,
# each input image has to go through three convolution layers
# of 32 neurons each. Then it has to go through a max pooling layer and finally through a fully connected layer of 16384
# neurons. It executed for 200 epochs, with a batch size of 32,
# and a learning rate of 0.0001"

    def build_m3(self, img_h, img_w):
        return Sequential([
        layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_h, img_w, 3)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        ])

# **M-4** Conv + Pool + Conv + Dense + Dense **|**
# *Epoch=120, LR=0.01*


    def build_m4(self, img_h, img_w):
        return Sequential([
        layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_h, img_w, 3)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        ])

# **M-5** Conv + Globalpool + Dense + Dense **|**
# *Epoch=50, LR=0.01*

    def build_m5(self, img_h, img_w):
        M5 = Sequential([
        layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_h, img_w, 3)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        tf.keras.layers.GlobalMaxPooling2D(data_format=None, keepdims=False),
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        ])

# **M-6** Conv + Conv + Globalpool + Dense + Dense **|**
# 
# *Epoch=50, LR=0.01*

    def build_m6(self, img_h, img_w):
        return Sequential([
        layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_h, img_w, 3)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        tf.keras.layers.GlobalMaxPooling2D(data_format=None, keepdims=False),
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        ])

# **M-7** Conv + Pool + Conv + Pool + Dense + Dense **|**
# 
# *Epoch=250, LR=0.01*
    def build_m7(self, img_h, img_w):
        return Sequential([
        layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_h, img_w, 3)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        ])

# **M-8** Conv + Pool + Conv + Pool + Conv + Pool + Dense + Dense **|**
# 
# *Epoch=200, LR=0.01*

    def build_m8(self, img_h, img_w):
        M8 = Sequential([
        layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_h, img_w, 3)),
        layers.Conv2D(16, 3, padding='same', activation='relu'), 
        layers.MaxPooling2D(),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        ])


# VGG ref: https://arxiv.org/pdf/1409.1556.pdf
# 
# https://machinelearningmastery.com/how-to-implement-major-architecture-innovations-for-convolutional-neural-networks/
# 
# 
# "VGG3 is the visual geometry group and includes three
# stages.[...]Each stage contains two convolution layers and a pooling
# layer. The size of the convolution layer used is 3 × 3 and the
# pooling layer is 1 × 1. Using a 3 × 3 filter allows for expressing more information about the image across all the channels
# while keeping the size of the convolution layer consistent with
# the size of the image. The filter size of the pooling layer is 1×1"
# 
# HOW MANY FILTERS IN A CONV LAYER?
# n STRIDES?

# **M-9** Conv + Conv + Pool + Conv + Conv + Pool + Conv +
# Conv + Pool + Dense + Dense + Dense + Dense **|**
# 
# *Epoch=30, LR=0.0001*
    def build_m9(self, img_h, img_w):
        return Sequential([
        layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_h, img_w, 3)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu')
        ])

#  **M-10** Conv + Conv + Pool + Dropout + Conv +
# Conv + Pool + Dropout + Conv + Conv + Pool + Dropout + Dense +
# Dropout + Dense + Dropout + Dense + Dropout + Dense **|**
# 
# https://keras.io/api/layers/regularization_layers/dropout/
# 
# *Epoch=30, LR=0.0001*
    def build_m10(self, img_h, img_w, dropout=0.2):
        return Sequential([
        layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_h, img_w, 3)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(dropout, input_shape=(2,)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(dropout, input_shape=(2,)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(dropout, input_shape=(2,)),
        layers.Dense(128, activation='relu'),
        layers.Dropout(dropout, input_shape=(2,)),
        layers.Dense(128, activation='relu'),
        layers.Dropout(dropout, input_shape=(2,)),
        layers.Dense(128, activation='relu'),
        layers.Dropout(dropout, input_shape=(2,)),
        layers.Dense(128, activation='relu')
        ])


#  **M-11**  Conv +
# Batchnorm + Conv + Batchnorm + Pool + Dropout + Conv + Batchnorm + Conv + Batchnorm + Pool + Dropout + Conv + Batchnorm +Conv +
# Batchnorm + Pool + Dropout + Dense + Batchnorm + Dropout + Dense + Batchnorm + Dropout+ Dense + Batchnorm + Dropout + Dense **|**
# 
# https://keras.io/api/layers/normalization_layers/batch_normalization/
# 
# *Epoch=30, LR=0.0001*
    def build_m11(self, img_h, img_w,dropout=0.2):
            return Sequential([
            layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_h, img_w, 3)),
            layers.Conv2D(16, 3, padding='same', activation='relu'),
            layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer="zeros", 
                                                gamma_initializer="ones", moving_mean_initializer="zeros", moving_variance_initializer="ones", 
                                                beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None),

            layers.Conv2D(16, 3, padding='same', activation='relu'),
            layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer="zeros", 
                                                gamma_initializer="ones", moving_mean_initializer="zeros", moving_variance_initializer="ones", 
                                                beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None),
            layers.MaxPooling2D(),
            layers.Dropout(dropout, input_shape=(2,)),

            layers.Conv2D(16, 3, padding='same', activation='relu'),
            layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer="zeros", 
                                                gamma_initializer="ones", moving_mean_initializer="zeros", moving_variance_initializer="ones", 
                                                beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None),
            layers.Conv2D(16, 3, padding='same', activation='relu'),
            layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer="zeros", 
                                                gamma_initializer="ones", moving_mean_initializer="zeros", moving_variance_initializer="ones", 
                                                beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None),
            layers.MaxPooling2D(),
            layers.Dropout(dropout, input_shape=(2,)),

            layers.Conv2D(16, 3, padding='same', activation='relu'),
            layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer="zeros", 
                                                gamma_initializer="ones", moving_mean_initializer="zeros", moving_variance_initializer="ones", 
                                                beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None),
            layers.Conv2D(16, 3, padding='same', activation='relu'),
            layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer="zeros", 
                                                gamma_initializer="ones", moving_mean_initializer="zeros", moving_variance_initializer="ones", 
                                                beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None),
            layers.MaxPooling2D(),
            layers.Dropout(dropout, input_shape=(2,)),

            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer="zeros", 
                                                gamma_initializer="ones", moving_mean_initializer="zeros", moving_variance_initializer="ones", 
                                                beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None),
            layers.Dropout(dropout, input_shape=(2,)),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer="zeros", 
                                                gamma_initializer="ones", moving_mean_initializer="zeros", moving_variance_initializer="ones", 
                                                beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None),
            layers.Dropout(dropout, input_shape=(2,)),
            layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer="zeros", 
                                                gamma_initializer="ones", moving_mean_initializer="zeros", moving_variance_initializer="ones", 
                                                beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer="zeros", 
                                                gamma_initializer="ones", moving_mean_initializer="zeros", moving_variance_initializer="ones", 
                                                beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None),
            layers.Dropout(dropout, input_shape=(2,)),
            layers.Dense(128, activation='relu')
            ])






# **M-12** ResNet-50 + Dense + Dense + Dense + Dense **|**
# 
# https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet50/ResNet50
# 
# *Epochs=30, LR=0.0001*

 

    def build_m12(self, img_h, img_w):
        return Sequential([
        layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_h, img_w, 3)),
        applications.resnet50.ResNet50( include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000),
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        ])
