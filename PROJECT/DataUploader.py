
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import os, os.path
from matplotlib.pyplot import figure, imshow, axis
import matplotlib.pyplot as plt

class UploadData():
  def __init__(self):
    
    # self.TRAINING_MAL_DIRECTORY = '/home/qian/Masterproject/dataset/train_images/GRAY/m_greyscale/'
    # self.TRAINING_BEN_DIRECTORY = '/home/qian/Masterproject/dataset/train_images/GRAY/b_greyscale/'
    
    # self.TEST_MAL_DIRECTORY = '/content/dataset/Train/GRAY/m_greyscale/'
    # self.TEST_BEN_DIRECTORY = '/content/dataset/Train/GRAY/b_greyscale/'
    
    self.TRAINING_MAL_DIRECTORY = '/home/qian/Masterproject/dataset/catsdogs/dataset/training_set/gatti/'
    self.TRAINING_BEN_DIRECTORY = '/home/qian/Masterproject/dataset/catsdogs/dataset/training_set/cani/'
    
    self.TEST_MAL_DIRECTORY = '/home/qian/Masterproject/dataset/catsdogs/dataset/test_set/gatti/'
    self.TEST_BEN_DIRECTORY = '/home/qian/Masterproject/dataset/catsdogs/dataset/test_set/cani/'
    # path joining version for other paths
    
    # self.mal_train_size = len([name for name in os.listdir(self.TRAINING_MAL_DIRECTORY+'mm_greyscale') if os.path.isfile(os.path.join(self.TRAINING_MAL_DIRECTORY+'mm_greyscale', name))])
    # self.ben_train_size = len([name for name in os.listdir(self.TRAINING_BEN_DIRECTORY+'bb_greyscale') if os.path.isfile(os.path.join(self.TRAINING_BEN_DIRECTORY+'bb_greyscale', name))])
    # self.mal_test_size = len([name for name in os.listdir(self.TEST_MAL_DIRECTORY+'mm_greyscale') if os.path.isfile(os.path.join(self.TEST_MAL_DIRECTORY+'mm_greyscale', name))])
    # self.ben_test_size = len([name for name in os.listdir(self.TEST_BEN_DIRECTORY+'bb_greyscale') if os.path.isfile(os.path.join(self.TEST_BEN_DIRECTORY+'bb_greyscale', name))])
    
    self.mal_train_size = 4000 #len([name for name in os.listdir(self.TRAINING_MAL_DIRECTORY+'cats') if os.path.isfile(os.path.join(self.TRAINING_MAL_DIRECTORY+'cats', name))])
    self.ben_train_size = 4000 #len([name for name in os.listdir(self.TRAINING_BEN_DIRECTORY+'dogs') if os.path.isfile(os.path.join(self.TRAINING_BEN_DIRECTORY+'dogs', name))])
    self.mal_test_size = 1000 #len([name for name in os.listdir(self.TEST_MAL_DIRECTORY+'cats') if os.path.isfile(os.path.join(self.TEST_MAL_DIRECTORY+'cats', name))])
    self.ben_test_size = 1000 #len([name for name in os.listdir(self.TEST_BEN_DIRECTORY+'dogs') if os.path.isfile(os.path.join(self.TEST_BEN_DIRECTORY+'dogs', name))])

    self.COLOR_MODE = 'grayscale'
    self.IMAGE_HEIGHT = 256
    self.IMAGE_WIDTH = 256
    self.BATCH_SIZE = 32
    self.VAL_SPLIT = 0.15
    self.SEED = 1337
    self.trian_mal_lbls = [1] * self.mal_train_size
    self.trian_ben_lbls = [0] * self.ben_train_size
    self.test_mal_lbls = [1] * self.mal_test_size
    self.test_ben_lbls = [0] * self.ben_test_size


  '''def __init__(self):
  training_mal_set,vali_mal_set = self.upload_mal_set()
  training_ben_set,vali_ben_set = self.upload_ben_set()
  training_set,vali_set = self.upload_set()'''

  def upload_train_mal_set(self):
    training_mal_set = keras.preprocessing.image_dataset_from_directory(
            self.TRAINING_MAL_DIRECTORY,
            labels = self.trian_mal_lbls ,
            label_mode = 'int',
            color_mode=self.COLOR_MODE,
            validation_split=self.VAL_SPLIT,
            subset='training',
            seed=self.SEED,
            interpolation="area",
            batch_size=self.BATCH_SIZE,
            image_size=(self.IMAGE_HEIGHT, self.IMAGE_WIDTH),
            shuffle=True
    )

    val_mal_set = keras.preprocessing.image_dataset_from_directory(
      self.TRAINING_MAL_DIRECTORY,
      labels = self.trian_mal_lbls ,
      label_mode = 'int', 
      color_mode=self.COLOR_MODE,
      subset='validation',
      validation_split=self.VAL_SPLIT,
      seed=self.SEED,
      interpolation="area",
      batch_size=self.BATCH_SIZE,
      image_size=(self.IMAGE_HEIGHT, self.IMAGE_WIDTH),
      shuffle=True
    )
    return training_mal_set, val_mal_set

  def upload_train_ben_set(self):
    training_ben_set = keras.preprocessing.image_dataset_from_directory(
      self.TRAINING_BEN_DIRECTORY,            
      labels = self.trian_ben_lbls,
      label_mode = 'int',
      color_mode=self.COLOR_MODE,
      validation_split=self.VAL_SPLIT,
      subset='training',
      seed=self.SEED,
      interpolation="area",
      batch_size=self.BATCH_SIZE,
      image_size=(self.IMAGE_HEIGHT, self.IMAGE_WIDTH),
      shuffle=True,
    )
    val_ben_set = keras.preprocessing.image_dataset_from_directory(
      self.TRAINING_BEN_DIRECTORY,
      labels= self.trian_ben_lbls,
      label_mode="int",
      color_mode=self.COLOR_MODE,
      validation_split=self.VAL_SPLIT,
      subset='validation',
      seed=self.SEED,
      interpolation="area",
      batch_size=self.BATCH_SIZE,
      image_size=(self.IMAGE_HEIGHT, self.IMAGE_WIDTH),
      shuffle=True
    )
    return training_ben_set, val_ben_set

  def upload_training_set(self, training_ben_set, vali_ben_set, training_mal_set, val_mal_set):
    training_set = training_ben_set.concatenate(training_mal_set)
    training_set = training_set.unbatch()
    training_set = training_set.shuffle(int((1-self.VAL_SPLIT)*(float(self.ben_train_size+self.mal_train_size))))
    training_set = training_set.batch(self.BATCH_SIZE)

    val_set = vali_ben_set.concatenate(val_mal_set)
    val_set = val_set.unbatch()
    val_set = val_set.shuffle(int((self.VAL_SPLIT)*(float(self.ben_train_size+self.mal_train_size))))
    val_set = val_set.batch(self.BATCH_SIZE)
    return training_set, val_set



  def upload_test_mal_set(self):
    test_mal_set = keras.preprocessing.image_dataset_from_directory(
            self.TEST_MAL_DIRECTORY,
            labels = self.test_mal_lbls ,
            label_mode = 'int',
            color_mode=self.COLOR_MODE,
            seed=self.SEED,
            interpolation="area",
            batch_size=self.BATCH_SIZE,
            image_size=(self.IMAGE_HEIGHT, self.IMAGE_WIDTH),
            shuffle=True
    )
    return test_mal_set

  def upload_test_ben_set(self):
    test_ben_set = keras.preprocessing.image_dataset_from_directory(
            self.TEST_BEN_DIRECTORY,
            labels = self.test_ben_lbls ,
            label_mode = 'int',
            color_mode=self.COLOR_MODE,
            seed=self.SEED,
            interpolation="area",
            batch_size=self.BATCH_SIZE,
            image_size=(self.IMAGE_HEIGHT, self.IMAGE_WIDTH),
            shuffle=True
    )
    return test_ben_set

  def upload_test_set(self, test_ben_set, test_mal_set):
    test_set = test_ben_set.concatenate(test_mal_set)
    test_set = test_set.unbatch()
    test_set = test_set.shuffle(int((1-self.VAL_SPLIT)*(float(self.ben_test_size+self.mal_test_size))))
    test_set = test_set.batch(self.BATCH_SIZE)
    return test_set




    # how to get the dataset:    
    # https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/load_data/images.ipynb?hl=zh-tw#scrollTo=ucMoYase6URl
  def shuffle_nomalize(self, mixdata_set, mal_set, ben_set):
    AUTOTUNE = tf.data.AUTOTUNE
    mixdata_ds = mixdata_set.cache().shuffle(11929).prefetch(buffer_size=AUTOTUNE)
    mal_ds = mal_set.cache().shuffle(11929).prefetch(buffer_size=AUTOTUNE)
    ben_ds = ben_set.cache().shuffle(11929).prefetch(buffer_size=AUTOTUNE)
    # test_ds = test_set.cache().shuffle(11929).prefetch(buffer_size=AUTOTUNE)
    # validation_ds = validation_set.cache().prefetch(buffer_size=AUTOTUNE)

    # The RGB channel values are in the [0, 255] range. This is not ideal for a neural network;
    #  in general you should seek to make your input values small.
    #  Here, you will standardize values to be in the [0, 1] range by using the tf.keras.layers.experimental.preprocessing.Rescaling layer.
    normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255)

    normalized_ds = mixdata_ds.map(lambda x, y: (normalization_layer(x), y))
    normalized_mal_ds = mal_ds.map(lambda x, y: (normalization_layer(x), y))
    normalized_ben_ds = ben_ds.map(lambda x, y: (normalization_layer(x), y))
    # normalized_test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))
    # normalized_validatioin_ds = validation_ds.map(lambda x, y: (normalization_layer(x), y))

    return normalized_ds, normalized_mal_ds, normalized_ben_ds

