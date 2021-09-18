import tensorflow as tf
import tensorflow.keras as keras
import numpy as np



class UploadData():

    def __init__(self):
        self.TRAINING_DIRECTORY = '/home/qian/Masterproject/dataset/train_images/GRAY'
        self.TRAINING_MAL_DIRECTORY ='/home/qian/Masterproject/dataset/train_images/GRAY/m_greyscale'
        self.TRAINING_BEN_DIRECTORY = '/home/qian/Masterproject/dataset/train_images/GRAY/b_greyscale'
        #TEST_DIRECTORY = '/home/qian/Masterproject/dataset/malware/processed_image_malicous/greyscale'
        self.COLOR_MODE = 'grayscale'
        self.IMAGE_HEIGHT = 256
        self.IMAGE_WIDTH = 256
        self.BATCH_SIZE = 100
        self.SEED = 1337
        

    def upload_mal_set(self):
        training_set = keras.preprocessing.image_dataset_from_directory(
        self.TRAINING_MAL_DIRECTORY,
        labels="inferred",
        label_mode="int",
        color_mode= self.COLOR_MODE,
        validation_split=0.15,
        subset='training',
        seed= self.SEED,
        interpolation="area",
        batch_size = self.BATCH_SIZE,   
        image_size=(self.IMAGE_HEIGHT, self.IMAGE_WIDTH), 
        shuffle=True,
        )

        vali_mal_set = keras.preprocessing.image_dataset_from_directory(
        self.TRAINING_MAL_DIRECTORY,
        labels="inferred",
        label_mode="int",
        subset='validation',
        validation_split=0.15,
        seed=self.SEED,
        interpolation="area",
        batch_size=self.BATCH_SIZE,
        image_size=(self.IMAGE_HEIGHT, self.IMAGE_WIDTH),
        )
        return training_mal_set,vali_mal_set
    
    def upload_ben_set(self):
        training_ben_set = keras.preprocessing.image_dataset_from_directory(
        self.TRAINING_BEN_DIRECTORY,
        labels="inferred",
        label_mode="int",
        color_mode= self.COLOR_MODE,
        validation_split=0.15,
        subset='training',
        seed= self.SEED,
        interpolation="area",
        batch_size = self.BATCH_SIZE,   
        image_size=(self.IMAGE_HEIGHT, self.IMAGE_WIDTH), 
        shuffle=True,
        )
        vali_ben_set = keras.preprocessing.image_dataset_from_directory(
        self.TRAINING_BEN_DIRECTORY,
        labels="inferred",
        label_mode="int",
        color_mode= self.COLOR_MODE,
        validation_split=0.15,
        subset='validation',
        seed= self.SEED,
        interpolation="area",
        batch_size = self.BATCH_SIZE,   
        image_size=(self.IMAGE_HEIGHT, self.IMAGE_WIDTH), 
        shuffle=True,
        )
        return training_ben_set,vali_ben_set

    def upload_set(self):
        training_set = keras.preprocessing.image_dataset_from_directory(
        self.TRAINING_BEN_DIRECTORY,
        labels="inferred",
        label_mode="int",
        color_mode= self.COLOR_MODE,
        validation_split=0.15,
        subset='training',
        seed= self.SEED,
        interpolation="area",
        batch_size = self.BATCH_SIZE,   
        image_size=(self.IMAGE_HEIGHT, self.IMAGE_WIDTH), 
        shuffle=True,
        )
        vali_set = keras.preprocessing.image_dataset_from_directory(
        self.TRAINING_BEN_DIRECTORY,
        labels="inferred",
        label_mode="int",
        color_mode= self.COLOR_MODE,
        validation_split=0.15,
        subset='validation',
        seed= self.SEED,
        interpolation="area",
        batch_size = self.BATCH_SIZE,   
        image_size=(self.IMAGE_HEIGHT, self.IMAGE_WIDTH), 
        shuffle=True,
        )
        return training_set,vali_set

    

    #how to get the dataset:    
    #https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/load_data/images.ipynb?hl=zh-tw#scrollTo=ucMoYase6URl
    def shuffle_nomalize(self,mixdata_set,mal_set,ben_set):

        AUTOTUNE = tf.data.AUTOTUNE
        mixdata_ds = mixdata_set.cache().shuffle(11929).prefetch(buffer_size=AUTOTUNE)
        mal_ds = mal_set.cache().shuffle(11929).prefetch(buffer_size=AUTOTUNE)
        ben_ds = ben_set.cache().shuffle(11929).prefetch(buffer_size=AUTOTUNE)
        #test_ds = test_set.cache().shuffle(11929).prefetch(buffer_size=AUTOTUNE)
        #validation_ds = validation_set.cache().prefetch(buffer_size=AUTOTUNE)

        #The RGB channel values are in the [0, 255] range. This is not ideal for a neural network;
        #  in general you should seek to make your input values small.
        #  Here, you will standardize values to be in the [0, 1] range by using the tf.keras.layers.experimental.preprocessing.Rescaling layer.
        normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)

        normalized_ds = mixdata_ds.map(lambda x, y: (normalization_layer(x), y))
        normalized_mal_ds = mal_ds.map(lambda x, y: (normalization_layer(x), y))
        normalized_ben_ds = ben_ds.map(lambda x, y: (normalization_layer(x), y))
        #normalized_test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))
        #normalized_validatioin_ds = validation_ds.map(lambda x, y: (normalization_layer(x), y))

        return normalized_ds,normalized_mal_ds,normalized_ben_ds



    # def get_train_dataset():
    #     train_ds,train_mal_ds,train_ben_ds = shuffle_nomalize(training_set,training_mal_set,training_ben_set)
    #     return train_ds,train_mal_ds,train_ben_ds
    
    # def get_vali_dataset():
    #     vali_ds,vali_mal_ds,vali_ben_ds = shuffle_nomalize(vali_set,vali_mal_set,vali_ben_set)
    #     return vali_ds,vali_mal_ds,vali_ben_ds
if __name__ =="__main__":
    uploader = UploadData()
    training_mal_set,vali_mal_set = uploader.upload_mal_set()
    training_ben_set,vali_ben_set = uploader.upload_ben_set()
    training_set,vali_set = uploader.upload_set()
    train_ds,train_mal_ds,train_ben_ds = uploader.shuffle_nomalize(training_set,training_mal_set,training_ben_set)
