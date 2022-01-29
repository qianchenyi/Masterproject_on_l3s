from Blackbox import MalwareDetectionModels
from DataUploader import UploadData
from MasterGAN import MasterGAN
import ImageProcessing
import numpy as np
import os
import tensorflow.keras as keras
from tensorflow.keras import layers
import tensorflow as tf
import math


bb_detector = MalwareDetectionModels(None, None, 'M11',None, None,None,loaded=True)
@tf.function
def ds(exe_batch):
    return tf.stack(exe_batch)
exe_batch = []
malware_sample = ImageProcessing.readBytes2('/home/qian/Masterproject/dataset/Malimg_exe_train/00340500c4c4ddc4eaad3f4e7ab8fb45.exe') #<- is a tensor
height = math.ceil(malware_sample.shape[0]/256)

for i in range(0,64):
    exe_batch.append(malware_sample)
malware_sample_1 = ds(exe_batch)
malware_sample_1=tf.reshape(malware_sample_1, [64,height,256,1])



COLOR_MODE = 'grayscale'        
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
BATCH_SIZE = 64
SEED = 1337
DIRECTORY = '/home/qian/Masterproject/dataset/Benimg_noclmp_test/'
gan_set = keras.preprocessing.image_dataset_from_directory(
    DIRECTORY,
    label_mode = None,
    color_mode=COLOR_MODE,
    seed=SEED,
    interpolation="bilinear",
    batch_size=BATCH_SIZE,
    image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    shuffle=False
)
bb_on_real = np.load('/home/qian/Masterproject/dataset/bb_on_real/BB_ON_REAL_RIGHT.npy',allow_pickle=True)


GAN = MasterGAN(True,bb_detector,height, 256, 256)
GAN.train(gan_set, malware_sample_1,mal_test, bb_on_real)

