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
bb_on_real = np.load('/home/qian/Masterproject/dataset/bb_on_real/BB_ON_REAL_RIGHT_test.npy',allow_pickle=True)

#####should be changed later#########
mal_test = '/home/qian/Masterproject/ae.exe'
GAN = MasterGAN(True,bb_detector,height, 256, 256)
GAN.train(gan_set, malware_sample_1,mal_test, bb_on_real)

# noise_generator = tf.random.Generator.from_seed(2)
# new_noise = noise_generator.normal(shape=[64,65536])

# generator = tf.keras.models.load_model('/home/qian/Masterproject/PROJECT/saved_model/generator_model.h5')
# sub_detector =  tf.keras.models.load_model('/home/qian/Masterproject/PROJECT/saved_model/sub_detector.h5')

# adv_sample = generator.predict([malware_sample_1, new_noise])
# bb_on_adv = bb_detector.make_prediction(adv_sample,100)
# sub_on_adv = sub_detector.predict(adv_sample)