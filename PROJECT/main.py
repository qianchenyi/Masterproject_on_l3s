from Blackbox import MalwareDetectionModels
from DataUploader import UploadData
from MasterGAN import MasterGAN
import os
import zipfile
# import ImageProcessing
from numpy import asarray
from numpy import save
from kaggle.api.kaggle_api_extended import KaggleApi
import os
import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

width, height = 256, 256
# colormap = ImageProcessing.defineColorMap()
# R_colormap = ImageProcessing.defineColorMap()
# G_colormap = ImageProcessing.defineColorMap()
# B_colormap = ImageProcessing.defineColorMap()
# save('/colormap.npy', colormap)
# save('/R_colormap.npy', R_colormap)
# save('/G_colormap.npy', G_colormap)
# save('/B_colormap.npy', B_colormap)

# A file named kaggle.json will be downloaded. Move this file in to ~/.kaggle/ folder in Mac and Linux or to C:\Users\.kaggle\ on windows.
# api = KaggleApi()
# api.authenticate()
# # Signature: dataset_download_files(dataset, path=None, force=False, quiet=True, unzip=False)
# api.dataset_download_files('chetankv/dogs-cats-images', 'C:\Users\Lara\Desktop\Master_project\New Version\main.py',unzip=True)
# os.system('ls -l')
# os.system('mv /content/data/dataset/training_set/dogs /content/data/dataset/training_set/cani')
# os.system('mv /content/data/dataset/training_set/cats /content/data/dataset/training_set/gatti')

# os.system('mv /content/data/dataset/test_set/dogs /content/data/dataset/test_set/cani')
# os.system('mv /content/data/dataset/test_set/cats /content/data/dataset/test_set/gatti')

uploader = UploadData()
mal_train, mal_val = uploader.upload_train_mal_set()
mal_test =uploader.upload_test_mal_set()

ben_train, ben_val = uploader.upload_train_ben_set()
mixed_train, mixed_val = uploader.upload_training_set(ben_train, ben_val,mal_train, mal_val)

bb_detector = MalwareDetectionModels(width, height, 'M11',mixed_train, mixed_val)
GAN = MasterGAN(bb_detector,256, 256)
GAN.train(ben_train,mal_train,mal_test)
GAN.loss_plot()


