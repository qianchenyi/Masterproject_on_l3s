import os
import ImageProcessing
import random
import numpy as np
import math
from tensorflow import keras
import tensorflow as tf

# used to shuffle and convert benign executable files into images with and without the use of a colormap 

exe_dir = 'dataset/ben_exe_test/BEN_exe_test/'
img_dir_no_cm = 'dataset/test_images_wo_cm/Ben_img_no_colormap/ben_img_without_clmap/'
img_dir_cm= 'dataset/test_images_with_colormap/GRAY/b_greyscale/bb_greyscale/'

colormap = np.load('dataset/colormap/gray_colormap.npy')
width = 256
exe_names = os.listdir(exe_dir)
random.shuffle(exe_names)
i=0   

for name in exe_names:
    or_dir = exe_dir+name 
    img_bin_array = ImageProcessing.readBytes(or_dir)
    grayscale_array = ImageProcessing.to1DArray_grayscale(img_bin_array,colormap)
    height = math.ceil(len(img_bin_array)/width)
    height_cm = math.ceil(len(grayscale_array)/width)
    if i<=9:
        ImageProcessing.saveImg(img_dir_no_cm+'000'+str(i)+'.png', img_bin_array, (width,height),'L')
        ImageProcessing.saveImg(img_dir_cm+'000'+str(i)+'.png', grayscale_array, (width,height_cm),'L')
    elif i>=10 and i<100:
        ImageProcessing.saveImg(img_dir_no_cm+'00'+str(i)+'.png', img_bin_array, (width,height),'L')
        ImageProcessing.saveImg(img_dir_cm+'00'+str(i)+'.png', grayscale_array, (width,height_cm),'L')
    elif i>=100 and i<1000:
        ImageProcessing.saveImg(img_dir_no_cm+'0'+str(i)+'.png', img_bin_array, (width,height),'L')
        ImageProcessing.saveImg(img_dir_cm+'0'+str(i)+'.png', grayscale_array, (width,height_cm),'L')
    else:
        ImageProcessing.saveImg(img_dir_no_cm+str(i)+'.png', img_bin_array, (width,height),'L')
        ImageProcessing.saveImg(img_dir_cm+str(i)+'.png', grayscale_array, (width,height_cm),'L')
    i=i+1

