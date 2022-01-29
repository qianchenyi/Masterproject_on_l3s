
# ref: https://github.com/ncarkaci/binary-to-image
import numpy as np
import os, math
import argparse
from os.path import join as pjoin
from PIL import Image
import math
from numpy import asarray
from numpy import save
from itertools import zip_longest
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import tensorflow as tf
import random
from random import randint 
# ben_data_dir = '/home/qian/Masterproject/dataset/benign_dataset/'
# ben_gray_save_dir='/home/qian/Masterproject/dataset/train_images/GRAY/b_greyscale/bb_greyscale'
# mal_data_dir ='/home/qian/Masterproject/dataset/malware'
# mal_gray_save_dir = '/home/qian/Masterproject/dataset/train_images/GRAY/m_greyscale/mm_greyscale'
# def defineColorMap():
#   rows  = 256
#   columns = 256
#   min = 0 
#   max = 256
#   step = 2
#   colormap = np.random.randint(min, max, size=rows * columns, dtype='l')
#   colormap.resize(rows,columns)
#   print(colormap)
#   print("\n\n ",colormap.shape)
#   return colormap

# gray_colormap = defineColorMap()
# R_colormap = defineColorMap()
# G_colormap = defineColorMap()
# B_colormap = defineColorMap()
# save('/home/qian/Masterproject/dataset/colormap/gray_colormap.npy', gray_colormap)
# save('/home/qian/Masterproject/dataset/colormap/R_colormap.npy', R_colormap)
# save('/home/qian/Masterproject/dataset/colormap/G_colormap.npy', G_colormap)
# save('/home/qian/Masterproject/dataset/colormap/B_colormap.npy', B_colormap)

def readBytes (filename):
  img_bin_data = []
  with open(filename, 'rb') as file:
     #this sintax is to be read as "with the output of the function open considered as a file"
     # wb as in read binary
    while True:
      # as long as we can read one byte
      b = file.read(1)
      if not b:
        break
      img_bin_data.append(int.from_bytes(b, byteorder='big'))
  return img_bin_data

def readBytes_fromNumpy (filename):
  return np.load(filename)
def readBytes1 (filename):
  img_bin_data = []
  with open(filename, 'rb') as file:
     #this sintax is to be read as "with the output of the function open considered as a file"
     # wb as in read binary
    while True:
      # as long as we can read one byte
      b = file.read(1)
      if not b:
        break
      img_bin_data.append(int.from_bytes(b, byteorder='big'))
  while len(img_bin_data)%256!=0:
    img_bin_data.append(randint(0, 255))
  return img_bin_data

def readBytes2 (filename):
  img_bin_data = []
  with open(filename, 'rb') as file:
     #this sintax is to be read as "with the output of the function open considered as a file"
     # wb as in read binary
    while True:
      # as long as we can read one byte
      b = file.read(1)
      if not b:
        break
      img_bin_data.append(int.from_bytes(b, byteorder='big'))
  while len(img_bin_data)%256!=0:
    img_bin_data.append(randint(0, 255))
  return tf.convert_to_tensor(img_bin_data)

#print(img_bin_data)

#img_bin_data = readBytes_fromNumpy('/content/123.npy')
#print(img_bin_data)


def grouper(iterable, n, fillvalue=None):
  "Collect data into fixed-length chunks or blocks"
  # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
  args = [iter(iterable)] * n
  return zip_longest(fillvalue = fillvalue, *args)


def to1DArray_grayscale(img_bin_data,colormap):
  pixel_array = []
  for x, y in grouper(img_bin_data, 2) :
    new_pixel = colormap[x][y]
    pixel_array.append(new_pixel)
    
  return pixel_array

#Gray_array = to1DArray_RGB(img_bin_data)


def saveImg (filename, data, size, img_type):
  try:
    image = Image.new(img_type, size)
    #tuples = [tuple(x) for x in data]
    #image.putdata(tuples)
    image.putdata(data)
    '''ref : https://stackoverflow.com/questions/68642888/issues-converting-rgb-to-image-with-correct-size'''
    ''' ref: https://github.com/ncarkaci/binary-to-image
    setup output filename
    dirname     = os.path.dirname(filename)
    name, _     = os.path.splitext(filename)
    name        = os.path.basename(name)
    imagename   = dirname + os.sep + img_type + os.sep + name + '_'+img_type+ '.png'
    os.makedirs(os.path.dirname(imagename), exist_ok=True)'''
    image.save(filename)
    # print('The file', filename, 'saved.')
  except Exception as err:
    print(err)



# mal_exe_dir='/home/qian/Masterproject/dataset/exe_malimg'
# width = 256
# colormap = readBytes_fromNumpy('/home/qian/Masterproject/dataset/colormap/gray_colormap.npy')
# for i in os.listdir(mal_exe_dir):#read the img name in this family
#     img_dir= pjoin(mal_exe_dir,i)#img address
#     img_bin_data = readBytes(img_dir)
#     grayscale_array = to1DArray_grayscale(img_bin_data,colormap)
#     height = math.ceil(len(grayscale_array)/width)
   
#     saveImg(pjoin(mal_gray_save_dir,i[:-4])+'.png', grayscale_array, (width,height),'L')


# for i in os.listdir(mal_data_dir):#read the img name in this family
#     img_dir= pjoin(mal_data_dir,i)#img address
#     img = cv2.imread(img_dir,cv2.IMREAD_GRAYSCALE)#read the img
#     img_bin_data = img.flatten()
#     grayscale_array = to1DArray_grayscale(img_bin_data,colormap)
#     height = math.ceil(len(grayscale_array)/width)

#     saveImg(pjoin(mal_gray_save_dir,i[:-4])+'.png', grayscale_array, (width,height),'L')


