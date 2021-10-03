
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



def to1DArray_RGB(img_bin_data,R,G,B):
  pixel_array = []
  for x, y in grouper(img_bin_data, 2) :
    pixel_array.append((R_colormap[x][y], G_colormap[x][y], B_colormap[x][y]))
    #print(pixel_array[index])
  return pixel_array

#RGB_array = to1DArray_RGB(img_bin_data)

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



#saveImg('/content/outputt.png', RGB_array, (512,512),'RGB')

def findMax(matrix):
  max = 0
  for i in range(0,256):
    for j in range(0,256):
      if matrix[i][j]>max:
        max =  matrix[i][j]
  return int(max)

def generateMarkovImg(binary_data):
  # input B (binary_data) = {b1, b2, b3...bn} is a set where bi represents the decimal value of a byte.
  # TM[i][j] represents the probability that byte bi is followed by bj
  TM = np.zeros((256,256))
  S = np.zeros((256,1))
  L = len(binary_data)
  #we determine the frequency of occurrence of byte bi followed by bi+1 and bi followed by bk where 0 ≤ k ≤ 255. 
  i=0
  while i < L-1:
    r = binary_data[i]
    
    c = binary_data[i+1]
    
    TM[r][c] = TM[r][c] +1
    S[r] = S[r] + 1
    
    i = i+1

  i = 0
  j = 0
  #compute the probability that byte bi is followed by bi+1
  while i < 256:
    rs = S[i]
    while j < 256:
      TM[i][j] = TM[i][j]/rs
      j = j+1
    i = i+1

  print("TM shape", TM.shape)
  MP = findMax(TM)
  print('max',MP)
  i = 0
  j = 0
  M =  np.zeros((256,256))
  # compute pixels in Markov image
  while i < 256:
    while j < 256:
      
      p = (TM[i][j]*(255/MP))%256
      M[i][j] = p
      j = j+1
    i = i+1
  print(M)
  return M
  #Output: M = {m1, m2, m3...mn} is a set where mi represents a pixel value in Markov image.

#saveImg('/content/outputtt.png', generateMarkovImg(img_bin_data), (512,512),'L')
# def bin_to_img(original_file, data, size, img_type):
#   readBytes (original_file)
#   saveImg (filename, data, size, img_type)

# mal_exe_dir='/home/qian/Masterproject/dataset/exe_malimg'
# mal_gray_save_dir = '/home/qian/Masterproject/dataset/train_images/MARKOV/m_markov'
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

#what we need to convert is only the benign .exe files, the malware use Malimg, it's already image 
# width = 256
# binary_ben_dir='/home/qian/Masterproject/dataset/benign_dataset'
# Benimg ='/home/qian/Masterproject/dataset/Benimg'
# for i in os.listdir(ben_data_dir):#read the img name in this family
#   exe_dir= pjoin(ben_data_dir,i)#img address
#   bin_data = readBytes(exe_dir)
#   height = math.ceil(len(bin_data)/width)
  
#   saveImg(pjoin(Benimg,i[:-4])+'.png', bin_data, (width,height),'L')

# width = 128
# ben_data_dir = '/home/qian/Masterproject/dataset/exe_comb/mal1.exe'
# img_bin_data = readBytes(ben_data_dir)
# height = math.ceil(len(img_bin_data)/width)
# saveImg('/home/qian/Masterproject/dataset/exe_comb/com/malbig.png', img_bin_data, (width,height),'L')
