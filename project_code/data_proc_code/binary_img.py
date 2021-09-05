import numpy as np
import os, math
from os.path import join as pjoin
import argparse
from PIL import Image

read_path = '/home/qian/Masterproject/dataset/malware'
#img_bin_data = readBytes('/Users/jessica/Documents/masterproject/malimg/binary_dataset_9010/train/Adialer.C/00bb6b6a7be5402fcfce453630bfff19.npy')


def defineColorMap():
    rows  = 256
    columns = 256
    min = 0 
    max = 255
    step = 2
    colormap = np.random.randint(min, max, size=rows * columns, dtype='l')
    colormap.resize(rows,columns)
    #print(colormap)
    #print("\n\n ",colormap.shape)
    return colormap

colormap = defineColorMap()

R_colormap = defineColorMap()
G_colormap = defineColorMap()
B_colormap = defineColorMap()

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



def to1DArray_greyscale(img_bin_data):
    pixel_array = []
    for index in range(0, len(img_bin_data)-2) :
        pixel_array.append(colormap[img_bin_data[index]][img_bin_data[index+1]])
    return pixel_array



def to1DArray_RGB(img_bin_data):
    pixel_array = []
    for index in range(0, len(img_bin_data)-2) :
        pixel_array.append((R_colormap[img_bin_data[index]][img_bin_data[index+1]], G_colormap[img_bin_data[index]][img_bin_data[index+1]], B_colormap[img_bin_data[index]][img_bin_data[index+1]]))
        print(pixel_array[index])
    return pixel_array 



def saveImg (filename, data, size, img_type):
    try:
        image = Image.new(img_type, size)
        image.putdata(data)
        ''' ref: https://github.com/ncarkaci/binary-to-image
        setup output filename
        dirname     = os.path.dirname(filename)
        name, _     = os.path.splitext(filename)
        name        = os.path.basename(name)
        imagename   = dirname + os.sep + img_type + os.sep + name + '_'+img_type+ '.png'
        os.makedirs(os.path.dirname(imagename), exist_ok=True)'''
        image.save(filename)
        print('The file', filename, 'saved.')
    except Exception as err:
        print(err)

def GABOR():
    ksize = 3 # kernel size
    sigma = 3 # standard deviation
    theta = 180 # orientation of the Gabor function
    lambd = 180 # width of the strips of the Gabor function
    gamma = 0.5 # aspect ratio
    psi = 0 # phase offset
    ktype = ktype=cv2.CV_32F # 	Type of filter coefficients. It can be CV_32F or CV_64F . ?? not specified in the paper

    # ref: https://docs.opencv.org/3.4/d4/d86/group__imgproc__filter.html#gae84c92d248183bd92fa713ce51cc3599
    gabor_kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F)
    return gabor_kernel
    
files = os.listdir(read_path)
for file in files:
    file_path=pjoin(read_path,file)
    img_bin_data = readBytes(file_path)
    greyscale_array = to1DArray_greyscale(img_bin_data)
    RGB_array = to1DArray_RGB(img_bin_data)
#print(img_bin_data)
saveImg('/Users/jessica/Documents/masterproject/malimg/Bin_im/greyscale_img.png', greyscale_array, (512,512),'L')
saveImg('/Users/jessica/Documents/masterproject/malimg/Bin_im/RGB_img.png', RGB_array, (512,512),'RGB')

import numpy as np
import cv2 
import matplotlib.pyplot as plt



plt.imshow(gabor_kernel)

image = cv2.imread('/Users/jessica/Documents/masterproject/malimg/Bin_im/greyscale_img.png') # reading image

image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.imshow(image, cmap='gray') 

filtered_image = cv2.filter2D(image, cv2.CV_8UC3, gabor_kernel)
plt.imshow(filtered_image, cmap='gray') 