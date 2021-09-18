import numpy as np
import os, math
from os.path import join as pjoin
import argparse
from PIL import Image

read_path = '/home/qian/Masterproject/dataset/benign_dataset'
save_path = '/home/qian/Masterproject/dataset/train_images/GRAY/b_greyscale'
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

def readBytes_fromNumpy (filename):
    return np.load(filename) 
    img_bin_data = readBytes_fromNumpy(filename)
    print(img_bin_data)


def to1DArray_greyscale(img_bin_data):
    pixel_array = []
    for index in range(0, len(img_bin_data)-2) :
        pixel_array.append(colormap[img_bin_data[index]][img_bin_data[index+1]])
    return pixel_array



def to1DArray_RGB(img_bin_data):
    pixel_array = []
    for index in range(0, len(img_bin_data)-2) :
        pixel_array.append((R_colormap[img_bin_data[index]][img_bin_data[index+1]], G_colormap[img_bin_data[index]][img_bin_data[index+1]], B_colormap[img_bin_data[index]][img_bin_data[index+1]]))
        #print(pixel_array[index])
    return pixel_array 

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
    #print(M)
    return M
  #Output: M = {m1, m2, m3...mn} is a set where mi represents a pixel value in Markov image.


def saveImg (filename,dirname, data, size, img_type):
    try:
        save_dir = pjoin(save_path,dirname,filename)
        image = Image.new(img_type, size)
        image.putdata(data)
        ''' ref: https://github.com/ncarkaci/binary-to-image
        setup output filename
        dirname     = os.path.dirname(filename)
        name, _     = os.path.splitext(filename)
        name        = os.path.basename(name)
        imagename   = dirname + os.sep + img_type + os.sep + name + '_'+img_type+ '.png'
        os.makedirs(os.path.dirname(imagename), exist_ok=True)'''
        image.save(save_dir)
        print('The file', save_dir, 'saved.')
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
    markov_arry = generateMarkovImg(img_bin_data)
    #add the gaber filter
    #filtered_grey = cv2.filter2D(greyscale_array, cv2.CV_8UC3, gabor_kernel)
    #filtered_RGB = cv2.filter2D(RGB_array, cv2.CV_8UC3, gabor_kernel)
    #filtered_markov = cv2.filter2D(markov_arry, cv2.CV_8UC3, gabor_kernel)
#print(img_bin_data)
    filename = file.split(".")[0]+'.png'
    saveImg(filename, 'bb_greyscale/',greyscale_array, (512,512),'L')
    saveImg(filename, 'RGB',RGB_array, (512,512),'RGB')
    saveImg(filename, 'markov',markov_arry, (512,512),'L')

# import numpy as np
# import cv2 
# import matplotlib.pyplot as plt
# plt.imshow(gabor_kernel)
# image = cv2.imread('/Users/jessica/Documents/masterproject/malimg/Bin_im/greyscale_img.png') # reading image
# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# plt.imshow(image, cmap='gray') 
# filtered_image = cv2.filter2D(image, cv2.CV_8UC3, gabor_kernel)
# plt.imshow(filtered_image, cmap='gray') 