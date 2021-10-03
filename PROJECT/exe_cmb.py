import codecs
import ImageProcessing
import math
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow.keras as keras
import cv2
import os
from os.path import join as pjoin


# with open('/home/qian/Masterproject/dataset/folder_mal_test/mal/00df6a70facd0a1cf8571bd399d88840.png', 'rb') as file:
#     #this sintax is to be read as "with the output of the function open considered as a file"
#     # wb as in read binary
#   while True:
#     # as long as we can read one byte
#     b = file.read(1)
#     if not b:
#       break

#     img_bin_data.append(int.from_bytes(b, byteorder='big'))
# print(img_bin_data[0:15]) 

########convert the malimg in to the exe files###########
mal_data_dir = '/home/qian/Masterproject/dataset/Malimg_orig_test/test'
for i in os.listdir(mal_data_dir):#read the img name in this family
  img_dir= pjoin(mal_data_dir,i)#img address  
  img1 = cv2.imread(img_dir,cv2.IMREAD_GRAYSCALE)#read the img
  img_bin_data = img1.flatten() 

  hex_file = []

  for v in img_bin_data:
    s = int(v).to_bytes(1, 'little')
    hex_file.append(s)
  
  #listToStr = ' '.join([str(elem) for elem in hex_file])
  f = open(pjoin("/home/qian/Masterproject/dataset/exe_comb/",i[:-4])+'.exe', "wb")
  for value in hex_file:
    f.write(value)
  f.close()
#############################################################

#################concatonate two mal and ben imgs################
# img1= cv2.imread('/home/qian/Masterproject/dataset/folder_mal_test/mal/00a0d4f008a1b2f17716ca3d4fef4c92.png',cv2.IMREAD_GRAYSCALE)
# img_bin_data1 = img1.flatten() 
# img2 = cv2.imread('/home/qian/Masterproject/dataset/train_images/GRAY/b_greyscale/bb_greyscale/mblctr.png',cv2.IMREAD_GRAYSCALE)#read the img
# img_bin_data2 = img2.flatten() 
# cat = np.concatenate((img_bin_data1 , img_bin_data2), axis=None)
###############################################################
# img_bin_data1 = []
# with open('/home/qian/Masterproject/dataset/train_images/GRAY/b_greyscale/bb_greyscale/mblctr.png', 'rb') as file:
#     #this sintax is to be read as "with the output of the function open considered as a file"
#     # wb as in read binary
#   while True:
#     # as long as we can read one byte
#     bb = file.read(1)
#     if not bb:
#       break
#     img_bin_data1.append(int.from_bytes(bb, byteorder='big'))
# print(img_bin_data1[0:15])

i=0

# for el in img_bin_data1:
#   #encoded = el.to_bytes(2, byteorder='big')
#   ##print(encoded)
#   if(i<5):
#     print(el)
#   img_bin_data.append(el)
#   i= i+1
# #print(img_bin_data)
# print(len(img_bin_data))



# f = "/home/qian/Masterproject/dataset/comb_exe/com/ag.png"
# height = math.ceil(len(cat)/256)
# ImageProcessing.saveImg(f,cat,(256,height),'L')

def load_image(img_path):
    
    img = image.load_img(img_path, target_size=(256, 256),color_mode='grayscale',interpolation='area')
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]
    return img_tensor

def load_image( DIRECTORY):
    COLOR_MODE = 'grayscale'
    IMAGE_HEIGHT = 256
    IMAGE_WIDTH = 256
    BATCH_SIZE = 1
    SEED = 1337
    mid_img_set= keras.preprocessing.image_dataset_from_directory(
        DIRECTORY,
        label_mode = None,
        color_mode=COLOR_MODE,
        seed=SEED,
        interpolation="area",
        batch_size=BATCH_SIZE,
        image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
        shuffle=True
    )
    return mid_img_set
bb_detector = load_model('/home/qian/Masterproject/PROJECT/saved_model/new_model.h5')
new_image = load_image('/home/qian/Masterproject/dataset/comb_exe')
ben_imge = load_image('/home/qian/Masterproject/dataset/folder_ben_test')
mal_imge = load_image('/home/qian/Masterproject/dataset/folder_mal_test')

res = bb_detector.predict(new_image )
res_b = bb_detector.predict(ben_imge )
res_m = bb_detector.predict(mal_imge)
print("the combined file")
print(res)
print("the benign file")
print(res_b)
print("the malious file")
print(res_m)