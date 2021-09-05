import os
from os.path import join as pjoin
import cv2
import numpy as np

data_dir  = '/home/qian/Masterproject/dataset/dataset_9010/test/validation_ds_update_924/malimg_dataset/validation'
binary_dir = '/home/qian/Masterproject/dataset/malware'


def savefile(guy,i,binary_file):
    # save_dir = pjoin(binary_dir, guy)  # the save address
    save_dir = '/home/qian/Masterproject/dataset/malware'
    # if not os.path.exists(save_dir):  # if the save address doesn't exsit, create
    #     os.makedirs(save_dir)

    name_binary = "{}/{}.npy".format(save_dir,i[:-4])#the name of .npy file
    np.save(name_binary,binary_file)


def load_data(data_dir):
    for guy in os.listdir(data_dir):#read the name of img_family in the data_dir
        family_dir = pjoin(data_dir,guy)#the malware family document direciton
        for i in os.listdir(family_dir):#read the img name in this family
            img_dir= pjoin(family_dir,i)#img address
            img = cv2.imread(img_dir,cv2.IMREAD_GRAYSCALE)#read the img
            img_1D = img.flatten()
            savefile(guy,i,img_1D)

load_data(data_dir)




