# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 17:53:54 2021

The downloaded images 432 * 288 Size 
@author: Raj
# conversion of images to 128 * 128 size and feed to CNN for training/testing 
"""
import os
from PIL import Image
from datetime import datetime
#import cv2
for image_file_name in os.listdir('D:\\Ecg_Image_Data\\train\\F'):
    if image_file_name.endswith(".png"):
        now = datetime.now().strftime('%Y%m%d-%H%M%S-%f')
        im = Image.open('D:\\Ecg_Image_Data\\train\\F\\'+image_file_name)
        new_width  = 128
        new_height = 128
        im = im.resize((new_width, new_height), Image.ANTIALIAS)
       # im1 = im.resize((128, 128) , Image.ANTIALIAS)
        im.save('D:\\Output\\train\\F\\' + now + '.png')