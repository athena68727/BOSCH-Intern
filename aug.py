import os
from PIL import Image
import tensorflow as tf
import numpy as np
import cv2
import time
import random

# Augment the data as you like

def darker(image,percetage=0.8):
    image_copy = image.copy()
    w = image.shape[1]
    h = image.shape[0]
    #get darker
    for xi in range(0,w):
        for xj in range(0,h):
            image_copy[xj,xi,0] = int(image[xj,xi,0]*percetage)
            image_copy[xj,xi,1] = int(image[xj,xi,1]*percetage)
            image_copy[xj,xi,2] = int(image[xj,xi,2]*percetage)
    return image_copy

# 亮度
def brighter(image, percetage=1.5):
    image_copy = image.copy()
    w = image.shape[1]
    h = image.shape[0]
    #get brighter
    for xi in range(0,w):
        for xj in range(0,h):
            image_copy[xj,xi,0] = np.clip(int(image[xj,xi,0]*percetage),a_max=255,a_min=0)
            image_copy[xj,xi,1] = np.clip(int(image[xj,xi,1]*percetage),a_max=255,a_min=0)
            image_copy[xj,xi,2] = np.clip(int(image[xj,xi,2]*percetage),a_max=255,a_min=0)
    return image_copy

# 旋转
def rotate(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]
    # If no rotation center is specified, the center of the image is set as the rotation center
    if center is None:
        center = (w / 2, h / 2)
    m = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, m, (w, h))
    return rotated

# 翻转
def flip(image):
    flipped_image = np.fliplr(image)
    return flipped_image

lst = ["\\can", "\\button", "\\line"]
round = 5
for part in lst:
    for i in range(round):
        for filename in os.listdir(os.getcwd()+ part):
            if filename.endswith(".jpg"):
                if "_new" not in filename:
                    img = cv2.imread(os.getcwd()+ part + "\\"+filename)
                    cv2.imwrite(os.getcwd()+ part + "\\"+ filename[:-4] + str(i) + "a_new.jpg", darker(img , percetage=random.random()*0.5 + 0.5))
                    cv2.imwrite(os.getcwd()+ part + "\\"+ filename[:-4] + str(i) + "b_new.jpg", brighter(img , percetage=random.random()*0.5 + 1.0))
                    cv2.imwrite(os.getcwd()+ part + "\\"+ filename[:-4] + str(i) + "c_new.jpg", rotate(img, angle = random.random() * 360))
                
        


