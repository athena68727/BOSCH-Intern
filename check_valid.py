from re import I
import cv2
import imutils
import numpy as np
import joblib
import os
from standardcrop import mycrop # remember to work on it
from PIL import Image
from numpy import average, dot, linalg
import time


def process(image, size=(100,100)):
    image = cv2.resize(image, size)
    '''cv2.namedWindow('1',0)
    cv2.imshow("1", image)
    while(1):
        if cv2.waitKey(1)==ord('n'):
            break'''
    return image

def image_similarity_vectors_via_numpy(image1, image2):
    image1 = process(image1)
    image2 = process(image2)
    images = [image1, image2]
    vectors = []
    norms = []
    for image in images:
        vector = []
        for pixel_tuple in image:
            vector.append(average(pixel_tuple))
        vectors.append(vector)
        norms.append(linalg.norm(vector, 2))
    a, b = vectors
    a_norm, b_norm = norms
    # dot返回的是点积，对二维数组（矩阵）进行计算
    res = dot(a / a_norm, b / b_norm)
    #print("Cosine similarity:", res)
    return res

def if_valid(img, type):
    if type[-1] == '4':
        img_154 = cv2.imread('dump\\ref_154.jpg')
        img_154 = mycrop(img_154, '154_c')
        img1 = mycrop(img, '154_c')
        cosin1 = image_similarity_vectors_via_numpy(img1, img_154)
        if cosin1 > 0.95:
            return 1
    elif type[-1] == '2':
        img_152 = cv2.imread('dump\\ref_152.jpg')
        #print(par + "\\dump\\reference_m.jpg")
        img_152 = mycrop(img_152, '152')
        img2 = mycrop(img, '152')
        cosin2 = image_similarity_vectors_via_numpy(img2, img_152)
    #print(cosin1, cosin2)
        if cosin2 > 0.99:
            return 1
    return 0

# For test, remember to edit the import
if __name__ == '__main__':
    
    print("For normal image:")
    for filename in os.listdir("image_154"):
        start = time.time()
        if filename.endswith('jpg'):
            img = cv2.imread("image_154\\"+filename)
            if_valid(img, '154')
        end = time.time()
        #print("Time consumed:", end - start)