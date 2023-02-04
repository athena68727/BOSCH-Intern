import cv2
import imutils
import numpy as np
import joblib
import os
from tools.standardcrop import mycrop # remember to work on it
from PIL import Image
from numpy import average, dot, linalg
import time

par = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


def process(image, size=(100,100)):
    # 利用image对图像大小重新设置, Image.ANTIALIAS为高质量的
    image = cv2.resize(image, size)
    image = image[50:80, 50:80]
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

def which_side(img):
    img_m = cv2.imread(par + '\\dump\\reference_m.jpg')
    img_n = cv2.imread(par + '\\dump\\reference_n.jpg')
    #print(par + "\\dump\\reference_m.jpg")
    img_m = mycrop(img_m, 'm')
    img_n = mycrop(img_n, 'n')
    img1 = mycrop(img, 'm')
    img2 = mycrop(img, 'n')
    cosin1 = image_similarity_vectors_via_numpy(img1, img_m)
    cosin2 = image_similarity_vectors_via_numpy(img2, img_n)
    #print(cosin1, cosin2)
    if cosin1 > 0.98:
        return 'm'
    elif cosin2 > 0.98:
        return 'n'
    else:
        return None

# For test, remember to edit the import
if __name__ == '__main__':
    '''print("For rightside:")
    for filename in os.listdir(par + "\\image_right"):
        start = time.time()
        if filename.endswith('jpg'):
            img = cv2.imread(par + "\\image_right\\"+filename)
            which_side(img)
        end = time.time()
        #print("Time consumed:", end - start)

    print("\n \n For leftside:")
    for filename in os.listdir(par + "\\image_left"):
        start = time.time()
        if filename.endswith('jpg'):
            img = cv2.imread(par + "\\image_left\\"+filename)
            which_side(img)
        end = time.time()
        #print("Time consumed:", end - start)'''
    print("\n \n For Ineffective:")
    '''for filename in os.listdir(par + "\\image_no_effect"):
        start = time.time()
        if filename.endswith('jpg'):
            img = cv2.imread(par + "\\image_no_effect\\"+filename)
            which_side(img)
        end = time.time()
        #print("Time consumed:", end - start)'''
    
    print("For normal image:")
    for filename in os.listdir(par + "\\image"):
        start = time.time()
        if filename.endswith('jpg'):
            img = cv2.imread(par + "\\image\\"+filename)
            which_side(img)
        end = time.time()
        #print("Time consumed:", end - start)