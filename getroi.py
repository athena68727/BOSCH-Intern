from lib2to3.refactor import get_all_fix_names
from pickletools import uint8
from threading import main_thread
import cv2
import imutils
import numpy as np
import joblib
import os
import math
import torch
from tools.standardcrop import mycrop
from check_valid import if_valid
from PIL import Image
import time
from unet import get_model
import torchvision
#from unetpp import getL3, getL2
import datetime
import matplotlib.pyplot as plt


def trans(img, size):
    # transform to square and padding
    h, w = img.shape[0:2]
    _w = _h = size
    scale = min(_h / h, _w / w)
    h = int(h * scale)
    w = int(w * scale)
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
    top = (_h - h) // 2
    left = (_w - w) // 2
    bottom = _h - h - top
    right = _w - w - left
    new_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return new_img

def get_mask_154(img):
    seg_model = get_model(training=False)
    roi = mycrop(img, "154")
    roi = trans(roi, 1440)
    totensor = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    roi = totensor(roi)
    roi = roi.unsqueeze(0)
    start = time.time()
    with torch.no_grad():
        out = seg_model(roi)

    end = time.time()
    #print("Judge cost", end - start)

    sample_img = roi[0].transpose(0, 1).transpose(1, 2).numpy()*255
    sample_img = sample_img.astype(np.uint8)
    out = out[0].transpose(0,1).transpose(1,2).numpy()
    labels = np.argmax(out, axis=2)

    can = np.zeros_like(labels).astype(np.uint8)
    button = np.zeros_like(labels).astype(np.uint8)
    line = np.zeros_like(labels).astype(np.uint8)
    can[labels == 1] = 255
    button[labels == 2] = 255
    line[labels == 3] = 255

    lst = [can, button, line]
    result = []
    locations = []
    
    mask_show = np.zeros_like(sample_img).astype(np.uint8)
    for i, area in enumerate(lst):
        mask = np.zeros_like(sample_img).astype(np.uint8)
        contours, _= cv2.findContours(area, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            continue
        main_area = max(contours, key = lambda i: len(i))
        x, y, w, h = cv2.boundingRect(main_area)
        main_area = cv2.convexHull(main_area)

        kernel = np.ones((7, 7), np.uint8)
        mask = np.ascontiguousarray(mask)
        mask_show = np.ascontiguousarray(mask_show)
        if i == 0:
            mask_show = cv2.fillConvexPoly(mask_show, main_area, (255, 0, 0))
            mask = cv2.fillConvexPoly(mask, main_area, (255, 255, 255))
            mask = cv2.dilate(mask, kernel = kernel, iterations=1)
            _roi = cv2.bitwise_and(mask, sample_img)
            result.append(_roi[y: y + h, x: x + w, :])
            locations.append([y, h, x, w])
            cv2.imwrite(r'can/' + filename, _roi[y: y + h, x: x + w, :])

        elif i == 1:
            mask_show = cv2.fillConvexPoly(mask_show, main_area, (0, 255, 0))
            mask = cv2.fillConvexPoly(mask, main_area, (255, 255, 255))
            mask = cv2.dilate(mask, kernel = kernel, iterations=1)
            _roi = cv2.bitwise_and(mask, sample_img)
            result.append(_roi[y: y + h, x: x + w, :])
            locations.append([y, h, x, w])
            cv2.imwrite(r'button/' +filename, _roi[y: y + h, x: x + w, :])

        elif i == 2:
            mask_show = cv2.fillConvexPoly(mask_show, main_area, (0, 0, 255))
            mask = cv2.fillConvexPoly(mask, main_area, (255, 255, 255))
            mask = cv2.dilate(mask, kernel = kernel, iterations=1)
            _roi = cv2.bitwise_and(mask, sample_img)
            result.append(_roi[y: y + h, x: x + w, :])
            locations.append([y, h, x, w])
            cv2.imwrite(r'line/' +filename, _roi[y: y + h, x: x + w, :])
            
    seg_result = cv2.addWeighted(src1=sample_img, \
        alpha=0.5, src2=mask_show, beta=0.5, gamma=0)


    seg_result[:50, :400] = np.ones_like(seg_result[:50, :400])*255
    now = datetime.datetime.now()
    cv2.putText(seg_result, now.strftime("%Y-%m-%d %H:%M:%S"), (1,30 ), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.imwrite("current_seg\\current.jpg", seg_result)


    return result, locations
    #cv2.imwrite("seg_result154\\"+filename, result)
    '''cv2.imwrite("seg_result154\\"+filename, result)
    cv2.namedWindow('1', 0)
    cv2.imshow("1", result)
    while(1):
        if cv2.waitKey(0)==ord('n'):
            break'''


if __name__ == '__main__':
    
    #print(*torch.__config__.show().split("\n"), sep="\n")
    #seg_model = mkldnn_utils.to_mkldnn(seg_model)
    print(os.listdir("seg_test154"))
    for filename in os.listdir("seg_test154"):
        if filename.endswith('jpg'):
            img = cv2.imread("seg_test154\\"+filename)
            result, _ = get_mask_154(img)
            cv2.imshow('Seg_result', img)
            cv2.waitKey(0)
            # plt.imshow(result)
            # plt.show()
            # cv2.imwrite("seg_result154\\"+filename, result)





            
            