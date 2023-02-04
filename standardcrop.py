import cv2
import imutils
import numpy as np
import joblib
import os
import argparse

#par = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

def mycrop(img, side = "154"):
    print("side:", side)
    area = joblib.load('dump\\area_' + side + '.pkl')
    img = img[area['top']:area['bottom'], area['left']:area['right']]

    '''cv2.namedWindow('1',0)
    cv2.imshow("1", img)
    while(1):
        if cv2.waitKey(1)==ord('n'):
            break'''
    img = cv2.resize(img, (1440, 1440))
    return img


def nothing(x):
    pass

# This module enables adjusting the tracking area in case of relocation of cameras.

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Type')
    parser.add_argument("--type", type=str, default="154")
    args = parser.parse_args()
    type = args.type
    if type != '154' and type != '152' and type != "154_c":
        print("Invalid type!")
    if type[-1] == 'c':
        img = cv2.imread("dump\\ref_" + type[:-2] + ".jpg")
    else:
        img = cv2.imread("dump\\ref_" + type + ".jpg")
    #img = imutils.resize(img, width=2800)
    if type == "154":
        cv2.namedWindow('img', cv2.WINDOW_NORMAL)
        cv2.createTrackbar('left','img',50,2550, nothing)
        cv2.createTrackbar('right','img',1000,2850, nothing)
        cv2.createTrackbar('top','img',50,1439, nothing)
        cv2.createTrackbar('bottom','img',500,1439, nothing) # Trackbars for changing the area.
    elif type == "152":
        cv2.namedWindow('img', cv2.WINDOW_NORMAL)
        cv2.createTrackbar('left','img',50,1910, nothing)
        cv2.createTrackbar('right','img',1000,1910, nothing)
        cv2.createTrackbar('top','img',50,1439, nothing)
        cv2.createTrackbar('bottom','img',500,1439, nothing) # Trackbars for changing the area.
    elif type == "154_c":
        cv2.namedWindow('img', cv2.WINDOW_NORMAL)
        cv2.createTrackbar('left','img',50,1910, nothing)
        cv2.createTrackbar('right','img',1000,1910, nothing)
        cv2.createTrackbar('top','img',50,1430, nothing)
        cv2.createTrackbar('bottom','img',500,1430, nothing) # Trackbars for changing the area.
    print("[INFO] 按‘S’确定选择区域保存并退出")
    # Please maximize the window. Press 's' to save the area
    while(1):
        left = cv2.getTrackbarPos('left','img')
        right = cv2.getTrackbarPos('right','img')
        top = cv2.getTrackbarPos('top','img')
        bottom = cv2.getTrackbarPos('bottom','img')
        if top < bottom and left < right:
            sample = img[top:bottom, left:right]
        cv2.imshow('img',sample)
        
        if cv2.waitKey(1)==ord('s'): # save the parameters
            area = {"left":left, "right":right, "top":top, "bottom":bottom}
            joblib.dump(area, 'dump\\area_'+ type + '.pkl')
            check = joblib.load('dump\\area_'+ type + '.pkl')
            print(check)
            print("已保存")
            break





