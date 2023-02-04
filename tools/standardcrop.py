import cv2
import imutils
import numpy as np
import joblib
import os
import argparse
import matplotlib.pyplot as plt
par = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

def mycrop(img, side="152"):
    print("side:",side)
    area = joblib.load(par + '\\dump\\area_' + side + '.pkl')
    print("img",img.shape)
    area['top'] = 850
    area['right']= 2200
    area["left"] = 1300
    area["bottom"] = 1350
    print("area:",area)
    img = img[area['top']:area['bottom'], area['left']:area['right']]
    '''cv2.namedWindow('1',0)
    cv2.imshow("1", img)
    while(1):
        if cv2.waitKey(1)==ord('n'):
            break'''
    img = cv2.resize(img, (1000, 1000))
    plt.imshow(img)
    plt.show()
    return img

def nothing(x):
    pass

# This module enables adjusting the tracking area in case of relocation of cameras.

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Type')
    parser.add_argument("--type", type=str, default="m")
    args = parser.parse_args()
    side = args.type
    if side != 'm' and side != 'n':
        print("Invalid type!")
    img = cv2.imread(par +"\\dump\\reference_" + side + ".jpg")
    #img = imutils.resize(img, width=2800)

    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.createTrackbar('left','img',50,2550, nothing)
    cv2.createTrackbar('right','img',1000,2550, nothing)
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
            joblib.dump(area, par  + '\\dump\\area_'+ side + '.pkl')
            check = joblib.load(par  + '\\dump\\area_'+ side + '.pkl')
            print(check)
            print("已保存")
            break