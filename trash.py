import cv2
import imutils
import numpy as np
import joblib
import os
import math
import torch
from standardcrop import mycrop
from check_valid import if_valid
from PIL import Image
import time
from sympy import *
from sympy.abc import x, y

# 多边形提取ROI大失败 >_<

if __name__ == '__main__':
    for filename in os.listdir("image_154"):
        if filename.endswith('jpg'):
            img = cv2.imread("image_154\\"+filename)
            if not if_valid(img, "154"):
                continue
            roi = mycrop(img, "154")
            roi = roi[:,:600]
            cv2.namedWindow('1', 0)
            cv2.imshow("1", roi)
            while(1):
                if cv2.waitKey(1)==ord('n'):
                    break
            #for line part
            roi_canny = cv2.Canny(roi, 40, 80, apertureSize=3)
            # (1000, 600)
            cv2.namedWindow('1', 0)
            cv2.imshow("1", roi_canny)
            while(1):
                if cv2.waitKey(1)==ord('n'):
                    break
            kernel = np.ones((5, 5), np.uint8)
            roi_canny = cv2.dilate(roi_canny, kernel = kernel, iterations=1)
            minLineLength = 100
            maxLineGap = 50

            split = [roi_canny[:330,:200], roi_canny[250: 600, 150:,], roi_canny[600: , 250: ], roi_canny[200:, :300]]
            # UP
            equations = []
            canvas = np.zeros_like(roi_canny)
            for i, r in enumerate(split):
                lines = cv2.HoughLinesP(r, 1, np.pi/180, 100, minLineLength, maxLineGap)
                #print(len(lines))                
                if lines is None:
                    continue
                if i == 0:
                    candidate_score = 0
                    candidate = None
                    for i in range(len(lines)):
                        for x1,y1,x2,y2 in lines[i]:
                            if x1==x2: continue
                            k = (y1-y2)/(x1-x2)
                            b = (y1-y2)/(x1-x2)*(-x1)+y1
                            area = -b**2/k
                            #print(area)
                            if k < 0 and k > -2.5 and area > candidate_score and x1 > 50 and x2 < 120:
                                candidate = ((x1,y1), (x2,y2))
                                candidate_score = area
                    if candidate is not None:
                        cv2.line(canvas, candidate[0], candidate[1], (255,255,255), 3) # Save the line lowest
                        x1, y1 = candidate[0]
                        x2, y2 = candidate[1]
                        equation = (y1-y2)*(x1-x2)*(x - x1) + y1 - y
                        equations.append(equation)
                    else:
                        pass
                elif i == 1:
                    candidate_score = -1000
                    candidate = None
                    for i in range(len(lines)):
                        for x1,y1,x2,y2 in lines[i]:
                            if x1==x2: continue
                            b = (y1-y2)/(x1-x2)*(-x1-150)+y1+250
                            k = (y1-y2)/(x1-x2)
                            if k >0.5 and b > candidate_score:
                                candidate = ((x1+150, y1+250), (x2+150, y2+250))
                                candidate_score = b
                    if candidate is not None:
                        x1, y1 = candidate[0]
                        x2, y2 = candidate[1]
                        cv2.line(canvas, candidate[0], candidate[1], (255,255,255), 3)
                        equation = (y1-y2)/(x1-x2)*(x - x1) + y1 - y
                        equations.append(equation)
                
                elif i == 2:
                    candidate_score = -1000
                    candidate = None
                    for i in range(len(lines)):
                        for x1,y1,x2,y2 in lines[i]:
                            if x1==x2: continue
                            b = (y1-y2)/(x1-x2)*(-x1-250)+y1+600
                            k = (y1-y2)/(x1-x2)
                            score = y1 + y2
                            if k < -0.5 and score > candidate_score:
                                candidate = ((x1+250, y1+600), (x2+250, y2+600))
                                candidate_score = score
                    if candidate is not None:
                        x1, y1 = candidate[0]
                        x2, y2 = candidate[1]                        
                        cv2.line(canvas, candidate[0], candidate[1], (255,255,255), 3)
                        equation = (y1-y2)/(x1-x2)*(x - x1) + y1 - y
                        equations.append(equation)
                
                elif i ==3:
                    candidate_score = -1000
                    candidate = None
                    for i in range(len(lines)):
                        for x1,y1,x2,y2 in lines[i]:
                            if x1==x2: continue
                            b = (y1-y2)/(x1-x2)*(-x1)+ y1 + 200
                            k = (y1-y2)/(x1-x2)
                            if k > 0 and b > candidate_score:
                                candidate_score = b
                                candidate = ((x1, y1+200), (x2, y2+200))
                    if candidate is not None:
                        x1, y1 = candidate[0]
                        x2, y2 = candidate[1]                        
                        cv2.line(canvas, candidate[0], candidate[1], (255,255,255), 3)
                        equation = (y1-y2)/(x1-x2)*(x - x1) + y1 - y
                        equations.append(equation)                    
            if len(equations) < 4: continue
            
            for i in range(4):
                if i == 3:
                    print(solve([equations[3], equations[0]], [x, y]))
                else:
                    print(solve([equations[i], equations[i+1]], [x, y]))

            cv2.namedWindow('1', 0)
            cv2.imshow("1", canvas)
            while(1):
                if cv2.waitKey(1)==ord('n'):
                    break
# up: (300, 200)