from lib2to3.refactor import get_all_fix_names
from pickletools import uint8
from re import M
from threading import main_thread
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
from unet import get_model
import torchvision
#from unetpp import getL3, getL2
from model import get_model
from torchvision import transforms
from getroi import get_mask_154
from alarm import raise_alarm

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def check(area, part):
    BGRarea = area[:, :, ::-1]
    a=Image.fromarray(BGRarea)
    input_tensor = preprocess(a)
    input_batch = input_tensor.unsqueeze(0)
    model_pth = os.getcwd() + r'/ckpt/'+ part + '_mobilenet_v2-grease-detection.pth'
    model = get_model(model_pth, False)
    model.eval()

    # weight is for activate/deactivate alarm, when testing, you can deactivate alarm to avoid interupting production
    #  weight = torch.tensor([1.0, 1.0]).to(torch.float32)  active
    weight = torch.tensor([0.0, 1.0]).to(torch.float32)
    with torch.no_grad():
        output = model(input_batch)
    probabilities = torch.nn.functional.softmax(output[0], dim=0) * weight
    result = torch.argmax(probabilities)

    if not result.item():
        cv2.imwrite(os.getcwd() + '\\detected\\' + str(int(time.time()*1000)) + '_ok.jpg', area)
        return 0
    else:
        return 1

def draw_anomaly(img, location, normal):
    final_img = mycrop(img, "154")
    if normal:
        final_img = cv2.resize(final_img, (1000, 1000))
        final_img[:200, -200:] = [0, 255, 0]
        final_img = cv2.putText(final_img, "OK", (850, 130), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)
        cv2.imwrite(r'cur_report/report.jpg', final_img)

    else:    
        y, h, x, w = location[0], location[1], location[2], location[3]
        final_img = cv2.rectangle(final_img, (x, y), (x+w, y+h), (0, 0, 255), 3, 4)
        final_img = cv2.resize(final_img, (1000, 1000))
        final_img[:200, -200:] = [0, 0, 255]
        final_img = cv2.putText(final_img, "NG", (850, 130), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)
        cv2.imwrite(r'cur_report/report.jpg', final_img)
    
    return

if __name__ == '__main__':
    for filename in os.listdir("seg_test154"):
        if filename.endswith('jpg'):
            img = cv2.imread("seg_test154\\"+filename)
            areas, locations = get_mask_154(img)
            if len(areas) != 3:
                continue
            
            result = 1
            namelist = ["can", "button", "line"]
            for i in range(3):
                result = check(areas[i], namelist[i])
                if not result:
                    print("Damage detected")
                    draw_anomaly(img, locations[i], result) #currently designed for one damage
                    raise_alarm()
                    break
            if result:
                draw_anomaly(img, locations[i], result)
            if cv2.waitKey(2000) == 27:
                exit()