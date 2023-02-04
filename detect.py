from opcua import Client
import threading
import time
import cv2
import imutils
import numpy as np
import joblib
import os
import math
import torch
from PIL import Image
import sys
import keyboard
from getroi import get_mask_154
import datetime
from alarm import raise_alarm
from check import check, draw_anomaly

class ImageDection(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        try:
            #cap = cv2.VideoCapture('rtsp://sys9bh:njpservice@123@172.17.227.152:554')
            #_, frame = cap.read()
            #print("Collecting... Press ENTER to quit")
            #cv2.imwrite(os.getcwd() + '\\image_152\\' + str(int(time.time())) + '.jpg', frame)
            cap = cv2.VideoCapture('rtsp://sys9bh:njpservice@123@172.17.227.154:554')
            _, frame = cap.read()
            process = Image_process(frame=frame)
            process.start()
            #cv2.imwrite(os.getcwd() + '\\image_154\\' + str(int(time.time())) + '.jpg', frame)
            self.stop()

        except Exception as e:
            print('Capture exception', e)
            self.stop()

    def stop(self):
        print("停止读入")

class Image_process(threading.Thread):
    def __init__(self, frame):
        threading.Thread.__init__(self)
        self.frame = frame

    def run(self):
        try:
            now = datetime.datetime.now()
            print("Checking" + now.strftime("%Y-%m-%d %H:%M:%S"))
            areas, locations = get_mask_154(self.frame)
            if len(areas) != 3:
                return
            namelist = ["can", "button", "line"]
            for i in range(3):
                result = check(areas[i], namelist[i])
                if not result:
                    print("Damage detected")
                    raise_alarm()
                draw_anomaly(self.frame, locations[i], result)

            self.stop()

        except Exception as e:
            print('Process exception', e)
            self.stop()

    def stop(self):
        print("Detection ends")

class IOHandler(object):
    def datachange_notification(self, node, val, attr):
        if val:
            #print("exchanging")
            imgdetect=ImageDection()
            imgdetect.daemon = True
            imgdetect.start()

class OpcuaClient(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.client = Client("opc.tcp://172.17.227.111:49320/")
        self.client.set_security_string(r"Basic256Sha256,SignAndEncrypt,license/certificate.der,license/private_key.pem")
        self.client.application_uri = "urn:example.org:FreeOpcUa:python-opcua"
        self.client.set_user('sys9bh')
        self.client.set_password('njpservice@123')
        self.client.secure_channel_timeout = 600000
        self.client.session_timeout = 15000

    def InitClient(self):
        try:
            self.client.connect()
            iohanlder = IOHandler()
            ioVar = self.client.get_node('ns=2;s=ST035.PLC.GLOBAL.M114_M921ZVisu.InSeparator')
            ioChange = self.client.create_subscription(1000, iohanlder)
            ioChange.subscribe_data_change(ioVar)
        except Exception as e:
            print('Client exception', e)


    def run(self):
        self.InitClient()
        print ("client线程启动")

    def stop(self):
        self.client.disconnect()
        print("client线程退出")


if __name__ == "__main__":
    global cnt
    start = time.time()
    client = OpcuaClient()
    client.daemon = True
    while(True):
        try:
            client = OpcuaClient()
            client.daemon = True
            client.start()
            
            while(True):
                end = time.time()
                wait_time = end - start
                if wait_time > 60*5:
                    sys.exit()
                time.sleep(0.1)

            #cv2.imwrite(os.getcwd() + '\\cur_report\\' + 'final.jpg', cv2.imread(os.getcwd() + "\\OK.jpg"))
            #cv2.imwrite(os.getcwd() + '\\current\\' + 'current.jpg', cv2.imread(os.getcwd() + "\\waiting.jpg"))
            #cv2.imwrite(os.getcwd() + '\\current\\' + 'gr.jpg', cv2.imread(os.getcwd() + "\\waiting.jpg"))
        except Exception as e:
            print("Server error, thread len: ", len(threading.enumerate()))
            if cv2.waitKey(2000) == 27:
                exit()
            continue