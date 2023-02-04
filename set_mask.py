from logging import raiseExceptions
import cv2
import imutils
import numpy as np
import joblib
import os
from tools.standardcrop import mycrop
from check_valid import if_valid
import argparse

global pts # 用于存放点
def draw_roi(event, x, y, flags, param):
    img_click = img_cp.copy()
    if event == cv2.EVENT_LBUTTONDOWN:  # 点击左键，选择点
        pts.append((x, y))
    if event == cv2.EVENT_RBUTTONDOWN:  # 点击右键，撤销上一次选择的点
        pts.pop()
    if event == cv2.EVENT_MBUTTONDOWN:  # 点击中键，绘制并显示轮廓
        mask = np.zeros(img_cp.shape, np.uint8) #初始化mask为原图大小，像素全为黑
        points = np.array(pts, np.int32)
        points = points.reshape((-1, 1, 2))

        mask = cv2.polylines(mask, [points], True, (255, 255, 255), 2) #黑色背景，白色边框，未填充
        mask2 = cv2.fillPoly(mask.copy(), [points], (0, 255, 0))  # 黑色背景，绿色填充区域

        #把mask2叠加到原图像
        show_image = cv2.addWeighted(src1=img_cp, alpha=0.8, src2=mask2, beta=0.2, gamma=0)
        if len(pts) > 1:
            # 画线
            for i in range(len(pts) - 1):
                cv2.circle(show_image, pts[i], 3, (255, 255, 255), -1)  # x ,y 为鼠标点击地方的坐标
                cv2.line(img=show_image, pt1=pts[i], pt2=pts[i + 1], color=(0, 255, 255), thickness=1)
        cv2.imshow("cut_image", show_image)
        #cv2.waitKey(0)

    if len(pts) > 0:
        # 将pts中的最后一点画出来
        cv2.circle(img_click, pts[-1], 3, (0, 255, 0), -1)

    if len(pts) > 1:
        # 画线
        for i in range(len(pts) - 1):
            cv2.circle(img_click, pts[i],3, (0, 255, 0), -1)  # x ,y 为鼠标点击地方的坐标
            cv2.line(img=img_click, pt1=pts[i], pt2=pts[i + 1], color=(255, 0, 0), thickness=1)

    cv2.imshow(current_window, img_click)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Type')
    parser.add_argument("--type", type=str, default="154")
    args = parser.parse_args()
    if args.type[-1] == '4':
        for filename in os.listdir("154"):
            if filename.endswith('jpg'):
                pts = []  # 用于存放点
                img = cv2.imread("154\\"+filename)
                #if not if_valid(img, "154"):
                    #continue
                img = mycrop(img, "154")
                # cv2.imwrite(r"dataset_154/img/" + filename, img)

                img_cp = img.copy()
                current_window = filename + "can"
                cv2.namedWindow(current_window, cv2.WINDOW_FREERATIO)
                cv2.moveWindow(current_window,100,10)
                cv2.setMouseCallback(current_window, draw_roi)
                print("[INFO] 单击左键: 选择点, 单击右键: 删除上一次选择的点, 单击中键: 确定ROI区域")
                #print("[INFO] 按‘S’确定选择区域并保存")
                print("[INFO] 按 ESC 保存退出")
                while True:
                    if cv2.waitKey(1) == 27: #按ESC保存区域坐标并退出
                        cv2.destroyAllWindows()
                        print("[INFO] ROI坐标已保存到本地")
                        break
                
                points = np.array(pts, np.int32)
                mask = np.zeros_like(cv2.cvtColor(img,cv2.COLOR_RGB2GRAY), np.uint8)
                mask = cv2.fillPoly(mask, [points], 50)
                pts = []

                img_cp = img.copy()
                current_window = filename + "button"
                cv2.namedWindow(current_window, cv2.WINDOW_FREERATIO)
                cv2.moveWindow(current_window,100,10)
                cv2.setMouseCallback(current_window, draw_roi)
                print("[INFO] 单击左键: 选择点, 单击右键: 删除上一次选择的点, 单击中键: 确定ROI区域")
                #print("[INFO] 按‘S’确定选择区域并保存")
                print("[INFO] 按 ESC 保存退出")
                while True:
                    if cv2.waitKey(1) == 27: #按ESC保存区域坐标并退出
                        cv2.destroyAllWindows()
                        print("[INFO] ROI坐标已保存到本地")
                        break
                
                points = np.array(pts, np.int32)
                #mask = np.zeros_like(cv2.cvtColor(img,cv2.COLOR_RGB2GRAY), np.uint8)
                mask = cv2.fillPoly(mask, [points], 100)
                pts = []

                img_cp = img.copy()
                current_window = filename + "line"
                cv2.namedWindow(current_window, cv2.WINDOW_FREERATIO)
                cv2.moveWindow(current_window,100,10)
                cv2.setMouseCallback(current_window, draw_roi)
                print("[INFO] 单击左键: 选择点, 单击右键: 删除上一次选择的点, 单击中键: 确定ROI区域")
                #print("[INFO] 按‘S’确定选择区域并保存")
                print("[INFO] 按 ESC 保存退出")
                while True:
                    if cv2.waitKey(1) == 27: #按ESC保存区域坐标并退出
                        cv2.destroyAllWindows()
                        print("[INFO] ROI坐标已保存到本地")
                        break
                
                points = np.array(pts, np.int32)
                #mask = np.zeros_like(cv2.cvtColor(img,cv2.COLOR_RGB2GRAY), np.uint8)
                mask = cv2.fillPoly(mask, [points], 200)

                cv2.imwrite(r"dataset_154/mask/" + filename[:-4]+"_m.jpg", mask)
            #mask_rgb = cv2.addWeighted(src1=img_cp, alpha=0.9, src2=mask2, beta=0.1, gamma=0)#cv2.fillPoly(img_cp, [points],  (0, 255, 0))
        
        else:
            for filename in os.listdir("154"):
                if filename.endswith('jpg'):
                    pts = []  # 用于存放点
                    img = cv2.imread("154\\"+filename)
                
                    img = mycrop(img, "154")
                    cv2.imwrite(r"dataset_154/img/" + filename, img)

                    img_cp = img.copy()
                    current_window = filename + "can"
                    cv2.namedWindow(current_window, cv2.WINDOW_FREERATIO)
                    cv2.moveWindow(current_window,100,10)
                    cv2.setMouseCallback(current_window, draw_roi)
                    print("[INFO] 单击左键: 选择点, 单击右键: 删除上一次选择的点, 单击中键: 确定ROI区域")
                    #print("[INFO] 按‘S’确定选择区域并保存")
                    print("[INFO] 按 ESC 保存退出")
                    while True:
                        if cv2.waitKey(1) == 27: #按ESC保存区域坐标并退出
                            cv2.destroyAllWindows()
                            print("[INFO] ROI坐标已保存到本地")
                            break
                    
                    points = np.array(pts, np.int32)
                    mask = np.zeros_like(cv2.cvtColor(img,cv2.COLOR_RGB2GRAY), np.uint8)
                    mask = cv2.fillPoly(mask, [points], 50)
                    pts = []

                    img_cp = img.copy()
                    current_window = filename + "button"
                    cv2.namedWindow(current_window, cv2.WINDOW_FREERATIO)
                    cv2.moveWindow(current_window,100,10)
                    cv2.setMouseCallback(current_window, draw_roi)
                    print("[INFO] 单击左键: 选择点, 单击右键: 删除上一次选择的点, 单击中键: 确定ROI区域")
                    #print("[INFO] 按‘S’确定选择区域并保存")
                    print("[INFO] 按 ESC 保存退出")
                    while True:
                        if cv2.waitKey(1) == 27: #按ESC保存区域坐标并退出
                            cv2.destroyAllWindows()
                            print("[INFO] ROI坐标已保存到本地")
                            break
                    
                    points = np.array(pts, np.int32)
                    #mask = np.zeros_like(cv2.cvtColor(img,cv2.COLOR_RGB2GRAY), np.uint8)
                    mask = cv2.fillPoly(mask, [points], 100)

                    img_cp = img.copy()
                    current_window = filename + "line"
                    cv2.namedWindow(current_window, cv2.WINDOW_FREERATIO)
                    cv2.moveWindow(current_window, 100, 10)
                    cv2.setMouseCallback(current_window, draw_roi)
                    print("[INFO] 单击左键: 选择点, 单击右键: 删除上一次选择的点, 单击中键: 确定ROI区域")
                    # print("[INFO] 按‘S’确定选择区域并保存")
                    print("[INFO] 按 ESC 保存退出")
                    while True:
                        if cv2.waitKey(1) == 27:  # 按ESC保存区域坐标并退出
                            cv2.destroyAllWindows()
                            print("[INFO] ROI坐标已保存到本地")
                            break

                    points = np.array(pts, np.int32)
                    # mask = np.zeros_like(cv2.cvtColor(img,cv2.COLOR_RGB2GRAY), np.uint8)
                    mask = cv2.fillPoly(mask, [points], 200)

                    cv2.imwrite(r"dataset_154/mask/" + filename[:-4]+"_m.jpg", mask)

