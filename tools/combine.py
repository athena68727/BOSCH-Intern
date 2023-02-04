import numpy as np
import joblib
import os
import cv2
import datetime

par = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

# 读取ROI区域坐标
def conbine(side):
    configdata1 =joblib.load(par+'\\dump\\Area_configU.pkl').get('ROI')
    points1 = np.array(configdata1, np.int32)
    points1 = points1.reshape((-1, 1, 2))
    configdata2 =joblib.load(par+'\\dump\\Area_configD.pkl').get('ROI')
    points2 = np.array(configdata2, np.int32)
    points2 = points2.reshape((-1, 1, 2))
    points=np.vstack((points1,points2))

    saved_data = {
                    "ROI": points
                }
    joblib.dump(value=saved_data, filename=par+"\\dump\\roi_" + side +".pkl")


    '''configdata =joblib.load(par+'\\dump\\Area_config.pkl').get('ROI')
    roi_S = np.array(configdata, np.int32)
    roi_S = roi_S.reshape((-1, 1, 2))

    saved_data = {
                    "ROI": roi_S
                }
    joblib.dump(value=saved_data, filename=par+"\\dump\\roi_S.pkl")'''

def get_roi(img, side):
    configdata1 =joblib.load(par+'\\dump\\Area_configU_' + side + '.pkl').get('ROI')
    points1 = np.array(configdata1, np.int32)
    points1 = points1.reshape((-1, 1, 2))
    configdata2 =joblib.load(par+'\\dump\\Area_configD_' + side + '.pkl').get('ROI')
    points2 = np.array(configdata2, np.int32)
    points2 = points2.reshape((-1, 1, 2))

    mask_rgb = np.zeros(img.shape, np.uint8)
    mask_rgb = cv2.fillPoly(mask_rgb, [points1], (255, 255, 255))
    mask_rgb = cv2.fillPoly(mask_rgb, [points2], (255, 255, 255))
    mask = cv2.fillPoly(np.zeros(img.shape, np.uint8), [points1], (0, 255, 0))
    mask = cv2.fillPoly(mask, [points2], (0, 255, 0))

    _roi = cv2.bitwise_and(mask_rgb, img, )
    _roi_withblack = _roi.copy()
    '''cv2.namedWindow('1', 0)
    cv2.imshow("1", mask)
    while(1):
        if cv2.waitKey(1)==ord('n'):
            break'''
    # 对ROI区域作平滑处理(高斯滤波)
    _roi = cv2.GaussianBlur(_roi, (3, 3), 0.1)

    _roi = cv2.Canny(_roi, 80, 160)
    img_gr = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gr = cv2.Canny(img_gr, 80, 160)
    #img_gr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    '''cv2.namedWindow('1', 0)
    cv2.imshow("1", img_gr)
    while(1):
        if cv2.waitKey(1)==ord('n'):
            break'''
    # 消除ROI边界
    _roi = cv2.polylines(_roi, [points1], isClosed=True, color=(0, 0, 0), thickness=3)
    _roi = cv2.polylines(_roi, [points2], isClosed=True, color=(0, 0, 0), thickness=3)
    #img_gr = cv2.polylines(img_gr, [points1], isClosed=True, color=(2, 255, 2), thickness=5)
    #img_gr = cv2.polylines(img_gr, [points2], isClosed=True, color=(2, 255, 2), thickness=5)
    # 膨胀处理
    kernel = np.ones((3, 3), np.uint8)
    _roi = cv2.dilate(_roi, kernel = kernel, iterations=1)
    '''cv2.namedWindow('1',0)
    cv2.imshow("1", _roi)
    while(1):
        if cv2.waitKey(1)==ord('n'):
            break'''
    gr = _roi + img_gr
    gr[:50, :300] = np.ones_like(gr[:50, :300])*255
    now = datetime.datetime.now()
    gr = cv2.putText(gr, now.strftime("%Y-%m-%d %H:%M:%S"), (1,30 ), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
    cv2.imwrite(par + '\\current\\' + 'gr.jpg', gr)
    return _roi, _roi_withblack, mask







