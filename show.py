import cv2
import os
import numpy as np
import datetime

current_pth = r'current_seg/current.jpg'
report_pth = r'cur_report/report.jpg'


if __name__ == "__main__":
    '''m_u = joblib.load(os.getcwd()+'\\dump\\Area_configU_m.pkl').get('ROI')
    m_d = joblib.load(os.getcwd()+'\\dump\\Area_configD_m.pkl').get('ROI')
    points_mu = np.array(m_u, np.int32)
    points_mu = points_mu.reshape((-1, 1, 2))
    points_md = np.array(m_d, np.int32)
    points_md = points_md.reshape((-1, 1, 2))
    points_m = [points_mu, points_md]

    n_u = joblib.load(os.getcwd()+'\\dump\\Area_configU_n.pkl').get('ROI')
    n_d = joblib.load(os.getcwd()+'\\dump\\Area_configD_n.pkl').get('ROI')
    points_nu = np.array(n_u, np.int32)
    points_nu = points_nu.reshape((-1, 1, 2))
    points_nd = np.array(n_d, np.int32)
    points_nd = points_nd.reshape((-1, 1, 2))
    points_n = [points_nu, points_nd]'''

    while(True):
        points = None
        img = cv2.imread(current_pth)
        report = cv2.imread(report_pth)
        if img is None or report is None: # Read fails
            if cv2.waitKey(2000) == 27:
                exit()
            continue

        img = cv2.resize(img, (500, 500))
        report = cv2.resize(report, (500, 500))
        imgs = np.hstack([img, report])
        cv2.namedWindow('mutil_pic', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("mutil_pic", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("mutil_pic", imgs)
        if cv2.waitKey(2000) == 27:
            exit()
            