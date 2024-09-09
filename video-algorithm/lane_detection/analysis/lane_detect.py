import copy
import os
import time

import cv2

from ai_common import setting
from ai_common.alarm_util import package_business_alarm
from ai_common.log_util import logger
from analysis.extract_param import param


def compare_img_hist(img1, img2):
    """
    Compare the similarity of two pictures using histogram
        Attention: this is a comparision of similarity, using histogram to calculate

        For example:
         1. img1 and img2 are both 720P .PNG file,
            and if compare with img1, img2 only add a black dot(about 9*9px),
            the result will be 0.999999999953

    :param img1: img1 in MAT format(img1 = cv2.imread(image1))
    :param img2: img2 in MAT format(img2 = cv2.imread(image2))
    :return: the similarity of two pictures
    """
    # Get the histogram data of image 1, then using normalize the picture for better compare
    degree = 0
    H1 = cv2.calcHist([img1], [1], None, [256], [0, 256])
    H2 = cv2.calcHist([img2], [1], None, [256], [0, 256])
    for i in range(len(H1)):
        if H1[i] != H2[i]:
            degree = degree + (1 - abs(H1[i] - H2[i]) / max(H1[i], H2[i]))
        else:
            degree += 1
    degree = degree / len(H1)
    return degree


def process_result(img_array, data):
    params = param(data)
    flag = True
    img_array_copy = copy.copy(img_array)
    pos_point = [(params.w1, params.h1), (params.w2, params.h2), (params.w3, params.h3), (params.w4, params.h4)]
    x_min = pos_point[0][0]
    x_max = pos_point[1][0]
    y_min = pos_point[0][1]
    y_max = pos_point[2][1]
    if data['k8sName'] in setting.background.keys() and setting.frame.keys():
        if setting.frame[data['k8sName']] == 0:
            background_cut = img_array_copy[int(y_min):int(y_max), int(x_min):int(x_max)]
            cv2.rectangle(img_array_copy, pos_point[0], pos_point[3], (0, 255, 0), params.line_thickness)
            setting.background[data['k8sName']] = background_cut
            setting.frame[data['k8sName']] += 1
        # a hour clear frame, redefine 1
        if int(time.time() - setting.starttime) > 3600:
            setting.frame[data['k8sName']] = 1
            setting.starttime = time.time()
            return img_array_copy
        else:
            img_cut = img_array_copy[int(y_min):int(y_max), int(x_min):int(x_max)]
            same = compare_img_hist(setting.background[data['k8sName']], img_cut)
            logger.debug(f"same:{same}")
            if same > params.same:
                # normal
                cv2.rectangle(img_array_copy, pos_point[0], pos_point[3], (0, 255, 0), params.line_thickness)
                logger.info(f"检测无异常")
            else:
                # warnning
                flag = False
                cv2.rectangle(img_array_copy, pos_point[0], pos_point[3], (0, 0, 255), params.line_thickness)
                cv2.putText(img_array_copy, 'Do not place objects', pos_point[0], cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), params.line_thickness)
                logger.info(f"检测异常，区域内被占用")

        setting.frame[data['k8sName']] += 1

    # --------------------------------------- #
    if setting.IMG_VERIFY == 1:
        cv2.imshow('laneoccupy_detect.jpg', img_array_copy)
        cv2.waitKey()
        cv2.destroyAllWindows()

    if setting.SAVE_IMG == 1:
        if not os.path.exists(setting.image_save_path):
            os.makedirs(setting.image_save_path)
        cv2.imwrite(f"{setting.image_save_path}/{int(time.time())}.jpg", img_array_copy)

    if not flag:
        alarm_label = 'lane'
        logger.info('Driveway found to be occupied')
        alarm_type = "车道占用"
        alarm_info = "{\"AlarmMsg\":\"Driveway found to be occupied!!!\"}"
        # 组装告警信息
        package_business_alarm(alarm_info, alarm_type, img_array_copy, img_array, data['deviceId'], alarm_label)

    return img_array_copy






