import copy
import os
import time

import cv2

from ai_common import setting
from ai_common.alarm_util import package_business_alarm
from ai_common.log_util import logger
from ai_common.np_util import load_poly_area_data, cal_iou
from ai_common.polylines import put_region
from ai_common.util.plots import plot_one_box
from analysis.extract_param import param


def judgment(hand, head, smoke, data, img, img_array):
    pts, w1, h1 = load_poly_area_data(data)
    params = param(data)
    img_array_copy = copy.copy(img_array)
    flag = True

    if params.hand_verify_switch == False and params.head_verify_switch == False: # 不进行校验
        for s in smoke:
            if s is not None:
                plot_one_box(s, img_array_copy, label=smoke[s], color=(255, 0, 0), line_thickness=params.line_thickness)
                flag = False
                pass
            pass
        pass
    if params.hand_verify_switch == False and params.head_verify_switch == True: # 头校验
        iou_list = []
        # 对头和香烟进行判断
        for h in head:
            for s in smoke:
                iou = cal_iou(h, s)
                if iou > 0:
                    iou_list.append(iou)
                    s_h = (h[2] - h[0]) * (h[3] - h[1])
                    s_s = (s[2] - s[0]) * (s[3] - s[1])
                    h_divide_s = s_s / s_h
                    if h_divide_s < params.s_h_ratio:
                        flag = False
                        plot_one_box(h, img_array_copy, label=head[h], color=(255, 0, 0), line_thickness=params.line_thickness)
                        plot_one_box(s, img_array_copy, label=smoke[s], color=(0, 0, 255), line_thickness=params.line_thickness)
                        pass
                    pass
                pass
            pass
        pass
    if params.hand_verify_switch == True and params.head_verify_switch == False: # 手校验
        for h in hand:
            for s in smoke:
                iou = cal_iou(h, s)
                if iou > 0:
                    flag = False
                    plot_one_box(h, img_array_copy, label=hand[h], color=(255, 0, 0), line_thickness=params.line_thickness)
                    plot_one_box(s, img_array_copy, label=smoke[s], color=(0, 0, 255), line_thickness=params.line_thickness)
                    pass
                pass
            pass
        pass
    if params.hand_verify_switch == True and params.head_verify_switch == True: # 手、头校验
        iou_list = []
        for h in hand:
            for s in smoke:
                iou = cal_iou(h, s)
                if iou > 0:
                    for h0 in head:
                        iou = cal_iou(h0, s)
                        if iou > 0:
                            iou_list.append(iou)
                            s_h0 = (h0[2] - h0[0]) * (h0[3] - h0[1])
                            s_s = (s[2] - s[0]) * (s[3] - s[1])
                            h0_divide_s = s_s / s_h0
                            if h0_divide_s < params.s_h_ratio:
                                flag = False
                                plot_one_box(h, img_array_copy, label=hand[h], color=(255, 0, 0), line_thickness=params.line_thickness)
                                plot_one_box(s, img_array_copy, label=smoke[s], color=(0, 0, 255), line_thickness=params.line_thickness)
                                plot_one_box(h0, img_array_copy, label=head[h0], color=(255, 0, 0), line_thickness=params.line_thickness)
                                pass
                            pass
                        pass
                    pass
                pass
            pass
        pass
    if setting.IMG_VERIFY == 1:
        if params.detect_area_flag:
            if params.polyline:
                img_array_copy = put_region(img_array_copy, w1, h1, pts, params.line_thickness)
        cv2.imshow('smoke_detect', img_array_copy)
        cv2.waitKey()
        cv2.destroyAllWindows()


    if setting.SAVE_IMG == 1:
        if not os.path.exists(setting.image_save_path):
            os.makedirs(setting.image_save_path)
        cv2.imwrite(f"{setting.image_save_path}/{int(time.time())}.jpg", img_array_copy)


    if not flag:
        if params.detect_area_flag:
            if params.polyline:
                img_array_copy = put_region(img_array_copy, w1, h1, pts, params.line_thickness)
        logger.info('Detect stuff smoking!')
        alarm_label = 'smoke'
        alarm_type_smoking = "抽烟"
        alarm_info_smoking = "{\"AlarmMsg\": \"Detect stuff smoking.\"}"
        # 组装告警信息
        package_business_alarm(alarm_info_smoking, alarm_type_smoking, img_array_copy, img_array, data['deviceId'], alarm_label)