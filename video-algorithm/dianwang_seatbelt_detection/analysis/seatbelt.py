import os
import time

import cv2

from ai_common import setting
from ai_common.alarm_util import package_business_alarm
from ai_common.log_util import logger
from ai_common.np_util import person_in_poly_area, is_poi_in_poly, tuple_xyxy, load_poly_area_data, cal_iou
from ai_common.polylines import put_region
from ai_common.util.general import scale_coords
from analysis.extract_param import param

from ai_common.util.plots import plot_one_box
from ai_common.util.torch_utils import time_synchronized


def process_result_seatbelt(high, person, pred_seatbelt, names_seatbelt, data, img, img_array, colors):
    params = param(data)
    img_array_copy = img_array.copy()
    seatbelt = {}
    # s = ''
    pts, w1, h1 = load_poly_area_data(data)
    flag = True
    for i, det in enumerate(pred_seatbelt):
        # s += '%g x %g ' % img.shape[2:]  # logger.info string, 取第三和第四维，对应宽和高 512 x 640

        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img_array_copy.shape).round()

            # # logger.info results
            # for c in det[:,-1].unique():
            #     n = (det[:, -1] == c).sum()
            #     s += f"{n} {names_seatbelt[int(c)]}{'s' * (n > 1)}, "

            # judge results
            for *xyxy, conf, cls in reversed(det):
                # seatbelt_list = ['seatbelt']
                c = int(cls)  # integer class 1， 0
                xyxy = [xyxy[0].item(), xyxy[1].item(), xyxy[2].item(), xyxy[3].item()]

                # label
                if params.hide_labels:
                    label = None
                elif params.hide_conf:
                    if c == 3:
                        label = names_seatbelt[1]
                    else:
                        label = names_seatbelt[0]
                else:
                    if c == 3:
                        label = names_seatbelt[1] + "%.2f" % conf
                    else:
                        label = names_seatbelt[0] + "%.2f" % conf

                if params.detect_area_flag:
                    # 求物体框的中心点
                    object_cx, object_cy = person_in_poly_area(xyxy)
                    # 判断中心点是否在检测框内部
                    if not is_poi_in_poly([object_cx, object_cy], pts):
                        # 不在感兴趣的框内，则继续判断下一个物体。
                        continue
                    if c == 3:
                        if data['tools']['Target_filter']['filter_switch'] == True:
                            if int(data['tools']['Target_filter']['max_seatbelt_xyxy']) > (int(xyxy[2]) - int(xyxy[0])) * (int(xyxy[3]) - int(xyxy[1])) > int(data['tools']['Target_filter']['min_seatbelt_xyxy']):
                                seatbelt[tuple_xyxy(xyxy)] = label
                                pass
                            pass
                        else:
                            seatbelt[tuple_xyxy(xyxy)] = label
                            pass
                        pass
                    pass
                else:
                    if c == 3:
                        if data['tools']['Target_filter']['filter_switch'] == True:
                            if int(data['tools']['Target_filter']['max_seatbelt_xyxy']) > (int(xyxy[2]) - int(xyxy[0])) * (int(xyxy[3]) - int(xyxy[1])) > int(data['tools']['Target_filter']['min_seatbelt_xyxy']):
                                seatbelt[tuple_xyxy(xyxy)] = label
                                pass
                            pass
                        else:
                            seatbelt[tuple_xyxy(xyxy)] = label
                            pass
                        pass
                    pass
                pass
            pass
        pass
    # logger.info(f'seatbelt detect result: {s}Done.')
    # judge
    for seat in seatbelt:
        for p in person:
            if cal_iou(seat, p) > 0:
                pass
            else:
                flag = False
                plot_one_box(p, img_array_copy, label=person[p], color=(0, 0, 255), line_thickness=params.line_thickness)

    # -------------------------------------------- #
    if setting.IMG_VERIFY == 1 or setting.VID_VERIFY == 1:
        if params.detect_area_flag:
            if params.polyline:
                img_array_copy = put_region(img_array_copy, w1, h1, pts, params.line_thickness)
        for p in person:
            plot_one_box(p, img_array_copy, label=person[p], color=(0, 0, 255), line_thickness=params.line_thickness)
        for h in high:
            plot_one_box(h, img_array_copy, label=high[h], color=(0, 0, 255), line_thickness=params.line_thickness)
        for s in seatbelt:
            plot_one_box(s, img_array_copy, label=seatbelt[s], color=(0, 0, 255), line_thickness=params.line_thickness)
        if setting.IMG_VERIFY == 1:
            cv2.imshow('detect_seatbelt.jpg', img_array_copy)
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
        for p in person:
            plot_one_box(p, img_array_copy, label=person[p], color=(255, 0, 0), line_thickness=params.line_thickness)
        alarm_type = "未戴安全帽"
        alarm_label = 'helmet'
        alarm_info = "{\"AlarmMsg\":\"Detected people without work helmet!\"}"
        # 组装告警信息
        package_business_alarm(alarm_info, alarm_type, img_array_copy, img_array, data['deviceId'], alarm_label)

        t3 = time_synchronized()
        logger.info(f'设备 {data["k8sName"]} Detected people without work helmet!。')
    return img_array_copy
