import copy
import os
import time

import cv2

from ai_common import setting
from ai_common.alarm_util import package_business_alarm
from ai_common.log_util import logger
from ai_common.np_util import person_in_poly_area, is_poi_in_poly, load_poly_area_data, tuple_xyxy
from ai_common.polylines import put_region
from ai_common.util.general import scale_coords
from ai_common.util.plots import plot_one_box

def process_result_invaion(pred, names, hide_labels, hide_conf, line_thickness, data, img, img_array, polyline):
    flag = True
    img_array_copy = copy.copy(img_array)
    pts, w1, h1 = load_poly_area_data(data)
    people = {}
    head_helmet = {}
    s = ''

    for i, det in enumerate(pred):  # detections per image, det Tensor:(3,6)
        s += '%g x %g ' % img.shape[2:]  # logger.info() string, 取第三和第四维，对应宽和高 512 x 640
        if len(det):  # len(det): 2
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img_array_copy.shape).round()

            # logger.info() results
            for c in det[:,-1].unique():
                n = (det[:, -1] == c).sum()
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

            # judge results
            for *xyxy, conf, cls in reversed(det):
                people_list = ['person']
                head_helmet_list = ['helmet', 'head']
                c = int(cls)  # integer class 1， 0

                # label
                if hide_labels:
                    label = None
                elif hide_conf:
                    label = names[c]
                else:
                    label = names[c] + "%.2f" % conf

                xyxy = tuple_xyxy([xyxy[0].item(), xyxy[1].item(), xyxy[2].item(), xyxy[3].item()])
                if polyline:
                    put_region(img_array_copy, w1, h1, pts, line_thickness)
                # 求物体框的中心点
                object_cx, object_cy = person_in_poly_area(xyxy)
                # 判断中心点是否在检测框内部
                if is_poi_in_poly([object_cx, object_cy], pts):
                    if data['tools']['Target_filter']['filter_switch'] == True:
                        if names[c] in people_list:
                            if int(data['tools']['Target_filter']['max_people_xyxy']) > (
                                    int(xyxy[2]) - int(xyxy[0])) * (int(xyxy[3]) - int(xyxy[1])) > int(
                                    data['tools']['Target_filter']['min_people_xyxy']):
                                people[xyxy] = label
                        if names[c] in head_helmet_list:
                            if int(data['tools']['Target_filter']['max_head_helmet_xyxy']) > (
                                    int(xyxy[2]) - int(xyxy[0])) * (int(xyxy[3]) - int(xyxy[1])) > int(
                                    data['tools']['Target_filter']['min_head_helmet_xyxy']):
                                head_helmet[xyxy] = label
                    else:
                        if names[c] in people_list:
                            people[xyxy] = label
                        if names[c] in head_helmet_list:
                            head_helmet[xyxy] = label

    if len(people) != 0 or len(head_helmet) != 0:
        flag = False
    logger.info(f"Detect result: {s}Done.")

    # --------------------------------------------- #
    if setting.IMG_VERIFY == 1:
        for drow_p in people:
            plot_one_box(drow_p, img_array_copy, label=people[drow_p], color=(0, 0, 255), line_thickness=line_thickness)
        for drow_h in head_helmet:
            plot_one_box(drow_h, img_array_copy, label=head_helmet[drow_h], color=(0, 0, 255), line_thickness=line_thickness)

        cv2.imshow('detect_invaion.jpg', img_array_copy)
        cv2.waitKey()
        cv2.destroyAllWindows()

    if setting.SAVE_IMG == 1:
        if not os.path.exists(setting.image_save_path):
            os.makedirs(setting.image_save_path)
        cv2.imwrite(f"{setting.image_save_path}/{int(time.time())}.jpg", img_array_copy)

    # 危险区域检测到人时，组装告警数据。
    if not flag:
        for drow_p in people:
            plot_one_box(drow_p, img_array_copy, label=people[drow_p], color=(0, 0, 255), line_thickness=line_thickness)
        for drow_h in head_helmet:
            plot_one_box(drow_h, img_array_copy, label=head_helmet[drow_h], color=(0, 0, 255), line_thickness=line_thickness)
        logger.info(f'Alarm! Someone in the danger region.')
        alarm_info = "{\"AlarmMsg\": \"Someone in the danger region.\"}"
        alarm_label = 'invaion'
        alarm_type = "区域入侵"
        # 组装告警信息
        package_business_alarm(alarm_info, alarm_type, img_array_copy, img_array, data['deviceId'], alarm_label)

    return img_array_copy

