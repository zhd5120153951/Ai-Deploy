import os
import time
import cv2
import numpy as np

from ai_common import setting
from ai_common.alarm_util import package_business_alarm
from ai_common.log_util import logger
from ai_common.np_util import person_in_poly_area, is_poi_in_poly, load_poly_area_data, tuple_xyxy
from ai_common.polylines import put_region
from ai_common.util.general import scale_coords
from ai_common.util.plots import plot_one_box


def process_result_leave(pred_person, names_person, data, img, img_array, params):
    flag_person = True
    img_array_copy = img_array.copy()
    h = float(img_array.shape[0])
    w = float(img_array.shape[1])
    now_time = time.time()
    during_time = params.interval_time * 60
    sg = ''
    pts, w1, h1 = load_poly_area_data(data)
    people = {}
    head_helmet = {}
    
    for i, det in enumerate(pred_person):
        sg += '%g x %g ' % img.shape[2:]  # logger.info string, 取第三和第四维，对应宽和高 512 x 640

        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img_array_copy.shape).round()

            # logger.info results
            for cg in det[:,-1].unique():
                ng = (det[:, -1] == cg).sum()
                sg += f"{ng} {names_person[int(cg)]}{'s' * (ng > 1)}, "

            # judge results
            for *xyxy, conf, cls in reversed(det):
                people_list = ['person']
                head_helmet_list = ['head', 'helmet']
                cg = int(cls)  # integer class 1， 0
                xyxy = [xyxy[0].item(), xyxy[1].item(), xyxy[2].item(), xyxy[3].item()]

                # label
                if params.hide_labels:
                    label = None
                elif params.hide_conf:
                    label = names_person[cg]
                else:
                    label = names_person[cg] + "%.2f" % conf

                if params.detect_area_flag:
                    # 求物体框的中心点
                    object_cx, object_cy = person_in_poly_area(xyxy)
                    # 判断中心点是否在检测框内部
                    if not is_poi_in_poly([object_cx, object_cy], pts):
                        # 不在感兴趣的框内，则继续判断下一个物体。
                        continue
                    if data['tools']['Target_filter']['filter_switch'] == True:
                        if names_person[cg] in people_list:
                            if params.max_people_xyxy > (int(xyxy[2]) - int(xyxy[0])) * (int(xyxy[3]) - int(xyxy[1])) > params.min_people_xyxy:
                                people[tuple_xyxy(xyxy)] = label
                        if names_person[cg] in head_helmet_list:
                            if params.max_head_helmet_xyxy > (int(xyxy[2]) - int(xyxy[0])) * (int(xyxy[3]) - int(xyxy[1])) > params.min_head_helmet_xyxy:
                                head_helmet[tuple_xyxy(xyxy)] = label
                    else:
                        if names_person[cg] in people_list:
                            people[tuple_xyxy(xyxy)] = label
                        if names_person[cg] in head_helmet_list:
                            head_helmet[tuple_xyxy(xyxy)] = label
                            pass
                        pass
                    pass
                else:
                    if data['tools']['Target_filter']['filter_switch'] == True:
                        if names_person[cg] in people_list:
                            if params.max_people_xyxy > (int(xyxy[2]) - int(xyxy[0])) * (int(xyxy[3]) - int(xyxy[1])) > params.min_people_xyxy:
                                people[tuple_xyxy(xyxy)] = label
                        if names_person[cg] in head_helmet_list:
                            if params.max_head_helmet_xyxy > (int(xyxy[2]) - int(xyxy[0])) * (int(xyxy[3]) - int(xyxy[1])) > params.min_head_helmet_xyxy:
                                head_helmet[tuple_xyxy(xyxy)] = label
                    else:
                        if names_person[cg] in people_list:
                            people[tuple_xyxy(xyxy)] = label
                        if names_person[cg] in head_helmet_list:
                            head_helmet[tuple_xyxy(xyxy)] = label

    logger.info(f'person detect result: {sg}Done.')

    # 统计人数
    # ------------------------------------------------------------- #
    flag_list = []
    num = len(head_helmet)
    for p in people:
        p_pts = [[p[0], p[3]], [p[2], p[3]], [p[2], p[1]], [p[0], p[1]]]
        for h in head_helmet:
            h_x, h_y = person_in_poly_area(h)
            flag = is_poi_in_poly([h_x, h_y], p_pts)
            flag_list.append(flag)
        # 对falg_list中元素进行计数
        flag_num = flag_list.count(False)
        if flag_num == len(flag_list):
            num += 1
        flag_list.clear()
    logger.info(f"人数统计为：{num}人")

    cv2.putText(img_array_copy, f"detect {num} people", (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

    # 时间间隔检测
    # ------------------------------------------------------------- #
    if params.interval_switch == True:
        if int(now_time - setting.old_leave_time) > during_time:
            setting.old_leave_time = time.time()
            # 检测人数大于等于设定人数，视为有人值岗。
            if num < params.people_num:
                flag_person = False
    else:
        if num < params.people_num:
            flag_person = False
    # ------------------------------------------------------------- #
    # 结果可视化与保存
    if setting.IMG_VERIFY == 1 or setting.VID_VERIFY == 1:
        for drow_p in people:
            plot_one_box(drow_p, img_array_copy, label=people[drow_p], color=(0, 0, 255), line_thickness=params.line_thickness)
        for drow_h in head_helmet:
            plot_one_box(drow_h, img_array_copy, label=head_helmet[drow_h], color=(0, 0, 255), line_thickness=params.line_thickness)
        if params.detect_area_flag:
            if params.polyline:
                img_array_copy = put_region(img_array_copy, w1, h1, pts, params.line_thickness)
    if setting.IMG_VERIFY == 1:
        cv2.imshow('detect_person.jpg', img_array_copy)
        cv2.waitKey()
        cv2.destroyAllWindows()

    if setting.SAVE_IMG == 1:
        if not os.path.exists(setting.image_save_path):
            os.makedirs(setting.image_save_path)
        cv2.imwrite(f"{setting.image_save_path}/{int(time.time())}.jpg", img_array_copy)

    # ------------------------------------------------------------- #
    if not flag_person:
        if params.detect_area_flag:
            if params.polyline:
                img_array_copy = put_region(img_array_copy, w1, h1, pts, params.line_thickness)
        for drow_p in people:
            plot_one_box(drow_p, img_array_copy, label=people[drow_p], color=(0, 0, 255), line_thickness=params.line_thickness)
        for drow_h in head_helmet:
            plot_one_box(drow_h, img_array_copy, label=head_helmet[drow_h], color=(0, 0, 255), line_thickness=params.line_thickness)
        logger.info(f'Alarm! Found staff leave!')
        alarm_info = "{\"AlarmMsg\": \"No one detected in certain area. Staff analysis alarm.\"}"
        alarm_label = 'leave'
        alarm_type = "人员离岗"
        # 组装告警信息
        package_business_alarm(alarm_info, alarm_type, img_array_copy, img_array, data['deviceId'], alarm_label)

    return img_array_copy
