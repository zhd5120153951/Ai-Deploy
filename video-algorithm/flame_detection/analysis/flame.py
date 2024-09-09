import copy
import os
import time
import cv2

from ai_common import setting
from ai_common.alarm_util import package_business_alarm
from ai_common.log_util import logger
from ai_common.np_util import tuple_xyxy, same, person_in_poly_area, is_poi_in_poly, load_poly_area_data
from ai_common.polylines import put_region
from ai_common.util.general import scale_coords
from ai_common.util.plots import plot_one_box


def process_result_flame(pred_flame, names_flame, hide_labels, hide_conf, line_thickness, data, img, img_array, detect_area_flag, polyline):
    flag_flame = True
    flag_flame0 = True
    img_array_copy = copy.deepcopy(img_array)
    flame_img1 = {}
    flame_xyxy_lab = {}
    s = ''
    pts, w1, h1 = load_poly_area_data(data)
    interval_time = float(data['tools']['flame_second_clasify']['clear_time']) * 60 # 清空frame字典计数

    for i, det_flame in enumerate(pred_flame):  # detections per image
        if data['k8sName'] in setting.flame_img.keys() and setting.flame_frame.keys():
            s += '%g x %g ' % img.shape[2:]  # logger.info string, 取第三和第四维，对应宽和高 512 x 640

            if det_flame is not None:
                det_flame[:, :4] = scale_coords(img.shape[2:], det_flame[:, :4], img_array_copy.shape).round()

                # logger.info results
                for c in det_flame[:, -1].unique():
                    n = (det_flame[:, -1] == c).sum()  # 检测到目标的总和  # tensor([ True, False, False])
                    s += f"{n} {names_flame[int(c)]}{'s' * (n > 1)}, "  # add to string '512 x 640 1 person, 2 heads, '

                for *xyxy, conf, cls in reversed(det_flame):
                    c = int(cls)  # integer class 1， 0

                    # label
                    if hide_labels:
                        label = None
                    elif hide_conf:
                        label = names_flame[c]
                    else:
                        label = names_flame[c] + "%.2f" % conf

                    xyxy = [xyxy[0].item(), xyxy[1].item(), xyxy[2].item(), xyxy[3].item()]
                    flame_warning_list = ['smoke', 'fire']

                    if detect_area_flag:
                        # 求物体框的中心点
                        object_cx, object_cy = person_in_poly_area(xyxy)
                        # 判断中心点是否在检测框内部
                        if not is_poi_in_poly([object_cx, object_cy], pts):
                            # 不在感兴趣的框内，则继续判断下一个物体。
                            continue
                        if names_flame[c] in flame_warning_list:
                            flame_img1[tuple_xyxy(xyxy)] = img_array_copy[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]
                            flame_xyxy_lab[tuple_xyxy(xyxy)] = label
                            flag_flame = False
                    else:
                        if names_flame[c] in flame_warning_list:
                            flame_img1[tuple_xyxy(xyxy)] = img_array_copy[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]
                            flame_xyxy_lab[tuple_xyxy(xyxy)] = label
                            flag_flame = False
# --------------------------------------------------------------------------------------------------------- #
#             本张图片box与上一张图片box做iou对比，再做same对比。
#             if setting.flame_frame[data['k8sName']] == 1:
#                 setting.flame_img[data['k8sName']] = flame_img1
#                 setting.flame_frame[data['k8sName']] += 1
#                 flag_flame0 = False
#             else:
#                 for xyxy_img1 in flame_img1:
#                     for xyxy_img in setting.flame_img[data['k8sName']]:
#                         det_same = same(setting.flame_img[data['k8sName']][xyxy_img], flame_img1[xyxy_img1])
#                         if det_same > float(data['det_same']):
#                             flag_flame0 = False
#                         setting.flame_img[data['k8sName']] = flame_img1
#                 setting.flame_frame[data['k8sName']] += 1
# ----------------------------------------------------------------------------------------------------------- #
            if data['tools']['flame_second_clasify']['clasify_switch']:
                # 本张图片与上一张图片box相同地方做same对比。
                if setting.flame_frame[data['k8sName']] == 1:
                    setting.flame_img[data['k8sName']] = flame_img1
                    setting.flame_frame[data['k8sName']] += 1
                    flag_flame0 = False
                else:
                    for xyxy0 in setting.flame_img[data['k8sName']]:
                        img0 = setting.flame_img[data['k8sName']][xyxy0]
                        img1 = img_array_copy[int(xyxy0[1]):int(xyxy0[3]), int(xyxy0[0]):int(xyxy0[2])]
                        det_same = same(img0, img1)
                        setting.flame_frame[data['k8sName']] += 1
                        logger.debug(f"设备：{data['k8sName']}；flame_frame:{setting.flame_frame[data['k8sName']]}")
                        logger.debug(f"设备：{data['k8sName']}；det_same:{det_same}")
                        if det_same < float(data['tools']['flame_second_clasify']['det_same']):
                            flag_flame0 = True
                            break
                        else:
                            flag_flame0 = False
                        setting.flame_img[data['k8sName']] = flame_img1
                # clear frame, redefine 1
                if int(time.time() - setting.starttime) > interval_time:
                    setting.flame_frame[data['k8sName']] = 1
                    setting.starttime = time.time()
# ----------------------------------------------------------------------------------------------------------- #
    logger.info(f'Flame detect result: {s}Done.')

    if setting.IMG_VERIFY == 1:
        if detect_area_flag:
            if polyline:
                img_array_copy = put_region(img_array_copy, w1, h1, pts, line_thickness)
        for xyxy_ in flame_xyxy_lab:
            plot_one_box(xyxy_, img_array_copy, label=flame_xyxy_lab[xyxy_], color=(0, 0, 255), line_thickness=line_thickness)
        cv2.imshow('Flame_detect.jpg', img_array_copy)
        cv2.waitKey()
        cv2.destroyAllWindows()

    if setting.SAVE_IMG == 1:
        if not os.path.exists(setting.image_save_path):
            os.makedirs(setting.image_save_path)
        cv2.imwrite(f"{setting.image_save_path}/{int(time.time())}.jpg", img_array_copy)


    if data['tools']['flame_second_clasify']['clasify_switch']:
        if not flag_flame and flag_flame0:
            logger.info("warning!warning!Found flame!")
            if detect_area_flag:
                if polyline:
                    img_array_copy = put_region(img_array_copy, w1, h1, pts, line_thickness)
            for xyxy_ in flame_xyxy_lab:
                plot_one_box(xyxy_, img_array_copy, label=flame_xyxy_lab[xyxy_], color=(0, 0, 255), line_thickness=line_thickness)
            alarm_label = 'flame'
            alarm_type_flame = "明火烟雾检测"
            alarm_info_flame = "{\"AlarmMsg\": \"Warning!Found flame!\"}"
            # 组装告警信息
            package_business_alarm(alarm_info_flame, alarm_type_flame, img_array_copy, img_array, data['deviceId'], alarm_label)
    else:
        if not flag_flame:
            logger.info("warning!warning!Found flame!")
            if detect_area_flag:
                if polyline:
                    img_array_copy = put_region(img_array_copy, w1, h1, pts, line_thickness)
            for xyxy_ in flame_xyxy_lab:
                plot_one_box(xyxy_, img_array_copy, label=flame_xyxy_lab[xyxy_], color=(0, 0, 255), line_thickness=line_thickness)
            alarm_label = 'flame'
            alarm_type_flame = "明火烟雾检测"
            alarm_info_flame = "{\"AlarmMsg\": \"Warning!Found flame!\"}"
            # 组装告警信息
            package_business_alarm(alarm_info_flame, alarm_type_flame, img_array_copy, img_array, data['deviceId'], alarm_label)

    return img_array_copy
