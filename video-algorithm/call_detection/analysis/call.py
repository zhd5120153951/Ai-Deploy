import os
import time

import cv2

from ai_common import setting
from ai_common.alarm_util import package_business_alarm
from ai_common.log_util import logger
from ai_common.np_util import person_in_poly_area, is_poi_in_poly, tuple_xyxy, cal_iou, load_poly_area_data
from ai_common.polylines import put_region
from ai_common.util.general import scale_coords
from ai_common.util.plots import plot_one_box
# import dlib


def process_result_call(head, pred_call, names_call, detect_area_flag, hide_labels, hide_conf, line_thickness, data, img, img_array, polyline):
    call = {}
    flag_call = True
    img_array_copy = img_array.copy()

    # ---------------------------------------- #
    # 用人脸进行判断
    # detector = dlib.get_frontal_face_detector()
    # faces = detector(img_array)
    # for i, d in enumerate(faces):
    #     # 人脸的左上和右下角坐标
    #     left = d.left()
    #     top = d.top()
    #     right = d.right()
    #     bottom = d.bottom()
    #     print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(i, left, top, right, bottom))
    #     cv2.rectangle(img_array_copy, (left, top), (right, bottom), color=(0, 255, 255), thickness=1)
    #     cv2.imshow('detect result', img_array_copy)
    #     cv2.waitKey(0)
    # ---------------------------------------- #
    pts, w1, h1 = load_poly_area_data(data)
    sm = ''
    for i, det_call in enumerate(pred_call):
        img_array_copy = img_array.copy()
        sm += '%g x %g ' % img.shape[2:]  # print string, 取第三和第四维，对应宽和高 512 x 640
        if len(det_call):
            det_call[:, :4] = scale_coords(img.shape[2:], det_call[:, :4], img_array_copy.shape).round()

        # logger.info results
        for cm in det_call[:, -1].unique():
            nm = (det_call[:, -1] == cm).sum()
            sm += f"{nm} {names_call[int(cm)]}{'s' * (nm > 1)}, "

        # Write results
        for *xyxy, conf, cls in det_call:
            cm = int(cls)
            xyxy = [xyxy[0].item(), xyxy[1].item(), xyxy[2].item(), xyxy[3].item()]

            # label
            if hide_labels:
                label = None
            elif hide_conf:
                label = names_call[cm]
            else:
                label = names_call[cm] + "%.2f" % conf

            if detect_area_flag:
                # 求物体框的中心点
                object_cx, object_cy = person_in_poly_area(xyxy)
                # 判断中心点是否在检测框内部
                if not is_poi_in_poly([object_cx, object_cy], pts):
                    # 不在感兴趣的框内，则继续判断下一个物体。
                    continue
                if data['tools']['Target_filter']['filter_switch'] == True:
                    if int(data['tools']['Target_filter']['max_call_xyxy']) > (int(xyxy[2]) - int(xyxy[0])) * (int(xyxy[3]) - int(xyxy[1])):
                        call[tuple_xyxy(xyxy)] = label
                        pass
                    pass
                else:
                    call[tuple_xyxy(xyxy)] = label
                    pass
                pass
            else:
                if data['tools']['Target_filter']['filter_switch'] == True:
                    if int(data['tools']['Target_filter']['max_call_xyxy']) > (int(xyxy[2]) - int(xyxy[0])) * (int(xyxy[3]) - int(xyxy[1])):
                        call[tuple_xyxy(xyxy)] = label
                        pass
                    pass
                else:
                    call[tuple_xyxy(xyxy)] = label
                pass
            pass
        pass
    iou_list = []
    # 对头和电话进行判断
    for h in head:
        for s in call:
            iou = cal_iou(h, s)
            if iou > 0:
                iou_list.append(iou)
                s_h = (h[2] - h[0]) * (h[3] - h[1])
                s_s = (s[2] - s[0]) * (s[3] - s[1])
                h_divide_s = s_s / s_h
                # print(h_divide_s)
                if h_divide_s < float(data['model_args']['s/h']):
                    flag_call = False
                    plot_one_box(h, img_array_copy, label=head[h], color=(255, 0, 0), line_thickness=line_thickness)
                    plot_one_box(s, img_array_copy, label=call[s], color=(0, 0, 255), line_thickness=line_thickness)
                    pass
                pass
            pass
        pass
    logger.info(f'call detect result: {sm}Done.')

    if setting.IMG_VERIFY == 1:
        if detect_area_flag:
            if polyline:
                img_array_copy = put_region(img_array_copy, w1, h1, pts, line_thickness)
        cv2.imshow('call_detect', img_array_copy)
        cv2.waitKey()
        cv2.destroyAllWindows()

    if setting.SAVE_IMG == 1:
        if not os.path.exists(setting.image_save_path):
            os.makedirs(setting.image_save_path)
        cv2.imwrite(f"{setting.image_save_path}/{int(time.time())}.jpg", img_array_copy)

    if not flag_call:
        if detect_area_flag:
            if polyline:
                img_array_copy = put_region(img_array_copy, w1, h1, pts, line_thickness)
        logger.info('Detect stuff calling!')
        alarm_label = 'call'
        alarm_type_smoking = "打电话"
        alarm_info_smoking = "{\"AlarmMsg\": \"Detect stuff calling.\"}"
        # 组装告警信息
        package_business_alarm(alarm_info_smoking, alarm_type_smoking, img_array_copy, img_array, data['k8sName'], alarm_label)

    return img_array_copy