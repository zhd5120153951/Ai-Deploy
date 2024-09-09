import copy
import os
import time
import cv2

from car_detection.analysis.OCR_det_utils import ocr_det
from car_detection.models.LPRNET import CHARS
from ai_common import setting
from ai_common.alarm_util import package_business_alarm
from ai_common.log_util import logger
from ai_common.np_util import person_in_poly_area, is_poi_in_poly, load_poly_area_data
from ai_common.polylines import put_region
from ai_common.util.general import scale_coords
from ai_common.util.plots import plot_one_box


def process_result_car(pred_car, names_car, hide_labels, hide_conf, line_thickness, data, img, img_array, MODEL_rec, detect_area_flag, colors, polyline):
    flag_car = True
    img_array_copy = copy.copy(img_array)
    s = ''
    pts, w1, h1 = load_poly_area_data(data)

    # ocr车牌号码识别
    pred, plat_num = ocr_det(pred_car, MODEL_rec, img, img_array_copy) # [tensor([[146.1150, 258.6115, 206.3078, 284.8014,   0.9074,   0.0000]])]

    for i, det_car in enumerate(pred_car):  # detections per image
        s += '%g x %g ' % img.shape[2:]  # logger.info string, 取第三和第四维，对应宽和高 512 x 640

        if det_car is not None and len(det_car):
            det_car[:, :4] = scale_coords(img.shape[2:], det_car[:, :4], img_array_copy.shape).round()

            # logger.info results
            for c in det_car[:, -1].unique():
                n = (det_car[:, -1] == c).sum()  # 检测到目标的总和  # tensor([ True, False, False])
                s += f"{n} {names_car[int(c)]}{'s' * (n > 1)}, "  # add to string '512 x 640 1 person, 2 heads, '

            for det, lic_plat in zip(det_car, plat_num):
                *xyxy, conf, cls = det
                c = int(cls)  # integer class 1， 0

                xyxy = [xyxy[0].item(), xyxy[1].item(), xyxy[2].item(), xyxy[3].item()]

                warning_list = ['plate']

                if detect_area_flag:
                    # 求物体框的中心点
                    object_cx, object_cy = person_in_poly_area(xyxy)
                    # 判断中心点是否在检测框内部
                    if not is_poi_in_poly([object_cx, object_cy], pts):
                        # 不在感兴趣的框内，则继续判断下一个物体。
                        continue
                    if names_car[c] in warning_list:
                        if data['tools']['Target_filter']['filter_switch']: # 目标过滤
                            if int(data['tools']['Target_filter']['max_plate_xyxy']) > (int(xyxy[2]) - int(xyxy[0])) * (
                                    int(xyxy[3]) - int(xyxy[1])) > int(data['tools']['Target_filter']['min_plate_xyxy']):
                                if polyline:
                                    img_array_copy = put_region(img_array_copy, w1, h1, pts, line_thickness)
                                flag_car = False
                                lb = ""
                                for a, i in enumerate(lic_plat):
                                    lb += CHARS[int(i)]
                                LABEL = '%s %.2f' % (lb, conf) # 苏CFY773 0.91
                                # label
                                if hide_labels:
                                    label = None
                                elif hide_conf:
                                    label = '%s' % lb
                                else:
                                    label = LABEL
                                logger.debug(f"license id:{label}")
                                plot_one_box(xyxy, img_array_copy, label=label, color=(0, 0, 255), line_thickness=line_thickness)
                        else:
                            if polyline:
                                img_array_copy = put_region(img_array_copy, w1, h1, pts, line_thickness)
                            flag_car = False
                            lb = ""
                            for a, i in enumerate(lic_plat):
                                lb += CHARS[int(i)]
                            LABEL = '%s %.2f' % (lb, conf)  # 苏CFY773 0.91
                            # label
                            if hide_labels:
                                label = None
                            elif hide_conf:
                                label = '%s' % lb
                            else:
                                label = LABEL
                            logger.debug(f"license id:{label}")
                            plot_one_box(xyxy, img_array_copy, label=label, color=(0, 0, 255),
                                         line_thickness=line_thickness)
                else:
                    if names_car[c] in warning_list:
                        if data['tools']['Target_filter']['filter_switch']:
                            if int(data['tools']['Target_filter']['max_plate_xyxy']) > (int(xyxy[2]) - int(xyxy[0])) * (
                                    int(xyxy[3]) - int(xyxy[1])) > int(data['tools']['Target_filter']['min_plate_xyxy']):
                                flag_car = False
                                if names_car[c] in warning_list:
                                    lb = ""
                                    for a, i in enumerate(lic_plat):
                                        lb += CHARS[int(i)]
                                    LABEL = '%s %.2f' % (lb, conf) # 苏CFY773 0.91
                                    # label
                                    if hide_labels:
                                        label = None
                                    elif hide_conf:
                                        label = '%s'% lb
                                    else:
                                        label = LABEL
                                    logger.debug(f"license id:{label}")
                                    plot_one_box(xyxy, img_array_copy, label=label, color=(0, 0, 255), line_thickness=line_thickness)
                            else:
                                flag_car = False
                                if names_car[c] in warning_list:
                                    lb = ""
                                    for a, i in enumerate(lic_plat):
                                        lb += CHARS[int(i)]
                                    LABEL = '%s %.2f' % (lb, conf)  # 苏CFY773 0.91
                                    # label
                                    if hide_labels:
                                        label = None
                                    elif hide_conf:
                                        label = '%s' % lb
                                    else:
                                        label = LABEL
                                    logger.debug(f"license id:{label}")
                                    plot_one_box(xyxy, img_array_copy, label=label, color=(0, 0, 255), line_thickness=line_thickness)

# ----------------------------------------------------------------------------------------------------------- #
    logger.info(f'car detect result: {s}Done.')

    if setting.IMG_VERIFY == 1:
        if detect_area_flag:
            if polyline:
                img_array_copy = put_region(img_array_copy, w1, h1, pts, line_thickness)
        cv2.imshow('car_detect', img_array_copy)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()

    if setting.SAVE_IMG == 1:
        if not os.path.exists(setting.image_save_path):
            os.makedirs(setting.image_save_path)
        cv2.imwrite(f"{setting.image_save_path}/{int(time.time())}.jpg", img_array_copy)

    if not flag_car:
        logger.info("Got a license plate and I.D. the plate")
        alarm_label = 'car'
        alarm_type_car = "车牌识别检测"
        alarm_info_car = "{\"AlarmMsg\": \"Got a license plate and I.D. the plate\"}"
        # 组装告警信息
        package_business_alarm(alarm_info_car, alarm_type_car, img_array_copy, img_array, data['deviceId'], alarm_label)

    return img_array_copy
