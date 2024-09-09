import copy

from ai_common import setting
from ai_common.np_util import *
from ai_common.polylines import put_region
from ai_common.util.general import *
from ai_common.log_util import logger
from ai_common.util.plots import *
from ai_common.alarm_util import package_business_alarm
from ai_common.util.torch_utils import time_synchronized


def process_result_helmet(pred, detect_area_flag, names, hide_labels, hide_conf, line_thickness, data, img, img_array, polyline):

    """安全帽检测"""

    ts = time_synchronized()
    flag_h = True  # 异常为False
    img_array_copy = copy.copy(img_array)
    head = {}
    helmet = {}
    s = ''
    pts, w1, h1 = load_poly_area_data(data)

    for i, det in enumerate(pred):  # detections per image
        s += '%g x %g ' % img.shape[2:]  # print string, 取第三和第四维，对应宽和高 512 x 640

        if det is not None:  # Tensor:(6, 11)
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img_array_copy.shape).round()

            # # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # 检测到目标的总和  # tensor([ True, False, False])
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

            for *xyxy, conf, cls in reversed(det):
                head_list = ['head']
                helmet_list = ['helmet']
                c = int(cls)  # integer class 1， 0

                # label
                if hide_labels:
                    label = None
                elif hide_conf:
                    label = names[c]
                else:
                    label = names[c] + "%.2f" % conf

                xyxy = [xyxy[0].item(), xyxy[1].item(), xyxy[2].item(), xyxy[3].item()]

                # 指定了检测区域
                if detect_area_flag:
                    # 求物体框的中心点
                    object_cx, object_cy = person_in_poly_area(xyxy)
                    # 判断中心点是否在检测框内部
                    if not is_poi_in_poly([object_cx, object_cy], pts):
                        # 不在感兴趣的框内，则继续判断下一个物体。
                        continue
                    if data['tools']['Target_filter']['filter_switch'] == 'True':
                        if names[c] in head_list:
                            if int(data['tools']['Target_filter']['max_head_helmet_xyxy']) >= (int(xyxy[2]) - int(xyxy[0])) * (int(xyxy[3]) - int(xyxy[1]))\
                                    >= int(data['tools']['Target_filter']['min_head_helmet_xyxy']):
                                head[tuple_xyxy(xyxy)] = label
                        if names[c] in helmet_list:
                            if int(data['tools']['Target_filter']['max_head_helmet_xyxy']) >= (int(xyxy[2]) - int(xyxy[0])) * (int(xyxy[3]) - int(xyxy[1]))\
                                    >= int(data['tools']['Target_filter']['min_head_helmet_xyxy']):
                                helmet[tuple_xyxy(xyxy)] = label
                    else:
                        if names[c] in head_list:
                            head[tuple_xyxy(xyxy)] = label
                        if names[c] in helmet_list:
                            helmet[tuple_xyxy(xyxy)] = label
                else:
                    if data['tools']['Target_filter']['filter_switch'] == 'True':
                        if names[c] in head_list:
                            if int(data['tools']['Target_filter']['max_head_helmet_xyxy']) >= (int(xyxy[2]) - int(xyxy[0])) * (int(xyxy[3]) - int(xyxy[1]))\
                                    >= int(data['tools']['Target_filter']['min_head_helmet_xyxy']):
                                head[tuple_xyxy(xyxy)] = label
                        if names[c] in helmet_list:
                            if int(data['tools']['Target_filter']['max_head_helmet_xyxy']) >= (int(xyxy[2]) - int(xyxy[0])) * (int(xyxy[3]) - int(xyxy[1]))\
                                    >= int(data['tools']['Target_filter']['min_head_helmet_xyxy']):
                                helmet[tuple_xyxy(xyxy)] = label
                    else:
                        if names[c] in head_list:
                            head[tuple_xyxy(xyxy)] = label
                        if names[c] in helmet_list:
                            helmet[tuple_xyxy(xyxy)] = label
                        pass
                    pass
                pass
            pass
        pass

    logger.info(f'Helmet detect result:： {s}Done.')

    logger.debug(f"head:{head}")
    logger.debug(f"helmet:{helmet}")
    if data['tools']['helmet_head_verify']['switch'] == 'True': # 安全帽和头的互斥校验
        for h_item in head:
            if h_item is not None:  # 有头
                h_h_iou_list = []
                for hh_item in helmet:
                    if hh_item is not None:  # 有安全帽
                        h_h_iou = cal_iou(h_item, hh_item)
                        h_h_iou_list.append(h_h_iou)
                h_h_iou_list.sort()
                if len(h_h_iou_list) > 0:
                    if h_h_iou_list[-1] < float(data['tools']['helmet_head_verify']['iou_h_h']):
                        plot_one_box(h_item, img_array_copy, label=head[h_item], color=(0, 0, 255), line_thickness=line_thickness)  # red
                        flag_h = False
                        pass
                    pass
                else:
                    plot_one_box(h_item, img_array_copy, label=head[h_item], color=(0, 0, 255), line_thickness=line_thickness)  # red
                    flag_h = False
                    pass
                pass
            pass
        for h in helmet:
            plot_one_box(h, img_array_copy, label=helmet[h], color=(255, 0, 0), line_thickness=line_thickness)
    else:
        for h0 in head:
            plot_one_box(h0, img_array_copy, label=head[h0], color=(0, 0, 255), line_thickness=line_thickness)
        for h in helmet:
            plot_one_box(h, img_array_copy, label=helmet[h], color=(255, 0, 0), line_thickness=line_thickness)
        if len(head) != 0:
            flag_h = False


    # 验证、保存检测结果
    # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- #
    if setting.IMG_VERIFY == 1:
        if detect_area_flag:
            if polyline:
                img_array_copy = put_region(img_array_copy, w1, h1, pts, line_thickness)
        cv2.imshow('detect_helmet.jpg', img_array_copy)
        cv2.waitKey()
        cv2.destroyAllWindows()

    if setting.SAVE_IMG == 1:
        if not os.path.exists(setting.image_save_path):
            os.makedirs(setting.image_save_path)
        cv2.imwrite(f"{setting.image_save_path}/{int(time.time())}.jpg", img_array_copy)
    # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- #

    if flag_h:
        t2 = time_synchronized()
        logger.info(f'设备 {data["k8sName"]} 分析结果无异常。耗时： ({t2 - ts:.3f})s.')
        return img_array_copy
    else:
        if detect_area_flag:
            if polyline:
                img_array_copy = put_region(img_array_copy, w1, h1, pts, line_thickness)
        alarm_type = "未戴安全帽"
        alarm_label = 'helmet'
        alarm_info = "{发现员工未佩戴安全帽}"
        # 组装告警信息
        package_business_alarm(alarm_info, alarm_type, img_array_copy, img_array, data['deviceId'], alarm_label)

        t3 = time_synchronized()
        logger.info(f'设备 {data["k8sName"]} Detected people without work helmet!。耗时： ({t3 - ts:.3f})s.')

    return img_array_copy
