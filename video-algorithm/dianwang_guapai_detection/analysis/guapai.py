import copy

from ai_common import setting
from ai_common.np_util import *
from ai_common.polylines import put_region
from ai_common.util.general import *
from ai_common.log_util import logger
from ai_common.util.plots import *
from ai_common.alarm_util import package_business_alarm
from ai_common.util.torch_utils import time_synchronized


def process_result_guapai(pred, data, img, img_array, names, params):
    ts = time_synchronized()
    flag = True  # 异常为False
    img_array_copy = copy.copy(img_array)
    s = ''
    guapai = {}
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
                c = int(cls)  # integer class 1， 0

                # label
                if params.hide_labels:
                    label = None
                elif params.hide_conf:
                    label = names[c]
                else:
                    label = names[c] + "%.2f" % conf

                xyxy = [xyxy[0].item(), xyxy[1].item(), xyxy[2].item(), xyxy[3].item()]

                # 指定了检测区域
                if params.detect_area_flag:
                    # 求物体框的中心点
                    object_cx, object_cy = person_in_poly_area(xyxy)
                    # 判断中心点是否在检测框内部
                    if not is_poi_in_poly([object_cx, object_cy], pts):
                        # 不在感兴趣的框内，则继续判断下一个物体。
                        continue
                    if data['tools']['Target_filter']['filter_switch'] == True:
                        if c == 0:
                            if params.max_guapai_xyxy >= (int(xyxy[2]) - int(xyxy[0])) * (int(xyxy[3]) - int(xyxy[1])) >= params.min_guapai_xyxy:
                                guapai[tuple_xyxy(xyxy)] = label
                        if c == 1:
                            flag = False
                    else:
                        if c == 0:
                            guapai[tuple_xyxy(xyxy)] = label
                        if c == 1:
                            flag = False
                else:
                    if data['tools']['Target_filter']['filter_switch'] == True:
                        if c == 0:
                            if params.max_guapai_xyxy >= (int(xyxy[2]) - int(xyxy[0])) * (int(xyxy[3]) - int(xyxy[1])) >= params.min_guapai_xyxy:
                                guapai[tuple_xyxy(xyxy)] = label
                        if c == 1:
                            flag = False
                    else:
                        if c == 0:
                            guapai[tuple_xyxy(xyxy)] = label
                        if c == 1:
                            flag = False
                        pass
                    pass
                pass
            pass
        pass
    for g in guapai:
        plot_one_box(g, img_array_copy, label=guapai[g], color=(0, 0, 255), line_thickness=params.line_thickness)

    logger.info(f'Guapai detect result:： {s}Done.')
    logger.debug(guapai)

    # 验证、保存检测结果
    # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- #
    if setting.IMG_VERIFY == 1:
        if params.detect_area_flag:
            if params.polyline:
                img_array_copy = put_region(img_array_copy, w1, h1, pts, params.line_thickness)
        cv2.imshow('detect_guapai.jpg', img_array_copy)
        cv2.waitKey()
        cv2.destroyAllWindows()

    if setting.SAVE_IMG == 1:
        if not os.path.exists(setting.image_save_path):
            os.makedirs(setting.image_save_path)
        cv2.imwrite(f"{setting.image_save_path}/{int(time.time())}.jpg", img_array_copy)
    # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- #

    if flag:
        t2 = time_synchronized()
        logger.info(f'设备 {data["k8sName"]} 分析结果无异常。耗时： ({t2 - ts:.3f})s.')
        return img_array_copy
    else:
        if params.detect_area_flag:
            if params.polyline:
                img_array_copy = put_region(img_array_copy, w1, h1, pts, params.line_thickness)
        alarm_type = "未挂牌"
        alarm_label = 'guapai'
        alarm_info = "{\"AlarmMsg\":\"Not guapai.\"}"
        # 组装告警信息
        package_business_alarm(alarm_info, alarm_type, img_array_copy, img_array, data['deviceId'], alarm_label)

        t3 = time_synchronized()
        logger.info(f'设备 {data["k8sName"]} Detected notguapai。耗时： ({t3 - ts:.3f})s.')

    return img_array_copy
