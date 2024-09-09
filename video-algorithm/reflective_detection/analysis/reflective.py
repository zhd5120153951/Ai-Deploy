import copy
from ai_common import setting
from ai_common.polylines import put_region
from ai_common.np_util import *
from ai_common.util.general import *
from ai_common.log_util import logger
from ai_common.util.plots import *
from ai_common.alarm_util import package_business_alarm
from ai_common.util.torch_utils import time_synchronized


def process_result_reflective(pred_reflective, names_reflective, data, img, img_array, person, head_helmet, params):

    '''反光衣检测'''

    ts = time_synchronized()
    img_array_copy = copy.deepcopy(img_array)
    reflective = {}
    noreflective = {}
    flag = True
    s0 = ''
    pts, w1, h1 = load_poly_area_data(data)

    for i, det in enumerate(pred_reflective):  # detections per image
        det = copy.deepcopy(det)
        s0 += '%g x %g ' % img.shape[2:]  # print string, 取第三和第四维，对应宽和高 512 x 640

        if det is not None:  # Tensor:(6, 11)
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img_array_copy.shape).round()

            for c0 in det[:, -1].unique():
                n0 = (det[:, -1] == c0).sum()  # 检测到目标的总和  # tensor([ True, False, False])
                s0 += f"{n0} {names_reflective[int(c0)]}{'s' * (n0 > 1)}, "  # add to string '512 x 640 1 person, 2 heads, '
            for *xyxy, conf, cls in reversed(det):
                c0 = int(cls)  # integer class 1， 0

                # label
                if params.hide_labels:
                    label = None
                elif params.hide_conf:
                    label = names_reflective[c0]
                else:
                    label = names_reflective[c0] + "%.2f" % conf

                xyxy = [xyxy[0].item(), xyxy[1].item(), xyxy[2].item(), xyxy[3].item()]
                # 指定了检测区域
                if params.detect_area_flag:
                    # 求物体框的中心点
                    object_cx, object_cy = person_in_poly_area(xyxy)
                    # 判断中心点是否在检测框内部
                    if not is_poi_in_poly([object_cx, object_cy], pts):
                        # 不在感兴趣的框内，则继续判断下一个物体。
                        continue
                    if c0 == 0:
                        reflective[tuple_xyxy(xyxy)] = label
                    if c0 == 1:
                        flag = False
                        noreflective[tuple_xyxy(xyxy)] = label
                else:
                    if c0 == 0:
                        reflective[tuple_xyxy(xyxy)] = label
                    if c0 == 1:
                        flag = False
                        noreflective[tuple_xyxy(xyxy)] = label
                    pass
                pass
            pass
    logger.info(f'reflective detect result: {s0}Done.')

    # real_reflective = {}
    # real_people = {}
    # flag_u = []
    # # ------------------ #
    # logger.debug(f"person:{person}")
    # logger.debug(f"reflective:{reflective}")
    #
    # # 人和头二重校验
    # for p in person:
    #     if p is not None:
    #         for h in head_helmet:
    #             if h is not None:
    #                 iou_ph = cal_iou(p, h)
    #                 if iou_ph > params.p_h_iou_min and iou_ph < params.p_h_iou_max:
    #                     real_people[p] = person[p]
    #                     break
    #                     pass
    #                 pass
    #             pass
    #         pass
    #     pass
    # pass
    # # 人和反光衣二重校验
    # for u in reflective:
    #     for r_p in real_people:
    #         iou = cal_iou(r_p, u)
    #         if iou > params.p_u_iou:
    #             real_reflective[u] = reflective[u]
    #             break
    #             pass
    #         pass
    #     pass
    # pass
    # # --------判断人数--------- #
    # # if len(real_people) <= 1:
    # #     return img_array_copy
    # # ----------------------- #
    # # else:
    # # 判断未穿反光衣的人
    # for pp in real_people:
    #     flag = False
    #     flag_u.append(flag)
    #     for uu in real_reflective:
    #         iou_pu = cal_iou(pp, uu)
    #         if iou_pu > params.p_u_iou:
    #             flag = True
    #             flag_u.append(flag)
    #             break
    #     if not flag:
    #         plot_one_box(pp, img_array_copy, label='noreflective', color=(0, 0, 255), line_thickness=params.line_thickness)
    #
    # if flag_u.count(False) > flag_u.count(True):
    #     flag = False # 人 > 反光衣
    # else:
    #     flag = True # 人 < 反光衣
    #
    # for real_u in real_reflective:
    #     plot_one_box(real_u, img_array_copy, label=real_reflective[real_u], color=(255, 0, 0),
    #                  line_thickness=params.line_thickness)

    # 验证检测结果
    # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- #
    if setting.IMG_VERIFY == 1 or setting.VID_VERIFY == 1:
        if params.detect_area_flag:
            if params.polyline:
                img_array_copy = put_region(img_array_copy, w1, h1, pts, params.line_thickness)
        for f in reflective:
            plot_one_box(f, img_array_copy, label=reflective[f], color=(255, 0, 0), line_thickness=params.line_thickness)
        for nf in noreflective:
            plot_one_box(nf, img_array_copy, label=noreflective[nf], color=(0, 0, 255), line_thickness=params.line_thickness)
    if setting.IMG_VERIFY == 1:
        cv2.imshow('detect_reflective.jpg', img_array_copy)
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
                img_array_copy = put_region(img_array_copy, w1, h1, pts, params.polyline)
        for f in reflective:
            plot_one_box(f, img_array_copy, label=reflective[f], color=(255, 0, 0), line_thickness=params.line_thickness)
        for nf in noreflective:
            plot_one_box(nf, img_array_copy, label=noreflective[nf], color=(0, 0, 255), line_thickness=params.line_thickness)
        alarm_type = "未穿反光衣"
        alarm_label = 'reflective'
        alarm_info = "{\"AlarmMsg\":\"Detected people without work clothes!\"}"
        # 组装告警信息
        package_business_alarm(alarm_info, alarm_type, img_array_copy, img_array, data['deviceId'], alarm_label)

        t2 = time_synchronized()
        logger.info(f'设备 {data["k8sName"]} Detected people without work clothes!!! 耗时： ({t2 - ts:.3f})s.')

    return img_array_copy


