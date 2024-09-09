import copy
import math
import os
import time
import cv2

from ai_common import setting
from ai_common.alarm_util import package_business_alarm
from ai_common.log_util import logger
from ai_common.np_util import person_in_poly_area, is_poi_in_poly, load_poly_area_data, tuple_xyxy, cal_iou
from ai_common.polylines import put_region
from ai_common.util.general import scale_coords


# from analysis.falldown_infer import falldown_inferrnce

from ai_common.util.plots import plot_one_box, colors
from analysis.analysis_utils import bbx_ratios, cal_iou_head, angle


def process_result(pred, names, detect_area_flag, line_thickness, data, img, img_array, polyline, standbbx_ratio, clear_t, h_h_conf, cal_ious, d_frame):
    flag = True
    clear_time = clear_t * 60 * 60
    # im0S为原画
    ang = standbbx_ratio
    ang = math.atan(ang)
    ang = math.degrees(ang)
    stand_angle = (ang + 0.5)
    img_array_copy = img_array.copy()

    pts, w1, h1 = load_poly_area_data(data)
    s = ''
    if data['k8sName'] in setting.frame_t.keys() and setting.person.keys() and setting.person_bbxr.keys() and setting.person_frame.keys() and setting.person_new.keys() and setting.person_newbbx.keys() and setting.person_linshi.keys() and setting.head1.keys() and setting.head2.keys():
        for i, det in enumerate(pred):
            # ------------- 清空字典储存的内容 ------------- #
            time_now = time.time()
            if int(time_now - setting.starttime) % clear_time == 0:
                setting.person[data['k8sName']].clear()
                setting.person_frame[data['k8sName']].clear()
                setting.person_bbxr[data['k8sName']].clear()
            setting.person_new[data['k8sName']].clear()
            setting.person_newbbx[data['k8sName']].clear()
            # ------------------------------------------ #
            s += '%g x %g ' % img.shape[2:]  # print string, 取第三和第四维，对应宽和高 512 x 640
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img_array_copy.shape).round()

                # logger.info results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                head_helmet_list = ['head', 'helmet']
                person_list = ['person']

                # Write results
                for *xyxy, conf, cls in det:
                    c = int(cls)
                    xyxy = [xyxy[0].item(), xyxy[1].item(), xyxy[2].item(), xyxy[3].item()]

                    if detect_area_flag:
                        # 求物体框的中心点
                        object_cx, object_cy = person_in_poly_area(xyxy)
                        # 判断中心点是否在检测框内部
                        if not is_poi_in_poly([object_cx, object_cy], pts):
                            # 不在感兴趣的框内，则继续判断下一个物体。
                            continue
                        # flag = falldown_inferrnce(data, names, c, head_helmet_list, conf, h_h_conf, xyxy, person_list, img_array_copy, cal_ious, standbbx_ratio, d_frame, stand_angle, flag)
                        xyxy = tuple_xyxy(xyxy)
                        if names[c] in head_helmet_list and conf > h_h_conf:
                            setting.head1[data['k8sName']].append(xyxy)
                        if names[c] in person_list:
                            # 第一帧图片
                            if setting.frame_t[data['k8sName']] == 0:
                                setting.person[data['k8sName']][xyxy] = 0  # 人的坐标{xyxy:0}
                                bbx = bbx_ratios(xyxy)  # 计算坐标的比例
                                setting.person_bbxr[data['k8sName']][xyxy] = bbx  # {xyxy:坐标比例}
                                setting.person_frame[data['k8sName']][xyxy] = 0  # 第一帧
                            else:  # 正常帧（及大于第一帧）
                                new_person = 0  # new_person 相用来记录当前帧与上一帧是否有对应的人，充当媒介。无则为0，有则为当前帧坐标框坐标
                                bbx_r = bbx_ratios(xyxy)
                                plot_one_box(xyxy, img_array_copy, label=str(bbx_r), color=colors(c, True),
                                             line_thickness=1)
                                for p1, flag1 in setting.person[data['k8sName']].items():  # 上一帧图片中的人坐标
                                    if p1 is not None:
                                        # ----- 追踪上一帧图片中的相同目标，如果没有相同目标，结束本次循环 ----- #
                                        ious = cal_iou(xyxy, p1)
                                        if ious > cal_ious:
                                            print('ious ', ious)
                                            new_person = p1
                                            setting.person[p1][data['k8sName']] = p1
                                            # ------------------------------------ #
                                            # ----- 如果比例框【持续】小于阈值，则判断摔倒 ----- #
                                            if bbx_r < standbbx_ratio:  # 比例小于阈值
                                                if setting.person_frame[data['k8sName']][p1] >= d_frame:  # 比例框持续小于阈值
                                                    one_head = 0  # one_head 用来记录这个人是否检测到头，充当媒介，无则为0，有则为1.
                                                    # ---- 判断是否同时检测到身体和头。有头根据比例框和头的角度判断、无头根据比例狂判断 ---- #
                                                    for head_ in setting.head2[data['k8sName']]:
                                                        print('head_person ', cal_iou_head(head_, p1))
                                                        if cal_iou_head(head_, p1) > 0.95:
                                                            one_head = 1
                                                            angles = angle(head_, p1)
                                                            print('angle', angles)
                                                            if angles < stand_angle:
                                                                plot_one_box(xyxy, img_array_copy, label='falling',
                                                                             color=colors(c, True))
                                                                flag = False
                                                                print('this person is falling!')
                                                                break
                                                            else:
                                                                setting.person_frame[data['k8sName']][p1] = 0
                                                                break
                                                        else:
                                                            continue
                                                    if one_head == 0:  # 检测不到头的时候，只用比例框判定
                                                        plot_one_box(xyxy, img_array_copy, label='falling',
                                                                     color=colors(c, True))
                                                        flag = False
                                                        print('this person is falling!')
                                                        break
                                                else:
                                                    setting.person_frame[data['k8sName']][p1] += 1  # 持续帧数加一
                                                    break
                                            else:
                                                setting.person_frame[data['k8sName']][p1] = 0  # 没有持续增大，持续帧数重置为0
                                                break
                                        else:
                                            continue

                                if new_person == 0:  # 如果在基础框中没有找到对应的人
                                    setting.person_new[data['k8sName']][xyxy] = 0
                                    setting.person_newbbx[data['k8sName']][xyxy] = bbx_r
                                else:  # 更新坐标和比例
                                    setting.person[data['k8sName']].pop(new_person)
                                    frame = setting.person_frame[data['k8sName']].pop(new_person)
                                    b = setting.person_bbxr[data['k8sName']].pop(new_person)
                                    setting.person[data['k8sName']][xyxy] = 1
                                    setting.person_bbxr[data['k8sName']][xyxy] = b
                                    setting.person_frame[data['k8sName']][xyxy] = frame
                    else:
                        # flag = falldown_inferrnce(data, names, c, head_helmet_list, conf, h_h_conf, xyxy, person_list, img_array_copy, cal_ious, standbbx_ratio, d_frame, stand_angle, flag)
                        xyxy = tuple_xyxy(xyxy)
                        if names[c] in head_helmet_list and conf > h_h_conf:
                            setting.head1[data['k8sName']].append(xyxy)
                        if names[c] in person_list:
                            # 第一帧图片
                            if setting.frame_t[data['k8sName']] == 0:
                                setting.person[data['k8sName']][xyxy] = 0  # 人的坐标{xyxy:0}
                                bbx = bbx_ratios(xyxy)  # 计算坐标的比例
                                setting.person_bbxr[data['k8sName']][xyxy] = bbx  # {xyxy:坐标比例}
                                setting.person_frame[data['k8sName']][xyxy] = 0  # 第一帧
                            else:  # 正常帧（及大于第一帧）
                                new_person = 0  # new_person 相用来记录当前帧与上一帧是否有对应的人，充当媒介。无则为0，有则为当前帧坐标框坐标
                                bbx_r = bbx_ratios(xyxy)
                                plot_one_box(xyxy, img_array_copy, label=str(bbx_r), color=colors(c, True), line_thickness=1)
                                print(setting.person[data['k8sName']])
                                for p1, flag1 in setting.person[data['k8sName']].items():  # 上一帧图片中的人坐标
                                    if p1 is not None:
                                        # ----- 追踪上一帧图片中的相同目标，如果没有相同目标，结束本次循环 ----- #
                                        ious = cal_iou(xyxy, p1)
                                        if ious > cal_ious:
                                            print('ious ', ious)
                                            new_person = p1
                                            setting.person[data['k8sName']][p1] = p1
                                            # ------------------------------------ #
                                            # ----- 如果比例框【持续】小于阈值，则判断摔倒 ----- #
                                            if bbx_r < standbbx_ratio:  # 比例小于阈值
                                                if setting.person_frame[data['k8sName']][p1] >= d_frame:  # 比例框持续小于阈值
                                                    one_head = 0  # one_head 用来记录这个人是否检测到头，充当媒介，无则为0，有则为1.
                                                    # ---- 判断是否同时检测到身体和头。有头根据比例框和头的角度判断、无头根据比例狂判断 ---- #
                                                    for head_ in setting.head2[data['k8sName']]:
                                                        print('head_person ', cal_iou_head(head_, p1))
                                                        if cal_iou_head(head_, p1) > 0.95:
                                                            one_head = 1
                                                            angles = angle(head_, p1)
                                                            print('angle', angles)
                                                            if angles < stand_angle:
                                                                plot_one_box(xyxy, img_array_copy, label='falling',
                                                                             color=colors(c, True))
                                                                flag = False
                                                                print('this person is falling!')
                                                                break
                                                            else:
                                                                setting.person_frame[data['k8sName']][p1] = 0
                                                                break
                                                        else:
                                                            continue
                                                    if one_head == 0:  # 检测不到头的时候，只用比例框判定
                                                        plot_one_box(xyxy, img_array_copy, label='falling',
                                                                     color=colors(c, True))
                                                        flag = False
                                                        print('this person is falling!')
                                                        break
                                                else:
                                                    setting.person_frame[data['k8sName']][p1] += 1  # 持续帧数加一
                                                    break
                                            else:
                                                setting.person_frame[data['k8sName']][p1] = 0  # 没有持续增大，持续帧数重置为0
                                                break
                                        else:
                                            continue

                                if new_person == 0:  # 如果在基础框中没有找到对应的人
                                    setting.person_new[data['k8sName']][xyxy] = 0
                                    setting.person_newbbx[data['k8sName']][xyxy] = bbx_r
                                else:  # 更新坐标和比例
                                    setting.person[data['k8sName']].pop(new_person)
                                    frame = setting.person_frame[data['k8sName']].pop(new_person)
                                    b = setting.person_bbxr[data['k8sName']].pop(new_person)
                                    setting.person[data['k8sName']][xyxy] = 1
                                    setting.person_bbxr[data['k8sName']][xyxy] = b
                                    setting.person_frame[data['k8sName']][xyxy] = frame

                setting.head2[data['k8sName']].clear()
                setting.head2[data['k8sName']] = copy.deepcopy(setting.head1[data['k8sName']])
                setting.head1[data['k8sName']].clear()

                # ----------- 删除后续不出现的 ------------ #
                setting.person_linshi[data['k8sName']].update(setting.person[data['k8sName']])
                for p1, flag1 in setting.person_linshi[data['k8sName']].items():
                    if p1 is not None:
                        if setting.person[data['k8sName']][p1] == 0:
                            setting.person[data['k8sName']].pop(p1)
                            setting.person_frame[data['k8sName']].pop(p1)
                            setting.person_bbxr[data['k8sName']].pop(p1)
                        else:
                            setting.person[data['k8sName']][p1] = 0
                setting.person_linshi[data['k8sName']].clear()
                # ------------------------------------- #
                for p2, flag2 in setting.person_new[data['k8sName']].items():  # 加入新框
                    if p2 is not None:
                        setting.person[data['k8sName']][p2] = 0
                        setting.person_frame[data['k8sName']][p2] = 0
                        setting.person_bbxr[data['k8sName']][p2] = setting.person_newbbx[data['k8sName']][p2]

                setting.frame_t[data['k8sName']] += 1
                logger.debug(f"frame_t:{setting.frame_t[data['k8sName']]}")

    logger.info(f'fall down detect result: {s}Done.')


    if setting.IMG_VERIFY == 1:
        if detect_area_flag:
            if polyline:
                img_array_copy = put_region(img_array_copy, w1, h1, pts, line_thickness)
        cv2.imshow('falldown_detect', img_array_copy)
        cv2.waitKey()
        cv2.destroyAllWindows()

    if setting.SAVE_IMG == 1:
        if not os.path.exists(setting.image_save_path):
            os.makedirs(setting.image_save_path)
        cv2.imwrite(f"{setting.image_save_path}/{int(time.time())}.jpg", img_array_copy)

    if not flag:
        if detect_area_flag:
            if polyline:
                img_array_copy = put_region(img_array_copy, w1, h1, pts, line_thickness)
        logger.info('Detect stuff falling down!')
        alarm_label = 'falldown'
        alarm_type = "跌倒检测"
        alarm_info = "{\"AlarmMsg\": \"Detect stuff falling down.\"}"
        # 组装告警信息
        package_business_alarm(alarm_info, alarm_type, img_array_copy, img_array, data['deviceId'], alarm_label)

    return img_array_copy