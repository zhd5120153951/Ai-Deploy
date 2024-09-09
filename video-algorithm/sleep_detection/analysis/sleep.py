import copy
import os
import time
import cv2

from ai_common import setting
from ai_common.alarm_util import package_business_alarm
from ai_common.log_util import logger
from ai_common.np_util import is_poi_in_poly, person_in_poly_area, cal_iou, same, load_poly_area_data
from ai_common.polylines import put_region
from utils.general import scale_boxes
from utils.plots import save_one_box, plot_one_box

# 根据前后帧的iou来判断别有没有动，再根据不动帧的总数是否大于某一阈值判定是否睡岗


def process_result_sleep(pred, names, data, img, img_array, params):
    flag_sleep = False
    img_array_copy = copy.copy(img_array)
    ss = ''
    clear_t = 20  # 分钟
    clear_time = clear_t * 60
    interval_frame = 2
    during_frames = params.sleep_interval_time * 60  # 间隔_秒

    pts, w1, h1 = load_poly_area_data(data)
    # 这里直接跳过，有bug
    if data['k8sName'] in setting.person_dic.keys() and setting.person1_dic.keys() and setting.person_img_dic.keys() and setting.person_img1_dic.keys() and setting.person_frame_dic.keys():
        for j, dets in enumerate(pred):  # detections per image
            time_now = time.time()
            # clear_time:5*60*60 每隔clear_time清空一次列表
            if (int(time_now) - int(setting.starttime)) % int(clear_time) == 0:
                setting.person1_dic[data['k8sName']].clear()
                setting.person_frame_dic[data['k8sName']].clear()
                setting.person_img1_dic[data['k8sName']].clear()
            setting.person_dic[data['k8sName']].clear()
            setting.person_img_dic[data['k8sName']].clear()
            ss += '%gx%g ' % img.shape[2:]  # logger.info() string设置打印图片的信息

            if len(dets):
                # Rescale boxes from img_size to im0 size  将预测信息映射到原图中
                dets[:, :4] = scale_boxes(
                    img.shape[2:], dets[:, :4], img_array_copy.shape).round()

                # logger.info() results 打印检测到的类别数量
                for cs in dets[:, -1].unique():
                    ns = (dets[:, -1] == cs).sum()  # detections per class
                    # add to string
                    ss += f"{ns} {names[int(cs)]}{'s' * (ns > 1)}, "

                # Write results
                for *xyxys, confs, clss in reversed(dets):
                    warninglist = ['person']
                    cs = int(clss)  # integer class
                    # labels = None if hide_labels else (names[cs] if hide_conf else f'{names[cs]}') + "%.2f" % confs
                    if params.detect_area_flag:
                        # 求物体框的中心点
                        object_cx, object_cy = person_in_poly_area(xyxys)
                        # 判断中心点是否在检测框内部
                        if not is_poi_in_poly([object_cx, object_cy], pts):
                            # 不在感兴趣的框内，则继续判断下一个物体。
                            continue
                        # 1.对图像进行{坐标：裁剪目标}
                        xyxys = (xyxys[0].item(), xyxys[1].item(),
                                 xyxys[2].item(), xyxys[3].item())
                        img_s = img_array_copy[int(xyxys[1]):int(
                            xyxys[3]), int(xyxys[0]):int(xyxys[2])]
                        if names[cs] in warninglist:
                            setting.person_dic[data['k8sName']
                                               ][xyxys] = 0  # 保存视频第一帧的xyxy坐标
                            # 视频第一帧的xyxy框住的图像
                            setting.person_img_dic[data['k8sName']
                                                   ][xyxys] = img_s
                            # plot_one_box(xyxys, img_array_copy, label=labels, color=colors(cs, True), line_thickness=line_thickness)
                    else:
                        xyxys = (xyxys[0].item(), xyxys[1].item(),
                                 xyxys[2].item(), xyxys[3].item())
                        img_s = img_array_copy[int(xyxys[1]):int(
                            xyxys[3]), int(xyxys[0]):int(xyxys[2])]
                        if names[cs] in warninglist:
                            # 保存视频第一帧的xyxy坐标:{(0.0, 122.0, 157.0, 477.0): 0}
                            setting.person_dic[data['k8sName']][xyxys] = 0
                            # 视频第一帧的xyxy框住的图像
                            setting.person_img_dic[data['k8sName']
                                                   ][xyxys] = img_s
                            # plot_one_box(xyxys, img_array_copy, label=labels, color=colors(cs, True), line_thickness=line_thickness)
            logger.info(f'Sleep detect result: {ss}Done.')

            # frame_t为0或字典person1为空
            if setting.frame_t_dic[data['k8sName']] == 0 or not bool(setting.person1_dic[data['k8sName']]):
                # A.updata(B) -> 将B合并到A中
                # 第一帧图像
                setting.person1_dic[data['k8sName']].update(
                    setting.person_dic[data['k8sName']])
                setting.person_frame_dic[data['k8sName']].update(
                    setting.person_dic[data['k8sName']])  # {(0.0, 122.0, 157.0, 477.0): 0}
                setting.person_img1_dic[data['k8sName']].update(
                    setting.person_img_dic[data['k8sName']])
            else:
                # 上一帧
                for p1, flag1 in setting.person1_dic[data['k8sName']].items():
                    if p1 is not None:
                        change = 0
                        # 当前帧
                        for p2, flag2 in setting.person_dic[data['k8sName']].items():
                            if p2 is not None:
                                iou = cal_iou(p1, p2)
                                if iou > params.cal_ious:
                                    logger.debug(f'iou is: {iou}')
                                    # 通过person1上一帧和person当前帧的坐标框IOU，判断同一对象
                                    change = 1
                                    # {(0.0, 122.0, 157.0, 477.0): 3}
                                    setting.person_dic[data['k8sName']][p2] = 3
                                    # 上一帧目标图像
                                    image1 = setting.person_img1_dic[data['k8sName']][p1]
                                    # 当前帧目标图像
                                    image2 = setting.person_img_dic[data['k8sName']][p2]
                                    # 通过person1_img上一帧和person_img当前帧的目标图像image，判断同一对象的前后重合度
                                    s = same(image1, image2)
                                    logger.debug(f's is: {s}')
                                    if s > params.sames:
                                        if flag1 == 0:  # 更新该员工画面进行下一轮寻找，更新标记
                                            # frame清零
                                            setting.person_frame_dic[data['k8sName']
                                                                     ][p1] = setting.frame_t_dic[data['k8sName']]
                                            # {(0.0, 122.0, 157.0, 477.0): 1}
                                            setting.person1_dic[data['k8sName']][p1] = 1
                                        # 该员工sleeping
                                        else:
                                            # 当前时间和目标被标记为1的时间截
                                            interval = setting.frame_t_dic[data['k8sName']] - \
                                                setting.person_frame_dic[data['k8sName']][p1]
                                            # logger.debug(f'睡岗people:{setting.person1_dic[person1]}时间间隔为{interval}')
                                            # if (interval>=during_frames and interval<=(during_frames+5*interval_frame)):
                                            if interval >= during_frames:  # 若大于规定设置睡岗时间
                                                flag_sleep = True
                                                x_ = list(p2)
                                                plot_one_box(x_, img_array_copy, label='sleeping', color=(
                                                    0, 0, 255), line_thickness=params.line_thickness)
                                            break
                                    else:  # 若前后两帧同一对象重合度小于sames
                                        # flag重新标记为0
                                        setting.person1_dic[data['k8sName']][p1] = 0
                                        # 时间截重新标记为0
                                        setting.person_frame_dic[data['k8sName']][p1] = 0
                                    # 标记上一帧图像为image2
                                    setting.person_img1_dic[data['k8sName']
                                                            ][p1] = image2
                        if change == 0:
                            # flag重新标记为0
                            setting.person1_dic[data['k8sName']][p1] = 0
                            # 时间截重新标记为0
                            setting.person_frame_dic[data['k8sName']][p1] = 0
                for p in setting.person_dic[data['k8sName']]:
                    if setting.person_dic[data['k8sName']][p] != 3:
                        setting.person1_dic[data['k8sName']
                                            ][p] = setting.person_dic[data['k8sName']][p]
                        setting.person_img1_dic[data['k8sName']
                                                ][p] = setting.person_img_dic[data['k8sName']][p]
                        setting.person_frame_dic[data['k8sName']][p] = 0

    if setting.IMG_VERIFY == 1:
        if params.detect_area_flag:
            if params.polyline:
                img_array_copy = put_region(
                    img_array_copy, w1, h1, pts, params.line_thickness)
        cv2.imshow('detect_sleep.jpg', img_array_copy)
        cv2.waitKey()
        cv2.destroyAllWindows()

    if setting.SAVE_IMG == 1:
        if not os.path.exists(setting.image_save_path):
            os.makedirs(setting.image_save_path)
        cv2.imwrite(
            f"{setting.image_save_path}/{int(time.time())}.jpg", img_array_copy)

    if flag_sleep:
        logger.info('Stuff were found sleeping in the certain area!')
        alarm_info_sleep = "{\"AlarmMsg\":\"Stuff were found sleeping in the certain area.\"}"
        alarm_type_sleep = "人员睡岗"
        alarm_label = 'sleep'
        # 组装告警信息
        package_business_alarm(alarm_info_sleep, alarm_type_sleep,
                               img_array_copy, img_array, data['deviceId'], alarm_label)
        # ====================================== #
        # 间隔报警 - 防止一直报警
        # logger.debug("Clear person1")
        # setting.person1_dic[data['k8sName']].clear()
        # setting.person_frame_dic[data['k8sName']].clear()
        # setting.person_img1_dic[data['k8sName']].clear()
        # ====================================== #
    # setting.frame_t_dic[data['k8sName']
    #                     ] = setting.frame_t_dic[data['k8sName']] + str(interval_frame)
    setting.frame_t_dic = {data['k8sName']                           : data['k8sName']+str(interval_frame)}
    logger.info(
        f"frame_t_{data['k8sName']}:{setting.frame_t_dic[data['k8sName']]}")

    return img_array_copy
