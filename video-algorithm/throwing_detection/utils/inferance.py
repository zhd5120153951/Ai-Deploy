import copy

import cv2
import numpy as np
from utils.MOG2Detector import MOG2Detector
from ai_common import setting
from ai_common.alarm_util import package_business_alarm
from utils.sort import Sort

from ai_common.log_util import logger


def start_detect(dist_threshold, direct_threshold, frame_threshold, frame_pick):
    '''
    dist_threshold：判断运动距离不符合抛物运动的阈值。
    direct_threshold：判断方向改变的阈值，若当前方向与上一帧方向不同且位移变化量大于等于direct_threshold，则认为运动方向发生变化。
    frame_threshold：捕获目标帧数的最小值。
    frame_pick：抽帧频率，每frame_pick帧图像中抽一帧做计算。
    '''

    detector = MOG2Detector(history=7, dist2Threshold=1000, minArea=100) # MOG2实例
    sort = Sort(max_age=3, min_hits=1, iou_threshold=0.1) # Sort实例

    frameID = 0
    match_ID_dict = {}  # 字典，目标ID：目标在每帧的Y轴坐标
    process_dict = {}  # 字典，目标ID：[上次运动方向，方向曾经是否变化，上次的距离变化量，是否为抛物]
    throw_list = []  # 抛物ID列表
    frame_threshold /= frame_pick

    if setting.image_queue.qsize() > 0:
        while True:
            logger.info(f"image_queue.qsize():{setting.image_queue.qsize()}; -> Get images form the image_queue -> image_queue.full():{setting.image_queue.full()}\n")
            capture = setting.image_queue.get()
            img_array = capture['img_array']
            img_array_copy = copy.copy(img_array)
            if frameID > 0 and frameID % frame_pick != 0:
                continue
            frameID += 1

            mask, bboxs = detector.detectOneFrame(img_array) # mask为前景掩码，bboxs为轮廓数组
            if bboxs != []:
                bboxs = np.array(bboxs)
                bboxs[:, 2:4] += bboxs[:, 0:2]
                trackBox = sort.update(bboxs) # 预测bboxs坐标
            else:
                trackBox = sort.update()

            for bbox in trackBox:
                bbox = [int(bbox[i]) for i in range(5)]
                y = (bbox[1] + bbox[3])/2
                if(match_ID_dict.__contains__(bbox[4])): # 判断键是否存在于字典中
                    # 计算 距离变化量 及 运动方向
                    dist = y - match_ID_dict[bbox[4]][-1]
                    if(dist > 0):
                        direction = 0  # 方向向下
                    elif(dist == 0):
                        direction = 1  # 相对静止
                    else:
                        direction = 2  # 方向向上

                    match_ID_dict[bbox[4]].append(y)

                    if(process_dict[bbox[4]][0] == -1):
                        # 首次记录方向
                        process_dict[bbox[4]][0] = direction
                    else:
                        if(process_dict[bbox[4]][0] != direction and abs(dist) >= direct_threshold):
                            # 需要更新方向
                            if(direction == 0 or process_dict[bbox[4]][1] == True):
                                # 方向由下变上or方向已经变化过，不是抛物
                                process_dict[bbox[4]][3] = False
                            else:
                                process_dict[bbox[4]][0] = direction
                                process_dict[bbox[4]][1] = True
                    process_dict[bbox[4]][2] = abs(dist)
                else:
                    match_ID_dict.update({bbox[4]: [y]})
                    process_dict.update({bbox[4]: [-1, False, 0, True]})

                cv2.rectangle(img_array_copy, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 6)
                cv2.putText(img_array_copy, "ID:"+str(bbox[4]), (bbox[0], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


                for key in match_ID_dict:
                    if(len(match_ID_dict[key]) >= frame_threshold and process_dict[key][3] == True and len(bbox) > 1):
                        alarm_label = 'throwing'
                        alarm_type_throwing = "高空抛物检测"
                        alarm_info_throwing = "{\"AlarmMsg\": \"Warning!Found something filling!\"}"
                        # 组装告警信息
                        package_business_alarm(alarm_info_throwing, alarm_type_throwing, img_array_copy, img_array, capture['deviceId'], alarm_label)

    if setting.TEST_FLAG == 1:
        for key in match_ID_dict:
            print("ID:", key)
            print(match_ID_dict[key])
            print(process_dict[key])
        for key in match_ID_dict:
            if(len(match_ID_dict[key]) >= frame_threshold and process_dict[key][3] == True):
                throw_list.append(key)
        print("Throw List:")
        print(throw_list)

# json_list = {'deviceId': '222222222222222222', 'k8sName': 'jjyconv-dev28-d8f9b8', 'interval': '3', 'dist_threshold': '10', 'direct_threshold': '5', 'frame_threshold': '10', 'frame_pick': '1'}
# start_detect(json_list, path='../data/2.mp4', dist_threshold=10, direct_threshold=5, frame_threshold=10, frame_pick=1)
