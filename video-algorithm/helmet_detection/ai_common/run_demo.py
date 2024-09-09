import copy
import os
import cv2

from ai_common import setting
from ai_common.img_capture import cap_rtsp_native
from ai_common.log_util import logger


def run_demo_main(process_data):
    for device_args_dict in setting.device_args_list:
        k8s_Name = device_args_dict['k8sName']
        setting.last_hist_dic[k8s_Name] = None
        # 各算法中需要用到的全局变量
        # --------------------------------------- #
        # 烟雾明火
        setting.flame_img[k8s_Name] = {}
        setting.flame_frame[k8s_Name] = 1
        # --------------------------------------- #
        # 摔倒检测
        setting.frame_t[k8s_Name] = 0
        setting.person[k8s_Name] = {}  # 人
        setting.person_bbxr[k8s_Name] = {}  # 框的比例
        setting.person_frame[k8s_Name] = {}  # 变大持续的帧数
        setting.person_new[k8s_Name] = {}  # 存储新人
        setting.person_newbbx[k8s_Name] = {}  # 存储新人的比例框
        setting.person_linshi[k8s_Name] = {}
        setting.head1[k8s_Name] = []
        setting.head2[k8s_Name] = []
        # ---------------------------------------- #
        # 高空抛物
        setting.frameID[k8s_Name] = 0
        setting.match_ID_dict[k8s_Name] = {}  # 字典，目标ID：目标在每帧的Y轴坐标
        # 字典，目标ID：[上次运动方向，方向曾经是否变化，上次的距离变化量，是否为抛物]
        setting.process_dict[k8s_Name] = {}
        setting.throw_list[k8s_Name] = []  # 抛物ID列表
        # ---------------------------------------- #
        # 车道占用
        setting.background[k8s_Name] = {}
        setting.frame[k8s_Name] = 0
        # 根据算法名称判断读rtsp流还是读图片
        ai_name = os.getcwd().replace('\\', '/').split('/')[-1]
        logger.debug(
            f"----------------------------{ai_name}---------------------------")
        if setting.TEST_FLAG == 0 and setting.REMOTE_DEBUG == 0:
            rtsp_ai_name = ['throwing_detection_v3', 'falldown_detection_v3']
        else:
            rtsp_ai_name = ['throwing_detection', 'falldown_detection']
        capture = cap_rtsp_native(
            setting.device_args_list, rtsp_ai_name, ai_name)
        fps = 25
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        if not os.path.exists(setting.video_save_path):
            os.makedirs(setting.video_save_path)
        out = cv2.VideoWriter(
            f"{setting.video_save_path}/out.mp4", fourcc, fps, size)
        while True:
            if setting.image_queue.qsize() > 0:
                try:
                    img_array_copy = process_data()
                except RuntimeError:
                    continue
                img = copy.deepcopy(img_array_copy)
                if setting.VID_VERIFY == 1:
                    cv2.imshow("video", img)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                if setting.SAVE_VIDEO == 1:
                    out.write(img)
            else:
                print('处理视频完成')
                capture.release()
                out.release()
                break
