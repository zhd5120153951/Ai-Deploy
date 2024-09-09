import ast
import shutil

from ai_common import setting
from ai_common.img_util import *
from ai_common.log_util import logger
from ai_common.result_path_util import result_path
import requests
import os
import time


def send_alarm():
    # 获取图片异常告警
    cap_img_alarm()

    # 检测图片异常告警
    if not os.path.exists(setting.project_result_dir):
        os.makedirs(setting.project_result_dir, exist_ok=True)
    det_class = os.listdir(os.path.abspath(setting.project_result_dir)) # ['flame_gathered_detection', 'leave_sleep_detection', 'overalls_helmet_detection']
    special_name = os.getcwd().replace('\\', '/').split('/')[-1]
    for cla in det_class:
        if cla == special_name:
            alarm_data = result_path(setting.project_result_dir, cla)  # D:/yunbian_3.0_git/results/leave_sleep_detection
            alarm_data_list = os.listdir(result_path(setting.project_result_dir, cla))
            for data in alarm_data_list:
                alarm_data_path = result_path(alarm_data, data) # D:/yunbian_3.0_git/results/leave_sleep_detection/2022-09-28
                alarm_data_txt_unsend = result_path(alarm_data_path, 'txt/unsend') # D:/yunbian_3.0_git/results/leave_sleep_detection/2022-09-28/txt/unsend
                alarm_data_txt_sended = result_path(alarm_data_path, 'txt/sended')  # D:/yunbian_3.0_git/results/leave_sleep_detection/2022-09-28/txt/sended
                if not os.path.exists(alarm_data_txt_unsend):
                    os.makedirs(alarm_data_txt_unsend, exist_ok=True)
                if os.listdir(alarm_data_txt_unsend) == 0:
                    logger.info(f"{data}: No alarm message is available.")
                else:
                    for unsend_alarm in os.listdir(alarm_data_txt_unsend):
                        # ---------------------------------组装发送告警信息--------------------------------- #
                        # 获取告警图片目录与告警原图目录
                        with open(result_path(alarm_data_txt_unsend, unsend_alarm), 'r', encoding='utf-8') as F: # ../results/Y-m-d/txt/unsend/unsend_2022-09-28_09_46_56_leave_22222222222222222.txt
                            alarm_dic = F.read()
                        alarm_dic = ast.literal_eval(alarm_dic)
                        alarmImage_path = alarm_dic['alarmImage']
                        alarmOriginalImage_path = alarm_dic['alarmOriginalImage']

                        if os.path.exists(alarmImage_path) and os.path.exists(alarmOriginalImage_path):
                            # 读取告警原图目录 -> 转换成base64 -> 组装新的告警信息
                            with open(alarmImage_path, 'rb') as f:
                                byte_alarmImage = f.read()
                            base64_alarmImage = base64.b64encode(byte_alarmImage).decode("ascii")
                            alarm_dic['alarmImage'] = base64_alarmImage
                            # 读取告警图片目录 -> 转换成base64 -> 组装新的告警信息
                            with open(alarmOriginalImage_path, 'rb') as f:
                                byte_alarmOriginalImage = f.read()
                            base64_alarmOriginalImage = base64.b64encode(byte_alarmOriginalImage).decode("ascii")
                            alarm_dic['alarmOriginalImage'] = base64_alarmOriginalImage
                        else:
                            shutil.rmtree(result_path(alarm_data_txt_unsend, unsend_alarm))
                        # ------------------------------------------------------------------------------------ #
                        # 请求告警url,发送告警
                        alarm_list = []
                        alarm_list.append(alarm_dic)
                        response = requests.post(setting.ALARM_PATH + '/alarm/algorithm-alarm/add', json=alarm_list)
                        if response.status_code == 200:
                            if not os.path.exists(alarm_data_txt_sended):
                                os.makedirs(alarm_data_txt_sended)
                            # 移动文件txt/unsend至txt/sended
                            shutil.move(alarm_data_txt_unsend + '/' + unsend_alarm, alarm_data_txt_sended)
                            logger.info('Sending alarm seccessfully.')
                        else:
                            logger.info(f'The alarm service return exception information, the HTTP error code is {response.status_code}.')
                            pass
                        pass
                    pass
                pass
            pass
        pass
    pass


def send_imgnum():
    if setting.TEST_FLAG == 0:
        try:
            for devide_code in setting.NUM:
                num_dic = {'detectionTimes': f"{setting.NUM[devide_code]}", 'appId':f"{setting.APP_ID}", 'deviceId':f"{devide_code}"}
                logger.debug(num_dic)
                response = requests.post(setting.SYNC_PATH + '/sync/appDeviceModel/updateDetectionTimes', json=num_dic)
                logger.debug(f"post请求返回结果{response.text}")
                if response.status_code == 200:
                    # 如果发送成功，则清零，否则一直计数。
                    logger.info(f'应用id: {setting.APP_ID}， 设备编码: {devide_code}， Send number of images successfully.Number of {setting.NUM[devide_code]}')
                    setting.NUM[devide_code] = 0
        except Exception as e:
            raise Exception(f"Failed to connect to the detect img_num service. The error message is {e}.")
    else:
        for devide_code in setting.NUM:
            logger.info(f'应用id: {setting.APP_ID}， 设备编码: {devide_code}， send number of images successfully.number of {setting.NUM[devide_code]}')
            setting.NUM[devide_code] = 0
        return 'res_num_data', 0

'''
第二步，生成告警信息保存至本地
'''
def package_business_alarm(alarm_info, alarm_type, img_alarm, img_ori, device_id, alarm_label):
    alarm_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())  # 告警时间
    local_time = time.strftime('%Y-%m-%d', time.localtime(time.time()))
    # 对图片进行保存，将原始图片保存到当前执行的容器中，以及将处理后的图片保存到容后发送到平台
    special_name = os.getcwd().replace('\\', '/').split('/')[-1]
    save_img_data_dir = result_path(setting.project_result_dir, special_name + '/' + local_time) # D:/yunbian_3.0_git/results/leave_sleep_detection/2022-09-28
    save_txt_data_dir = result_path(setting.project_result_dir, special_name + '/' + local_time)
    save_img_dir = result_path(save_img_data_dir, setting.alarm_img_dir) # D:/yunbian_3.0_git/results/leave_sleep_detection/2022-09-28/img
    os.makedirs(save_img_dir, exist_ok=True)
    save_txt_dir = result_path(save_txt_data_dir, setting.alarm_dic_dir)
    save_txt_dir = result_path(save_txt_dir, 'unsend') # D:/yunbian_3.0_git/results/leave_sleep_detection/2022-09-28/txt/unsend
    os.makedirs(save_txt_dir, exist_ok=True)
    # 路径不能有空格，不能有冒号，否则图片无法保存。
    alarm_time_re = str(alarm_time).replace(' ', '_')
    alarm_time_re = alarm_time_re.replace(':', '_')
    # 图片保存到容器内部
    img_alarm_name = save_img_dir + f'/{alarm_time_re}_img_alarm_{alarm_label}_{device_id}.jpg'
    img_ori_name = save_img_dir + f'/{alarm_time_re}_img_ori_{alarm_label}_{device_id}.jpg'
    cv2.imwrite(img_alarm_name, img_alarm) # 保存路径中不能有中文
    cv2.imwrite(img_ori_name, img_ori)
    alarm_dic = {
        # "companyId": setting.COMPANY_ID,
        # "companyName": setting.COMPANY_NAME,
        # "nodeId": setting.NODE_ID,
        # "nodeName": setting.NODE_NAME,
        "deviceId": device_id,
        # "deviceName": device_name,
        "appId": setting.APP_ID,
        # "appName": setting.APP_NAME,
        # "modelId": setting.MODEL_ID,
        # "modelName": setting.MODEL_NAME,
        "alarmTime": alarm_time,
        "alarmInfo": alarm_info,
        "alarmType": alarm_type,
        "type": setting.NORMAL_ALARM,
        "alarmImage": img_alarm_name,
        "alarmOriginalImage": img_ori_name
    }

    # 告警信息以字典形式保存在容器内指定路径的txt文件
    with open(save_txt_dir + f"/{alarm_time_re}_{alarm_label}_{device_id}.txt", 'w', encoding='utf-8') as f:
        f.write(str(alarm_dic))


def cap_img_alarm():
    if setting.TEST_FLAG == 0:
        if setting.alarm_queue.qsize() > 0:
            img_alarm_dic = setting.alarm_queue.get()
            try:
                response = requests.post(setting.ALARM_PATH + '/alarm/algorithm-alarm/add', json=img_alarm_dic)
                if response.status_code == 200:
                    res_data = response.json()['desc']
                    res_code = response.json()['code']
                    logger.info(f"res_data:{res_data}, res_code:{res_code}")
                else:
                    raise Exception(f'The alarm service return exception information, the HTTP error code is {response.status_code}. ')
            except Exception as e:
                raise Exception(f"Failed to connect to the alarm service. The error message is {e}.")
