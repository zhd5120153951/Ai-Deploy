import os

from ai_common import setting
from ai_common.img_util import *
import requests
from ai_common.exception_util import package_code_exception
from ai_common.log_util import logger
from utils.torch_utils import time_sync


def cap_rtsp(json, job_id):
    try:
        response = requests.get(setting.SERVICE_PATH +
                                f"/rtsp/get/{json['k8sName']}")
        response.content.replace(
            bytes('\\', encoding='utf-8'), bytes('/', encoding='utf-8'))
        logger.debug(
            f"设备{json['k8sName']},请求状态为{response},请求返回内容为{response.content}")
    except Exception as e:
        logger.info(
            f'Job_id{job_id}failed to connect to the rtsp agent service. Send the following exception information to the alarm service{e}.')
        alarm_dic = package_code_exception(
            f'Job_id{job_id}failed to connect to the rtsp agent service, exit the analysis and perform the next check. Exception information is{e}.',
            101, json['deviceId'][0])
        setting.alarm_queue.put(list(alarm_dic))
        raise Exception(
            f'Job_id{job_id}failed to connect to the rtsp agent service. Exit the analysis and perform the next check.')
    else:
        res = response.json()
        if res['code'] == 0:
            res_rtsp = res['data']  # rtsp地址
            print(res_rtsp)
            res_code = res['code']  # 0
            res_msg = res['msg']  # success
            logger.info(
                f'Job_id{job_id}get rtsp successfully. The return message is{res_msg}, Return code is{res_code}.')
            if json['k8sName'] in res_rtsp.keys():  # 该算法配置多个设备时，将获取的rtsp_path与设备进行对映
                capture = cv2.VideoCapture(res_rtsp[json['k8sName']])

                while capture.isOpened():  # 判断是否俘获到rtsp
                    ret, frame = capture.read()
                    if ret:
                        letter_img = letterbox(frame)[0]
                        img_RGB = letter_img[:, :, ::-1].transpose(2, 0, 1)
                        imgresize = np.ascontiguousarray(img_RGB)
                        json_tmp = copy.copy(json)
                        h0w0 = frame.shape[0:2]
                        json_tmp['img_array'] = frame
                        json_tmp['img'] = imgresize
                        json_tmp['h0w0'] = h0w0
                        setting.image_queue.put(json_tmp)
                        setting.NUM[json['k8sName']] += 1  # 统计获取图片张数
                        pass
                    pass
                pass
            pass
        else:
            alarm_dic = package_code_exception(
                f'Job_id{job_id}connect to the rtsp agent service successfully but fails to be obtained. Error code is{response.status_code}. Exit the analysis and perform the next check.',
                response.status_code, json['deviceId'][0])
            setting.alarm_queue.put(list(alarm_dic))
            raise Exception(
                f'Job_id{job_id}failed to get rtsp. Error code is{response.status_code}. Exit the analysis and perform the next check.')


def cap_img(json_list, job_id, k8sName_list, device_list):
    try:
        response = requests.post(
            setting.SERVICE_PATH + '/image_base64/get/list', json=k8sName_list)
    except Exception as e:
        logger.info(
            f'Job_id{job_id}failed to connect to the image agent service. Send the following exception information to the alarm service{e}.')
        alarm_dic = package_code_exception(
            f'Job_id{job_id}failed to connect to the image agent service, exit the analysis and perform the next check. Exception information is{e}.', 101, device_list[0])
        setting.alarm_queue.put(list(alarm_dic))
        raise Exception(
            f'Job_id{job_id}failed to connect to the image agent service. Exit the analysis and perform the next check.')
    else:
        res = response.json()
        if res['code'] == 0:
            res_data = res['data']
            res_code = res['code']
            res_msg = res['msg']
            # 重新组装数据，放入全局队列里
            for json in json_list:
                for key in res_data:
                    if key == json['k8sName']:
                        json['img_str'] = res_data[key]
                setting.image_queue.put(json)
                # 对获取图片张数进行计数
                setting.NUM[json['k8sName']] += 1
                logger.debug(
                    f"设备 {json['k8sName']} 检测图片张数为 {setting.NUM[json['k8sName']]}")
            logger.info(
                f'Job_id{job_id}get images successfully. The return message is{res_msg}, Return code is{res_code}.')
        else:
            alarm_dic = package_code_exception(
                f'Job_id{job_id}connect to the image agent service successfully but fails to be obtained. Error code is{response.status_code}. Exit the analysis and perform the next check.',
                response.status_code, device_list[0])
            setting.alarm_queue.put(list(alarm_dic))
            raise Exception(
                f'Job_id{job_id}failed to get images. Error code is{response.status_code}. Exit the analysis and perform the next check.')


def cap_rtsp_native(json_list, rtsp_ai_name, ai_name):
    capture = cv2.VideoCapture(setting.video_path, cv2.CAP_FFMPEG)
    if ai_name in rtsp_ai_name:
        for json in json_list:
            while capture.isOpened():
                ret, frame = capture.read()
                if ret:
                    letter_img = letterbox(frame)[0]
                    img_RGB = letter_img[:, :, ::-1].transpose(2, 0, 1)
                    imgresize = np.ascontiguousarray(img_RGB)
                    json_tmp = copy.copy(json)
                    h0w0 = frame.shape[0:2]
                    json_tmp['img_array'] = frame
                    json_tmp['img'] = imgresize
                    json_tmp['h0w0'] = h0w0
                    # print("json_tmp: ", json_tmp)
                    setting.image_queue.put(json_tmp)
                else:
                    break
                pass
            pass
        pass
    else:
        for json in json_list:
            while capture.isOpened():
                ret, frame = capture.read()
                if ret:
                    letter_img = letterbox(frame)[0]
                    img_RGB = letter_img[:, :, ::-1].transpose(2, 0, 1)
                    imgresize = np.ascontiguousarray(img_RGB)
                    json_tmp = copy.copy(json)
                    h0w0 = frame.shape[0:2]
                    json_tmp['img'] = imgresize
                    json_tmp['h0w0'] = h0w0
                    json_tmp['img_array'] = frame
                    # print("json_tmp:", json_tmp)
                    setting.image_queue.put(json_tmp)
                else:
                    break
                pass
            pass
        pass
    return capture


def cap_img_native(json_list):
    try:
        # 批量读取
        img_name_list = os.listdir(setting.image_path)
        for json in json_list:
            for img_name in img_name_list:
                img_path = os.path.join(setting.image_path, img_name)
                # 本地读取图片测试
                img = cv2.imread(img_path)
                letter_img = letterbox(img, new_shape=1920)[0]
                img_RGB = letter_img[:, :, ::-1].transpose(2, 0, 1)
                imgresize = np.ascontiguousarray(img_RGB)
                json_tmp = copy.copy(json)
                h0w0 = img.shape[0:2]
                json_tmp['img'] = imgresize
                json_tmp['h0w0'] = h0w0
                json_tmp['img_array'] = img
                setting.image_queue.put(json_tmp)
                pass
            pass
        pass
    except:
        # 单张测试
        for json in json_list:
            img = cv2.imread(setting.image_path)
            letter_img = letterbox(img)[0]
            img_RGB = letter_img[:, :, ::-1].transpose(2, 0, 1)
            imgresize = np.ascontiguousarray(img_RGB)
            json_tmp = copy.copy(json)
            h0w0 = img.shape[0:2]
            json_tmp['img'] = imgresize
            json_tmp['h0w0'] = h0w0
            json_tmp['img_array'] = img
            setting.image_queue.put(json_tmp)
            pass
        pass
    pass


def cap_data(json_list, job_id):
    # 根据算法名称判断读rtsp流还是读图片
    ai_name = os.getcwd().replace('\\', '/').split('/')[-1]
    logger.debug(
        f"----------------------------{ai_name}---------------------------")
    if setting.TEST_FLAG == 0 and setting.REMOTE_DEBUG == 0:
        rtsp_ai_name = ['throwing_detection_v3', 'falldown_detection_v3']
    else:
        rtsp_ai_name = ['throwing_detection', 'falldown_detection']

    if setting.TEST_FLAG == 0:
        k8sName_list = []
        device_list = []
        dic_list = []
        for dic in json_list:
            k8sName_list.append(dic['k8sName'])
            device_list.append(dic['deviceId'])
            dic_list.append(dic)

        # 请求图片服务代理
        logger.info(
            f"Job_id{job_id}start requesting the image service agent:{setting.SERVICE_PATH}. The contained device list is {k8sName_list}.")

        # 读rtsp流
        if ai_name in rtsp_ai_name:
            for json in json_list:
                cap_rtsp(json, job_id)
                pass
            pass
        else:
            cap_img(json_list, job_id, k8sName_list, device_list)
            pass
        pass
    else:
        # 本地测试
        if setting.IMAGE == 1:
            cap_img_native(json_list)
        if setting.VIDEO == 1:
            cap_rtsp_native(json_list, rtsp_ai_name, ai_name)
            pass
        pass
    pass


def capture_data(json_list, job_id):
    t1 = time_sync()
    cap_data(json_list, job_id)
    t2 = time_sync()
    logger.info(f'The first step is time-consuming ({t2 - t1:.3f}s).')
