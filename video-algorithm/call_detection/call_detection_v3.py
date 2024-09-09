# -*- coding: utf-8 -*-
"""Run inference with a YOLOv5 model on images, videos, directories

Usage:
    $ python throwing.py --source path/to/img.jpg --weights yolov5s.pt --img 640

modify：
1. 通过HTTP方式取base64格式的图片或者流，老版本是通过 cap = cv2.VideoCapture(CAMERA_URL)  取RTSP流
    1.1 新增了流转发服务，该服务提供的是HTTP请求，所以要引入HTTP客户端；
    1.2 HTTP客户端对比，最简单和流行的是 Requests，但其不支持异步；
        异步性能最高的是AIOHTTP，但其要使用异步编程方式，且支持异常请求方式；
        HTTPX 支持异步，且只需使用同步的编程方式，与 Requests一样易用。但同步的性能又稍差于 Requests。
    1.3 暂时还不需要一次性把所有设备的图片取回来，现在通过for循环一个个取，后面设备很多取流出现性能问题再说，并且切换为 HTTPX 也不用改请求代码，因为API一致。
    1.4 综上，当前选用：Requests。

2. 通过HTTP方式输出告警图片结果和应用信息，老版本是先保存到容器内的本地路径，再通过MQTT将base64转码后的图片发到平台
    2.1 2.0的架构不在使用基于MQTT的 ecci_client，而是使用HTTP将结果发送至云端告警服务。
    2.2 告警图片仍然是先转base64。
    2.3 告警要区分算法结果的正常告警和算法代码异常告警。
    2.4 同样，在发送告警前，先在本地容器路径保存告警图片，以（时间-项目ID-设备ID）命名方式保存。

3. 调度组件选用 Apscheduler
    3.1 该调度初始化一个后台调度器用于每隔一定时间全量读取 /usr/app/args.json 文件最新内容。
    3.2 再初始化一个前台阻塞调度器，用来执行视频分析主逻辑。
    3.3 定时全量读取json文件有一定的延迟性，鉴于该文件不会改动太频繁，30秒左右同步一次比较合适。如果要及时，后续可引入监听组件 pyinotify。

4. 异常处理策略
    4.1 除了告警服务本身异常无法连接外，其余的代码异常全部转发至告警服务后，如果不影响主逻辑则继续执行后面的代码，否则也往最上层抛出异常，不再执行后续代码。
    4.2 告警服务本身的连接异常，直接一直往最外层抛出，不再执行异常后面的代码。

5. 并行运行作业策略
    5.1 当前一个处理函数是传入一个列表，即一组设备信息；其中每个字典包含一张对应的设备图片及其算法参数。
    5.2 要对不同设备并行处理对应的图片，当前的处理函数run_analysis的入参为单个字典，有几个设备就在Apscheduler中add几个job，用不同的job_id区分。

6. 算法参数提取到json文件中，便于从平台灵活下发，具体分为三类参数：
    6.1. 环境变量，界面不可改变的参数。
        图片获取地址 SERVICE_PATH
        告警输出地址 ALARM_PATH
        应用ID setting.APP_ID

    6.2. 非算法类json参数，界面下发的参数。
        检测频率：interval
        json文件解析频率：FILE_INTERVAL
        deviceId
        k8sName

    6.3. 算法类json参数，界面下发的参数args。
        检测区域框选坐标 hi,wi
        检测人数 people_num
        检测种类 warning_list
        模型路径 weights
        置信度   conf_thres
        重叠度：  iou_thres,
        检测类别： classes
        class-agnostic NMS： agnostic_nms False
        每张图片最大检测个数： max_det

"""
import codecs
from pathlib import Path
import sys

from ai_common.img_queue_get import images_queue_get
from ai_common.run_demo import run_demo_main
from analysis.head import process_result_head
from analysis.process import run_process
from analysis.extract_param import param
from models.load import attempt_load, load_engine
from ai_common.util.general import strip_optimizer
from ai_common.util.torch_utils import select_device, time_synchronized
from analysis.call import process_result_call
from analysis.img_analysis import img_preprocess
from ai_common.parse import load_config
from ai_common.alarm_util import *
from ai_common.exception_util import *

sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())


def run_analysis(data, update=False):
    # 动态参数
    detect_area_flag, hide_labels, hide_conf, polyline, conf_thres_head, conf_thres_call, iou_thres_head, iou_thres_call, line_thickness = param(data)

    # 图片预处理
    pred_call, pred_head, names_call, names_head, img, img_array, degree = img_preprocess(device, data, conf_thres_head, conf_thres_call, iou_thres_head, iou_thres_call, MODEL_head, MODEL_call, engine_call, engine_head)
    if len(pred_call) == 0 and len(pred_head) == 0:
        logger.info(f"设备 {data['k8sName']} 相似帧过滤")
        # 相似帧，或者未获取到图片直接跳过后续动作
        return img_array

    # 人头检测
    head, img_array_copy = process_result_head(pred_head, detect_area_flag, names_head, hide_labels, hide_conf, data, img, img_array)
    if len(head) != 0:
        # 电话检测
        img_array_copy = process_result_call(head, pred_call, names_call, detect_area_flag, hide_labels, hide_conf, line_thickness, data, img, img_array, polyline)

    if update:
        for weights in [weights_call_path, weights_head_path]:
            strip_optimizer(weights)

    return img_array_copy


def process_data():
    t1 = time_synchronized()
    capture_dic = images_queue_get()
    img_array_copy = run_analysis(capture_dic)
    t2 = time_synchronized()
    logger.info(f'设备 {capture_dic["k8sName"]} 推理处理总流程耗时： ({t2 - t1:.3f})s.')

    return img_array_copy


def main():
    # 解析配置
    load_config()
    # 加载模型
    load_model()
    # run demo
    if setting.TEST_FLAG == 1 and setting.VIDEO == 1:
        run_demo_main(process_data)
    else:
        run_process(process_data)


def load_model():
    global device
    global MODEL_call
    global MODEL_head
    global engine_call
    global engine_head

    engine_call, engine_head = load_engine(engine_call_path, engine_head_path)

    MODEL_call = attempt_load(weights_call_path, device)  # load FP32 model
    MODEL_head = attempt_load(weights_head_path, device)



device = select_device('')
weights_call_path = './weights/call.pt'
weights_head_path ='./weights/helmet_head_person_l.pt'
engine_call_path = './weights/call.engine'
engine_head_path = './weights/helmet_head_person_l.engine'
MODEL_call = None
MODEL_head = None
engine_call = None
engine_head = None
FILE = Path(__file__).absolute()  # D:\Code\CloudEdge1.0\leave_detection\test2.0.py
sys.path.append(FILE.parents[0].as_posix())  # 'D:/Code/CloudEdge1.0/leave_detection'


# 主函数
if __name__ == '__main__':
    main()
